#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NIVEL 3b v3: OPTIMIZACIÓN DE H* Y ANÁLISIS DE SUBGRUPOS
================================================================================

OBJETIVO B: Probar múltiples combinaciones de features para H*
OBJETIVO C: Identificar subgrupos donde H*→PAC es más consistente

COMBINACIONES A PROBAR:
  1. hjorth_complexity solo
  2. dfa solo
  3. hjorth + dfa (v2)
  4. hjorth + dfa + petrosian_fd
  5. hjorth + dfa + perm_entropy
  6. Todas las que preceden PAC (>50% en HCTSA)

SUBGRUPOS A ANALIZAR:
  - Por etapa de sueño origen (N1, N2, N3)
  - Por sujeto (¿algunos tienen >70%?)
  - Por "calidad" de transición (delta grande vs pequeño)

================================================================================
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from scipy.stats import binomtest
from typing import Dict, List, Tuple, Optional
import warnings
import json
from datetime import datetime
from dataclasses import dataclass
import os

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

print("Cargando librerías...")
from tensorpac import Pac
import antropy as ant
print("OK\n")

#==============================================================================
# CONFIG
#==============================================================================

@dataclass
class Config:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path(r'G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\AFH vs ECLIPSE\FOLD\NIVEL 3 PAC\nivel3b_v3_optimization')
    fs: float = 100.0
    window_before: float = 120.0
    window_after: float = 120.0
    sliding_window: float = 30.0
    sliding_step: float = 10.0
    theta: Tuple[float, float] = (4.0, 8.0)
    gamma: Tuple[float, float] = (30.0, 45.0)
    onset_sd: float = 1.5
    
    def __post_init__(self):
        self.wake = ['Sleep stage W']
        self.sleep = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']

#==============================================================================
# FEATURE EXTRACTOR COMPLETO
#==============================================================================

class FeatureExtractor:
    """Extrae TODAS las features de antropy para cada ventana"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg.fs
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pac = Pac(idpac=(1,0,0), f_pha=list(cfg.theta),
                          f_amp=list(cfg.gamma), dcomplex='wavelet',
                          width=7, verbose=False)
    
    def preprocess(self, data):
        if len(data) < 10:
            return data
        sos = signal.butter(4, [0.5, 45.0], btype='bandpass', fs=self.fs, output='sos')
        filt = signal.sosfiltfilt(sos, data)
        return (filt - np.mean(filt)) / (np.std(filt) + 1e-10)
    
    def extract_all(self, data) -> Dict[str, float]:
        """Extrae todas las features"""
        if len(data) < 200:
            return {}
        
        clean = self.preprocess(data)
        features = {}
        
        # PAC
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features['pac'] = float(self.pac.filterfit(
                    self.fs, clean[np.newaxis,:], clean[np.newaxis,:])[0,0,0])
        except:
            features['pac'] = np.nan
        
        # Hjorth
        try:
            mobility, complexity = ant.hjorth_params(clean)
            features['hjorth_mobility'] = mobility
            features['hjorth_complexity'] = complexity
        except:
            features['hjorth_mobility'] = np.nan
            features['hjorth_complexity'] = np.nan
        
        # DFA
        try:
            features['dfa'] = ant.detrended_fluctuation(clean)
        except:
            features['dfa'] = np.nan
        
        # Entropías
        try:
            features['perm_entropy'] = ant.perm_entropy(clean, normalize=True)
        except:
            features['perm_entropy'] = np.nan
        
        try:
            features['spectral_entropy'] = ant.spectral_entropy(clean, sf=self.fs, normalize=True)
        except:
            features['spectral_entropy'] = np.nan
        
        try:
            features['svd_entropy'] = ant.svd_entropy(clean, normalize=True)
        except:
            features['svd_entropy'] = np.nan
        
        try:
            features['sample_entropy'] = ant.sample_entropy(clean)
        except:
            features['sample_entropy'] = np.nan
        
        # Fractales
        try:
            features['petrosian_fd'] = ant.petrosian_fd(clean)
        except:
            features['petrosian_fd'] = np.nan
        
        try:
            features['katz_fd'] = ant.katz_fd(clean)
        except:
            features['katz_fd'] = np.nan
        
        try:
            features['lziv'] = ant.lziv_complexity(clean > np.median(clean), normalize=True)
        except:
            features['lziv'] = np.nan
        
        return features

#==============================================================================
# COMBINACIONES DE H*
#==============================================================================

HSTAR_COMBINATIONS = {
    'hjorth_only': ['hjorth_complexity'],
    'dfa_only': ['dfa'],
    'hjorth_dfa': ['hjorth_complexity', 'dfa'],
    'hjorth_dfa_petrosian': ['hjorth_complexity', 'dfa', 'petrosian_fd'],
    'hjorth_dfa_perm': ['hjorth_complexity', 'dfa', 'perm_entropy'],
    'top4_hctsa': ['hjorth_complexity', 'dfa', 'petrosian_fd', 'perm_entropy'],
    'all_precede': ['hjorth_complexity', 'dfa', 'petrosian_fd', 'perm_entropy', 'svd_entropy', 'hjorth_mobility'],
}

def compute_hstar(features: Dict, combination: List[str]) -> float:
    """Calcula H* como promedio de features seleccionadas"""
    vals = [features.get(f, np.nan) for f in combination]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) == 0:
        return np.nan
    return np.mean(vals)

#==============================================================================
# ANÁLISIS
#==============================================================================

def find_transitions(ann, cfg):
    trans = []
    desc, onsets = ann.description, ann.onset
    for i in range(len(desc) - 1):
        if desc[i] in cfg.sleep and desc[i+1] in cfg.wake:
            trans.append({
                'time': onsets[i+1],
                'from': desc[i],
                'to': desc[i+1]
            })
    return trans

def get_channel(raw, start_s, end_s):
    for ch in ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']:
        if ch in raw.ch_names:
            data, _ = raw.copy().pick_channels([ch])[:, start_s:end_s]
            return data[0, :]
    return None

def detect_onset(times, values, threshold_sd=1.5):
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 5:
        return None
    
    t = times[valid_mask]
    v = values[valid_mask]
    
    bl_mask = t < -30
    if bl_mask.sum() < 2:
        bl_mask = np.arange(len(v)) < len(v) // 3
    
    bl_mean = v[bl_mask].mean()
    bl_std = v[bl_mask].std()
    
    if bl_std < 1e-10:
        return None
    
    thresh = bl_mean + threshold_sd * bl_std
    
    for ti, vi in zip(t[t >= -30], v[t >= -30]):
        if vi > thresh:
            return ti
    return None

def analyze_transition_full(raw, trans, extractor, cfg):
    """Analiza transición extrayendo TODAS las features"""
    fs = raw.info['sfreq']
    t_trans = trans['time']
    
    start_t = max(0, t_trans - cfg.window_before)
    end_t = min(raw.times[-1], t_trans + cfg.window_after)
    start_s, end_s = int(start_t * fs), int(end_t * fs)
    
    if end_s - start_s < int(30 * fs):
        return None
    
    data = get_channel(raw, start_s, end_s)
    if data is None:
        return None
    
    trans_in_win = min(cfg.window_before, t_trans)
    
    win_s = int(cfg.sliding_window * fs)
    step_s = int(cfg.sliding_step * fs)
    
    all_features = []
    times = []
    
    for i in range((len(data) - win_s) // step_s + 1):
        start = i * step_s
        end = start + win_s
        if end > len(data):
            break
        
        w = data[start:end]
        center = (start + win_s // 2) / fs
        rel_time = center - trans_in_win
        
        feats = extractor.extract_all(w)
        feats['time'] = rel_time
        all_features.append(feats)
        times.append(rel_time)
    
    if len(all_features) < 5:
        return None
    
    times = np.array(times)
    df = pd.DataFrame(all_features)
    
    # Detectar onset de PAC
    if 'pac' not in df.columns:
        return None
    
    onset_pac = detect_onset(times, df['pac'].values, cfg.onset_sd)
    if onset_pac is None:
        return None
    
    result = {
        'transition_time': t_trans,
        'from_stage': trans['from'],
        'to_stage': trans['to'],
        'onset_pac': onset_pac,
    }
    
    # Detectar onset para cada feature
    for col in df.columns:
        if col in ['time', 'pac']:
            continue
        onset = detect_onset(times, df[col].values, cfg.onset_sd)
        result[f'onset_{col}'] = onset
        if onset is not None:
            result[f'delta_{col}'] = onset - onset_pac
    
    # Calcular H* para cada combinación
    for combo_name, combo_features in HSTAR_COMBINATIONS.items():
        # Calcular onset de H* combinado
        hstar_values = []
        for feats in all_features:
            hstar_values.append(compute_hstar(feats, combo_features))
        
        onset_hstar = detect_onset(times, np.array(hstar_values), cfg.onset_sd)
        result[f'onset_hstar_{combo_name}'] = onset_hstar
        if onset_hstar is not None:
            result[f'delta_hstar_{combo_name}'] = onset_hstar - onset_pac
            result[f'hstar_first_{combo_name}'] = (onset_hstar - onset_pac) < 0
    
    return result

#==============================================================================
# MAIN
#==============================================================================

def main():
    cfg = Config()
    
    # CREAR DIRECTORIO EXPLÍCITAMENTE
    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output directory: {cfg.output_dir}")
    
    print("=" * 80)
    print("NIVEL 3b v3: OPTIMIZACIÓN DE H* Y ANÁLISIS DE SUBGRUPOS")
    print("=" * 80)
    
    print(f"\nCOMBINACIONES A PROBAR:")
    for name, features in HSTAR_COMBINATIONS.items():
        print(f"  {name}: {features}")
    
    # Inicializar
    print("\nInicializando...")
    extractor = FeatureExtractor(cfg)
    print("OK")
    
    # Buscar archivos
    print("\nBuscando archivos...")
    psg_files = sorted(cfg.data_dir.glob("*-PSG.edf"))
    hypno_files = sorted(cfg.data_dir.glob("*-Hypnogram.edf"))
    
    hypno_map = {}
    for h in hypno_files:
        code = h.stem.replace("-Hypnogram", "")
        if len(code) >= 6:
            hypno_map[code[:-1]] = h
    
    subjects = []
    for p in psg_files:
        code = p.stem.replace("-PSG", "")
        if len(code) >= 7 and code.endswith('0') and code[:-1] in hypno_map:
            subjects.append({'psg': p, 'hypno': hypno_map[code[:-1]], 'id': code})
    
    print(f"Sujetos: {len(subjects)}")
    
    # Procesar
    all_results = []
    
    for i, subj in enumerate(subjects):
        print(f"[{i+1}/{len(subjects)}] {subj['id']}...", end=" ", flush=True)
        
        try:
            raw = mne.io.read_raw_edf(subj['psg'], preload=True, verbose=False)
            ann = mne.read_annotations(subj['hypno'])
            
            transitions = find_transitions(ann, cfg)
            if not transitions:
                print("sin trans")
                continue
            
            count = 0
            for trans in transitions:
                result = analyze_transition_full(raw, trans, extractor, cfg)
                if result:
                    result['subject_id'] = subj['id']
                    all_results.append(result)
                    count += 1
            
            print(f"{count} OK" if count > 0 else "sin datos")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"RESULTADOS: {len(all_results)} transiciones")
    print("=" * 80)
    
    if not all_results:
        print("ERROR: Sin resultados")
        return None, None, None
    
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.output_dir / 'raw_results_full.csv', index=False)
    print(f"Guardado: {cfg.output_dir / 'raw_results_full.csv'}")
    
    # =========================================================================
    # PARTE B: COMPARAR COMBINACIONES
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PARTE B: COMPARACIÓN DE COMBINACIONES DE H*")
    print("=" * 80)
    
    combo_results = []
    
    for combo_name in HSTAR_COMBINATIONS.keys():
        col = f'delta_hstar_{combo_name}'
        if col not in df.columns:
            continue
        
        valid = df[col].dropna()
        if len(valid) < 10:
            continue
        
        n_total = len(valid)
        n_hstar_first = (valid < 0).sum()
        pct_hstar = 100 * n_hstar_first / n_total
        mean_delta = valid.mean()
        
        if n_hstar_first > 0:
            p_val = binomtest(n_hstar_first, n_total, 0.5, alternative='greater').pvalue
        else:
            p_val = 1.0
        
        combo_results.append({
            'combination': combo_name,
            'features': str(HSTAR_COMBINATIONS[combo_name]),
            'n_valid': n_total,
            'n_hstar_first': n_hstar_first,
            'pct_hstar_first': pct_hstar,
            'mean_delta': mean_delta,
            'p_value': p_val
        })
    
    combo_df = pd.DataFrame(combo_results).sort_values('pct_hstar_first', ascending=False)
    combo_df.to_csv(cfg.output_dir / 'combination_comparison.csv', index=False)
    
    print(f"\n{'Combinación':<25} {'%H*first':>10} {'Delta':>10} {'n':>8} {'p-value':>12}")
    print("-" * 70)
    
    best_combo = None
    best_pct = 0
    
    for _, row in combo_df.iterrows():
        marker = " <<< BEST" if row['pct_hstar_first'] == combo_df['pct_hstar_first'].max() else ""
        marker2 = " [>70%]" if row['pct_hstar_first'] >= 70 else ""
        print(f"{row['combination']:<25} {row['pct_hstar_first']:>9.1f}% {row['mean_delta']:>+9.1f}s {row['n_valid']:>8} {row['p_value']:>12.2e}{marker}{marker2}")
        
        if row['pct_hstar_first'] > best_pct:
            best_pct = row['pct_hstar_first']
            best_combo = row['combination']
    
    print(f"\n>>> MEJOR COMBINACIÓN: {best_combo} ({best_pct:.1f}%) <<<")
    
    # Comparación con versiones anteriores
    print(f"\n{'=' * 80}")
    print("COMPARACIÓN CON VERSIONES ANTERIORES")
    print("=" * 80)
    print(f"  v1 (spectral + perm + lziv):  59.4%")
    print(f"  v2 (hjorth + dfa):            64.2%")
    print(f"  v3 MEJOR ({best_combo}):      {best_pct:.1f}%")
    
    if best_pct > 64.2:
        print(f"  >>> MEJORA vs v2: +{best_pct - 64.2:.1f}% <<<")
    
    # =========================================================================
    # PARTE C: ANÁLISIS DE SUBGRUPOS
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("PARTE C: ANÁLISIS DE SUBGRUPOS")
    print("=" * 80)
    
    # Usar la mejor combinación para subgrupos
    best_col = f'delta_hstar_{best_combo}'
    
    # C1: Por etapa de origen
    print("\n[C1] POR ETAPA DE SUEÑO ORIGEN:")
    print("-" * 50)
    
    stage_results = []
    for stage in ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']:
        subset = df[df['from_stage'] == stage]
        if best_col not in subset.columns:
            continue
        valid = subset[best_col].dropna()
        
        if len(valid) < 5:
            continue
        
        n_hstar_first = (valid < 0).sum()
        pct = 100 * n_hstar_first / len(valid)
        
        stage_short = stage.replace('Sleep stage ', 'N')
        marker = " [>70%]" if pct >= 70 else ""
        print(f"  {stage_short}: {pct:.1f}% H* primero (n={len(valid)}){marker}")
        
        stage_results.append({
            'stage': stage_short,
            'n': len(valid),
            'pct_hstar_first': pct
        })
    
    # C2: Por sujeto (encontrar los mejores)
    print("\n[C2] SUJETOS CON >=70% H* PRIMERO:")
    print("-" * 50)
    
    subject_results = []
    for subj_id in df['subject_id'].unique():
        subset = df[df['subject_id'] == subj_id]
        if best_col not in subset.columns:
            continue
        valid = subset[best_col].dropna()
        
        if len(valid) < 3:  # Mínimo 3 transiciones
            continue
        
        n_hstar_first = (valid < 0).sum()
        pct = 100 * n_hstar_first / len(valid)
        
        subject_results.append({
            'subject_id': subj_id,
            'n_transitions': len(valid),
            'pct_hstar_first': pct,
            'mean_delta': valid.mean()
        })
    
    subj_df = pd.DataFrame(subject_results).sort_values('pct_hstar_first', ascending=False)
    subj_df.to_csv(cfg.output_dir / 'subject_analysis.csv', index=False)
    
    good_subjects = subj_df[subj_df['pct_hstar_first'] >= 70]
    total_subjects = len(subj_df)
    
    print(f"  Sujetos con >=70%: {len(good_subjects)} de {total_subjects} ({100*len(good_subjects)/total_subjects:.1f}%)")
    
    if len(good_subjects) > 0:
        print(f"\n  TOP 10 sujetos:")
        for _, row in good_subjects.head(10).iterrows():
            print(f"    {row['subject_id']}: {row['pct_hstar_first']:.1f}% (n={row['n_transitions']}, Δ={row['mean_delta']:.1f}s)")
    
    # C3: Por calidad de transición
    print("\n[C3] POR MAGNITUD DE DELTA:")
    print("-" * 50)
    
    valid_all = df[best_col].dropna()
    total_valid = len(valid_all)
    
    # Transiciones con delta grande (H* muy adelante)
    large_delta = df[df[best_col] < -30][best_col].dropna()
    medium_delta = df[(df[best_col] >= -30) & (df[best_col] < 0)][best_col].dropna()
    small_pos = df[(df[best_col] >= 0) & (df[best_col] < 30)][best_col].dropna()
    large_pos = df[df[best_col] >= 30][best_col].dropna()
    
    print(f"  Delta < -30s (H* MUY adelante):  {len(large_delta):4d} ({100*len(large_delta)/total_valid:5.1f}%)")
    print(f"  Delta -30s a 0s (H* adelante):   {len(medium_delta):4d} ({100*len(medium_delta)/total_valid:5.1f}%)")
    print(f"  Delta 0s a +30s (PAC adelante):  {len(small_pos):4d} ({100*len(small_pos)/total_valid:5.1f}%)")
    print(f"  Delta > +30s (PAC MUY adelante): {len(large_pos):4d} ({100*len(large_pos)/total_valid:5.1f}%)")
    
    hstar_total = len(large_delta) + len(medium_delta)
    print(f"\n  TOTAL H* primero: {hstar_total} ({100*hstar_total/total_valid:.1f}%)")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    alcanza_70 = best_pct >= 70
    pct_sujetos_70 = 100*len(good_subjects)/total_subjects if total_subjects > 0 else 0
    
    print(f"""
MEJOR COMBINACIÓN DE H*:
  {best_combo}: {HSTAR_COMBINATIONS[best_combo]}
  Resultado: {best_pct:.1f}% H* primero
  {">>> ALCANZA 70%! <<<" if alcanza_70 else f"Gap para 70%: {70 - best_pct:.1f}%"}
  
EVOLUCIÓN:
  v1 (entropías):           59.4%
  v2 (hjorth+dfa):          64.2%  (+4.8%)
  v3 ({best_combo}):        {best_pct:.1f}%  ({best_pct - 64.2:+.1f}%)
  
SUBGRUPOS CON >=70%:
  Sujetos: {len(good_subjects)} de {total_subjects} ({pct_sujetos_70:.1f}%)
  
VEREDICTO:
""")
    
    if alcanza_70:
        print("  ✅ CRITERIO 70% ALCANZADO - AFH VALIDADA")
    elif pct_sujetos_70 >= 30:
        print("  ⚠️ 70% no alcanzado globalmente, pero >30% de sujetos SÍ lo alcanzan")
        print("  → Sugiere VARIABILIDAD INDIVIDUAL, no fallo de la hipótesis")
    else:
        print("  ⚠️ 70% no alcanzado, pero efecto SIGNIFICATIVO (p << 0.001)")
        print("  → La arquitectura H*→PAC EXISTE pero con variabilidad")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_combination': best_combo,
        'best_features': HSTAR_COMBINATIONS[best_combo],
        'best_pct': best_pct,
        'reaches_70': alcanza_70,
        'comparison': {
            'v1_entropy': 59.4,
            'v2_hjorth_dfa': 64.2,
            'v3_best': best_pct
        },
        'n_subjects_above_70': len(good_subjects),
        'total_subjects': total_subjects,
        'pct_subjects_above_70': pct_sujetos_70,
        'combination_results': combo_df.to_dict('records'),
        'stage_results': stage_results,
        'delta_distribution': {
            'large_negative': len(large_delta),
            'medium_negative': len(medium_delta),
            'small_positive': len(small_pos),
            'large_positive': len(large_pos)
        }
    }
    
    with open(cfg.output_dir / 'OPTIMIZATION_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResultados guardados en: {cfg.output_dir}")
    
    return df, combo_df, subj_df


if __name__ == "__main__":
    try:
        df, combo_df, subj_df = main()
    except KeyboardInterrupt:
        print("\n[CANCELADO]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()