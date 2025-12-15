#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
HCTSA DISCOVERY: ¿Qué features capturan realmente H*?
================================================================================

OBJETIVO:
    Descubrir de forma DATA-DRIVEN cuáles features de series temporales
    preceden consistentemente a PAC durante transiciones Sleep->Wake.
    
    NO asumimos qué es H*. Dejamos que los datos revelen qué propiedades
    matemáticas anticipan la emergencia de consciencia.

METODOLOGÍA:
    1. Extraer ~40 features por ventana temporal (catch22 + antropy)
    2. Para cada transición: detectar onset de cada feature vs PAC
    3. Calcular Δt = onset(feature) - onset(PAC) para cada feature
    4. Ranking: ¿Cuáles features preceden PAC más consistentemente?
    5. Clustering: ¿Qué tienen en común las features "ganadoras"?

SALIDA ESPERADA:
    - Ranking de features por % de veces que preceden PAC
    - Identificación de "firma de H*" basada en datos
    - Posible descubrimiento de features inesperadas

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
from collections import Counter
import math

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

print("Cargando librerias...")
from tensorpac import Pac

# catch22
try:
    import pycatch22
    HAS_CATCH22 = True
    print("  pycatch22: OK")
except ImportError:
    HAS_CATCH22 = False
    print("  pycatch22: NO (pip install pycatch22)")

# antropy
try:
    import antropy as ant
    HAS_ANTROPY = True
    print("  antropy: OK")
except ImportError:
    HAS_ANTROPY = False
    print("  antropy: NO (pip install antropy)")

print("")

#==============================================================================
# CONFIG OPTIMIZADA
#==============================================================================

@dataclass
class Config:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./hctsa_discovery_results')
    fs: float = 100.0
    # Ventanas optimizadas para velocidad
    window_before: float = 120.0
    window_after: float = 120.0
    sliding_window: float = 30.0
    sliding_step: float = 10.0
    theta: Tuple[float, float] = (4.0, 8.0)
    gamma: Tuple[float, float] = (30.0, 45.0)
    onset_sd: float = 1.5
    max_subjects: int = None  # None = todos
    
    def __post_init__(self):
        self.wake = ['Sleep stage W']
        self.sleep = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']

#==============================================================================
# FEATURE EXTRACTOR RAPIDO
#==============================================================================

class FastFeatureExtractor:
    """Extrae features HCTSA de forma eficiente"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg.fs
        
        # PAC
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pac = Pac(idpac=(1,0,0), f_pha=list(cfg.theta), 
                          f_amp=list(cfg.gamma), dcomplex='wavelet', 
                          width=7, verbose=False)
        
        # Lista de features que extraeremos
        self.feature_names = ['PAC']
        if HAS_CATCH22:
            self.feature_names.extend([f'c22_{i}' for i in range(24)])  # catch24
        if HAS_ANTROPY:
            self.feature_names.extend([
                'ant_perm_entropy', 'ant_spectral_entropy', 'ant_svd_entropy',
                'ant_sample_entropy', 'ant_hjorth_complexity', 'ant_hjorth_mobility',
                'ant_lziv', 'ant_petrosian_fd', 'ant_katz_fd', 'ant_dfa'
            ])
    
    def preprocess(self, data):
        if len(data) < 10: return data
        sos = signal.butter(4, [0.5, 45.0], btype='bandpass', fs=self.fs, output='sos')
        filt = signal.sosfiltfilt(sos, data)
        return (filt - np.mean(filt)) / (np.std(filt) + 1e-10)
    
    def extract(self, data) -> Dict[str, float]:
        """Extrae todas las features de una ventana"""
        if len(data) < 200:
            return {}
        
        clean = self.preprocess(data)
        features = {}
        
        # 1. PAC
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features['PAC'] = float(self.pac.filterfit(
                    self.fs, clean[np.newaxis,:], clean[np.newaxis,:]
                )[0,0,0])
        except:
            features['PAC'] = np.nan
        
        # 2. catch22/24
        if HAS_CATCH22:
            try:
                c22 = pycatch22.catch22_all(clean.tolist(), catch24=True)
                for name, val in zip(c22['names'], c22['values']):
                    features[f'c22_{name}'] = val if not np.isnan(val) else np.nan
            except:
                pass
        
        # 3. Antropy
        if HAS_ANTROPY:
            try:
                features['ant_perm_entropy'] = ant.perm_entropy(clean, normalize=True)
                features['ant_spectral_entropy'] = ant.spectral_entropy(clean, sf=self.fs, normalize=True)
                features['ant_svd_entropy'] = ant.svd_entropy(clean, normalize=True)
                features['ant_sample_entropy'] = ant.sample_entropy(clean)
                params = ant.hjorth_params(clean)
                features['ant_hjorth_mobility'] = params[0]
                features['ant_hjorth_complexity'] = params[1]
                features['ant_lziv'] = ant.lziv_complexity(clean > np.median(clean), normalize=True)
                features['ant_petrosian_fd'] = ant.petrosian_fd(clean)
                features['ant_katz_fd'] = ant.katz_fd(clean)
                features['ant_dfa'] = ant.detrended_fluctuation(clean)
            except:
                pass
        
        return features

#==============================================================================
# ANALIZADOR
#==============================================================================

def find_transitions(ann, cfg):
    trans = []
    desc, onsets = ann.description, ann.onset
    for i in range(len(desc)-1):
        if desc[i] in cfg.sleep and desc[i+1] in cfg.wake:
            trans.append({'time': onsets[i+1], 'from': desc[i], 'to': desc[i+1]})
    return trans

def get_channel(raw, start_s, end_s):
    for ch in ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']:
        if ch in raw.ch_names:
            data, _ = raw.copy().pick_channels([ch])[:, start_s:end_s]
            return data[0,:]
    return None

def detect_onset(times, values, threshold_sd=1.5):
    """Detecta onset de activacion"""
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 5:
        return None
    
    t = times[valid_mask]
    v = values[valid_mask]
    
    bl_mask = t < -30
    if bl_mask.sum() < 2:
        bl_mask = np.arange(len(v)) < len(v)//3
    
    bl_mean = v[bl_mask].mean()
    bl_std = v[bl_mask].std()
    
    if bl_std < 1e-10:
        return None
    
    thresh = bl_mean + threshold_sd * bl_std
    
    for ti, vi in zip(t[t >= -30], v[t >= -30]):
        if vi > thresh:
            return ti
    return None

def analyze_transition(raw, trans, extractor, cfg):
    """Analiza una transicion extrayendo todas las features"""
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
    
    # Sliding windows
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
        center = (start + win_s//2) / fs
        rel_time = center - trans_in_win
        
        feats = extractor.extract(w)
        feats['time'] = rel_time
        all_features.append(feats)
        times.append(rel_time)
    
    if len(all_features) < 5:
        return None
    
    # Convertir a arrays
    times = np.array(times)
    df = pd.DataFrame(all_features)
    
    # Detectar onset de PAC
    if 'PAC' not in df.columns:
        return None
    
    onset_pac = detect_onset(times, df['PAC'].values, cfg.onset_sd)
    if onset_pac is None:
        return None
    
    # Detectar onset de cada feature y calcular delta
    result = {
        'transition_time': t_trans,
        'onset_PAC': onset_pac
    }
    
    for col in df.columns:
        if col in ['time', 'PAC']:
            continue
        
        onset = detect_onset(times, df[col].values, cfg.onset_sd)
        if onset is not None:
            delta = onset - onset_pac  # Negativo = feature antes que PAC
            result[f'delta_{col}'] = delta
    
    return result

#==============================================================================
# MAIN
#==============================================================================

def main():
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("HCTSA DISCOVERY: ¿Qué features capturan H*?")
    print("="*80)
    print(f"\nFeatures disponibles:")
    print(f"  catch22: {HAS_CATCH22}")
    print(f"  antropy: {HAS_ANTROPY}")
    
    if not HAS_CATCH22 and not HAS_ANTROPY:
        print("\nERROR: Necesitas al menos una libreria de features")
        print("  pip install pycatch22 antropy")
        return None, None
    
    # Inicializar
    print("\nInicializando extractor de features...")
    extractor = FastFeatureExtractor(cfg)
    print(f"  {len(extractor.feature_names)} features configuradas")
    
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
    
    if cfg.max_subjects:
        subjects = subjects[:cfg.max_subjects]
    
    print(f"Sujetos: {len(subjects)}")
    
    # Procesar
    all_results = []
    n_subj = 0
    
    for i, subj in enumerate(subjects):
        print(f"[{i+1}/{len(subjects)}] {subj['id']}...", end=" ", flush=True)
        
        try:
            raw = mne.io.read_raw_edf(subj['psg'], preload=True, verbose=False)
            ann = mne.read_annotations(subj['hypno'])
            
            transitions = find_transitions(ann, cfg)
            if not transitions:
                print("sin transiciones")
                continue
            
            subj_results = []
            for trans in transitions:
                result = analyze_transition(raw, trans, extractor, cfg)
                if result:
                    result['subject_id'] = subj['id']
                    subj_results.append(result)
            
            if subj_results:
                n_subj += 1
                all_results.extend(subj_results)
                print(f"{len(subj_results)} OK")
            else:
                print("sin datos")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS: {len(all_results)} transiciones de {n_subj} sujetos")
    print("="*80)
    
    if not all_results:
        print("ERROR: Sin resultados")
        return None, None
    
    # Compilar
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.output_dir / 'raw_results.csv', index=False)
    
    # Analizar cada feature
    print("\n" + "="*80)
    print("RANKING DE FEATURES (¿Cuáles preceden a PAC?)")
    print("="*80)
    
    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    
    rankings = []
    for col in delta_cols:
        feat_name = col.replace('delta_', '')
        valid = df[col].dropna()
        
        if len(valid) < 10:
            continue
        
        n_total = len(valid)
        n_before = (valid < 0).sum()  # Feature ANTES de PAC
        n_after = (valid > 0).sum()
        pct_before = 100 * n_before / n_total
        mean_delta = valid.mean()
        
        # Test estadistico
        if n_before + n_after > 0:
            p_val = binomtest(n_before, n_before + n_after, 0.5, alternative='greater').pvalue
        else:
            p_val = np.nan
        
        rankings.append({
            'feature': feat_name,
            'n_valid': n_total,
            'n_before_PAC': n_before,
            'pct_before_PAC': pct_before,
            'mean_delta_sec': mean_delta,
            'p_value': p_val,
            'significant': p_val < 0.05 if not np.isnan(p_val) else False
        })
    
    # Ordenar por % antes de PAC
    rankings_df = pd.DataFrame(rankings).sort_values('pct_before_PAC', ascending=False)
    rankings_df.to_csv(cfg.output_dir / 'feature_rankings.csv', index=False)
    
    # Mostrar top features
    print("\n[TOP 15 FEATURES QUE PRECEDEN A PAC (candidatos a H*)]")
    print("-"*80)
    print(f"{'Feature':<45} {'%Before':>8} {'Delta':>8} {'p-value':>10} {'Sig':>5}")
    print("-"*80)
    
    for _, row in rankings_df.head(15).iterrows():
        sig = "*" if row['significant'] else ""
        print(f"{row['feature'][:44]:<45} {row['pct_before_PAC']:>7.1f}% "
              f"{row['mean_delta_sec']:>+7.1f}s {row['p_value']:>10.4f} {sig:>5}")
    
    # Bottom features (siguen a PAC)
    print("\n[BOTTOM 10 FEATURES QUE SIGUEN A PAC]")
    print("-"*80)
    
    for _, row in rankings_df.tail(10).iterrows():
        print(f"{row['feature'][:44]:<45} {row['pct_before_PAC']:>7.1f}% "
              f"{row['mean_delta_sec']:>+7.1f}s")
    
    # Analisis de patrones
    print("\n" + "="*80)
    print("DESCUBRIMIENTOS")
    print("="*80)
    
    # Features significativas que preceden PAC
    sig_before = rankings_df[(rankings_df['significant']) & (rankings_df['pct_before_PAC'] > 60)]
    
    print(f"\nFeatures que CONSISTENTEMENTE preceden PAC (p<0.05, >60%):")
    if len(sig_before) > 0:
        for _, row in sig_before.iterrows():
            print(f"  - {row['feature']}: {row['pct_before_PAC']:.1f}% (Δ={row['mean_delta_sec']:+.1f}s)")
        
        # Buscar patrones
        print(f"\n  PATRON DETECTADO:")
        
        entropy_feats = [f for f in sig_before['feature'] if 'entropy' in f.lower()]
        complexity_feats = [f for f in sig_before['feature'] if 'complex' in f.lower() or 'lziv' in f.lower()]
        autocorr_feats = [f for f in sig_before['feature'] if 'auto' in f.lower() or 'ac' in f.lower()]
        
        if entropy_feats:
            print(f"    - Entropías: {entropy_feats}")
        if complexity_feats:
            print(f"    - Complejidad: {complexity_feats}")
        if autocorr_feats:
            print(f"    - Autocorrelación: {autocorr_feats}")
    else:
        print("  Ninguna feature significativa encontrada")
    
    # Features que siguen a PAC
    sig_after = rankings_df[(rankings_df['pct_before_PAC'] < 40)]
    
    print(f"\nFeatures que SIGUEN a PAC (potenciales consecuencias):")
    if len(sig_after) > 0:
        for _, row in sig_after.head(5).iterrows():
            print(f"  - {row['feature']}: {row['pct_before_PAC']:.1f}%")
    
    # Interpretacion
    print("\n" + "="*80)
    print("INTERPRETACION PARA AFH")
    print("="*80)
    
    if len(sig_before) >= 3:
        print("""
    HALLAZGO: Múltiples features de complejidad/entropía preceden
    consistentemente a PAC durante transiciones Sleep→Wake.
    
    IMPLICACION PARA H*:
    Esto sugiere que H* (coordinación organizacional) puede ser
    capturado por estas propiedades matemáticas. Las features
    "ganadoras" definen operacionalmente qué ES H*.
    
    SIGUIENTE PASO:
    Usar estas features específicas como proxy de H* en:
    - Predicción de despertar
    - Índice de profundidad de consciencia
    - Validación cruzada con otros datasets
        """)
    elif len(rankings_df[rankings_df['pct_before_PAC'] > 50]) > 0:
        print("""
    HALLAZGO: Algunas features preceden PAC pero sin significancia
    estadística fuerte.
    
    POSIBLES CAUSAS:
    - Muestra insuficiente
    - H* es más complejo que features individuales
    - Necesita combinación de features
        """)
    else:
        print("""
    HALLAZGO: Ninguna feature precede consistentemente a PAC.
    
    IMPLICACIONES:
    - La arquitectura H*→PAC podría no existir
    - O las features HCTSA no capturan H*
    - Considerar otras operacionalizaciones
        """)
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': n_subj,
        'n_transitions': len(all_results),
        'n_features_analyzed': len(rankings_df),
        'features_before_PAC_significant': sig_before['feature'].tolist() if len(sig_before) > 0 else [],
        'top_10_features': rankings_df.head(10)[['feature', 'pct_before_PAC', 'p_value']].to_dict('records')
    }
    
    with open(cfg.output_dir / 'DISCOVERY_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OUTPUT] {cfg.output_dir}")
    
    return df, rankings_df


if __name__ == "__main__":
    try:
        df, rankings = main()
    except KeyboardInterrupt:
        print("\n[CANCELADO]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()