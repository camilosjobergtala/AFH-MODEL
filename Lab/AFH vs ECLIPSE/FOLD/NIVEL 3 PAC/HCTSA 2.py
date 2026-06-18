#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NIVEL 3b v2: H* -> PAC PRECEDENCE (Sleep -> Wake)
NUEVOS PROXIES DE H* BASADOS EN HCTSA DISCOVERY
================================================================================

DESCUBRIMIENTO PREVIO:
    HCTSA Discovery reveló que los mejores proxies de H* son:
    - Hjorth Complexity (55.3% precede PAC)
    - DFA (53.0% precede PAC)
    
    Los proxies anteriores (spectral_entropy, lziv) NO eran óptimos:
    - spectral_entropy: 33.6% (SIGUE a PAC)
    - lziv: 36.4% (~empate)

HIPÓTESIS:
    Durante transiciones Sleep->Wake, H* se ACTIVA ANTES que PAC.
    
NUEVO PROXY H*:
    H* = mean(hjorth_complexity, dfa)
    
PREDICCIÓN:
    Delta_t = onset(H*) - onset(PAC) < 0 en >=70% de transiciones

================================================================================
Author: Camilo Sjoberg Tala
Date: 2025-12-13
Version: 2.0 (HCTSA-optimized)
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
import hashlib

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

print("Cargando tensorpac...")
from tensorpac import Pac

# Antropy para nuevas features
try:
    import antropy as ant
    HAS_ANTROPY = True
    print("antropy: OK")
except ImportError:
    HAS_ANTROPY = False
    print("ERROR: antropy requerido. pip install antropy")
    exit(1)

print("")

#==============================================================================
# CRITERIOS PRE-REGISTRADOS (ECLIPSE v3.0)
#==============================================================================

CRITERIA = [
    {'name': 'mean_delta_negative', 'threshold': 0.0, 'op': '<', 'required': True},
    {'name': 'pct_hstar_first', 'threshold': 70.0, 'op': '>=', 'required': True},
    {'name': 'p_value', 'threshold': 0.01, 'op': '<', 'required': True},
    {'name': 'pct_pac_first', 'threshold': 50.0, 'op': '<', 'required': True},
]

CRITERIA_HASH = hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()

#==============================================================================
# CONFIG
#==============================================================================

@dataclass
class Config:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./nivel3b_v2_results')
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
# METRICS CALCULATOR (NUEVOS PROXIES)
#==============================================================================

class MetricsCalculator:
    """Calcula H* con NUEVOS proxies optimizados por HCTSA Discovery"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg.fs
        
        # PAC calculator
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
    
    def compute_pac(self, data):
        """PAC theta-gamma (proxy de Fold)"""
        if len(data) < int(2 * self.fs):
            return np.nan
        clean = self.preprocess(data)
        if np.ptp(clean) > 8.0:
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(self.pac.filterfit(self.fs, clean[np.newaxis,:], clean[np.newaxis,:])[0,0,0])
        except:
            return np.nan
    
    def compute_hjorth_complexity(self, data):
        """
        Hjorth Complexity - MEJOR PROXY DE H* (55.3% en HCTSA Discovery)
        
        Mide el cambio en la "forma" de la señal.
        Alto = señal más compleja/organizada.
        """
        if len(data) < 100:
            return np.nan
        clean = self.preprocess(data)
        try:
            mobility, complexity = ant.hjorth_params(clean)
            return complexity
        except:
            return np.nan
    
    def compute_dfa(self, data):
        """
        Detrended Fluctuation Analysis - SEGUNDO MEJOR PROXY (53.0%)
        
        Mide correlaciones de largo alcance en la señal.
        Captura integración temporal.
        """
        if len(data) < 100:
            return np.nan
        clean = self.preprocess(data)
        try:
            return ant.detrended_fluctuation(clean)
        except:
            return np.nan
    
    def compute_hstar(self, data):
        """
        H* OPTIMIZADO = mean(hjorth_complexity, dfa)
        
        Basado en HCTSA Discovery que mostró que estas features
        son las que mejor preceden a PAC.
        """
        hjorth = self.compute_hjorth_complexity(data)
        dfa = self.compute_dfa(data)
        
        vals = [v for v in [hjorth, dfa] if not np.isnan(v)]
        if len(vals) == 0:
            return np.nan
        return np.mean(vals)
    
    def compute_all(self, data):
        """Computa todas las métricas para una ventana"""
        return {
            'pac': self.compute_pac(data),
            'hjorth_complexity': self.compute_hjorth_complexity(data),
            'dfa': self.compute_dfa(data),
            'hstar': self.compute_hstar(data)
        }

#==============================================================================
# ANALYZER
#==============================================================================

def find_transitions(ann, cfg):
    """Encuentra transiciones Sleep -> Wake"""
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
    """Extrae canal EEG"""
    for ch in ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']:
        if ch in raw.ch_names:
            data, _ = raw.copy().pick_channels([ch])[:, start_s:end_s]
            return data[0, :]
    return None

def detect_onset(times, values, threshold_sd=1.5):
    """Detecta onset de activación"""
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 5:
        return None
    
    t = times[valid_mask]
    v = values[valid_mask]
    
    # Baseline: antes de -30s
    bl_mask = t < -30
    if bl_mask.sum() < 2:
        bl_mask = np.arange(len(v)) < len(v) // 3
    
    bl_mean = v[bl_mask].mean()
    bl_std = v[bl_mask].std()
    
    if bl_std < 1e-10:
        return None
    
    thresh = bl_mean + threshold_sd * bl_std
    
    # Buscar primer cruce después de -30s
    for ti, vi in zip(t[t >= -30], v[t >= -30]):
        if vi > thresh:
            return ti
    return None

def analyze_transition(raw, trans, metrics, cfg):
    """Analiza una transición Sleep -> Wake"""
    fs = raw.info['sfreq']
    t_trans = trans['time']
    
    # Extraer ventana
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
    
    times = []
    pac_vals = []
    hstar_vals = []
    hjorth_vals = []
    dfa_vals = []
    
    for i in range((len(data) - win_s) // step_s + 1):
        start = i * step_s
        end = start + win_s
        if end > len(data):
            break
        
        w = data[start:end]
        center = (start + win_s // 2) / fs
        rel_time = center - trans_in_win
        
        m = metrics.compute_all(w)
        
        times.append(rel_time)
        pac_vals.append(m['pac'])
        hstar_vals.append(m['hstar'])
        hjorth_vals.append(m['hjorth_complexity'])
        dfa_vals.append(m['dfa'])
    
    if len(times) < 5:
        return None
    
    times = np.array(times)
    pac_vals = np.array(pac_vals)
    hstar_vals = np.array(hstar_vals)
    
    # Detectar onsets
    onset_pac = detect_onset(times, pac_vals, cfg.onset_sd)
    onset_hstar = detect_onset(times, hstar_vals, cfg.onset_sd)
    
    if onset_pac is None or onset_hstar is None:
        return None
    
    delta = onset_hstar - onset_pac
    
    return {
        'transition_time': t_trans,
        'from': trans['from'],
        'to': trans['to'],
        'onset_pac': onset_pac,
        'onset_hstar': onset_hstar,
        'delta': delta,
        'hstar_first': delta < 0
    }

#==============================================================================
# MAIN
#==============================================================================

def main():
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("NIVEL 3b v2: H* -> PAC PRECEDENCE (Sleep -> Wake)")
    print("NUEVOS PROXIES BASADOS EN HCTSA DISCOVERY")
    print("=" * 80)
    
    print(f"\nCriterios pre-registrados (hash: {CRITERIA_HASH[:16]}...):")
    for c in CRITERIA:
        print(f"  [{c['name']}] {c['op']} {c['threshold']}")
    
    print(f"\nNUEVO PROXY H*:")
    print(f"  H* = mean(hjorth_complexity, dfa)")
    print(f"  Basado en HCTSA Discovery (55.3% y 53.0% preceden PAC)")
    
    print(f"\nHIPÓTESIS: H* se activa ANTES que PAC durante despertar")
    print(f"PREDICCIÓN: Delta < 0 en >=70% de transiciones")
    
    # Inicializar
    print("\nInicializando métricas...")
    metrics = MetricsCalculator(cfg)
    print("  OK")
    
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
    
    print(f"Sujetos encontrados: {len(subjects)}")
    
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
                result = analyze_transition(raw, trans, metrics, cfg)
                if result:
                    result['subject_id'] = subj['id']
                    subj_results.append(result)
            
            if subj_results:
                n_subj += 1
                all_results.extend(subj_results)
                print(f"{len(subj_results)} trans OK")
            else:
                print("sin datos válidos")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    # Resultados
    print(f"\n{'=' * 80}")
    print(f"RESULTADOS: {len(all_results)} transiciones de {n_subj} sujetos")
    print("=" * 80)
    
    if not all_results:
        print("ERROR: Sin resultados válidos")
        return None
    
    # Compilar
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.output_dir / 'raw_results.csv', index=False)
    
    # Estadísticas
    valid = df['delta'].dropna()
    n_total = len(valid)
    n_hstar_first = (valid < 0).sum()
    n_pac_first = (valid > 0).sum()
    pct_hstar = 100 * n_hstar_first / n_total
    pct_pac = 100 * n_pac_first / n_total
    mean_delta = valid.mean()
    std_delta = valid.std()
    
    # Test estadístico
    if n_hstar_first + n_pac_first > 0:
        p_val = binomtest(n_hstar_first, n_hstar_first + n_pac_first, 0.5, alternative='greater').pvalue
    else:
        p_val = np.nan
    
    print(f"\nESTADÍSTICAS:")
    print(f"  n = {n_total}")
    print(f"  H* primero: {n_hstar_first} ({pct_hstar:.1f}%)")
    print(f"  PAC primero: {n_pac_first} ({pct_pac:.1f}%)")
    print(f"  Delta medio: {mean_delta:.2f} ± {std_delta:.2f} s")
    print(f"  p-value: {p_val:.6f}")
    
    # Comparación con v1
    print(f"\n{'=' * 80}")
    print("COMPARACIÓN CON PROXIES ANTERIORES")
    print("=" * 80)
    print(f"  v1 (spectral_entropy + perm_entropy + lziv): 59.4% H* primero")
    print(f"  v2 (hjorth_complexity + dfa):               {pct_hstar:.1f}% H* primero")
    if pct_hstar > 59.4:
        print(f"  >>> MEJORA: +{pct_hstar - 59.4:.1f}% <<<")
    elif pct_hstar < 59.4:
        print(f"  >>> PEOR: {pct_hstar - 59.4:.1f}% <<<")
    else:
        print(f"  >>> SIN CAMBIO <<<")
    
    # Evaluación de criterios
    print(f"\n{'=' * 80}")
    print("EVALUACIÓN DE CRITERIOS PRE-REGISTRADOS")
    print("=" * 80)
    
    metrics_vals = {
        'mean_delta_negative': mean_delta,
        'pct_hstar_first': pct_hstar,
        'p_value': p_val,
        'pct_pac_first': pct_pac
    }
    
    all_pass = True
    for c in CRITERIA:
        val = metrics_vals.get(c['name'])
        if c['op'] == '<':
            passed = val < c['threshold']
        elif c['op'] == '>=':
            passed = val >= c['threshold']
        elif c['op'] == '<=':
            passed = val <= c['threshold']
        else:
            passed = False
        
        icon = "[OK]" if passed else "[X]"
        print(f"  {icon} {c['name']} {c['op']} {c['threshold']} (observado: {val:.4f})")
        
        if c['required'] and not passed:
            all_pass = False
    
    # Veredicto
    if all_pass:
        verdict = "VALIDATED"
        desc = "Todos los criterios pasaron - AFH SOPORTADA"
    elif pct_pac >= 50:
        verdict = "FALSIFIED"
        desc = f"PAC primero en {pct_pac:.1f}% - AFH FALSIFICADA"
    else:
        verdict = "INCONCLUSIVE"
        desc = "Algunos criterios fallaron pero no falsificado"
    
    print(f"\n{'=' * 80}")
    print(f">>> VEREDICTO: {verdict} <<<")
    print(f"    {desc}")
    print("=" * 80)
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.0 (HCTSA-optimized)',
        'proxy_hstar': 'mean(hjorth_complexity, dfa)',
        'n_subjects': n_subj,
        'n_transitions': n_total,
        'statistics': {
            'n_hstar_first': int(n_hstar_first),
            'n_pac_first': int(n_pac_first),
            'pct_hstar_first': float(pct_hstar),
            'pct_pac_first': float(pct_pac),
            'mean_delta': float(mean_delta),
            'std_delta': float(std_delta),
            'p_value': float(p_val)
        },
        'comparison_v1': {
            'v1_pct_hstar_first': 59.4,
            'v2_pct_hstar_first': float(pct_hstar),
            'improvement': float(pct_hstar - 59.4)
        },
        'verdict': verdict,
        'description': desc,
        'criteria_hash': CRITERIA_HASH
    }
    
    with open(cfg.output_dir / 'SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResultados guardados en: {cfg.output_dir}")
    
    return df, summary


if __name__ == "__main__":
    try:
        df, summary = main()
    except KeyboardInterrupt:
        print("\n[CANCELADO]")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()