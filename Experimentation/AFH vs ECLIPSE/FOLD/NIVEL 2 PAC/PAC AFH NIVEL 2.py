#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIVEL 3b OPTIMIZADO: Test Precedencia H* -> PAC (Sleep -> Wake)
ECLIPSE v3.0 - VERSION RAPIDA
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
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib
import math

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

print("Cargando tensorpac...")
from tensorpac import Pac
print("OK")

#==============================================================================
# CONFIG - OPTIMIZADA PARA VELOCIDAD
#==============================================================================

@dataclass
class Config:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./nivel3b_results')
    fs: float = 100.0
    # VENTANAS MAS CORTAS = MAS RAPIDO
    window_before: float = 120.0  # 2 min antes (era 5 min)
    window_after: float = 120.0   # 2 min despues
    sliding_window: float = 30.0
    sliding_step: float = 10.0    # Paso mas grande (era 5s)
    theta: Tuple[float, float] = (4.0, 8.0)
    gamma: Tuple[float, float] = (30.0, 45.0)
    onset_sd: float = 1.5
    
    def __post_init__(self):
        self.wake = ['Sleep stage W']
        self.sleep = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']

#==============================================================================
# CRITERIOS PRE-REGISTRADOS
#==============================================================================

CRITERIA = [
    {'name': 'mean_delta_negative', 'threshold': 0.0, 'op': '<', 'required': True},
    {'name': 'pct_hstar_first', 'threshold': 70.0, 'op': '>=', 'required': True},
    {'name': 'p_value', 'threshold': 0.01, 'op': '<', 'required': True},
    {'name': 'pct_pac_first', 'threshold': 50.0, 'op': '<', 'required': True},
]

def evaluate_criterion(value, threshold, op):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    ops = {'>=': lambda x,y: x>=y, '<=': lambda x,y: x<=y, '>': lambda x,y: x>y, '<': lambda x,y: x<y}
    return ops[op](value, threshold)

#==============================================================================
# METRICAS SIMPLIFICADAS
#==============================================================================

class FastMetrics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg.fs
        print("  Inicializando PAC calculator...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pac = Pac(idpac=(1,0,0), f_pha=list(cfg.theta), f_amp=list(cfg.gamma),
                          dcomplex='wavelet', width=7, verbose=False)
        print("  PAC OK")
    
    def preprocess(self, data):
        if len(data) < 10: return data
        sos = signal.butter(4, [0.5, 45.0], btype='bandpass', fs=self.fs, output='sos')
        filt = signal.sosfiltfilt(sos, data)
        return (filt - np.mean(filt)) / (np.std(filt) + 1e-10)
    
    def pac_value(self, data):
        if len(data) < 200: return np.nan
        clean = self.preprocess(data)
        if np.ptp(clean) > 8: return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(self.pac.filterfit(self.fs, clean[np.newaxis,:], clean[np.newaxis,:])[0,0,0])
        except:
            return np.nan
    
    def spectral_entropy(self, data):
        if len(data) < 200: return np.nan
        clean = self.preprocess(data)
        _, psd = signal.welch(clean, fs=self.fs, nperseg=min(256, len(clean)//2))
        psd_n = psd / (np.sum(psd) + 1e-10)
        ent = -np.sum(psd_n * np.log2(psd_n + 1e-10))
        return ent / np.log2(len(psd))
    
    def perm_entropy(self, data, order=3, delay=1):
        if len(data) < 50: return np.nan
        clean = self.preprocess(data)
        n = len(clean)
        patterns = [tuple(np.argsort([clean[i+j*delay] for j in range(order)])) 
                   for i in range(n-(order-1)*delay)]
        counts = Counter(patterns)
        total = len(patterns)
        ent = -sum((c/total)*np.log2(c/total) for c in counts.values())
        return ent / np.log2(math.factorial(order))
    
    def lziv(self, data):
        if len(data) < 100: return np.nan
        clean = self.preprocess(data)
        binary = ''.join(['1' if x > np.median(clean) else '0' for x in clean])
        n = len(binary)
        c, pl, i = 1, 1, 0
        while i + pl <= n:
            if binary[i:i+pl] not in binary[:i+pl-1]:
                c += 1; i += pl; pl = 1
            else:
                pl += 1
                if i + pl > n: break
        return c / (n / np.log2(n)) if n > 1 else 0

#==============================================================================
# ANALIZADOR RAPIDO
#==============================================================================

def find_transitions(ann, cfg):
    """Encuentra transiciones Sleep -> Wake"""
    trans = []
    desc, onsets = ann.description, ann.onset
    for i in range(len(desc)-1):
        if desc[i] in cfg.sleep and desc[i+1] in cfg.wake:
            trans.append({'time': onsets[i+1], 'from': desc[i], 'to': desc[i+1]})
    return trans

def get_channel_data(raw, start_s, end_s):
    """Extrae datos del canal EEG"""
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
    
    times_v = times[valid_mask]
    vals_v = values[valid_mask]
    
    # Baseline: antes de -30s
    bl_mask = times_v < -30
    if bl_mask.sum() < 2:
        bl_mask = np.arange(len(vals_v)) < len(vals_v)//3
    
    bl_mean = vals_v[bl_mask].mean()
    bl_std = vals_v[bl_mask].std()
    
    if bl_std < 1e-10:
        return None
    
    threshold = bl_mean + threshold_sd * bl_std
    
    # Buscar primer cruce despues de -30s
    post_mask = times_v >= -30
    for t, v in zip(times_v[post_mask], vals_v[post_mask]):
        if v > threshold:
            return t
    return None

def analyze_transition(raw, trans, metrics, cfg):
    """Analiza una transicion"""
    fs = raw.info['sfreq']
    t_trans = trans['time']
    
    # Extraer ventana
    start_t = max(0, t_trans - cfg.window_before)
    end_t = min(raw.times[-1], t_trans + cfg.window_after)
    start_s, end_s = int(start_t * fs), int(end_t * fs)
    
    if end_s - start_s < int(30 * fs):
        return None
    
    data = get_channel_data(raw, start_s, end_s)
    if data is None:
        return None
    
    # Tiempo relativo a transicion
    trans_in_win = min(cfg.window_before, t_trans)
    
    # Sliding windows
    win_s = int(cfg.sliding_window * fs)
    step_s = int(cfg.sliding_step * fs)
    
    times, pacs, h_stars = [], [], []
    
    for i in range((len(data) - win_s) // step_s + 1):
        start = i * step_s
        end = start + win_s
        if end > len(data):
            break
        
        w = data[start:end]
        center = (start + win_s//2) / fs
        rel_time = center - trans_in_win
        
        times.append(rel_time)
        pacs.append(metrics.pac_value(w))
        
        # H* = promedio de los 3 proxies
        se = metrics.spectral_entropy(w)
        pe = metrics.perm_entropy(w)
        lz = metrics.lziv(w)
        h_star = np.nanmean([se, pe, lz])
        h_stars.append(h_star)
    
    if len(times) < 5:
        return None
    
    times = np.array(times)
    pacs = np.array(pacs)
    h_stars = np.array(h_stars)
    
    # Detectar onsets
    onset_pac = detect_onset(times, pacs, cfg.onset_sd)
    onset_hstar = detect_onset(times, h_stars, cfg.onset_sd)
    
    result = {
        'transition_time': t_trans,
        'from': trans['from'],
        'to': trans['to'],
        'onset_pac': onset_pac,
        'onset_hstar': onset_hstar,
    }
    
    if onset_pac is not None and onset_hstar is not None:
        result['delta'] = onset_hstar - onset_pac  # Negativo = H* primero
    
    return result

#==============================================================================
# MAIN
#==============================================================================

def main():
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("NIVEL 3b: H* -> PAC PRECEDENCE (Sleep -> Wake)")
    print("ECLIPSE v3.0 - VERSION OPTIMIZADA")
    print("="*80)
    
    # Pre-registro
    criteria_hash = hashlib.sha256(json.dumps(CRITERIA, sort_keys=True).encode()).hexdigest()
    print(f"\nCriterios pre-registrados (hash: {criteria_hash[:16]}...):")
    for c in CRITERIA:
        print(f"  [{c['name']}] {c['op']} {c['threshold']}")
    
    # Guardar criterios
    with open(cfg.output_dir / 'CRITERIA.json', 'w') as f:
        json.dump({'criteria': CRITERIA, 'hash': criteria_hash, 'date': datetime.now().isoformat()}, f, indent=2)
    
    print("\nHIPOTESIS: H* se activa ANTES que PAC durante despertar")
    print("PREDICCION: Delta < 0 en >=70% de transiciones\n")
    
    # Inicializar metricas
    print("Inicializando metricas...")
    metrics = FastMetrics(cfg)
    
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
    n_with_trans = 0
    
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
                if result and 'delta' in result:
                    result['subject_id'] = subj['id']
                    subj_results.append(result)
            
            if subj_results:
                n_with_trans += 1
                all_results.extend(subj_results)
                print(f"{len(subj_results)} trans OK")
            else:
                print("sin datos validos")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS: {len(all_results)} transiciones de {n_with_trans} sujetos")
    print("="*80)
    
    if not all_results:
        print("ERROR: Sin resultados")
        return
    
    # Compilar
    df = pd.DataFrame(all_results)
    df.to_csv(cfg.output_dir / 'raw_results.csv', index=False)
    
    # Estadisticas
    deltas = df['delta'].dropna()
    n_total = len(deltas)
    n_hstar_first = (deltas < 0).sum()
    n_pac_first = (deltas > 0).sum()
    pct_hstar = 100 * n_hstar_first / n_total
    pct_pac = 100 * n_pac_first / n_total
    mean_delta = deltas.mean()
    std_delta = deltas.std()
    
    # Test binomial
    if n_hstar_first + n_pac_first > 0:
        p_val = binomtest(n_hstar_first, n_hstar_first + n_pac_first, 0.5, alternative='greater').pvalue
    else:
        p_val = np.nan
    
    print(f"\nESTADISTICAS:")
    print(f"  n = {n_total}")
    print(f"  H* primero: {n_hstar_first} ({pct_hstar:.1f}%)")
    print(f"  PAC primero: {n_pac_first} ({pct_pac:.1f}%)")
    print(f"  Delta medio: {mean_delta:.2f} +/- {std_delta:.2f} s")
    print(f"  p-value: {p_val:.6f}")
    
    # Evaluar criterios
    print(f"\n{'='*80}")
    print("EVALUACION DE CRITERIOS PRE-REGISTRADOS")
    print("="*80)
    
    observed = {
        'mean_delta_negative': mean_delta,
        'pct_hstar_first': pct_hstar,
        'p_value': p_val,
        'pct_pac_first': pct_pac
    }
    
    all_pass = True
    for c in CRITERIA:
        val = observed[c['name']]
        passed = evaluate_criterion(val, c['threshold'], c['op'])
        if c['required'] and not passed:
            all_pass = False
        icon = "[OK]" if passed else "[X]"
        print(f"  {icon} {c['name']} {c['op']} {c['threshold']} (observado: {val:.4f})")
    
    # Veredicto
    if all_pass:
        verdict = "VALIDATED"
        desc = "Todos los criterios pasaron - H* precede PAC"
    elif pct_pac >= 50:
        verdict = "FALSIFIED"
        desc = f"PAC primero en {pct_pac:.1f}% - Arquitectura falsificada"
    else:
        verdict = "INCONCLUSIVE"
        desc = "Algunos criterios fallaron pero no falsificado"
    
    print(f"\n{'='*80}")
    print(f">>> VEREDICTO: {verdict} <<<")
    print(f"    {desc}")
    print("="*80)
    
    # Guardar
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': n_with_trans,
        'n_transitions': n_total,
        'statistics': {
            'n_hstar_first': int(n_hstar_first),
            'n_pac_first': int(n_pac_first),
            'pct_hstar_first': float(pct_hstar),
            'pct_pac_first': float(pct_pac),
            'mean_delta': float(mean_delta),
            'std_delta': float(std_delta),
            'p_value': float(p_val) if not np.isnan(p_val) else None
        },
        'verdict': verdict,
        'description': desc,
        'criteria_hash': criteria_hash
    }
    
    with open(cfg.output_dir / 'FINAL_SUMMARY.json', 'w') as f:
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