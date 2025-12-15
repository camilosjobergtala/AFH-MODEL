#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AFH NIVEL 3b - ANÁLISIS EXPLORATORIO DE CANDIDATOS H*
================================================================================

⚠️  DECLARACIÓN EXPLÍCITA:
    Este es un ANÁLISIS EXPLORATORIO para identificar candidatos H*.
    Los resultados NO son confirmatorios y requieren validación posterior
    en un dataset independiente.

OBJETIVO:
    Evaluar múltiples métricas EEG como posibles operacionalizaciones del
    "Autopsychic Fold" y comparar su precedencia temporal vs PAC.

MÉTRICAS CANDIDATAS:
    1. Hjorth Complexity (original)
    2. DFA - Detrended Fluctuation Analysis (original)
    3. H* v1 = (Hjorth + DFA) / 2 (original, ya testeado)
    4. Spectral Entropy
    5. Permutation Entropy
    6. Lempel-Ziv Complexity
    7. Sample Entropy
    8. Delta/Theta ratio
    9. Alpha power
    10. Beta power
    11. Delta-Beta coupling
    12. Theta-Gamma coupling (alternativo a PAC)

================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert, welch
from scipy.stats import entropy
from collections import defaultdict

warnings.filterwarnings('ignore')

__version__ = "exploratory_1.0"


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\nivel3b_exploratory"

ONSET_DETECTION = {
    'threshold_sd': 1.5,
    'consecutive_windows': 2,
    'window_size_s': 30,
    'window_step_s': 10,
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'min_wake_epochs_after': 2,
    'exclude_stages': ['?', 'M'],
}


# ============================================================================
# MÉTRICAS CANDIDATAS
# ============================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.999))
    high = max(low + 0.001, min(highcut / nyq, 0.999))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def compute_hjorth_complexity(signal):
    """Hjorth complexity - mide complejidad de la forma de onda."""
    try:
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var0, var1, var2 = np.var(signal), np.var(diff1), np.var(diff2)
        if var0 <= 0 or var1 <= 0:
            return np.nan
        return np.sqrt(var2 / var1) / np.sqrt(var1 / var0)
    except:
        return np.nan


def compute_hjorth_mobility(signal):
    """Hjorth mobility - pendiente media de la señal."""
    try:
        diff1 = np.diff(signal)
        var0, var1 = np.var(signal), np.var(diff1)
        if var0 <= 0:
            return np.nan
        return np.sqrt(var1 / var0)
    except:
        return np.nan


def compute_dfa(signal, min_box=4, max_box=None):
    """Detrended Fluctuation Analysis - detecta correlaciones de largo alcance."""
    try:
        n = len(signal)
        if max_box is None:
            max_box = n // 4
        y = np.cumsum(signal - np.mean(signal))
        scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
        scales = scales[(scales >= min_box) & (scales <= max_box)]
        if len(scales) < 4:
            return np.nan
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 1:
                fluctuations.append(np.nan)
                continue
            rms_list = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) < scale:
                    continue
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((segment - trend)**2)))
            fluctuations.append(np.mean(rms_list) if rms_list else np.nan)
        valid = ~np.isnan(fluctuations)
        if np.sum(valid) < 4:
            return np.nan
        alpha, _ = np.polyfit(np.log10(scales[valid]), np.log10(np.array(fluctuations)[valid]), 1)
        return alpha
    except:
        return np.nan


def compute_spectral_entropy(signal, fs, nperseg=256):
    """Entropía espectral - irregularidad del espectro de potencia."""
    try:
        freqs, psd = welch(signal, fs, nperseg=min(nperseg, len(signal)//2))
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm)) / np.log2(len(psd_norm))
    except:
        return np.nan


def compute_permutation_entropy(signal, order=3, delay=1):
    """Permutation entropy - complejidad basada en patrones ordinales."""
    try:
        n = len(signal)
        permutations = []
        for i in range(n - delay * (order - 1)):
            indices = [i + delay * j for j in range(order)]
            pattern = tuple(np.argsort([signal[idx] for idx in indices]))
            permutations.append(pattern)
        
        from collections import Counter
        counts = Counter(permutations)
        total = len(permutations)
        probs = np.array([c / total for c in counts.values()])
        return -np.sum(probs * np.log2(probs)) / np.log2(np.math.factorial(order))
    except:
        return np.nan


def compute_lempel_ziv(signal, threshold='median'):
    """Lempel-Ziv complexity - versión simplificada y rápida."""
    try:
        # Downsample para velocidad
        if len(signal) > 1000:
            step = len(signal) // 1000
            signal = signal[::step]
        
        if threshold == 'median':
            thresh = np.median(signal)
        else:
            thresh = np.mean(signal)
        
        # Binarizar
        binary = (signal > thresh).astype(int)
        n = len(binary)
        
        # Algoritmo simplificado de LZ
        complexity = 1
        i = 0
        k = 1
        while i + k <= n:
            # Buscar si el patrón actual existe antes
            pattern = tuple(binary[i:i+k])
            found = False
            for j in range(i):
                if j + k <= i and tuple(binary[j:j+k]) == pattern:
                    found = True
                    break
            if found:
                k += 1
            else:
                complexity += 1
                i = i + k
                k = 1
        
        # Normalizar
        if n > 1:
            return complexity / (n / np.log2(n))
        return complexity
    except:
        return np.nan


def compute_sample_entropy_fast(signal, m=2, r_factor=0.2, max_samples=500):
    """Sample entropy - versión rápida con downsampling."""
    try:
        # Downsample si es muy largo
        if len(signal) > max_samples:
            step = len(signal) // max_samples
            signal = signal[::step]
        
        n = len(signal)
        r = r_factor * np.std(signal)
        if r <= 0 or n < m + 2:
            return np.nan
        
        def count_matches(template_length):
            count = 0
            for i in range(n - template_length):
                for j in range(i + 1, n - template_length):
                    if np.max(np.abs(signal[i:i+template_length] - signal[j:j+template_length])) < r:
                        count += 1
            return count * 2  # Symmetric
        
        A = count_matches(m + 1)
        B = count_matches(m)
        
        if B == 0 or A == 0:
            return np.nan
        return -np.log(A / B)
    except:
        return np.nan


def compute_band_power(signal, fs, band):
    """Potencia en banda de frecuencia específica."""
    try:
        freqs, psd = welch(signal, fs, nperseg=min(256, len(signal)//2))
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        return np.trapz(psd[idx], freqs[idx])
    except:
        return np.nan


def compute_band_ratio(signal, fs, band1, band2):
    """Ratio entre dos bandas de frecuencia."""
    try:
        p1 = compute_band_power(signal, fs, band1)
        p2 = compute_band_power(signal, fs, band2)
        if p2 <= 0:
            return np.nan
        return p1 / p2
    except:
        return np.nan


def compute_pac_mvl(signal, fs, phase_freq=(4, 8), amp_freq=(30, 45)):
    """Phase-Amplitude Coupling via Mean Vector Length."""
    try:
        theta = bandpass_filter(signal, phase_freq[0], phase_freq[1], fs)
        theta_phase = np.angle(hilbert(theta))
        gamma = bandpass_filter(signal, amp_freq[0], amp_freq[1], fs)
        gamma_amp = np.abs(hilbert(gamma))
        n = len(theta_phase)
        mvl = np.abs(np.sum(gamma_amp * np.exp(1j * theta_phase))) / n
        return mvl / np.mean(gamma_amp) if np.mean(gamma_amp) > 0 else 0
    except:
        return np.nan


def compute_delta_gamma_pac(signal, fs):
    """PAC delta-gamma (alternativo)."""
    return compute_pac_mvl(signal, fs, phase_freq=(0.5, 4), amp_freq=(30, 45))


def compute_theta_gamma_pac(signal, fs):
    """PAC theta-gamma (el estándar)."""
    return compute_pac_mvl(signal, fs, phase_freq=(4, 8), amp_freq=(30, 45))


# ============================================================================
# DICCIONARIO DE MÉTRICAS
# ============================================================================

METRIC_FUNCTIONS = {
    'hjorth_complexity': lambda sig, fs: compute_hjorth_complexity(sig),
    'hjorth_mobility': lambda sig, fs: compute_hjorth_mobility(sig),
    'dfa': lambda sig, fs: compute_dfa(sig),
    'spectral_entropy': compute_spectral_entropy,
    'permutation_entropy': lambda sig, fs: compute_permutation_entropy(sig, order=3, delay=1),
    'lempel_ziv': lambda sig, fs: compute_lempel_ziv(sig),
    'sample_entropy': lambda sig, fs: compute_sample_entropy_fast(sig, m=2),
    'delta_power': lambda sig, fs: compute_band_power(sig, fs, (0.5, 4)),
    'theta_power': lambda sig, fs: compute_band_power(sig, fs, (4, 8)),
    'alpha_power': lambda sig, fs: compute_band_power(sig, fs, (8, 13)),
    'beta_power': lambda sig, fs: compute_band_power(sig, fs, (13, 30)),
    'gamma_power': lambda sig, fs: compute_band_power(sig, fs, (30, 45)),
    'delta_theta_ratio': lambda sig, fs: compute_band_ratio(sig, fs, (0.5, 4), (4, 8)),
    'delta_beta_ratio': lambda sig, fs: compute_band_ratio(sig, fs, (0.5, 4), (13, 30)),
    'theta_beta_ratio': lambda sig, fs: compute_band_ratio(sig, fs, (4, 8), (13, 30)),
    'pac_theta_gamma': compute_theta_gamma_pac,
    'pac_delta_gamma': compute_delta_gamma_pac,
}

# Métricas candidatas para H* (excluimos PAC que es el comparador)
HSTAR_CANDIDATES = [
    'hjorth_complexity',
    'hjorth_mobility', 
    'dfa',
    'spectral_entropy',
    'permutation_entropy',
    'lempel_ziv',
    'sample_entropy',
    'delta_power',
    'theta_power',
    'alpha_power',
    'beta_power',
    'delta_theta_ratio',
    'delta_beta_ratio',
    'theta_beta_ratio',
]

# Combinaciones a evaluar
HSTAR_COMBINATIONS = [
    ('hstar_v1', ['hjorth_complexity', 'dfa'], [0.5, 0.5]),
    ('hstar_v2_entropy', ['spectral_entropy', 'permutation_entropy'], [0.5, 0.5]),
    ('hstar_v3_complexity', ['hjorth_complexity', 'lempel_ziv', 'sample_entropy'], [0.33, 0.33, 0.34]),
    ('hstar_v4_spectral', ['delta_power', 'alpha_power'], [0.5, 0.5]),
    ('hstar_v5_ratios', ['delta_beta_ratio', 'theta_beta_ratio'], [0.5, 0.5]),
    ('hstar_v6_full', ['hjorth_complexity', 'dfa', 'spectral_entropy', 'permutation_entropy'], [0.25, 0.25, 0.25, 0.25]),
]


# ============================================================================
# SLEEP-EDF PROCESSOR (simplificado)
# ============================================================================

class SleepEDFProcessor:
    STAGE_MAPPING = {
        'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3', 'Sleep stage 4': 'N4', 'Sleep stage R': 'R',
        'Sleep stage ?': '?', 'Movement time': 'M'
    }
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.window_size = ONSET_DETECTION['window_size_s']
        self.window_step = ONSET_DETECTION['window_step_s']
    
    def find_psg_files(self):
        files = []
        psg_files = list(self.data_path.glob("*-PSG.edf"))
        if not psg_files:
            psg_files = list(self.data_path.glob("*PSG.edf"))
        
        for psg_file in sorted(psg_files):
            name = psg_file.name
            subject_id = name.replace('-PSG.edf', '').replace('PSG.edf', '')
            hyp_file = None
            
            hyp_candidates = list(self.data_path.glob(f"{subject_id}*Hypnogram.edf"))
            if hyp_candidates:
                hyp_file = hyp_candidates[0]
            
            if hyp_file is None and len(subject_id) >= 2:
                hyp_prefix = subject_id[:-1] + 'C'
                hyp_candidates = list(self.data_path.glob(f"{hyp_prefix}*Hypnogram.edf"))
                if hyp_candidates:
                    hyp_file = hyp_candidates[0]
            
            if hyp_file:
                files.append((subject_id, psg_file, hyp_file))
        
        return files
    
    def load_subject(self, psg_path, hyp_path):
        try:
            import mne
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            eeg_channel = None
            for ch in ['EEG Pz-Oz', 'EEG Fpz-Cz']:
                if ch in raw.ch_names:
                    eeg_channel = ch
                    break
            if eeg_channel is None:
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        eeg_channel = ch
                        break
            if eeg_channel is None:
                return None
            
            raw.pick_channels([eeg_channel])
            signal = raw.get_data()[0]
            fs = raw.info['sfreq']
            
            annotations = mne.read_annotations(str(hyp_path))
            total_duration = len(signal) / fs
            n_epochs = int(total_duration // 30)
            stages = ['?'] * n_epochs
            
            for ann in annotations:
                stage = self.STAGE_MAPPING.get(ann['description'], '?')
                start_epoch = int(ann['onset'] // 30)
                end_epoch = int((ann['onset'] + ann['duration']) // 30)
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            return {'signal': signal, 'fs': fs, 'stages': stages}
        except:
            return None
    
    def find_stable_transitions(self, stages):
        transitions = []
        sleep_stages = ['N1', 'N2', 'N3', 'N4']
        exclude = set(TRANSITION_CRITERIA['exclude_stages'])
        min_sleep = TRANSITION_CRITERIA['min_sleep_epochs_before']
        min_wake = TRANSITION_CRITERIA['min_wake_epochs_after']
        
        for i in range(min_sleep, len(stages) - min_wake):
            if stages[i] != 'W' or stages[i-1] not in sleep_stages:
                continue
            pre_stages = stages[max(0, i-min_sleep):i]
            if any(s in exclude or s == 'W' for s in pre_stages):
                continue
            if not all(s in sleep_stages for s in pre_stages):
                continue
            post_stages = stages[i:min(len(stages), i+min_wake)]
            if not all(s == 'W' for s in post_stages):
                continue
            transitions.append({'epoch_idx': i, 'from_stage': stages[i-1], 'time_seconds': i * 30})
        return transitions
    
    def extract_transition_window(self, signal, fs, transition_time, pre_seconds=120, post_seconds=120):
        start_sample = int((transition_time - pre_seconds) * fs)
        end_sample = int((transition_time + post_seconds) * fs)
        if start_sample < 0 or end_sample > len(signal):
            return None
        return signal[start_sample:end_sample]


# ============================================================================
# ANÁLISIS EXPLORATORIO
# ============================================================================

class ExploratoryAnalysis:
    def __init__(self, data_path: str, output_dir: str, max_subjects: int = 30):
        """
        max_subjects: limitar para análisis exploratorio rápido
        """
        self.processor = SleepEDFProcessor(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_subjects = max_subjects
        
        print(f"\n{'='*70}")
        print("⚠️  ANÁLISIS EXPLORATORIO - NO CONFIRMATORIO")
        print(f"{'='*70}")
        print(f"Objetivo: Evaluar candidatos H* alternativos")
        print(f"Max sujetos: {max_subjects} (para velocidad)")
        print(f"Output: {output_dir}")
        print(f"{'='*70}\n")
    
    def compute_all_metrics_timeseries(self, window, fs):
        """Calcula todas las métricas en ventanas deslizantes."""
        window_samples = int(ONSET_DETECTION['window_size_s'] * fs)
        step_samples = int(ONSET_DETECTION['window_step_s'] * fs)
        n_windows = (len(window) - window_samples) // step_samples + 1
        
        results = {'times': []}
        for metric_name in METRIC_FUNCTIONS.keys():
            results[metric_name] = []
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            segment = window[start:end]
            center_time = (start + window_samples/2) / fs - 120
            results['times'].append(center_time)
            
            for metric_name, func in METRIC_FUNCTIONS.items():
                try:
                    val = func(segment, fs)
                except:
                    val = np.nan
                results[metric_name].append(val)
        
        # Convertir a arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def normalize_baseline(self, values, times, baseline_end=-30):
        """Z-score usando solo baseline."""
        baseline_mask = times < baseline_end
        baseline_vals = values[baseline_mask]
        baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
        if len(baseline_vals) < 2:
            return values
        mu, sigma = np.mean(baseline_vals), np.std(baseline_vals)
        if sigma < 1e-10:
            return values - mu
        return (values - mu) / sigma
    
    def detect_onset(self, values, times, baseline_end=-30):
        """Detecta onset con 2 ventanas consecutivas > threshold."""
        threshold = ONSET_DETECTION['threshold_sd']
        k_consecutive = ONSET_DETECTION['consecutive_windows']
        
        mask = times >= baseline_end
        v = values[mask]
        t = times[mask]
        
        above = (~np.isnan(v)) & (v > threshold)
        
        run = 0
        for i, ok in enumerate(above):
            if ok:
                run += 1
                if run >= k_consecutive:
                    return t[i - k_consecutive + 1]
            else:
                run = 0
        return None
    
    def evaluate_metric_vs_pac(self, metric_z, pac_z, times):
        """Evalúa si métrica precede a PAC."""
        metric_onset = self.detect_onset(metric_z, times)
        pac_onset = self.detect_onset(pac_z, times)
        
        if metric_onset is not None and pac_onset is not None:
            delta = metric_onset - pac_onset
            return {
                'metric_onset': metric_onset,
                'pac_onset': pac_onset,
                'delta': delta,
                'metric_first': delta < 0
            }
        return None
    
    def run(self):
        print("📂 Buscando archivos...")
        files = self.processor.find_psg_files()
        print(f"   Total disponibles: {len(files)}")
        
        # Limitar para exploración rápida
        if len(files) > self.max_subjects:
            np.random.seed(42)
            indices = np.random.choice(len(files), self.max_subjects, replace=False)
            files = [files[i] for i in sorted(indices)]
        print(f"   Usando: {len(files)} sujetos")
        
        # Resultados por métrica
        metric_results = defaultdict(list)
        combination_results = defaultdict(list)
        all_transitions_data = []
        
        for i, (subject_id, psg_path, hyp_path) in enumerate(files):
            print(f"\n   [{i+1}/{len(files)}] {subject_id}...", flush=True)
            
            data = self.processor.load_subject(psg_path, hyp_path)
            if data is None:
                print(f"      ⚠️ No se pudo cargar")
                continue
            
            transitions = self.processor.find_stable_transitions(data['stages'])
            print(f"      Transiciones encontradas: {len(transitions)}", flush=True)
            
            for t_idx, trans in enumerate(transitions):
                if len(transitions) > 3:
                    print(f"\r      Procesando transición {t_idx+1}/{len(transitions)}...", end="", flush=True)
                
                window = self.processor.extract_transition_window(
                    data['signal'], data['fs'], trans['time_seconds']
                )
                if window is None:
                    continue
                
                # Calcular todas las métricas
                raw_metrics = self.compute_all_metrics_timeseries(window, data['fs'])
                times = raw_metrics['times']
                
                # Normalizar
                normalized = {}
                for name in METRIC_FUNCTIONS.keys():
                    normalized[name] = self.normalize_baseline(raw_metrics[name], times)
                
                # PAC de referencia
                pac_z = normalized['pac_theta_gamma']
                
                # Evaluar cada métrica individual
                for metric_name in HSTAR_CANDIDATES:
                    result = self.evaluate_metric_vs_pac(normalized[metric_name], pac_z, times)
                    if result:
                        metric_results[metric_name].append(result)
                
                # Evaluar combinaciones
                for combo_name, components, weights in HSTAR_COMBINATIONS:
                    combo_z = np.zeros_like(times)
                    valid = True
                    for comp, w in zip(components, weights):
                        if comp in normalized:
                            combo_z += w * normalized[comp]
                        else:
                            valid = False
                            break
                    if valid:
                        result = self.evaluate_metric_vs_pac(combo_z, pac_z, times)
                        if result:
                            combination_results[combo_name].append(result)
                
                # Guardar datos de transición
                trans_data = {
                    'subject_id': subject_id,
                    'from_stage': trans['from_stage'],
                }
                for metric_name in HSTAR_CANDIDATES:
                    result = self.evaluate_metric_vs_pac(normalized[metric_name], pac_z, times)
                    if result:
                        trans_data[f'{metric_name}_onset'] = result['metric_onset']
                        trans_data[f'{metric_name}_delta'] = result['delta']
                        trans_data[f'{metric_name}_first'] = result['metric_first']
                trans_data['pac_onset'] = self.detect_onset(pac_z, times)
                all_transitions_data.append(trans_data)
        
        print(f"\n   Total transiciones válidas: {len(all_transitions_data)}")
        
        # Generar ranking
        print(f"\n{'='*70}")
        print("📊 RANKING DE CANDIDATOS H*")
        print(f"{'='*70}")
        
        ranking = []
        
        # Métricas individuales
        print("\n📈 MÉTRICAS INDIVIDUALES:")
        for metric_name in HSTAR_CANDIDATES:
            results = metric_results[metric_name]
            if len(results) < 10:
                continue
            n_first = sum(1 for r in results if r['metric_first'])
            pct = 100 * n_first / len(results)
            mean_delta = np.mean([r['delta'] for r in results])
            ranking.append({
                'name': metric_name,
                'type': 'individual',
                'n_transitions': len(results),
                'pct_first': pct,
                'mean_delta': mean_delta
            })
        
        # Combinaciones
        print("\n📈 COMBINACIONES:")
        for combo_name, _, _ in HSTAR_COMBINATIONS:
            results = combination_results[combo_name]
            if len(results) < 10:
                continue
            n_first = sum(1 for r in results if r['metric_first'])
            pct = 100 * n_first / len(results)
            mean_delta = np.mean([r['delta'] for r in results])
            ranking.append({
                'name': combo_name,
                'type': 'combination',
                'n_transitions': len(results),
                'pct_first': pct,
                'mean_delta': mean_delta
            })
        
        # Ordenar por %first
        ranking = sorted(ranking, key=lambda x: x['pct_first'], reverse=True)
        
        print(f"\n{'Rank':<5} {'Candidato':<25} {'Tipo':<12} {'N':<6} {'%First':<10} {'Δ medio':<10}")
        print("-" * 70)
        for i, r in enumerate(ranking, 1):
            print(f"{i:<5} {r['name']:<25} {r['type']:<12} {r['n_transitions']:<6} {r['pct_first']:.1f}%     {r['mean_delta']:.1f}s")
        
        # Guardar resultados
        ranking_df = pd.DataFrame(ranking)
        ranking_df.to_csv(self.output_dir / "hstar_candidates_ranking.csv", index=False)
        
        transitions_df = pd.DataFrame(all_transitions_data)
        transitions_df.to_csv(self.output_dir / "exploratory_transitions.csv", index=False)
        
        # Guardar metadata
        metadata = {
            'analysis_type': 'EXPLORATORY - NOT CONFIRMATORY',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_subjects': len(files),
            'n_transitions': len(all_transitions_data),
            'metrics_evaluated': HSTAR_CANDIDATES,
            'combinations_evaluated': [c[0] for c in HSTAR_COMBINATIONS],
            'top_candidates': ranking[:5] if len(ranking) >= 5 else ranking,
            'warning': 'These results require independent validation before any confirmatory claims'
        }
        
        with open(self.output_dir / "exploratory_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n{'='*70}")
        print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
        print(f"{'='*70}")
        print(f"\n📁 Resultados en: {self.output_dir}")
        print("\n⚠️  RECORDATORIO: Estos resultados son EXPLORATORIOS.")
        print("   Cualquier candidato prometedor debe validarse en dataset independiente.")
        
        return ranking


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and p.is_dir():
            psg_files = list(p.glob("*PSG.edf")) + list(p.glob("*-PSG.edf"))
            if psg_files:
                return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AFH - Análisis exploratorio de candidatos H*")
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--max-subjects', type=int, default=30, help="Máximo sujetos (default: 30)")
    args = parser.parse_args()
    
    # Auto-detectar
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        print("🔍 Buscando datos...")
        data_path = find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontraron datos")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = DEFAULT_OUTPUT
    
    print(f"✅ Salida: {output_dir}")
    
    analysis = ExploratoryAnalysis(str(data_path), output_dir, args.max_subjects)
    ranking = analysis.run()


if __name__ == "__main__":
    main()