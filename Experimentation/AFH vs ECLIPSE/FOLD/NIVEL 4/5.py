#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AFH NIVEL 3b - ANÁLISIS EXPLORATORIO H* v3 (CORREGIDO)
Búsqueda sistemática de métricas proxy de convergencia talamocortical
================================================================================

CORRECCIONES APLICADAS (basadas en auditoría):
  1. Lectura de anotaciones MNE corregida (atributos, no dict)
  2. Verificación Nyquist antes de calcular bandas
  3. Sample entropy removida (O(N²) muy lento)
  4. Detección de onset bidireccional (aumentos Y disminuciones)
  5. Normalización robusta (mediana + MAD) para ratios/PAC
  6. Agregación por SUJETO (no solo por transición)
  7. Control por canal (Fpz-Cz vs Pz-Oz separados)
  8. Logging de progreso mejorado
  9. Verificación de potencias para PAC (control de sesgo)

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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.signal import butter, filtfilt, hilbert, welch
from collections import Counter
import hashlib
import traceback

warnings.filterwarnings('ignore')

__version__ = "exploratory_v3.2_with_slopes"

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\nivel3b_exploratory_v3"

SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.7

# Bandas de frecuencia (Hz) - conservadoras para evitar problemas Nyquist
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45),  # Conservador, verificar vs Nyquist
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'min_wake_epochs_after': 2,
    'exclude_stages': ['?', 'M'],
}

ONSET_DETECTION = {
    'threshold_sd': 1.5,
    'consecutive_windows': 2,
    'window_size_s': 30,
    'window_step_s': 10,
}

# ============================================================================
# FUNCIONES DE PROCESAMIENTO DE SEÑAL (CORREGIDAS)
# ============================================================================

def get_valid_bands(fs):
    """Retorna bandas válidas dado el sampling rate (Nyquist check)."""
    nyquist = fs / 2
    valid = {}
    for name, (low, high) in BANDS.items():
        if high < nyquist - 1:  # Margen de 1 Hz
            valid[name] = (low, high)
        else:
            # Ajustar banda si es posible
            if low < nyquist - 2:
                valid[name] = (low, min(high, nyquist - 2))
            # Si no, la banda no es válida para este fs
    return valid


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filtro pasa-banda Butterworth con verificación."""
    nyq = 0.5 * fs
    
    # Verificar que las frecuencias son válidas
    if highcut >= nyq:
        return None  # Banda inválida
    
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    
    if low >= high:
        return None
    
    try:
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)
    except:
        return None


def compute_band_power(signal, fs, band, nperseg=256):
    """Potencia en banda con verificación Nyquist."""
    nyquist = fs / 2
    if band[1] >= nyquist:
        return np.nan
    
    try:
        actual_nperseg = min(nperseg, len(signal)//2)
        if actual_nperseg < 16:
            return np.nan
        freqs, psd = welch(signal, fs, nperseg=actual_nperseg)
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if len(idx) == 0:
            return np.nan
        return np.trapz(psd[idx], freqs[idx])
    except:
        return np.nan


def robust_normalize(values, baseline_mask):
    """Normalización robusta usando mediana y MAD."""
    baseline_vals = values[baseline_mask]
    baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
    
    if len(baseline_vals) < 3:
        return values, False
    
    median = np.median(baseline_vals)
    mad = np.median(np.abs(baseline_vals - median))
    
    # MAD a escala de SD
    mad_scaled = mad * 1.4826
    
    if mad_scaled < 1e-10:
        return values - median, False
    
    normalized = (values - median) / mad_scaled
    return normalized, True


# ============================================================================
# MÉTRICAS CANDIDATAS (SIMPLIFICADAS Y CORREGIDAS)
# ============================================================================

def compute_spectral_ratios(signal, fs):
    """Ratios espectrales con log-transform para estabilidad."""
    valid_bands = get_valid_bands(fs)
    powers = {}
    
    for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        if name in valid_bands:
            powers[name] = compute_band_power(signal, fs, valid_bands[name])
        else:
            powers[name] = np.nan
    
    ratios = {}
    
    # Log-transform para estabilizar ratios
    def safe_log_ratio(num, den, name):
        if den > 0 and num > 0 and not np.isnan(num) and not np.isnan(den):
            ratios[name] = np.log(num / den)
        else:
            ratios[name] = np.nan
    
    safe_log_ratio(powers['theta'], powers['beta'], 'log_theta_beta')
    safe_log_ratio(powers['delta'], powers['beta'], 'log_delta_beta')
    safe_log_ratio(powers['alpha'], powers['beta'], 'log_alpha_beta')
    safe_log_ratio(powers['theta'], powers['delta'], 'log_theta_delta')
    safe_log_ratio(powers['alpha'], powers['delta'], 'log_alpha_delta')
    safe_log_ratio(powers['beta'], powers['delta'], 'log_beta_delta')
    safe_log_ratio(powers['beta'], powers['theta'], 'log_beta_theta')
    
    # Fast/slow ratio
    fast = (powers.get('beta', 0) or 0) + (powers.get('gamma', 0) or 0)
    slow = (powers.get('delta', 0) or 0) + (powers.get('theta', 0) or 0)
    if slow > 0 and fast > 0:
        ratios['log_fast_slow'] = np.log(fast / slow)
    else:
        ratios['log_fast_slow'] = np.nan
    
    # Guardar potencias absolutas (log) para diagnóstico
    for name, p in powers.items():
        if p is not None and p > 0:
            ratios[f'log_power_{name}'] = np.log(p)
        else:
            ratios[f'log_power_{name}'] = np.nan
    
    return ratios


def compute_pac_mvl(signal, fs, phase_band, amp_band):
    """PAC con verificación Nyquist y retorno de potencias para control."""
    nyquist = fs / 2
    
    if phase_band[1] >= nyquist or amp_band[1] >= nyquist:
        return np.nan, np.nan, np.nan
    
    try:
        phase_signal = bandpass_filter(signal, phase_band[0], phase_band[1], fs)
        if phase_signal is None:
            return np.nan, np.nan, np.nan
        
        amp_signal = bandpass_filter(signal, amp_band[0], amp_band[1], fs)
        if amp_signal is None:
            return np.nan, np.nan, np.nan
        
        phase = np.angle(hilbert(phase_signal))
        amplitude = np.abs(hilbert(amp_signal))
        
        # Potencias para control de sesgo
        phase_power = np.var(phase_signal)
        amp_power = np.var(amp_signal)
        
        n = len(phase)
        mvl = np.abs(np.sum(amplitude * np.exp(1j * phase))) / n
        
        mean_amp = np.mean(amplitude)
        if mean_amp > 0:
            pac_normalized = mvl / mean_amp
        else:
            pac_normalized = np.nan
        
        return pac_normalized, phase_power, amp_power
    except:
        return np.nan, np.nan, np.nan


def compute_all_pac(signal, fs):
    """PAC múltiples con control de potencias."""
    valid_bands = get_valid_bands(fs)
    results = {}
    
    combinations = [
        ('delta', 'gamma', 'pac_delta_gamma'),
        ('delta', 'beta', 'pac_delta_beta'),
        ('theta', 'gamma', 'pac_theta_gamma'),
        ('theta', 'beta', 'pac_theta_beta'),
        ('alpha', 'gamma', 'pac_alpha_gamma'),
    ]
    
    for phase_name, amp_name, result_name in combinations:
        if phase_name in valid_bands and amp_name in valid_bands:
            pac, phase_pow, amp_pow = compute_pac_mvl(
                signal, fs, valid_bands[phase_name], valid_bands[amp_name]
            )
            results[result_name] = pac
            results[f'{result_name}_phase_power'] = phase_pow
            results[f'{result_name}_amp_power'] = amp_pow
        else:
            results[result_name] = np.nan
            results[f'{result_name}_phase_power'] = np.nan
            results[f'{result_name}_amp_power'] = np.nan
    
    return results


def compute_permutation_entropy(signal, order=3, delay=1):
    """Permutation entropy - eficiente."""
    try:
        n = len(signal)
        if n < (order - 1) * delay + order:
            return np.nan
        
        # Construir patrones
        n_patterns = n - (order - 1) * delay
        patterns = np.zeros((n_patterns, order))
        
        for i in range(order):
            patterns[:, i] = signal[i * delay:i * delay + n_patterns]
        
        # Obtener rangos
        sorted_indices = np.argsort(patterns, axis=1)
        
        # Contar usando hash
        pattern_counts = Counter(tuple(row) for row in sorted_indices)
        
        # Calcular entropía
        probs = np.array(list(pattern_counts.values())) / n_patterns
        pe = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalizar
        import math
        max_entropy = np.log2(math.factorial(order))
        return pe / max_entropy if max_entropy > 0 else np.nan
    except:
        return np.nan


def compute_lempel_ziv(signal):
    """Lempel-Ziv complexity."""
    try:
        thresh = np.median(signal)
        binary = (signal > thresh).astype(int)
        
        n = len(binary)
        s = ''.join(map(str, binary))
        
        i, k = 0, 1
        c = 1
        
        while i + k <= n:
            if s[i:i+k] not in s[0:i+k-1]:
                c += 1
                i += k
                k = 1
            else:
                k += 1
        
        # Normalizar
        return c * np.log2(n) / n if n > 0 else np.nan
    except:
        return np.nan


def compute_hjorth(signal):
    """Hjorth parameters."""
    try:
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        
        var0 = np.var(signal)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        
        if var0 <= 0 or var1 <= 0:
            return {'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}
        
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else np.nan
        
        return {
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity,
        }
    except:
        return {'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}


def compute_dfa(signal, min_box=4, max_box=None):
    """DFA simplificado."""
    try:
        n = len(signal)
        if max_box is None:
            max_box = n // 4
        
        if max_box < min_box * 2:
            return np.nan
        
        y = np.cumsum(signal - np.mean(signal))
        scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 15).astype(int))
        scales = scales[(scales >= min_box) & (scales <= max_box)]
        
        if len(scales) < 4:
            return np.nan
        
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 1:
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
            
            if rms_list:
                fluctuations.append((scale, np.mean(rms_list)))
        
        if len(fluctuations) < 4:
            return np.nan
        
        scales_valid = np.array([f[0] for f in fluctuations])
        fluct_valid = np.array([f[1] for f in fluctuations])
        
        alpha, _ = np.polyfit(np.log10(scales_valid), np.log10(fluct_valid), 1)
        return alpha
    except:
        return np.nan


def compute_spectral_slope(signal, fs, freq_range=(2, 40)):
    """
    Spectral slope (1/f exponent) - proxy de balance E/I global.
    
    Más negativo = más "rosa" = más organizado/inhibido
    Menos negativo = más "blanco" = más desorganizado/excitado
    
    Durante despertar: slope se vuelve MENOS negativo (más excitación)
    """
    try:
        # Welch PSD
        nperseg = min(256, len(signal)//2)
        if nperseg < 32:
            return np.nan, np.nan
        
        freqs, psd = welch(signal, fs, nperseg=nperseg)
        
        # Filtrar rango de frecuencias
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]
        
        if len(freqs_fit) < 5 or np.any(psd_fit <= 0):
            return np.nan, np.nan
        
        # Fit lineal en log-log
        log_freqs = np.log10(freqs_fit)
        log_psd = np.log10(psd_fit)
        
        # Slope y offset
        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
        
        # R² del fit
        predicted = slope * log_freqs + intercept
        ss_res = np.sum((log_psd - predicted)**2)
        ss_tot = np.sum((log_psd - np.mean(log_psd))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        
        return slope, r_squared
    except:
        return np.nan, np.nan


def compute_spectral_slope_bands(signal, fs):
    """
    Spectral slope en diferentes rangos de frecuencia.
    
    - Low range (1-10 Hz): más sensible a ondas lentas/delta
    - Mid range (4-30 Hz): rango típico de actividad consciente  
    - High range (10-40 Hz): más sensible a actividad rápida
    """
    results = {}
    
    # Rango bajo (dominado por delta/theta)
    slope_low, r2_low = compute_spectral_slope(signal, fs, freq_range=(1, 10))
    results['slope_1_10Hz'] = slope_low
    results['slope_1_10Hz_r2'] = r2_low
    
    # Rango medio (theta a beta)
    slope_mid, r2_mid = compute_spectral_slope(signal, fs, freq_range=(4, 30))
    results['slope_4_30Hz'] = slope_mid
    results['slope_4_30Hz_r2'] = r2_mid
    
    # Rango alto (alpha a gamma)
    slope_high, r2_high = compute_spectral_slope(signal, fs, freq_range=(10, 40))
    results['slope_10_40Hz'] = slope_high
    results['slope_10_40Hz_r2'] = r2_high
    
    # Rango amplio (estándar)
    slope_full, r2_full = compute_spectral_slope(signal, fs, freq_range=(2, 40))
    results['slope_2_40Hz'] = slope_full
    results['slope_2_40Hz_r2'] = r2_full
    
    return results


def compute_theta_phase_variability(signal, fs):
    """
    Variabilidad de fase en theta - proxy de estabilidad talamocortical.
    
    Alta variabilidad = loops talamocorticales inestables
    Baja variabilidad = loops estables/sincronizados
    """
    try:
        theta_band = BANDS['theta']
        if theta_band[1] >= fs/2:
            return {'theta_phase_std': np.nan, 'theta_phase_entropy': np.nan}
        
        # Filtrar theta
        theta_signal = bandpass_filter(signal, theta_band[0], theta_band[1], fs)
        if theta_signal is None:
            return {'theta_phase_std': np.nan, 'theta_phase_entropy': np.nan}
        
        # Fase instantánea
        analytic = hilbert(theta_signal)
        phase = np.angle(analytic)
        
        # Variabilidad de fase (circular std)
        # Usando fórmula de estadística circular
        mean_vec = np.mean(np.exp(1j * phase))
        R = np.abs(mean_vec)  # Resultant length
        circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.nan
        
        # Entropía de fase (binned)
        n_bins = 18  # 20 grados por bin
        phase_hist, _ = np.histogram(phase, bins=n_bins, range=(-np.pi, np.pi))
        phase_prob = phase_hist / np.sum(phase_hist)
        phase_prob = phase_prob[phase_prob > 0]
        phase_entropy = -np.sum(phase_prob * np.log2(phase_prob))
        
        # Normalizar entropía por máximo teórico
        max_entropy = np.log2(n_bins)
        phase_entropy_norm = phase_entropy / max_entropy
        
        return {
            'theta_phase_circular_std': circular_std,
            'theta_phase_entropy': phase_entropy_norm,
        }
    except:
        return {'theta_phase_circular_std': np.nan, 'theta_phase_entropy': np.nan}


def compute_delta_beta_pac_ratio(signal, fs):
    """
    Ratio PAC delta-beta vs PAC delta-gamma.
    
    Hipótesis: Delta-beta podría ser más "talámico" que delta-gamma.
    Si delta-beta/delta-gamma cambia antes del despertar, podría
    indicar reorganización talamocortical previa a activación cortical.
    """
    try:
        valid_bands = get_valid_bands(fs)
        
        if 'delta' not in valid_bands or 'beta' not in valid_bands or 'gamma' not in valid_bands:
            return np.nan
        
        pac_db, _, _ = compute_pac_mvl(signal, fs, valid_bands['delta'], valid_bands['beta'])
        pac_dg, _, _ = compute_pac_mvl(signal, fs, valid_bands['delta'], valid_bands['gamma'])
        
        if pac_dg > 0 and not np.isnan(pac_dg) and not np.isnan(pac_db):
            return np.log(pac_db / pac_dg)  # Log ratio para estabilidad
        return np.nan
    except:
        return np.nan


def compute_all_metrics(signal, fs):
    """Calcula todas las métricas para una ventana."""
    metrics = {}
    
    # Ratios espectrales (log-transformed)
    ratios = compute_spectral_ratios(signal, fs)
    metrics.update(ratios)
    
    # PAC múltiples
    pac = compute_all_pac(signal, fs)
    metrics.update(pac)
    
    # Complejidad (sin sample entropy - muy lento)
    metrics['perm_entropy'] = compute_permutation_entropy(signal, order=3)
    metrics['lz_complexity'] = compute_lempel_ziv(signal)
    
    # Hjorth
    hjorth = compute_hjorth(signal)
    metrics.update(hjorth)
    
    # DFA global
    metrics['dfa'] = compute_dfa(signal)
    
    # === NUEVAS MÉTRICAS: Proxies talamocorticales ===
    
    # Spectral slope (1/f) - proxy de balance E/I global
    slopes = compute_spectral_slope_bands(signal, fs)
    metrics.update(slopes)
    
    # Variabilidad de fase theta - proxy de estabilidad talamocortical
    theta_phase = compute_theta_phase_variability(signal, fs)
    metrics.update(theta_phase)
    
    # Ratio PAC delta-beta/delta-gamma
    metrics['log_pac_db_dg_ratio'] = compute_delta_beta_pac_ratio(signal, fs)
    
    return metrics


# ============================================================================
# SLEEP-EDF PROCESSOR (CORREGIDO)
# ============================================================================

class SleepEDFProcessor:
    """Procesador de datos Sleep-EDF con correcciones."""
    
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
        """Encuentra pares PSG-Hypnogram."""
        files = []
        psg_files = list(self.data_path.rglob("*PSG*.edf"))
        
        print(f"   Encontrados {len(psg_files)} archivos PSG")
        
        hyp_files = list(self.data_path.rglob("*Hypnogram*.edf"))
        hyp_dict = {}
        for hyp in hyp_files:
            name = hyp.stem
            base = name.split('-')[0].split('_')[0]
            if len(base) >= 6:
                subject_base = base[:6]
                hyp_dict[subject_base] = hyp
                hyp_dict[base] = hyp
        
        for psg_file in sorted(psg_files):
            name = psg_file.stem
            base = name.split('-')[0].split('_')[0]
            
            subject_id = base if len(base) >= 6 else base
            subject_base = base[:6] if len(base) >= 6 else base
            
            hyp_file = None
            for variant in [base, subject_base, base.replace('E', 'C'), subject_base + 'C']:
                if variant in hyp_dict:
                    hyp_file = hyp_dict[variant]
                    break
            
            if hyp_file is None:
                candidates = list(psg_file.parent.glob(f"*{subject_base}*Hypnogram*.edf"))
                if candidates:
                    hyp_file = candidates[0]
            
            if hyp_file:
                files.append((subject_id, psg_file, hyp_file))
        
        # Eliminar duplicados
        seen = set()
        unique_files = []
        for sid, psg, hyp in files:
            if sid not in seen:
                seen.add(sid)
                unique_files.append((sid, psg, hyp))
        
        print(f"   Pareados únicos: {len(unique_files)}")
        return unique_files
    
    def load_subject(self, psg_path, hyp_path):
        """Carga datos con lectura corregida de anotaciones."""
        try:
            import mne
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            # Buscar canales EEG
            channels_found = {}
            for ch in raw.ch_names:
                ch_upper = ch.upper()
                if 'FPZ' in ch_upper and 'CZ' in ch_upper:
                    channels_found['Fpz-Cz'] = ch
                elif 'PZ' in ch_upper and 'OZ' in ch_upper:
                    channels_found['Pz-Oz'] = ch
            
            if not channels_found:
                # Fallback: primer canal EEG
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        channels_found['EEG'] = ch
                        break
            
            if not channels_found:
                return None
            
            fs = raw.info['sfreq']
            
            # Cargar señales por canal
            signals = {}
            for name, ch in channels_found.items():
                raw_ch = raw.copy().pick_channels([ch])
                signals[name] = raw_ch.get_data()[0]
            
            # CORRECCIÓN: Lectura de anotaciones usando atributos
            annotations = mne.read_annotations(str(hyp_path))
            
            total_duration = len(list(signals.values())[0]) / fs
            n_epochs = int(total_duration // 30)
            stages = ['?'] * n_epochs
            
            # Acceder por atributos, no por índice
            for onset, duration, description in zip(
                annotations.onset, 
                annotations.duration, 
                annotations.description
            ):
                stage = self.STAGE_MAPPING.get(description, '?')
                start_epoch = int(onset // 30)
                end_epoch = int((onset + duration) // 30)
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            return {
                'signals': signals,
                'fs': fs,
                'stages': stages,
            }
        except Exception as e:
            print(f"\n      Error loading: {e}")
            return None
    
    def find_stable_transitions(self, stages):
        """Encuentra transiciones sueño→vigilia estables."""
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
            
            transitions.append({
                'epoch_idx': i,
                'from_stage': stages[i-1],
                'time_seconds': i * 30
            })
        
        return transitions
    
    def extract_transition_window(self, signal, fs, transition_time, pre_seconds=120, post_seconds=120):
        """Extrae ventana alrededor de transición."""
        start_sample = int((transition_time - pre_seconds) * fs)
        end_sample = int((transition_time + post_seconds) * fs)
        if start_sample < 0 or end_sample > len(signal):
            return None
        return signal[start_sample:end_sample]
    
    def compute_metrics_timeseries(self, window, fs):
        """Calcula serie temporal de métricas."""
        window_samples = int(self.window_size * fs)
        step_samples = int(self.window_step * fs)
        n_windows = (len(window) - window_samples) // step_samples + 1
        
        all_results = []
        times = []
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            segment = window[start:end]
            center_time = (start + window_samples/2) / fs - 120
            
            times.append(center_time)
            metrics = compute_all_metrics(segment, fs)
            all_results.append(metrics)
        
        df = pd.DataFrame(all_results)
        df['time'] = times
        
        return df
    
    def normalize_metrics(self, df, baseline_end=-30):
        """Normalización robusta por baseline."""
        baseline_mask = df['time'].values < baseline_end
        
        normalized = pd.DataFrame()
        normalized['time'] = df['time']
        
        normalization_success = {}
        
        for col in df.columns:
            if col == 'time':
                continue
            
            values = df[col].values.copy()
            norm_values, success = robust_normalize(values, baseline_mask)
            normalized[col] = norm_values
            normalization_success[col] = success
        
        return normalized, normalization_success
    
    def detect_onset_bidirectional(self, values, times, baseline_end=-30, direction='increase'):
        """
        Detección de onset bidireccional.
        direction: 'increase' (> threshold), 'decrease' (< -threshold), 'any' (|z| > threshold)
        """
        threshold = ONSET_DETECTION['threshold_sd']
        k_consecutive = ONSET_DETECTION['consecutive_windows']
        
        mask = np.array(times) >= baseline_end
        v = np.array(values)[mask]
        t = np.array(times)[mask]
        
        if direction == 'increase':
            above = (~np.isnan(v)) & (v > threshold)
        elif direction == 'decrease':
            above = (~np.isnan(v)) & (v < -threshold)
        else:  # 'any'
            above = (~np.isnan(v)) & (np.abs(v) > threshold)
        
        run = 0
        for i, ok in enumerate(above):
            if ok:
                run += 1
                if run >= k_consecutive:
                    return t[i - k_consecutive + 1]
            else:
                run = 0
        
        return None


# ============================================================================
# ANÁLISIS EXPLORATORIO (CORREGIDO)
# ============================================================================

class HstarV3Exploration:
    """Análisis exploratorio H* v3 con agregación por sujeto."""
    
    def __init__(self, data_path: str, output_dir: str):
        self.processor = SleepEDFProcessor(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🔬 ANÁLISIS EXPLORATORIO H* v3 (CORREGIDO)")
        print(f"{'='*70}")
        print(f"Versión: {__version__}")
        print(f"Correcciones: Nyquist check, robust norm, bidirectional onset,")
        print(f"              subject-level aggregation, annotation fix")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def get_development_ids(self, all_subject_ids):
        """Obtiene IDs del development set."""
        np.random.seed(SACRED_SEED)
        shuffled = np.array(all_subject_ids).copy()
        np.random.shuffle(shuffled)
        
        n_dev = int(len(all_subject_ids) * DEVELOPMENT_RATIO)
        return set(shuffled[:n_dev].tolist())
    
    def run(self):
        """Ejecuta análisis exploratorio."""
        
        # 1. Buscar archivos
        print("📂 Buscando archivos...")
        files = self.processor.find_psg_files()
        
        if not files:
            print("❌ No se encontraron archivos")
            return None
        
        # 2. Filtrar development set
        all_ids = [f[0] for f in files]
        dev_ids = self.get_development_ids(all_ids)
        dev_files = [(sid, psg, hyp) for sid, psg, hyp in files if sid in dev_ids]
        
        print(f"   Total sujetos: {len(files)}")
        print(f"   Development set: {len(dev_files)} sujetos")
        
        # 3. Procesar transiciones
        print(f"\n{'='*70}")
        print("PROCESANDO TRANSICIONES")
        print(f"{'='*70}")
        
        all_transitions = []
        subject_stats = []
        failed_subjects = []
        
        for i, (subject_id, psg_path, hyp_path) in enumerate(dev_files):
            # Progress con más info
            print(f"\n[{i+1}/{len(dev_files)}] {subject_id}...", end="")
            sys.stdout.flush()
            
            try:
                data = self.processor.load_subject(psg_path, hyp_path)
                if data is None:
                    failed_subjects.append((subject_id, "load_failed"))
                    print(" ❌ load failed")
                    continue
                
                transitions = self.processor.find_stable_transitions(data['stages'])
                if not transitions:
                    failed_subjects.append((subject_id, "no_transitions"))
                    print(f" ❌ no transitions (stages: {Counter(data['stages'])})")
                    continue
                
                print(f" {len(transitions)} transitions, fs={data['fs']}", end="")
                
                subject_transitions = []
                
                # Procesar por canal
                for channel_name, signal in data['signals'].items():
                    for trans in transitions:
                        window = self.processor.extract_transition_window(
                            signal, data['fs'], trans['time_seconds']
                        )
                        if window is None:
                            continue
                        
                        # Calcular métricas
                        metrics_df = self.processor.compute_metrics_timeseries(window, data['fs'])
                        
                        # Normalizar
                        metrics_norm, norm_success = self.processor.normalize_metrics(metrics_df)
                        
                        times = metrics_norm['time'].values
                        
                        # PAC delta-gamma como referencia
                        if 'pac_delta_gamma' not in metrics_norm.columns:
                            continue
                        
                        # Onset del comparador (PAC aumenta al despertar)
                        pac_ref_onset = self.processor.detect_onset_bidirectional(
                            metrics_norm['pac_delta_gamma'].values, times, direction='increase'
                        )
                        
                        if pac_ref_onset is None:
                            continue
                        
                        trans_result = {
                            'subject_id': subject_id,
                            'channel': channel_name,
                            'from_stage': trans['from_stage'],
                            'pac_ref_onset': pac_ref_onset,
                        }
                        
                        # Detectar onset para cada métrica
                        for col in metrics_norm.columns:
                            if col == 'time' or col == 'pac_delta_gamma' or '_power' in col or '_r2' in col:
                                continue
                            
                            # Determinar dirección esperada
                            # Ratios beta/delta, fast/slow: aumentan al despertar
                            # Ratios theta/beta, delta/beta: disminuyen al despertar
                            # Slopes: se vuelven MENOS negativos (aumentan) al despertar
                            if any(x in col for x in ['beta_delta', 'fast_slow', 'alpha_delta', 'beta_theta', 
                                                       'slope_', 'log_pac_db_dg_ratio']):
                                direction = 'increase'
                            elif any(x in col for x in ['theta_beta', 'delta_beta', 'alpha_beta',
                                                         'theta_phase_circular_std', 'theta_phase_entropy']):
                                direction = 'decrease'
                            else:
                                direction = 'any'  # Para métricas sin dirección clara
                            
                            onset = self.processor.detect_onset_bidirectional(
                                metrics_norm[col].values, times, direction=direction
                            )
                            
                            trans_result[f'{col}_onset'] = onset
                            
                            if onset is not None:
                                trans_result[f'{col}_delta'] = onset - pac_ref_onset
                                trans_result[f'{col}_first'] = onset < pac_ref_onset
                            else:
                                trans_result[f'{col}_delta'] = np.nan
                                trans_result[f'{col}_first'] = np.nan
                        
                        all_transitions.append(trans_result)
                        subject_transitions.append(trans_result)
                
                # Estadísticas por sujeto
                if subject_transitions:
                    subject_stats.append({
                        'subject_id': subject_id,
                        'n_transitions': len(subject_transitions),
                        'channels': list(data['signals'].keys()),
                    })
                    print(f" ✅ {len(subject_transitions)} valid")
                else:
                    failed_subjects.append((subject_id, "no_valid_onsets"))
                    print(" ❌ no valid onsets")
                    
            except Exception as e:
                failed_subjects.append((subject_id, f"error: {str(e)[:50]}"))
                print(f" ❌ error: {str(e)[:50]}")
                continue
        
        print(f"\n\n{'='*70}")
        print(f"RESUMEN DE PROCESAMIENTO")
        print(f"{'='*70}")
        print(f"   ✅ Transiciones válidas: {len(all_transitions)}")
        print(f"   ✅ Sujetos con datos: {len(subject_stats)}")
        print(f"   ❌ Sujetos fallidos: {len(failed_subjects)}")
        
        if failed_subjects:
            reasons = Counter(reason for _, reason in failed_subjects)
            print(f"\n   Razones de fallo:")
            for reason, count in reasons.most_common():
                print(f"      {reason}: {count}")
        
        if not all_transitions:
            print("❌ No hay transiciones para analizar")
            return None
        
        # 4. Análisis de precedencia
        print(f"\n{'='*70}")
        print("ANÁLISIS DE PRECEDENCIA")
        print(f"{'='*70}")
        
        results_df = pd.DataFrame(all_transitions)
        
        # === NIVEL TRANSICIÓN ===
        print("\n📊 NIVEL TRANSICIÓN:")
        
        metric_cols = [col.replace('_first', '') for col in results_df.columns 
                       if col.endswith('_first') and 'pac_delta_gamma' not in col]
        
        transition_results = []
        
        for metric in metric_cols:
            first_col = f'{metric}_first'
            delta_col = f'{metric}_delta'
            
            if first_col not in results_df.columns:
                continue
            
            valid = results_df[first_col].notna()
            n_valid = valid.sum()
            
            if n_valid < 10:
                continue
            
            n_first = results_df.loc[valid, first_col].sum()
            pct_first = 100 * n_first / n_valid
            mean_delta = results_df.loc[valid, delta_col].mean()
            std_delta = results_df.loc[valid, delta_col].std()
            
            transition_results.append({
                'metric': metric,
                'n_valid': int(n_valid),
                'n_first': int(n_first),
                'pct_first_transition': pct_first,
                'mean_delta_s': mean_delta,
                'std_delta_s': std_delta,
            })
        
        trans_df = pd.DataFrame(transition_results)
        trans_df = trans_df.sort_values('pct_first_transition', ascending=False)
        
        # === NIVEL SUJETO (CRÍTICO) ===
        print("\n📊 NIVEL SUJETO (agregado):")
        
        subject_results = []
        
        for metric in metric_cols:
            first_col = f'{metric}_first'
            
            if first_col not in results_df.columns:
                continue
            
            # Agregar por sujeto
            subject_pcts = []
            for subject_id in results_df['subject_id'].unique():
                subj_data = results_df[results_df['subject_id'] == subject_id]
                valid = subj_data[first_col].notna()
                n_valid = valid.sum()
                
                if n_valid >= 1:  # Al menos 1 transición válida
                    n_first = subj_data.loc[valid, first_col].sum()
                    pct = 100 * n_first / n_valid
                    subject_pcts.append(pct)
            
            if len(subject_pcts) < 5:  # Mínimo 5 sujetos
                continue
            
            # Estadísticas a nivel sujeto
            mean_pct = np.mean(subject_pcts)
            std_pct = np.std(subject_pcts)
            median_pct = np.median(subject_pcts)
            n_subjects_dominant = sum(1 for p in subject_pcts if p > 50)
            pct_subjects_dominant = 100 * n_subjects_dominant / len(subject_pcts)
            
            subject_results.append({
                'metric': metric,
                'n_subjects': len(subject_pcts),
                'mean_pct_first_subject': mean_pct,
                'std_pct_first_subject': std_pct,
                'median_pct_first_subject': median_pct,
                'n_subjects_dominant': n_subjects_dominant,
                'pct_subjects_dominant': pct_subjects_dominant,
            })
        
        subject_df = pd.DataFrame(subject_results)
        subject_df = subject_df.sort_values('pct_subjects_dominant', ascending=False)
        
        # Mostrar top 15
        print(f"\n   TOP 15 MÉTRICAS (nivel sujeto):")
        print(f"   {'─'*75}")
        print(f"   {'Métrica':<30} {'%Dom':>7} {'Mean%':>7} {'Med%':>7} {'N_subj':>7}")
        print(f"   {'─'*75}")
        
        for _, row in subject_df.head(15).iterrows():
            print(f"   {row['metric']:<30} {row['pct_subjects_dominant']:>6.1f}% "
                  f"{row['mean_pct_first_subject']:>6.1f}% {row['median_pct_first_subject']:>6.1f}% "
                  f"{int(row['n_subjects']):>7}")
        
        # 5. Guardar resultados
        print(f"\n{'='*70}")
        print("GUARDANDO RESULTADOS")
        print(f"{'='*70}")
        
        results_df.to_csv(self.output_dir / "all_transitions_v3.csv", index=False)
        print(f"   ✅ all_transitions_v3.csv ({len(results_df)} rows)")
        
        trans_df.to_csv(self.output_dir / "precedence_by_transition.csv", index=False)
        print(f"   ✅ precedence_by_transition.csv")
        
        subject_df.to_csv(self.output_dir / "precedence_by_subject.csv", index=False)
        print(f"   ✅ precedence_by_subject.csv")
        
        # JSON summary
        summary = {
            'version': __version__,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sacred_seed': SACRED_SEED,
            'n_subjects_processed': len(subject_stats),
            'n_subjects_failed': len(failed_subjects),
            'n_transitions_total': len(all_transitions),
            'comparator': 'pac_delta_gamma',
            'analysis_levels': ['transition', 'subject'],
            'top_10_by_subject_dominance': subject_df.head(10).to_dict('records') if len(subject_df) > 0 else [],
            'criterion_target': '>55% subjects dominant (like validation criteria)',
        }
        
        with open(self.output_dir / "exploration_summary_v3.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   ✅ exploration_summary_v3.json")
        
        # 6. Resumen final
        print(f"\n{'='*70}")
        print("CANDIDATOS H* v3")
        print(f"{'='*70}")
        
        if len(subject_df) > 0:
            # Métricas con >55% sujetos dominantes (criterio de validación)
            strong_candidates = subject_df[subject_df['pct_subjects_dominant'] > 55]
            
            if len(strong_candidates) > 0:
                print(f"\n   🏆 CANDIDATOS FUERTES (>55% sujetos dominantes):")
                for _, row in strong_candidates.iterrows():
                    print(f"      • {row['metric']}: {row['pct_subjects_dominant']:.1f}% "
                          f"(mean={row['mean_pct_first_subject']:.1f}%, n={int(row['n_subjects'])})")
                
                best = strong_candidates.iloc[0]
                print(f"\n   🥇 MEJOR CANDIDATO H* v3: {best['metric']}")
                print(f"      - {best['pct_subjects_dominant']:.1f}% sujetos con métrica dominante")
                print(f"      - Media: {best['mean_pct_first_subject']:.1f}%")
                print(f"      - N sujetos: {int(best['n_subjects'])}")
            else:
                print(f"\n   ⚠️ Ninguna métrica alcanza >55% sujetos dominantes")
                print(f"   Top 3 más cercanos:")
                for _, row in subject_df.head(3).iterrows():
                    print(f"      • {row['metric']}: {row['pct_subjects_dominant']:.1f}%")
        
        print(f"\n   📁 Resultados en: {self.output_dir}")
        
        return {
            'transition_df': trans_df,
            'subject_df': subject_df,
            'raw_df': results_df,
            'summary': summary,
        }


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    """Busca automáticamente el directorio de datos."""
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and p.is_dir():
            psg_files = list(p.glob("*PSG*.edf")) + list(p.rglob("*PSG*.edf"))
            if psg_files:
                return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AFH H* v3 Exploratory Analysis (Corrected)")
    parser.add_argument('--sleep-edf-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    
    if args.sleep_edf_path:
        data_path = Path(args.sleep_edf_path)
    else:
        print("🔍 Buscando datos...")
        data_path = find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontraron datos Sleep-EDF")
        print("   Usa: python hstar_v3_exploratory.py --sleep-edf-path TU_RUTA")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    exploration = HstarV3Exploration(str(data_path), args.output_dir)
    results = exploration.run()
    
    if results:
        print("\n" + "="*70)
        print("✅ EXPLORACIÓN COMPLETADA")
        print("="*70)


if __name__ == "__main__":
    main()