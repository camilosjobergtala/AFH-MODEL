#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CASCADA DE DESPERTAR - ANÁLISIS DESCRIPTIVO (AUDITADO)
================================================================================

Versión runtime: ver __version__
================================================================================

CORRECCIONES APLICADAS (basadas en auditoría):

1. ARTEFACTOS:
   - Rechazo por amplitud pico-pico (>200 µV)
   - Rechazo por varianza excesiva (>4 SD)
   - Rechazo por potencia alta 30-45 Hz (proxy EMG, >3 SD)
   - Rechazo por kurtosis (>5)

2. PAC:
   - Implementación con surrogates (time-shift circular)
   - Reporte de z-score vs surrogates (zPAC)
   - Banda gamma reducida a 30-40 Hz para evitar EMG

3. LEMPEL-ZIV:
   - Complejidad heurística (binarización por mediana; NO es LZ76 canónico)

4. DFA:
   - Escalas en SEGUNDOS (0.5-10s), no muestras
   - Mínimo 2 segundos de escala

5. PERMUTATION ENTROPY:
   - Order=5 (más robusto)
   - Delay ajustado a frecuencia de muestreo

6. AGREGACIÓN:
   - Nivel SUJETO primero, luego global
   - Ranking por MEDIANA (no media)
   - Reporte de IQR

7. TEMPORAL:
   - t=0 explícitamente documentado como "inicio época W anotada"
   - Incertidumbre de ±15s reportada
   - Baseline temprano en ventana (t < -2400s = < -40 min)

8. ONSET DETECTION:
   - Requiere 3 ventanas consecutivas (no 2)
   - Ventanas sin solapamiento para onset (paso = tamaño)
   - Criterio de estabilidad post-onset

================================================================================
FIX: baseline_end_s = -2400s (baseline temprano: primeros 20 min de la ventana de 60 min)
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
from collections import Counter
import zlib  # Para hash estable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

__version__ = "cascade_v2.6_60min_stable_transient"

# Manifest de reproducibilidad
REPRODUCIBILITY_MANIFEST = {
    'version': __version__,
    'python_min': '3.8',
    'seeds': {
        'split': 'SACRED_SEED (2025)',
        'formula': (
            'subject_trans_seed = crc32(subject_id.encode()) + trans_idx; '
            'window_seed = subject_trans_seed*1000 + win_idx; '
            'pac_seed = window_seed*100 + pac_idx'
        ),
    },
    'dependencies': ['numpy', 'scipy', 'pandas', 'mne', 'matplotlib'],
}

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\awakening_cascade_v2_60min_typed"

SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.7

# Bandas de frecuencia (conservadoras)
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),  # Reducido para evitar EMG
}

# VENTANA MÁXIMA - 60 MINUTOS ANTES PARA VER TODO
WINDOW_CONFIG = {
    'pre_seconds': 3600,     # 60 minutos ANTES (1 hora)
    'post_seconds': 180,     # 3 minutos DESPUÉS
    'window_size_s': 30,     # Tamaño ventana análisis
    'window_step_s': 30,     # SIN SOLAPAMIENTO para onset robusto
}

# Detección de onset
# Baseline en los primeros 20 minutos de la ventana (t < -2400s = t < -40 min)
ONSET_CONFIG = {
    'baseline_end_s': -2400,     # Baseline: t < -40 min (primeros 20 min de ventana)
    'threshold_sd': 2.0,         # Umbral
    'consecutive_windows': 3,    # 3 ventanas consecutivas
    'stability_windows': 2,      # Ventanas post-onset para confirmar dirección
}

# Control de artefactos
ARTIFACT_CONFIG = {
    'max_pp_uv': 200,           # Máximo pico-pico en µV
    'max_variance_sd': 4,       # Máxima varianza en SD vs baseline
    'max_hf_power_sd': 3,       # Máxima potencia 30-45 Hz en SD (EMG proxy)
    'max_kurtosis': 5,          # Máxima kurtosis
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'stable_w_epochs': 4,          # estable = ≥4 W seguidas (120s)
    'transient_w_epochs_max': 2,   # transitorio = ≤2 W seguidas (30-60s)
    'exclude_gray_w_epochs': [3],  # excluir 3 W seguidas (zona gris)
    'exclude_stages': ['?', 'M'],
    'allowed_pre_sleep': ['N2', 'N3'],      # Solo N2/N3 antes (estricto)
    'allowed_return_sleep': ['N1', 'N2', 'N3', 'N4'],  # Sueño para retorno transient
}

# PAC surrogates
PAC_CONFIG = {
    'n_surrogates': 50,         # Número de surrogates
    'min_shift_s': 1,           # Mínimo shift en segundos
}


def get_runtime_versions():
    """Obtiene versiones de dependencias en runtime."""
    versions = {
        'python': sys.version.replace('\n', ' '),
    }
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except:
        versions['numpy'] = 'unknown'
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except:
        versions['scipy'] = 'unknown'
    try:
        import pandas
        versions['pandas'] = pandas.__version__
    except:
        versions['pandas'] = 'unknown'
    try:
        import mne
        versions['mne'] = mne.__version__
    except:
        versions['mne'] = 'unknown'
    try:
        import matplotlib
        versions['matplotlib'] = matplotlib.__version__
    except:
        versions['matplotlib'] = 'unknown'
    try:
        import platform
        versions['platform'] = platform.platform()
    except:
        versions['platform'] = 'unknown'
    return versions


# ============================================================================
# FUNCIONES DE SEÑAL BÁSICAS
# ============================================================================

def get_valid_bands(fs):
    """Retorna bandas válidas dado el sampling rate."""
    nyquist = fs / 2
    valid = {}
    for name, (low, high) in BANDS.items():
        if high < nyquist - 1:
            valid[name] = (low, high)
        elif low < nyquist - 2:
            valid[name] = (low, min(high, nyquist - 2))
    return valid


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filtro pasa-banda Butterworth."""
    nyq = 0.5 * fs
    if highcut >= nyq:
        return None
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
    """Potencia en banda de frecuencia."""
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


# ============================================================================
# CONTROL DE ARTEFACTOS
# ============================================================================

def check_artifacts(segment, fs, baseline_stats=None, signal_scale_uv=1.0):
    """
    Verifica si un segmento tiene artefactos.
    
    Args:
        segment: señal del segmento
        fs: sampling rate
        baseline_stats: estadísticas de baseline (opcional)
        signal_scale_uv: factor para convertir a µV (determinado externamente)
    
    Returns:
        (is_clean, artifact_type)
    """
    # 1. Pico-pico en µV (usando escala determinada externamente)
    pp = np.ptp(segment)
    pp_uv = pp * signal_scale_uv
    
    if pp_uv > ARTIFACT_CONFIG['max_pp_uv']:
        return False, 'high_amplitude'
    
    # 2. Kurtosis (con parámetros explícitos)
    kurt = stats.kurtosis(segment, fisher=True, bias=False)
    if kurt > ARTIFACT_CONFIG['max_kurtosis']:
        return False, 'high_kurtosis'
    
    # 3. Potencia alta frecuencia (EMG proxy)
    hf_power = compute_band_power(segment, fs, (30, 45))
    
    # Si hf_power es NaN, marcar como calidad insuficiente
    if not np.isfinite(hf_power):
        return False, 'hf_power_nan'
    
    if baseline_stats and 'hf_mean' in baseline_stats and 'hf_std' in baseline_stats:
        if baseline_stats['hf_std'] > 0:
            if hf_power > baseline_stats['hf_mean'] + ARTIFACT_CONFIG['max_hf_power_sd'] * baseline_stats['hf_std']:
                return False, 'high_emg'
    
    # 4. Varianza
    var = np.var(segment)
    if baseline_stats and 'var_mean' in baseline_stats and 'var_std' in baseline_stats:
        if baseline_stats['var_std'] > 0:
            if var > baseline_stats['var_mean'] + ARTIFACT_CONFIG['max_variance_sd'] * baseline_stats['var_std']:
                return False, 'high_variance'
    
    return True, 'clean'


def estimate_signal_scale(signal):
    """
    Estima factor de escala para convertir señal a µV.
    
    NOTA: Sleep-EDF con MNE típicamente entrega datos en Volt.
    Simplificamos asumiendo siempre Volt → µV (1e6).
    
    Si la señal ya está en µV (p99 > 10), no escalamos.
    """
    p99 = np.percentile(np.abs(signal), 99)
    
    # Si p99 > 10, probablemente ya está en µV o similar
    if p99 > 10:
        return 1.0
    # Si p99 < 0.01, definitivamente en Volt
    elif p99 < 0.01:
        return 1e6  # V -> µV
    # Caso intermedio: asumir Volt para Sleep-EDF
    else:
        return 1e6  # Conservador: asumir V -> µV


def compute_baseline_stats(segments, fs):
    """Calcula estadísticas de baseline para detección de artefactos."""
    variances = []
    hf_powers = []
    
    for seg in segments:
        variances.append(np.var(seg))
        hf_powers.append(compute_band_power(seg, fs, (30, 45)))
    
    variances = np.array([v for v in variances if not np.isnan(v)])
    hf_powers = np.array([p for p in hf_powers if not np.isnan(p)])
    
    stats_dict = {}
    if len(variances) > 2:
        stats_dict['var_mean'] = np.median(variances)  # Mediana más robusta
        stats_dict['var_std'] = np.median(np.abs(variances - stats_dict['var_mean'])) * 1.4826
    if len(hf_powers) > 2:
        stats_dict['hf_mean'] = np.median(hf_powers)
        stats_dict['hf_std'] = np.median(np.abs(hf_powers - stats_dict['hf_mean'])) * 1.4826
    
    return stats_dict


# ============================================================================
# MÉTRICAS CORREGIDAS
# ============================================================================

def compute_spectral_powers(signal, fs):
    """Potencias espectrales absolutas y relativas."""
    valid_bands = get_valid_bands(fs)
    results = {}
    
    total_power = compute_band_power(signal, fs, (0.5, 40))
    
    for name, band in valid_bands.items():
        power = compute_band_power(signal, fs, band)
        results[f'power_{name}'] = power
        # Fix: usar np.isfinite para evitar que power=0.0 evalúe False
        if np.isfinite(total_power) and total_power > 0 and np.isfinite(power):
            results[f'relpower_{name}'] = power / total_power
        else:
            results[f'relpower_{name}'] = np.nan
    
    return results


def compute_spectral_ratios(signal, fs):
    """Ratios entre bandas (log-transformed con protección de underflow y NaN)."""
    valid_bands = get_valid_bands(fs)
    powers = {}
    for name, band in valid_bands.items():
        powers[name] = compute_band_power(signal, fs, band)
    
    results = {}
    EPS = 1e-20  # Protección contra underflow
    
    # Helper para tratar NaN como 0
    def nz(x):
        return x if np.isfinite(x) else 0.0
    
    pairs = [
        ('theta', 'delta'), ('alpha', 'delta'), ('beta', 'delta'),
        ('alpha', 'theta'), ('beta', 'theta'), ('beta', 'alpha'),
    ]
    
    for num, den in pairs:
        if num in powers and den in powers:
            p_num = powers[num]
            p_den = powers[den]
            # Fix: usar np.isfinite y clip para evitar underflow
            if np.isfinite(p_num) and np.isfinite(p_den) and p_den > 0 and p_num > 0:
                p_num_safe = max(p_num, EPS)
                p_den_safe = max(p_den, EPS)
                results[f'ratio_{num}_{den}'] = np.log(p_num_safe / p_den_safe)
            else:
                results[f'ratio_{num}_{den}'] = np.nan
    
    # Fast/slow (con manejo robusto de NaN)
    fast = nz(powers.get('beta', np.nan)) + nz(powers.get('gamma', np.nan))
    slow = nz(powers.get('delta', np.nan)) + nz(powers.get('theta', np.nan))
    
    if slow > EPS and fast > EPS:
        results['ratio_fast_slow'] = np.log(fast / slow)
    else:
        results['ratio_fast_slow'] = np.nan
    
    return results


# --- PAC CON SURROGATES ---

def compute_pac_mvl_raw(phase_signal, amp_signal):
    """Calcula PAC MVL crudo (sin normalizar)."""
    phase = np.angle(hilbert(phase_signal))
    amplitude = np.abs(hilbert(amp_signal))
    n = len(phase)
    mvl = np.abs(np.sum(amplitude * np.exp(1j * phase))) / n
    return mvl


def compute_pac_with_surrogates(signal, fs, phase_band, amp_band, n_surrogates=50, seed=None):
    """
    PAC con surrogates para z-score.
    
    Usa time-shift circular de la amplitud para crear distribución nula.
    
    Args:
        signal: señal EEG
        fs: sampling rate
        phase_band: tupla (low, high) para fase
        amp_band: tupla (low, high) para amplitud
        n_surrogates: número de surrogates
        seed: semilla para reproducibilidad (opcional)
    """
    nyquist = fs / 2
    if phase_band[1] >= nyquist or amp_band[1] >= nyquist:
        return np.nan, np.nan
    
    try:
        phase_signal = bandpass_filter(signal, phase_band[0], phase_band[1], fs)
        if phase_signal is None:
            return np.nan, np.nan
        amp_signal = bandpass_filter(signal, amp_band[0], amp_band[1], fs)
        if amp_signal is None:
            return np.nan, np.nan
        
        # PAC real
        pac_real = compute_pac_mvl_raw(phase_signal, amp_signal)
        
        # Surrogates por time-shift circular (con RNG determinista)
        min_shift = int(PAC_CONFIG['min_shift_s'] * fs)
        max_shift = len(amp_signal) - min_shift
        
        if max_shift <= min_shift:
            return pac_real, np.nan
        
        # Usar RNG local para reproducibilidad
        rng = np.random.default_rng(seed)
        
        surrogate_pacs = []
        for _ in range(n_surrogates):
            shift = rng.integers(min_shift, max_shift)
            amp_shifted = np.roll(amp_signal, shift)
            pac_surr = compute_pac_mvl_raw(phase_signal, amp_shifted)
            surrogate_pacs.append(pac_surr)
        
        surrogate_pacs = np.array(surrogate_pacs)
        
        # Z-score
        mean_surr = np.mean(surrogate_pacs)
        std_surr = np.std(surrogate_pacs)
        
        if std_surr > 0:
            zpac = (pac_real - mean_surr) / std_surr
        else:
            zpac = 0
        
        return pac_real, zpac
        
    except:
        return np.nan, np.nan


def compute_all_pacs(signal, fs, window_seed=None):
    """Todos los PACs con z-score y semilla determinista."""
    valid_bands = get_valid_bands(fs)
    results = {}
    
    # Combinaciones relevantes (fase lenta → amplitud rápida)
    combinations = [
        ('delta', 'beta', 'pac_delta_beta'),
        ('delta', 'gamma', 'pac_delta_gamma'),
        ('theta', 'beta', 'pac_theta_beta'),
        ('theta', 'gamma', 'pac_theta_gamma'),
        ('alpha', 'beta', 'pac_alpha_beta'),
        ('alpha', 'gamma', 'pac_alpha_gamma'),
    ]
    
    for idx, (phase_name, amp_name, result_name) in enumerate(combinations):
        if phase_name in valid_bands and amp_name in valid_bands:
            # Seed único por combinación PAC dentro de la ventana
            pac_seed = None
            if window_seed is not None:
                pac_seed = window_seed * 100 + idx
            
            pac_raw, zpac = compute_pac_with_surrogates(
                signal, fs, 
                valid_bands[phase_name], 
                valid_bands[amp_name],
                n_surrogates=PAC_CONFIG['n_surrogates'],
                seed=pac_seed
            )
            results[f'{result_name}_raw'] = pac_raw
            results[f'{result_name}_z'] = zpac  # Z-score es la métrica principal
        else:
            results[f'{result_name}_raw'] = np.nan
            results[f'{result_name}_z'] = np.nan
    
    return results


# --- HJORTH ---

def compute_hjorth(signal):
    """Parámetros de Hjorth."""
    try:
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        
        var0 = np.var(signal)
        var1 = np.var(diff1)
        var2 = np.var(diff2)
        
        if var0 <= 0 or var1 <= 0:
            return {'hjorth_activity': np.nan, 'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}
        
        activity = var0
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else np.nan
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity,
        }
    except:
        return {'hjorth_activity': np.nan, 'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}


# --- DFA CORREGIDO (escalas en segundos) ---

def compute_dfa(signal, fs, min_scale_s=0.5, max_scale_s=10):
    """
    DFA con escalas en SEGUNDOS.
    """
    try:
        n = len(signal)
        
        # Convertir segundos a muestras
        min_box = max(4, int(min_scale_s * fs))
        max_box = min(n // 4, int(max_scale_s * fs))
        
        if max_box <= min_box * 2:
            return np.nan
        
        y = np.cumsum(signal - np.mean(signal))
        
        # Escalas logarítmicamente espaciadas
        n_scales = 15
        scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), n_scales).astype(int))
        scales = scales[(scales >= min_box) & (scales <= max_box)]
        
        if len(scales) < 4:
            return np.nan
        
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 2:
                continue
            
            rms_list = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) < scale:
                    continue
                x = np.arange(len(segment))
                try:
                    coeffs = np.polyfit(x, segment, 1)
                    trend = np.polyval(coeffs, x)
                    rms_list.append(np.sqrt(np.mean((segment - trend)**2)))
                except:
                    continue
            
            if rms_list:
                fluctuations.append((scale, np.mean(rms_list)))
        
        if len(fluctuations) < 4:
            return np.nan
        
        scales_valid = np.array([f[0] for f in fluctuations])
        fluct_valid = np.array([f[1] for f in fluctuations])
        
        # Evitar log de cero
        if np.any(fluct_valid <= 0):
            return np.nan
        
        alpha, _ = np.polyfit(np.log10(scales_valid), np.log10(fluct_valid), 1)
        return alpha
    except:
        return np.nan


# --- PERMUTATION ENTROPY CORREGIDO (order=5) ---

def compute_permutation_entropy(signal, order=5, delay=1):
    """
    Permutation entropy con order=5 (más robusto).
    """
    try:
        n = len(signal)
        n_patterns = n - (order - 1) * delay
        
        if n_patterns < order * 10:  # Mínimo de patrones
            return np.nan
        
        patterns = np.zeros((n_patterns, order))
        for i in range(order):
            patterns[:, i] = signal[i * delay:i * delay + n_patterns]
        
        sorted_indices = np.argsort(patterns, axis=1)
        pattern_counts = Counter(tuple(row) for row in sorted_indices)
        
        probs = np.array(list(pattern_counts.values())) / n_patterns
        pe = -np.sum(probs * np.log2(probs + 1e-10))
        
        import math
        max_entropy = np.log2(math.factorial(order))
        return pe / max_entropy if max_entropy > 0 else np.nan
    except:
        return np.nan


# --- LEMPEL-ZIV CORREGIDO (LZ76 estándar) ---

def compute_lempel_ziv_heuristic(signal):
    """
    Lempel-Ziv complexity - implementación heurística basada en búsqueda de subcadenas.
    
    NOTA: Esta es una aproximación, no el parsing LZ76 estándar de Kaspar & Schuster.
    Útil como métrica relativa de complejidad, pero no comparable directamente
    con valores de implementaciones LZ76 canónicas.
    
    Args:
        signal: señal EEG
        
    Returns:
        Complejidad normalizada (0-1 aproximadamente)
    """
    try:
        # Binarizar por mediana
        thresh = np.median(signal)
        binary = ''.join(['1' if x > thresh else '0' for x in signal])
        
        n = len(binary)
        if n == 0:
            return np.nan
        
        # Algoritmo heurístico de subcadenas
        i = 0
        c = 1  # Complejidad
        l = 1  # Longitud de la palabra actual
        
        while i + l <= n:
            substring = binary[i:i+l]
            prefix = binary[0:i+l-1]
            
            if substring in prefix:
                l += 1
            else:
                c += 1
                i += l
                l = 1
        
        # Normalización
        if n > 1:
            b = n / np.log2(n)
            lz_norm = c / b
        else:
            lz_norm = np.nan
        
        return lz_norm
    except:
        return np.nan


# --- SPECTRAL SLOPE ---

def compute_spectral_slope(signal, fs, freq_range=(2, 20), exclude_range=(8, 13)):
    """
    Spectral slope (1/f exponent) excluyendo banda alfa.
    
    Args:
        signal: señal EEG
        fs: sampling rate
        freq_range: rango total para fit
        exclude_range: rango a excluir (alfa por defecto)
    
    Returns:
        slope del fit log-log
    """
    try:
        nperseg = min(256, len(signal)//2)
        if nperseg < 32:
            return np.nan
        
        freqs, psd = welch(signal, fs, nperseg=nperseg)
        
        # Máscara que incluye freq_range pero excluye exclude_range
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if exclude_range:
            mask &= ~((freqs >= exclude_range[0]) & (freqs <= exclude_range[1]))
        
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]
        
        if len(freqs_fit) < 5 or np.any(psd_fit <= 0):
            return np.nan
        
        log_freqs = np.log10(freqs_fit)
        log_psd = np.log10(psd_fit)
        
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        return slope
    except:
        return np.nan


# --- TODAS LAS MÉTRICAS ---

def compute_all_metrics(signal, fs, window_seed=None):
    """Calcula TODAS las métricas para una ventana."""
    metrics = {}
    
    # Potencias espectrales
    powers = compute_spectral_powers(signal, fs)
    metrics.update(powers)
    
    # Ratios (log-transformed)
    ratios = compute_spectral_ratios(signal, fs)
    metrics.update(ratios)
    
    # PACs con z-score (semilla determinista)
    pacs = compute_all_pacs(signal, fs, window_seed=window_seed)
    metrics.update(pacs)
    
    # Hjorth
    hjorth = compute_hjorth(signal)
    metrics.update(hjorth)
    
    # DFA (escalas en segundos)
    metrics['dfa'] = compute_dfa(signal, fs, min_scale_s=0.5, max_scale_s=10)
    
    # Permutation entropy (order=5)
    delay = max(1, int(fs / 100))  # Delay adaptado a fs
    metrics['perm_entropy'] = compute_permutation_entropy(signal, order=5, delay=delay)
    
    # Lempel-Ziv (heurístico, no LZ76 estándar)
    metrics['lz_heuristic'] = compute_lempel_ziv_heuristic(signal)
    
    # Spectral slope (2-20 Hz excluyendo alfa 8-13 Hz)
    metrics['slope_aperiodic'] = compute_spectral_slope(signal, fs, (2, 20), exclude_range=(8, 13))
    
    return metrics


# ============================================================================
# SLEEP-EDF LOADER
# ============================================================================

class SleepEDFLoader:
    """Cargador de datos Sleep-EDF."""
    
    STAGE_MAPPING = {
        'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3', 'Sleep stage 4': 'N4', 'Sleep stage R': 'R',
        'Sleep stage ?': '?', 'Movement time': 'M'
    }
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.pairing_log = []  # Para auditoría
        self.duration_mismatches = []  # Registrar inconsistencias PSG↔Hyp
    
    def find_psg_files(self):
        """Encuentra pares PSG-Hypnogram con logging."""
        files = []
        psg_files = list(self.data_path.rglob("*PSG*.edf"))
        hyp_files = list(self.data_path.rglob("*Hypnogram*.edf"))
        
        hyp_dict = {}
        for hyp in hyp_files:
            name = hyp.stem
            base = name.split('-')[0].split('_')[0]
            if len(base) >= 6:
                hyp_dict[base[:6]] = hyp
                hyp_dict[base] = hyp
        
        for psg_file in sorted(psg_files):
            name = psg_file.stem
            base = name.split('-')[0].split('_')[0]
            subject_id = base
            subject_base = base[:6] if len(base) >= 6 else base
            
            hyp_file = None
            match_method = None
            
            for variant in [base, subject_base, base.replace('E', 'C'), subject_base + 'C']:
                if variant in hyp_dict:
                    hyp_file = hyp_dict[variant]
                    match_method = f'dict:{variant}'
                    break
            
            if hyp_file is None:
                candidates = list(psg_file.parent.glob(f"*{subject_base}*Hypnogram*.edf"))
                if candidates:
                    hyp_file = candidates[0]
                    match_method = 'glob'
            
            if hyp_file:
                files.append((subject_id, psg_file, hyp_file))
                self.pairing_log.append({
                    'subject_id': subject_id,
                    'psg': str(psg_file),
                    'hyp': str(hyp_file),
                    'method': match_method
                })
        
        # Eliminar duplicados
        seen = set()
        unique = []
        for sid, psg, hyp in files:
            if sid not in seen:
                seen.add(sid)
                unique.append((sid, psg, hyp))
        
        return unique
    
    def load_subject(self, psg_path, hyp_path, target_channel='Fpz-Cz'):
        """
        Carga datos de un sujeto.
        
        Prioriza el canal especificado para consistencia.
        Valida consistencia temporal PSG↔Hypnogram.
        """
        try:
            import mne
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            # Buscar canal objetivo
            channel = None
            for ch in raw.ch_names:
                ch_upper = ch.upper()
                if target_channel.upper().replace('-', '') in ch_upper.replace('-', ''):
                    channel = ch
                    break
            
            # Fallback a cualquier EEG
            if channel is None:
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        channel = ch
                        break
            
            if channel is None:
                return None
            
            fs = raw.info['sfreq']
            raw_ch = raw.copy().pick_channels([channel])
            signal = raw_ch.get_data()[0]
            
            # Duración del PSG
            psg_duration_s = len(signal) / fs
            
            # Cargar anotaciones
            annotations = mne.read_annotations(str(hyp_path))
            
            # Calcular duración total del hypnograma
            if len(annotations) > 0:
                hyp_duration_s = annotations.onset[-1] + annotations.duration[-1]
            else:
                return None
            
            # VALIDACIÓN: Verificar consistencia temporal (tolerancia = 1 época = 30s)
            duration_diff = abs(psg_duration_s - hyp_duration_s)
            if duration_diff > 60:  # Más de 2 épocas de diferencia
                # Registrar como sospechoso pero intentar continuar
                self.duration_mismatches.append({
                    'psg': str(psg_path),
                    'hyp': str(hyp_path),
                    'psg_duration_s': psg_duration_s,
                    'hyp_duration_s': hyp_duration_s,
                    'diff_s': duration_diff,
                })
            
            total_duration = len(signal) / fs
            n_epochs = int(total_duration // 30)
            stages = ['?'] * n_epochs
            
            for onset, duration, description in zip(
                annotations.onset, annotations.duration, annotations.description
            ):
                stage = self.STAGE_MAPPING.get(description, '?')
                start_epoch = int(np.floor(onset / 30))
                end_epoch = int(np.ceil((onset + duration) / 30))
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            # Estimar escala de señal para detección de artefactos
            signal_scale = estimate_signal_scale(signal)
            
            return {
                'signal': signal,
                'fs': fs,
                'stages': stages,
                'channel': channel,
                'signal_scale_uv': signal_scale,
                'psg_duration_s': psg_duration_s,
                'hyp_duration_s': hyp_duration_s,
            }
        except Exception as e:
            return None
    
    def find_transitions(self, stages):
        """
        Encuentra transiciones N2/N3 -> W y las clasifica como:
          - 'stable'    : >= 4 épocas W consecutivas (120s+)
          - 'transient' : <= 2 épocas W consecutivas y luego retorna a sueño
        Excluye zona gris: 3 épocas W consecutivas.
        """
        transitions = []
        exclude = set(TRANSITION_CRITERIA['exclude_stages'])
        min_sleep = TRANSITION_CRITERIA['min_sleep_epochs_before']

        stable_k = TRANSITION_CRITERIA['stable_w_epochs']            # 4
        transient_max = TRANSITION_CRITERIA['transient_w_epochs_max'] # 2
        gray_list = set(TRANSITION_CRITERIA['exclude_gray_w_epochs']) # {3}

        allowed_pre = set(TRANSITION_CRITERIA.get('allowed_pre_sleep', ['N2', 'N3']))
        allowed_return = set(TRANSITION_CRITERIA.get('allowed_return_sleep', ['N1', 'N2', 'N3', 'N4']))

        n = len(stages)

        def has_excluded(arr):
            return any(s in exclude for s in arr)

        i = min_sleep
        while i < n - 1:
            # Buscamos primer W que venga desde N2/N3
            if stages[i] != 'W':
                i += 1
                continue

            # Debe venir desde N2/N3 (estricto)
            if stages[i-1] not in allowed_pre:
                i += 1
                continue

            # Precondición: las min_sleep épocas previas deben ser sueño permitido
            pre = stages[i-min_sleep:i]
            if has_excluded(pre) or any(s == 'W' for s in pre):
                i += 1
                continue
            if not all(s in allowed_pre.union({'N1','N4'}) for s in pre):
                i += 1
                continue

            # Contar run de W consecutivas desde i
            run = 0
            j = i
            while j < n and stages[j] == 'W':
                run += 1
                j += 1

            # Excluir si hay ? o M en la vecindad inmediata (pre y post cercano)
            neighborhood = []
            neighborhood.extend(stages[max(0, i-2):i])     # 2 previas
            neighborhood.extend(stages[i:min(n, i+2)])     # 2 desde i
            if has_excluded(neighborhood):
                i += 1
                continue

            # Excluir si hay ? o M en el post-run inmediato (j y j+1), crucial para transient
            post_neighborhood = stages[j:min(n, j+2)]
            if has_excluded(post_neighborhood):
                i += 1
                continue

            # Clasificar
            label = None

            if run >= stable_k:
                label = 'stable'

            elif run <= transient_max:
                # transient: debe retornar a sueño permitido con ≥2 épocas de sueño
                if j+1 < n and stages[j] in allowed_return and stages[j+1] in allowed_return:
                    label = 'transient'
                else:
                    label = None

            elif run in gray_list:
                label = None  # zona gris excluida

            # Registrar transición si clasificó
            if label is not None:
                transitions.append({
                    'epoch_idx': i,
                    'from_stage': stages[i-1],
                    'time_seconds': i * 30,
                    'uncertainty_s': 15,
                    'type': label,
                    'w_run_epochs': run,
                })

                # Saltar hasta el final del bloque W para evitar duplicar
                i = j
            else:
                i += 1

        return transitions


# ============================================================================
# ANALIZADOR DE CASCADA (CORREGIDO)
# ============================================================================

class AwakeningCascadeAnalyzer:
    """Analizador descriptivo de la cascada de despertar (auditado)."""
    
    def __init__(self, data_path, output_dir):
        self.loader = SleepEDFLoader(data_path)
        self.output_dir = Path(output_dir)
        # Crear directorio con fallback a os.makedirs (más robusto en Google Drive)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            import os
            os.makedirs(str(self.output_dir), exist_ok=True)
        
        self.pre_s = WINDOW_CONFIG['pre_seconds']
        self.post_s = WINDOW_CONFIG['post_seconds']
        self.win_size = WINDOW_CONFIG['window_size_s']
        self.win_step = WINDOW_CONFIG['window_step_s']
        
        print(f"\n{'='*70}")
        print(f"🔬 CASCADA DE DESPERTAR {__version__}")
        print(f"{'='*70}")
        print(f"Versión: {__version__}")
        print(f"Ventana: [{-self.pre_s}s, +{self.post_s}s]")
        print(f"Ventana análisis: {self.win_size}s, paso: {self.win_step}s (sin solapamiento)")
        print(f"Baseline: t < {ONSET_CONFIG['baseline_end_s']}s")
        print(f"Threshold: {ONSET_CONFIG['threshold_sd']} SD, {ONSET_CONFIG['consecutive_windows']} ventanas")
        print(f"PAC surrogates: {PAC_CONFIG['n_surrogates']}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}")
        print(f"\n⚠️  NOTA: t=0 = inicio de época W anotada (±{15}s incertidumbre)")
        print(f"{'='*70}\n")
    
    def extract_window(self, signal, fs, transition_time):
        """Extrae ventana ampliada alrededor de transición."""
        start = int((transition_time - self.pre_s) * fs)
        end = int((transition_time + self.post_s) * fs)
        if start < 0 or end > len(signal):
            return None
        return signal[start:end]
    
    def segment_window(self, window, fs):
        """Divide ventana en segmentos sin solapamiento."""
        win_samples = int(self.win_size * fs)
        step_samples = int(self.win_step * fs)
        
        segments = []
        times = []
        
        i = 0
        while i + win_samples <= len(window):
            seg = window[i:i+win_samples]
            center_time = (i + win_samples/2) / fs - self.pre_s
            segments.append(seg)
            times.append(center_time)
            i += step_samples
        
        return segments, times
    
    def compute_timeseries(self, segments, times, fs, baseline_stats, signal_scale_uv=1.0, subject_trans_seed=None):
        """Computa métricas para cada segmento con control de artefactos."""
        results = []
        valid_times = []
        artifact_count = {'clean': 0, 'rejected': 0, 'by_type': {}}
        
        for win_idx, (seg, t) in enumerate(zip(segments, times)):
            # Verificar artefactos (con escala correcta)
            is_clean, artifact_type = check_artifacts(seg, fs, baseline_stats, signal_scale_uv)
            
            if not is_clean:
                artifact_count['rejected'] += 1
                artifact_count['by_type'][artifact_type] = artifact_count['by_type'].get(artifact_type, 0) + 1
                continue
            
            artifact_count['clean'] += 1
            
            # Seed determinista por ventana
            window_seed = None
            if subject_trans_seed is not None:
                window_seed = subject_trans_seed * 1000 + win_idx
            
            # Calcular métricas
            metrics = compute_all_metrics(seg, fs, window_seed)
            results.append(metrics)
            valid_times.append(t)
        
        if not results:
            return None, None, artifact_count
        
        df = pd.DataFrame(results)
        df['time'] = valid_times
        
        return df, valid_times, artifact_count
    
    def normalize_timeseries(self, df):
        """Normaliza por baseline usando MAD robusto."""
        baseline_mask = df['time'].values < ONSET_CONFIG['baseline_end_s']
        
        if baseline_mask.sum() < 3:
            return None
        
        normalized = pd.DataFrame()
        normalized['time'] = df['time']
        
        for col in df.columns:
            if col == 'time':
                continue
            
            baseline_vals = df.loc[baseline_mask, col].dropna()
            if len(baseline_vals) < 3:
                normalized[col] = np.nan
                continue
            
            median = np.median(baseline_vals)
            mad = np.median(np.abs(baseline_vals - median)) * 1.4826
            
            if mad < 1e-10:
                normalized[col] = df[col] - median
            else:
                normalized[col] = (df[col] - median) / mad
        
        return normalized
    
    def detect_onset(self, values, times):
        """
        Detecta onset: primer momento donde |z| > threshold 
        por k ventanas CONSECUTIVAS.
        """
        threshold = ONSET_CONFIG['threshold_sd']
        k = ONSET_CONFIG['consecutive_windows']
        
        values = np.array(values)
        times = np.array(times)
        
        above = (~np.isnan(values)) & (np.abs(values) > threshold)
        
        run = 0
        for i, ok in enumerate(above):
            if ok:
                run += 1
                if run >= k:
                    return times[i - k + 1]
            else:
                run = 0
        
        return np.nan
    
    def detect_direction(self, values, times, onset_time):
        """
        Determina dirección del cambio con criterio de estabilidad.
        """
        if np.isnan(onset_time):
            return 'none'
        
        times = np.array(times)
        values = np.array(values)
        
        # Ventanas post-onset para estabilidad
        n_stability = ONSET_CONFIG['stability_windows']
        mask = times >= onset_time
        post_values = values[mask][:n_stability * 2]
        
        if len(post_values) < n_stability:
            return 'unknown'
        
        mean_post = np.nanmean(post_values)
        
        if mean_post > ONSET_CONFIG['threshold_sd'] * 0.5:
            return 'increase'
        elif mean_post < -ONSET_CONFIG['threshold_sd'] * 0.5:
            return 'decrease'
        else:
            return 'unstable'
    
    def analyze_transition(self, df_norm):
        """Analiza una transición: extrae onset y dirección."""
        times = df_norm['time'].values
        results = {}
        
        for col in df_norm.columns:
            if col == 'time':
                continue
            
            values = df_norm[col].values
            onset = self.detect_onset(values, times)
            direction = self.detect_direction(values, times, onset)
            
            results[col] = {
                'onset_time': onset,
                'direction': direction,
            }
        
        return results
    
    def run(self):
        """Ejecuta análisis completo."""
        
        # 1. Buscar archivos
        print("📂 Buscando archivos...")
        files = self.loader.find_psg_files()
        print(f"   Encontrados: {len(files)} sujetos")
        
        # Guardar log de emparejamiento
        pd.DataFrame(self.loader.pairing_log).to_csv(
            self.output_dir / "psg_hyp_pairing.csv", index=False
        )
        
        # NOTA: duration_mismatches se guarda AL FINAL del run() después de procesar todos los sujetos
        
        # 2. Filtrar development set
        np.random.seed(SACRED_SEED)
        all_ids = [f[0] for f in files]
        shuffled = np.array(all_ids).copy()
        np.random.shuffle(shuffled)
        n_dev = int(len(all_ids) * DEVELOPMENT_RATIO)
        dev_ids = set(shuffled[:n_dev])
        holdout_ids = set(shuffled[n_dev:])
        dev_files = [(sid, psg, hyp) for sid, psg, hyp in files if sid in dev_ids]
        
        # Guardar split
        split_info = {
            'development': list(dev_ids),
            'holdout': list(holdout_ids),
            'seed': SACRED_SEED,
        }
        with open(self.output_dir / "train_test_split.json", 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"   Development: {len(dev_files)} sujetos")
        print(f"   Holdout: {len(holdout_ids)} sujetos (reservado)")
        
        # 3. Procesar transiciones
        print(f"\n{'='*70}")
        print("PROCESANDO TRANSICIONES")
        print(f"{'='*70}")
        
        # Estructura para agregación por sujeto
        subject_onsets = {}  # subject_id -> list of transition results
        total_artifacts = {'clean': 0, 'rejected': 0, 'by_type': Counter()}
        subject_artifact_stats = {}  # Por sujeto
        transition_counts = Counter()  # NUEVO: conteo stable vs transient
        
        for i, (subject_id, psg_path, hyp_path) in enumerate(dev_files):
            print(f"\r[{i+1}/{len(dev_files)}] {subject_id}...", end="", flush=True)
            
            data = self.loader.load_subject(psg_path, hyp_path)
            if data is None:
                continue
            
            transitions = self.loader.find_transitions(data['stages'])
            if not transitions:
                continue
            
            subject_results = []
            
            for trans_idx, trans in enumerate(transitions):
                # Contar tipo de transición
                transition_counts[trans.get('type', 'unknown')] += 1
                
                window = self.extract_window(data['signal'], data['fs'], trans['time_seconds'])
                if window is None:
                    continue
                
                # Segmentar
                segments, times = self.segment_window(window, data['fs'])
                
                # Calcular estadísticas de baseline para control de artefactos
                baseline_segments = [seg for seg, t in zip(segments, times) 
                                    if t < ONSET_CONFIG['baseline_end_s']]
                baseline_stats = compute_baseline_stats(baseline_segments, data['fs'])
                
                # Seed determinista por sujeto×transición (hash estable)
                subject_trans_seed = zlib.crc32(subject_id.encode()) + trans_idx
                
                # Calcular métricas con control de artefactos
                df_metrics, valid_times, artifact_count = self.compute_timeseries(
                    segments, times, data['fs'], baseline_stats,
                    signal_scale_uv=data.get('signal_scale_uv', 1.0),
                    subject_trans_seed=subject_trans_seed
                )
                
                total_artifacts['clean'] += artifact_count['clean']
                total_artifacts['rejected'] += artifact_count['rejected']
                # Acumular por tipo
                for atype, count in artifact_count.get('by_type', {}).items():
                    total_artifacts['by_type'][atype] += count
                
                # Por sujeto
                if subject_id not in subject_artifact_stats:
                    subject_artifact_stats[subject_id] = {'clean': 0, 'rejected': 0, 'by_type': Counter()}
                subject_artifact_stats[subject_id]['clean'] += artifact_count['clean']
                subject_artifact_stats[subject_id]['rejected'] += artifact_count['rejected']
                for atype, count in artifact_count.get('by_type', {}).items():
                    subject_artifact_stats[subject_id]['by_type'][atype] += count
                
                if df_metrics is None or len(df_metrics) < 5:
                    continue
                
                # Normalizar
                df_norm = self.normalize_timeseries(df_metrics)
                if df_norm is None:
                    continue
                
                # Analizar
                trans_results = self.analyze_transition(df_norm)
                trans_results['_meta'] = {
                    'from_stage': trans['from_stage'],
                    'transition_type': trans.get('type', 'unknown'),
                    'w_run_epochs': trans.get('w_run_epochs', None),
                    'n_clean_windows': artifact_count['clean'],
                }
                
                subject_results.append(trans_results)
            
            if subject_results:
                subject_onsets[subject_id] = subject_results
        
        print(f"\n\n   ✅ Sujetos con datos: {len(subject_onsets)}")
        print(f"   ✅ Ventanas limpias: {total_artifacts['clean']}")
        print(f"   ❌ Ventanas rechazadas (artefactos): {total_artifacts['rejected']}")
        print(f"   📊 Conteo transiciones: {dict(transition_counts)}")
        
        if not subject_onsets:
            print("❌ No hay datos para analizar")
            return None
        
        # 4. Agregar por SUJETO (crítico para evitar sobre-representación)
        print(f"\n{'='*70}")
        print("AGREGANDO POR SUJETO")
        print(f"{'='*70}")
        
        # Extraer métricas únicas
        sample_results = list(subject_onsets.values())[0][0]
        metrics = [k for k in sample_results.keys() if not k.startswith('_')]
        
        # Para cada métrica, calcular onset medio por sujeto
        subject_level_onsets = []
        
        for subject_id, transitions in subject_onsets.items():
            # Contar tipos de transición por sujeto
            types = []
            for t in transitions:
                m = t.get('_meta', {})
                types.append(m.get('transition_type', 'unknown'))
            c = Counter(types)
            
            subject_row = {
                'subject_id': subject_id, 
                'n_transitions': len(transitions),
                'n_stable': c.get('stable', 0),
                'n_transient': c.get('transient', 0),
            }
            
            for metric in metrics:
                onsets = [t[metric]['onset_time'] for t in transitions 
                         if metric in t and not np.isnan(t[metric]['onset_time'])]
                
                if onsets:
                    subject_row[f'{metric}_onset_median'] = np.median(onsets)
                    subject_row[f'{metric}_onset_mean'] = np.mean(onsets)
                    subject_row[f'{metric}_n_valid'] = len(onsets)
                else:
                    subject_row[f'{metric}_onset_median'] = np.nan
                    subject_row[f'{metric}_onset_mean'] = np.nan
                    subject_row[f'{metric}_n_valid'] = 0
            
            subject_level_onsets.append(subject_row)
        
        df_subject = pd.DataFrame(subject_level_onsets)
        
        # 5. Calcular ranking global (basado en medianas por sujeto)
        print(f"\n{'='*70}")
        print("CALCULANDO SECUENCIA TEMPORAL")
        print(f"{'='*70}")
        
        sequence_stats = []
        
        for metric in metrics:
            col = f'{metric}_onset_median'
            if col not in df_subject.columns:
                continue
            
            values = df_subject[col].dropna()
            
            if len(values) < 5:  # Mínimo 5 sujetos
                continue
            
            # Estadísticas robustas
            median_onset = np.median(values)
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            iqr = q75 - q25
            mean_onset = np.mean(values)
            n_subjects = len(values)
            
            # % sujetos con onset antes de t=0
            pct_before = 100 * (values < 0).sum() / len(values)
            
            sequence_stats.append({
                'metric': metric,
                'median_onset_s': median_onset,
                'iqr_s': iqr,
                'q25_s': q25,
                'q75_s': q75,
                'mean_onset_s': mean_onset,
                'n_subjects': n_subjects,
                'pct_before_t0': pct_before,
            })
        
        df_sequence = pd.DataFrame(sequence_stats)
        df_sequence = df_sequence.sort_values('median_onset_s')  # Ordenar por MEDIANA
        
        # 5b. Rankings SEPARADOS por tipo de transición (stable vs transient)
        print(f"\n{'='*70}")
        print("CALCULANDO SECUENCIAS POR TIPO DE TRANSICIÓN")
        print(f"{'='*70}")
        
        def compute_sequence_by_type(subject_onsets, trans_type, metrics):
            """Calcula secuencia temporal solo para un tipo de transición."""
            type_stats = []
            
            for metric in metrics:
                all_onsets = []
                for subject_id, transitions in subject_onsets.items():
                    # Filtrar por tipo
                    type_onsets = [
                        t[metric]['onset_time'] 
                        for t in transitions 
                        if t.get('_meta', {}).get('transition_type') == trans_type
                        and metric in t 
                        and not np.isnan(t[metric]['onset_time'])
                    ]
                    if type_onsets:
                        # Mediana por sujeto (evitar sobre-representación)
                        all_onsets.append(np.median(type_onsets))
                
                if len(all_onsets) >= 5:  # Mínimo 5 sujetos
                    values = np.array(all_onsets)
                    type_stats.append({
                        'metric': metric,
                        'median_onset_s': np.median(values),
                        'iqr_s': np.percentile(values, 75) - np.percentile(values, 25),
                        'q25_s': np.percentile(values, 25),
                        'q75_s': np.percentile(values, 75),
                        'mean_onset_s': np.mean(values),
                        'n_subjects': len(values),
                        'pct_before_t0': 100 * (values < 0).sum() / len(values),
                    })
            
            if type_stats:
                df = pd.DataFrame(type_stats)
                return df.sort_values('median_onset_s')
            return pd.DataFrame()
        
        df_sequence_stable = compute_sequence_by_type(subject_onsets, 'stable', metrics)
        df_sequence_transient = compute_sequence_by_type(subject_onsets, 'transient', metrics)
        
        print(f"   Stable: {len(df_sequence_stable)} métricas con ≥5 sujetos")
        print(f"   Transient: {len(df_sequence_transient)} métricas con ≥5 sujetos")
        
        # 6. Mostrar resultados
        print(f"\n   SECUENCIA TEMPORAL (ordenada por mediana de onset):")
        print(f"   {'─'*80}")
        print(f"   {'Métrica':<30} {'Mediana':>10} {'IQR':>10} {'%<t=0':>8} {'N_subj':>8}")
        print(f"   {'─'*80}")
        
        for _, row in df_sequence.head(25).iterrows():
            print(f"   {row['metric']:<30} {row['median_onset_s']:>+9.1f}s "
                  f"{row['iqr_s']:>9.1f}s {row['pct_before_t0']:>7.1f}% "
                  f"{int(row['n_subjects']):>8}")
        
        # 7. Guardar resultados
        print(f"\n{'='*70}")
        print("GUARDANDO RESULTADOS")
        print(f"{'='*70}")
        
        df_subject.to_csv(self.output_dir / "subject_level_onsets.csv", index=False)
        print(f"   ✅ subject_level_onsets.csv ({len(df_subject)} sujetos)")
        
        df_sequence.to_csv(self.output_dir / "sequence_ranking.csv", index=False)
        print(f"   ✅ sequence_ranking.csv ({len(df_sequence)} métricas)")
        
        # Rankings separados por tipo
        if len(df_sequence_stable) > 0:
            df_sequence_stable.to_csv(self.output_dir / "sequence_ranking_stable.csv", index=False)
            print(f"   ✅ sequence_ranking_stable.csv ({len(df_sequence_stable)} métricas)")
        
        if len(df_sequence_transient) > 0:
            df_sequence_transient.to_csv(self.output_dir / "sequence_ranking_transient.csv", index=False)
            print(f"   ✅ sequence_ranking_transient.csv ({len(df_sequence_transient)} métricas)")
        
        # Summary JSON
        runtime_versions = get_runtime_versions()
        summary = {
            'version': __version__,
            'reproducibility': REPRODUCIBILITY_MANIFEST,
            'runtime_versions': runtime_versions,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': {
                'window_pre_s': self.pre_s,
                'window_post_s': self.post_s,
                'window_size_s': self.win_size,
                'window_step_s': self.win_step,
                'baseline_end_s': ONSET_CONFIG['baseline_end_s'],
                'threshold_sd': ONSET_CONFIG['threshold_sd'],
                'consecutive_windows': ONSET_CONFIG['consecutive_windows'],
                'pac_surrogates': PAC_CONFIG['n_surrogates'],
            },
            'data': {
                'n_subjects': len(subject_onsets),
                'n_transitions_total': sum(len(t) for t in subject_onsets.values()),
                'transition_counts_development': dict(transition_counts),
                'n_subjects_with_stable': int((df_subject['n_stable'] > 0).sum()),
                'n_subjects_with_transient': int((df_subject['n_transient'] > 0).sum()),
                'windows_clean': total_artifacts['clean'],
                'windows_rejected': total_artifacts['rejected'],
                'rejection_rate': total_artifacts['rejected'] / (total_artifacts['clean'] + total_artifacts['rejected'] + 1e-10),
                'psg_hyp_mismatches': len(self.loader.duration_mismatches),
            },
            'temporal_uncertainty': '±15s (epoch resolution)',
            'sequence_top10': df_sequence.head(10).to_dict('records'),
            'sequence_bottom10': df_sequence.tail(10).to_dict('records'),
            'sequence_top10_stable': df_sequence_stable.head(10).to_dict('records') if len(df_sequence_stable) > 0 else [],
            'sequence_top10_transient': df_sequence_transient.head(10).to_dict('records') if len(df_sequence_transient) > 0 else [],
            'notes': [
                't=0 = inicio de primera época W anotada',
                'Onset = primer momento con |z| > threshold por k ventanas consecutivas',
                'Ranking basado en MEDIANA por sujeto (robusto a outliers)',
                'PAC reportado como z-score vs surrogates',
                'Ventanas con HF power no estimable se rechazan (hf_power_nan)',
                'lz_heuristic es aproximación basada en subcadenas, no LZ76 canónico',
                'Baseline temprano: baseline_end_s = -2400s (primeros 20 min de la ventana de 60 min)',
            ],
        }
        
        with open(self.output_dir / "cascade_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   ✅ cascade_summary.json")
        
        # Guardar duration mismatches (ahora sí, después de procesar todos los sujetos)
        if self.loader.duration_mismatches:
            pd.DataFrame(self.loader.duration_mismatches).to_csv(
                self.output_dir / "duration_mismatches.csv", index=False
            )
            print(f"   ✅ duration_mismatches.csv ({len(self.loader.duration_mismatches)} pares)")
        
        # Guardar resumen de artefactos (completo con breakdown por tipo)
        artifacts_summary = {
            'global': {
                'total_clean': total_artifacts['clean'],
                'total_rejected': total_artifacts['rejected'],
                'rejection_rate': total_artifacts['rejected'] / (total_artifacts['clean'] + total_artifacts['rejected'] + 1e-10),
                'by_type': dict(total_artifacts['by_type']),
            },
            'by_subject': {
                sid: {
                    'clean': subj_stats['clean'],
                    'rejected': subj_stats['rejected'],
                    'rejection_rate': subj_stats['rejected'] / (subj_stats['clean'] + subj_stats['rejected'] + 1e-10),
                    'by_type': dict(subj_stats['by_type']),
                }
                for sid, subj_stats in subject_artifact_stats.items()
            }
        }
        with open(self.output_dir / "artifacts_summary.json", 'w') as f:
            json.dump(artifacts_summary, f, indent=2)
        print(f"   ✅ artifacts_summary.json (global + {len(subject_artifact_stats)} sujetos)")
        
        # 8. Crear visualización
        self.create_cascade_plot(df_sequence)
        
        # Plots separados por tipo
        if len(df_sequence_stable) > 0:
            self.create_cascade_plot(df_sequence_stable, suffix='_stable', title_extra=' (STABLE)')
        if len(df_sequence_transient) > 0:
            self.create_cascade_plot(df_sequence_transient, suffix='_transient', title_extra=' (TRANSIENT)')
        
        # 9. Resumen final
        print(f"\n{'='*70}")
        print("RESUMEN DE LA CASCADA DE DESPERTAR")
        print(f"{'='*70}")
        
        print(f"\n   MÉTRICAS QUE CAMBIAN PRIMERO (onset más temprano):")
        for _, row in df_sequence.head(10).iterrows():
            print(f"      {row['median_onset_s']:>+7.1f}s (IQR={row['iqr_s']:.1f}s): {row['metric']}")
        
        print(f"\n   MÉTRICAS QUE CAMBIAN ÚLTIMO (onset más tardío):")
        for _, row in df_sequence.tail(5).iterrows():
            print(f"      {row['median_onset_s']:>+7.1f}s (IQR={row['iqr_s']:.1f}s): {row['metric']}")
        
        # Patrones
        before_zero = df_sequence[df_sequence['median_onset_s'] < 0]
        after_zero = df_sequence[df_sequence['median_onset_s'] >= 0]
        
        print(f"\n   PATRONES:")
        print(f"      Métricas con onset ANTES de anotación: {len(before_zero)}")
        print(f"      Métricas con onset DESPUÉS de anotación: {len(after_zero)}")
        
        # Comparación stable vs transient
        if len(df_sequence_stable) > 0 and len(df_sequence_transient) > 0:
            print(f"\n   COMPARACIÓN STABLE vs TRANSIENT:")
            # Métricas en común
            common_metrics = set(df_sequence_stable['metric']).intersection(set(df_sequence_transient['metric']))
            if common_metrics:
                print(f"      Métricas en común: {len(common_metrics)}")
                # Top 5 diferencias
                diffs = []
                for m in common_metrics:
                    s_row = df_sequence_stable.loc[df_sequence_stable['metric'] == m, 'median_onset_s']
                    t_row = df_sequence_transient.loc[df_sequence_transient['metric'] == m, 'median_onset_s']
                    if len(s_row) == 1 and len(t_row) == 1:
                        stable_onset = float(s_row.iloc[0])
                        trans_onset = float(t_row.iloc[0])
                        diffs.append((m, stable_onset, trans_onset, stable_onset - trans_onset))
                diffs.sort(key=lambda x: abs(x[3]), reverse=True)
                print(f"      Top diferencias (stable - transient):")
                for m, s, t, d in diffs[:5]:
                    print(f"         {m}: {s:+.0f}s vs {t:+.0f}s (Δ={d:+.0f}s)")
        
        if len(before_zero) > 0:
            first = before_zero.iloc[0]
            print(f"\n   🥇 Primera métrica en cambiar: {first['metric']}")
            print(f"      Mediana: {first['median_onset_s']:.1f}s, IQR: {first['iqr_s']:.1f}s")
        
        print(f"\n   📁 Resultados en: {self.output_dir}")
        print(f"\n   ⚠️  RECORDATORIO: t=0 tiene ±15s de incertidumbre")
        
        return {
            'subject_df': df_subject,
            'sequence_df': df_sequence,
            'sequence_stable_df': df_sequence_stable,
            'sequence_transient_df': df_sequence_transient,
            'summary': summary,
        }
    
    def create_cascade_plot(self, df_sequence, suffix='', title_extra=''):
        """Crea visualización de la cascada."""
        try:
            fig, ax = plt.subplots(figsize=(14, 12))
            
            df_plot = df_sequence.dropna(subset=['median_onset_s']).copy()
            
            if len(df_plot) == 0:
                return
            
            # Colores por tipo
            colors = []
            for metric in df_plot['metric']:
                if 'pac_' in metric:
                    colors.append('#e74c3c')
                elif 'power_' in metric or 'relpower_' in metric:
                    colors.append('#3498db')
                elif 'ratio_' in metric:
                    colors.append('#2ecc71')
                elif 'hjorth' in metric or 'dfa' in metric or 'entropy' in metric or 'lz_' in metric:
                    colors.append('#9b59b6')
                elif 'slope' in metric:
                    colors.append('#f39c12')
                else:
                    colors.append('#95a5a6')
            
            y_pos = np.arange(len(df_plot))
            
            # Barras con IQR como error
            ax.barh(y_pos, df_plot['median_onset_s'], 
                   xerr=[df_plot['median_onset_s'] - df_plot['q25_s'],
                         df_plot['q75_s'] - df_plot['median_onset_s']],
                   color=colors, alpha=0.7, capsize=3)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_plot['metric'], fontsize=8)
            
            # Línea en t=0 con banda de incertidumbre
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.axvspan(-15, 15, alpha=0.2, color='gray', label='Incertidumbre anotación (±15s)')
            
            ax.set_xlabel('Tiempo de onset (segundos, relativo a anotación)', fontsize=12)
            ax.set_title(f'CASCADA DE DESPERTAR {__version__}{title_extra}\n'
                        'Secuencia temporal de cambios (mediana ± IQR por sujeto)', fontsize=14)
            
            # Leyenda
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#e74c3c', alpha=0.7, label='PAC (z-score)'),
                Patch(facecolor='#3498db', alpha=0.7, label='Potencias'),
                Patch(facecolor='#2ecc71', alpha=0.7, label='Ratios'),
                Patch(facecolor='#9b59b6', alpha=0.7, label='Complejidad'),
                Patch(facecolor='#f39c12', alpha=0.7, label='Slope'),
                Patch(facecolor='gray', alpha=0.2, label='±15s incertidumbre'),
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            filename = f"cascade_plot{suffix}.png"
            plt.savefig(self.output_dir / filename, dpi=150)
            plt.close()
            
            print(f"   ✅ {filename}")
        except Exception as e:
            print(f"   ⚠️ No se pudo crear gráfico: {e}")


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and p.is_dir():
            if list(p.glob("*PSG*.edf")) or list(p.rglob("*PSG*.edf")):
                return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Awakening Cascade ({__version__})")
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
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    analyzer = AwakeningCascadeAnalyzer(str(data_path), args.output_dir)
    results = analyzer.run()
    
    if results:
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)


if __name__ == "__main__":
    main()