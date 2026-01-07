#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CASCADA DE DESPERTAR - ANÁLISIS DESCRIPTIVO (v3.6 - DURATION FIX)
================================================================================

CAMBIOS v3.6 vs v3.5:
1. max_duration_mismatch_s: 30 → 36000 (permite PSG/HYP con duraciones diferentes)

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
from scipy.signal import butter, filtfilt, hilbert, welch, detrend
from collections import Counter
import zlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

__version__ = "cascade_v3.6_duration_fix"

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.7

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\awakening_cascade_v3.6"

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),
}

WINDOW_CONFIG = {
    'pre_seconds': 1800,
    'post_seconds': 180,
    'window_size_s': 30,
    'window_step_s': 30,
}

ONSET_CONFIG = {
    'baseline_end_s': -900,
    'threshold_sd': 2.0,
    'consecutive_windows': 3,
    'direction_windows': 3,
    'min_baseline_windows': 10,
    'confirmation_threshold': 0.80,
    'oscillating_mean_abs_max': 0.5,
}

WELCH_CONFIG = {
    'nperseg_target': 256,
    'nperseg_min': 32,
}

ARTIFACT_CONFIG = {
    'max_pp_uv': 200,
    'max_variance_sd': 4,
    'max_hf_power_sd': 3,
    'max_kurtosis': 5,
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'stable_w_epochs': 4,
    'transient_w_epochs_max': 2,
    'exclude_gray_w_epochs': [3],
    'exclude_stages': ['?', 'M'],
    'allowed_pre_sleep': ['N2', 'N3'],
    'allowed_return_sleep': ['N1', 'N2', 'N3', 'N4'],
}

PAC_CONFIG = {
    'n_surrogates': 50,
    'min_shift_s': 1,
}

# v3.6: FIX CRÍTICO
STRICT_FILTERS = {
    'max_duration_mismatch_s': 36000,  # 10 horas (era 30s)
}

RANKING_CONFIG = {
    'min_subjects': 3,
}

SENSITIVITY_CONFIG = {
    'enabled': True,
    'threshold_sd_values': [1.5, 2.0, 2.5],
    'consecutive_windows_values': [2, 3, 4],
    'confirmation_thresholds': [0.66, 1.0],
}

REPRODUCIBILITY_MANIFEST = {
    'version': __version__,
    'python_min': '3.8',
    'seeds': {
        'split': 'SACRED_SEED (2025)',
        'formula': 'session_seed = crc32(session_id); trans_seed = session_seed + trans_idx',
    },
    'dependencies': ['numpy', 'scipy', 'pandas', 'mne', 'matplotlib'],
}


def get_full_config():
    return {
        'BANDS': BANDS,
        'WINDOW_CONFIG': WINDOW_CONFIG,
        'ONSET_CONFIG': ONSET_CONFIG,
        'ARTIFACT_CONFIG': ARTIFACT_CONFIG,
        'TRANSITION_CRITERIA': TRANSITION_CRITERIA,
        'PAC_CONFIG': PAC_CONFIG,
        'STRICT_FILTERS': STRICT_FILTERS,
        'WELCH_CONFIG': WELCH_CONFIG,
        'RANKING_CONFIG': RANKING_CONFIG,
        'SACRED_SEED': SACRED_SEED,
    }


def get_runtime_versions():
    versions = {'python': sys.version.replace('\n', ' ')}
    for pkg in ['numpy', 'scipy', 'pandas', 'mne', 'matplotlib']:
        try:
            mod = __import__(pkg)
            versions[pkg] = mod.__version__
        except:
            versions[pkg] = 'unknown'
    return versions


# ============================================================================
# FUNCIONES DE SEÑAL
# ============================================================================

def get_valid_bands(fs):
    nyquist = fs / 2
    valid = {}
    for name, (low, high) in BANDS.items():
        if high < nyquist - 1:
            valid[name] = (low, high)
        elif low < nyquist - 2:
            valid[name] = (low, min(high, nyquist - 2))
    return valid


def bandpass_filter(data, lowcut, highcut, fs, order=4):
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


def compute_band_power(signal, fs, band, nperseg=None):
    if nperseg is None:
        nperseg = WELCH_CONFIG['nperseg_target']
    nyquist = fs / 2
    if band[1] >= nyquist:
        return np.nan
    try:
        actual_nperseg = min(nperseg, len(signal))
        if actual_nperseg < WELCH_CONFIG['nperseg_min']:
            return np.nan
        freqs, psd = welch(signal, fs, nperseg=actual_nperseg, detrend='constant')
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if len(idx) == 0:
            return np.nan
        return np.trapz(psd[idx], freqs[idx])
    except:
        return np.nan


# ============================================================================
# ARTEFACTOS
# ============================================================================

def check_artifacts_minimal(segment, fs, signal_scale_uv=1.0):
    pp = np.ptp(segment)
    pp_uv = pp * signal_scale_uv
    if pp_uv > ARTIFACT_CONFIG['max_pp_uv']:
        return False, 'high_amplitude'
    
    kurt = stats.kurtosis(segment, fisher=True, bias=False)
    if kurt > ARTIFACT_CONFIG['max_kurtosis']:
        return False, 'high_kurtosis'
    
    return True, 'clean'


def compute_hf_power_emg_proxy(segment, fs):
    nyq = fs / 2
    high = min(45, nyq - 2)
    if high <= 30:
        return np.nan
    return compute_band_power(segment, fs, (30, high))


def check_artifacts_full(segment, fs, baseline_stats, signal_scale_uv=1.0):
    is_clean, artifact_type = check_artifacts_minimal(segment, fs, signal_scale_uv)
    if not is_clean:
        return is_clean, artifact_type
    
    hf_power = compute_hf_power_emg_proxy(segment, fs)
    if not np.isfinite(hf_power):
        return False, 'hf_power_nan'
    
    if baseline_stats and 'hf_mean' in baseline_stats and 'hf_std' in baseline_stats:
        if baseline_stats['hf_std'] > 0:
            if hf_power > baseline_stats['hf_mean'] + ARTIFACT_CONFIG['max_hf_power_sd'] * baseline_stats['hf_std']:
                return False, 'high_emg'
    
    var = np.var(segment)
    if baseline_stats and 'var_mean' in baseline_stats and 'var_std' in baseline_stats:
        if baseline_stats['var_std'] > 0:
            if var > baseline_stats['var_mean'] + ARTIFACT_CONFIG['max_variance_sd'] * baseline_stats['var_std']:
                return False, 'high_variance'
    
    return True, 'clean'


def compute_baseline_stats_clean(segments, fs, signal_scale_uv=1.0):
    clean_segments = []
    for seg in segments:
        is_clean, _ = check_artifacts_minimal(seg, fs, signal_scale_uv)
        if is_clean:
            clean_segments.append(seg)
    
    if len(clean_segments) < 3:
        return {}
    
    variances = []
    hf_powers = []
    
    for seg in clean_segments:
        variances.append(np.var(seg))
        hf_powers.append(compute_hf_power_emg_proxy(seg, fs))
    
    variances = np.array([v for v in variances if np.isfinite(v)])
    hf_powers = np.array([p for p in hf_powers if np.isfinite(p)])
    
    stats_dict = {'n_clean_baseline': len(clean_segments)}
    
    if len(variances) > 2:
        stats_dict['var_mean'] = np.median(variances)
        stats_dict['var_std'] = np.median(np.abs(variances - stats_dict['var_mean'])) * 1.4826
    
    if len(hf_powers) > 2:
        stats_dict['hf_mean'] = np.median(hf_powers)
        stats_dict['hf_std'] = np.median(np.abs(hf_powers - stats_dict['hf_mean'])) * 1.4826
    
    return stats_dict


def estimate_signal_scale(signal):
    p99 = np.percentile(np.abs(signal), 99)
    if p99 > 10:
        return 1.0
    else:
        return 1e6


# ============================================================================
# MÉTRICAS
# ============================================================================

def compute_spectral_powers(signal, fs):
    valid_bands = get_valid_bands(fs)
    results = {}
    total_power = compute_band_power(signal, fs, (0.5, 40))
    
    for name, band in valid_bands.items():
        power = compute_band_power(signal, fs, band)
        results[f'power_{name}'] = power
        if np.isfinite(total_power) and total_power > 0 and np.isfinite(power):
            results[f'relpower_{name}'] = power / total_power
        else:
            results[f'relpower_{name}'] = np.nan
    
    return results


def compute_spectral_ratios(signal, fs):
    valid_bands = get_valid_bands(fs)
    powers = {name: compute_band_power(signal, fs, band) for name, band in valid_bands.items()}
    
    results = {}
    EPS = 1e-20
    
    def nz(x):
        return x if np.isfinite(x) else 0.0
    
    pairs = [('theta', 'delta'), ('alpha', 'delta'), ('beta', 'delta'),
             ('alpha', 'theta'), ('beta', 'theta'), ('beta', 'alpha')]
    
    for num, den in pairs:
        if num in powers and den in powers:
            p_num, p_den = powers[num], powers[den]
            if np.isfinite(p_num) and np.isfinite(p_den) and p_den > 0 and p_num > 0:
                results[f'ratio_{num}_{den}'] = np.log(max(p_num, EPS) / max(p_den, EPS))
            else:
                results[f'ratio_{num}_{den}'] = np.nan
    
    fast = nz(powers.get('beta', np.nan)) + nz(powers.get('gamma', np.nan))
    slow = nz(powers.get('delta', np.nan)) + nz(powers.get('theta', np.nan))
    results['ratio_fast_slow'] = np.log(fast / slow) if slow > EPS and fast > EPS else np.nan
    
    return results


def compute_spectral_entropy(signal, fs, freq_range=(0.5, 40)):
    try:
        nperseg = min(WELCH_CONFIG['nperseg_target'], len(signal))
        if nperseg < WELCH_CONFIG['nperseg_min']:
            return np.nan
        freqs, psd = welch(signal, fs, nperseg=nperseg, detrend='constant')
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        psd_band = psd[mask]
        if len(psd_band) < 4:
            return np.nan
        psd_norm = psd_band / (np.sum(psd_band) + 1e-10)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        return entropy / np.log2(len(psd_norm)) if len(psd_norm) > 1 else np.nan
    except:
        return np.nan


def compute_pac_mvl_raw(phase_signal, amp_signal):
    phase = np.angle(hilbert(phase_signal))
    amplitude = np.abs(hilbert(amp_signal))
    return np.abs(np.sum(amplitude * np.exp(1j * phase))) / len(phase)


def compute_pac_with_surrogates(signal, fs, phase_band, amp_band, n_surrogates=50, seed=None):
    nyquist = fs / 2
    if phase_band[1] >= nyquist or amp_band[1] >= nyquist:
        return np.nan, np.nan
    
    try:
        phase_signal = bandpass_filter(signal, phase_band[0], phase_band[1], fs)
        amp_signal = bandpass_filter(signal, amp_band[0], amp_band[1], fs)
        if phase_signal is None or amp_signal is None:
            return np.nan, np.nan
        
        pac_real = compute_pac_mvl_raw(phase_signal, amp_signal)
        
        min_shift = int(PAC_CONFIG['min_shift_s'] * fs)
        max_shift = len(amp_signal) - min_shift
        if max_shift <= min_shift:
            return pac_real, np.nan
        
        rng = np.random.default_rng(seed)
        surrogate_pacs = []
        for _ in range(n_surrogates):
            shift = rng.integers(min_shift, max_shift)
            pac_surr = compute_pac_mvl_raw(phase_signal, np.roll(amp_signal, shift))
            surrogate_pacs.append(pac_surr)
        
        mean_surr = np.mean(surrogate_pacs)
        std_surr = np.std(surrogate_pacs)
        zpac = (pac_real - mean_surr) / std_surr if std_surr > 1e-10 else np.nan
        
        return pac_real, zpac
    except:
        return np.nan, np.nan


def compute_all_pacs(signal, fs, window_seed=None):
    valid_bands = get_valid_bands(fs)
    results = {}
    combinations = [
        ('delta', 'beta', 'pac_delta_beta'), ('delta', 'gamma', 'pac_delta_gamma'),
        ('theta', 'beta', 'pac_theta_beta'), ('theta', 'gamma', 'pac_theta_gamma'),
        ('alpha', 'beta', 'pac_alpha_beta'), ('alpha', 'gamma', 'pac_alpha_gamma'),
    ]
    
    for idx, (phase_name, amp_name, result_name) in enumerate(combinations):
        if phase_name in valid_bands and amp_name in valid_bands:
            pac_seed = (window_seed * 100 + idx) if (window_seed is not None) else None
            pac_raw, zpac = compute_pac_with_surrogates(
                signal, fs, valid_bands[phase_name], valid_bands[amp_name],
                n_surrogates=PAC_CONFIG['n_surrogates'], seed=pac_seed
            )
            results[f'{result_name}_raw'] = pac_raw
            results[f'{result_name}_z'] = zpac
        else:
            results[f'{result_name}_raw'] = np.nan
            results[f'{result_name}_z'] = np.nan
    return results


def compute_hjorth(signal):
    try:
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var0, var1, var2 = np.var(signal), np.var(diff1), np.var(diff2)
        if var0 <= 0 or var1 <= 0:
            return {'hjorth_activity': np.nan, 'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}
        mobility = np.sqrt(var1 / var0)
        complexity = np.sqrt(var2 / var1) / mobility if mobility > 0 else np.nan
        return {'hjorth_activity': var0, 'hjorth_mobility': mobility, 'hjorth_complexity': complexity}
    except:
        return {'hjorth_activity': np.nan, 'hjorth_mobility': np.nan, 'hjorth_complexity': np.nan}


def compute_dfa(signal, fs, min_scale_s=0.5, max_scale_s=10):
    try:
        signal = detrend(signal, type='linear')
        n = len(signal)
        min_box = max(4, int(min_scale_s * fs))
        max_box = min(n // 4, int(max_scale_s * fs))
        if max_box <= min_box * 2:
            return np.nan
        
        y = np.cumsum(signal - np.mean(signal))
        scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 15).astype(int))
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
                    rms_list.append(np.sqrt(np.mean((segment - np.polyval(coeffs, x))**2)))
                except:
                    continue
            if rms_list:
                fluctuations.append((scale, np.mean(rms_list)))
        
        if len(fluctuations) < 4:
            return np.nan
        scales_valid = np.array([f[0] for f in fluctuations])
        fluct_valid = np.array([f[1] for f in fluctuations])
        if np.any(fluct_valid <= 0):
            return np.nan
        alpha, _ = np.polyfit(np.log10(scales_valid), np.log10(fluct_valid), 1)
        return alpha
    except:
        return np.nan


def compute_permutation_entropy(signal, order=5, delay=1):
    try:
        signal = detrend(signal, type='linear')
        n = len(signal)
        n_patterns = n - (order - 1) * delay
        if n_patterns < order * 10:
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


def compute_lempel_ziv_heuristic(signal):
    try:
        signal = detrend(signal, type='linear')
        thresh = np.median(signal)
        binary = ''.join(['1' if x > thresh else '0' for x in signal])
        n = len(binary)
        if n == 0:
            return np.nan
        
        i, c, l = 0, 1, 1
        while i + l <= n:
            if binary[i:i+l] in binary[0:i+l-1]:
                l += 1
            else:
                c += 1
                i += l
                l = 1
        
        return c / (n / np.log2(n)) if n > 1 else np.nan
    except:
        return np.nan


def compute_spectral_slope(signal, fs, freq_range=(2, 20), exclude_range=(8, 13)):
    try:
        nperseg = min(WELCH_CONFIG['nperseg_target'], len(signal))
        if nperseg < WELCH_CONFIG['nperseg_min']:
            return np.nan
        freqs, psd = welch(signal, fs, nperseg=nperseg, detrend='constant')
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if exclude_range:
            mask &= ~((freqs >= exclude_range[0]) & (freqs <= exclude_range[1]))
        freqs_fit, psd_fit = freqs[mask], psd[mask]
        if len(freqs_fit) < 5 or np.any(psd_fit <= 0):
            return np.nan
        slope, _ = np.polyfit(np.log10(freqs_fit), np.log10(psd_fit), 1)
        return slope
    except:
        return np.nan


def compute_all_metrics(signal, fs, window_seed=None):
    metrics = {}
    metrics.update(compute_spectral_powers(signal, fs))
    metrics.update(compute_spectral_ratios(signal, fs))
    metrics.update(compute_all_pacs(signal, fs, window_seed))
    metrics.update(compute_hjorth(signal))
    metrics['dfa'] = compute_dfa(signal, fs)
    delay = max(1, int(fs / 100))
    metrics['perm_entropy'] = compute_permutation_entropy(signal, order=5, delay=delay)
    metrics['lz_heuristic'] = compute_lempel_ziv_heuristic(signal)
    metrics['slope_aperiodic'] = compute_spectral_slope(signal, fs)
    metrics['spectral_entropy'] = compute_spectral_entropy(signal, fs)
    return metrics


# ============================================================================
# SLEEP-EDF LOADER
# ============================================================================

class SleepEDFLoader:
    STAGE_MAPPING = {
        'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3', 'Sleep stage 4': 'N4', 'Sleep stage R': 'R',
        'Sleep stage ?': '?', 'Movement time': 'M'
    }
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.pairing_log = []
        self.duration_mismatches = []
        self.channel_log = []
    
    def find_psg_files(self):
        files = []
        psg_files = list(self.data_path.rglob("*PSG*.edf"))
        hyp_files = list(self.data_path.rglob("*Hypnogram*.edf"))
        
        hyp_by_folder = {}
        hyp_by_stem = {}
        
        for hyp in hyp_files:
            folder = str(hyp.parent)
            if folder not in hyp_by_folder:
                hyp_by_folder[folder] = []
            hyp_by_folder[folder].append(hyp)
            if hyp.stem not in hyp_by_stem:
                hyp_by_stem[hyp.stem] = []
            hyp_by_stem[hyp.stem].append(hyp)
        
        for psg_file in sorted(psg_files):
            psg_name = psg_file.stem
            psg_folder = str(psg_file.parent)
            psg_token = psg_name.split('-')[0]
            base = psg_token.split('_')[0]
            subject_id = base[:6] if len(base) >= 6 else base
            session_id = f"{subject_id}_{psg_name}"
            
            hyp_file = None
            match_method = None
            match_confidence = None
            
            expected_hyp_name = psg_name.replace('PSG', 'Hypnogram')
            if expected_hyp_name in hyp_by_stem:
                candidates = [h for h in hyp_by_stem[expected_hyp_name] if str(h.parent) == psg_folder]
                if len(candidates) == 1:
                    hyp_file = candidates[0]
                    match_method = 'exact_name'
                    match_confidence = 'high'
                elif len(candidates) > 1:
                    self.pairing_log.append({
                        'session_id': session_id, 'subject_id': subject_id,
                        'psg': str(psg_file), 'hyp': None,
                        'method': 'rejected_multiple_exact', 'match_confidence': 'rejected',
                    })
                    continue
            
            if hyp_file is None and psg_folder in hyp_by_folder:
                matching = [h for h in hyp_by_folder[psg_folder] if h.stem.split('-')[0] == psg_token]
                if len(matching) == 1:
                    hyp_file = matching[0]
                    match_method = 'same_token'
                    match_confidence = 'high'
                elif len(matching) > 1:
                    self.pairing_log.append({
                        'session_id': session_id, 'subject_id': subject_id,
                        'psg': str(psg_file), 'hyp': None,
                        'method': 'rejected_ambiguous_token', 'match_confidence': 'rejected',
                    })
                    continue
            
            if hyp_file is None and psg_folder in hyp_by_folder:
                matching = []
                for h in hyp_by_folder[psg_folder]:
                    h_token = h.stem.split('-')[0]
                    h_subj = h_token[:6] if len(h_token) >= 6 else h_token
                    if h_subj == subject_id:
                        matching.append(h)
                if len(matching) == 1:
                    hyp_file = matching[0]
                    match_method = 'same_subject'
                    match_confidence = 'medium'
                elif len(matching) > 1:
                    self.pairing_log.append({
                        'session_id': session_id, 'subject_id': subject_id,
                        'psg': str(psg_file), 'hyp': None,
                        'method': 'rejected_ambiguous_subject', 'match_confidence': 'rejected',
                    })
                    continue
            
            if hyp_file is None:
                self.pairing_log.append({
                    'session_id': session_id, 'subject_id': subject_id,
                    'psg': str(psg_file), 'hyp': None,
                    'method': 'no_match', 'match_confidence': 'rejected',
                })
                continue
            
            files.append({
                'session_id': session_id, 'subject_id': subject_id,
                'psg_path': psg_file, 'hyp_path': hyp_file,
                'match_confidence': match_confidence,
            })
            self.pairing_log.append({
                'session_id': session_id, 'subject_id': subject_id,
                'psg': str(psg_file), 'hyp': str(hyp_file),
                'method': match_method, 'match_confidence': match_confidence,
            })
        
        seen = set()
        unique = []
        for f in files:
            if str(f['psg_path']) not in seen:
                seen.add(str(f['psg_path']))
                unique.append(f)
        return unique
    
    def load_session(self, psg_path, hyp_path, session_id, target_channel='Fpz-Cz'):
        try:
            import mne
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            channel = None
            for ch in raw.ch_names:
                if target_channel.upper().replace('-', '') in ch.upper().replace('-', ''):
                    channel = ch
                    break
            if channel is None:
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        channel = ch
                        break
            if channel is None:
                return None
            
            self.channel_log.append({
                'session_id': session_id, 'channel': channel, 'n_channels': len(raw.ch_names)
            })
            
            fs = raw.info['sfreq']
            signal = raw.copy().pick_channels([channel]).get_data()[0]
            psg_duration_s = len(signal) / fs
            
            signal_scale = estimate_signal_scale(signal)
            p99_abs = float(np.percentile(np.abs(signal), 99))
            
            self.channel_log[-1].update({
                'fs': fs,
                'signal_scale_uv': signal_scale,
                'p99_abs': p99_abs,
            })
            
            annotations = mne.read_annotations(str(hyp_path))
            if len(annotations) == 0:
                return None
            
            hyp_duration_s = annotations.onset[-1] + annotations.duration[-1]
            duration_diff = abs(psg_duration_s - hyp_duration_s)
            
            # v3.6: Solo log, no rechazar
            if duration_diff > 30:
                self.duration_mismatches.append({
                    'session_id': session_id, 'diff_s': duration_diff
                })
            
            if duration_diff > STRICT_FILTERS['max_duration_mismatch_s']:
                return None
            
            n_epochs = int(psg_duration_s // 30)
            stages = ['?'] * n_epochs
            
            for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
                stage = self.STAGE_MAPPING.get(desc, '?')
                start_epoch = int(np.floor(onset / 30))
                end_epoch = int(np.ceil((onset + duration) / 30))
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            return {
                'signal': signal, 'fs': fs, 'stages': stages,
                'channel': channel, 'signal_scale_uv': estimate_signal_scale(signal),
            }
        except:
            return None
    
    def find_transitions(self, stages):
        transitions = []
        exclude = set(TRANSITION_CRITERIA['exclude_stages'])
        min_sleep = TRANSITION_CRITERIA['min_sleep_epochs_before']
        stable_k = TRANSITION_CRITERIA['stable_w_epochs']
        transient_max = TRANSITION_CRITERIA['transient_w_epochs_max']
        gray_list = set(TRANSITION_CRITERIA['exclude_gray_w_epochs'])
        allowed_pre = set(TRANSITION_CRITERIA['allowed_pre_sleep'])
        allowed_return = set(TRANSITION_CRITERIA['allowed_return_sleep'])
        
        n = len(stages)
        
        def has_excluded(arr):
            return any(s in exclude for s in arr)
        
        i = min_sleep
        while i < n - 1:
            if stages[i] != 'W' or stages[i-1] not in allowed_pre:
                i += 1
                continue
            
            pre = stages[i-min_sleep:i]
            if has_excluded(pre) or any(s == 'W' for s in pre):
                i += 1
                continue
            
            run = 0
            j = i
            while j < n and stages[j] == 'W':
                run += 1
                j += 1
            
            neighborhood = stages[max(0, i-2):i] + stages[i:min(n, i+2)]
            if has_excluded(neighborhood) or has_excluded(stages[j:min(n, j+2)]):
                i += 1
                continue
            
            label = None
            if run >= stable_k:
                label = 'stable'
            elif run <= transient_max:
                if j+1 < n and stages[j] in allowed_return and stages[j+1] in allowed_return:
                    label = 'transient'
            
            if label and run not in gray_list:
                transitions.append({
                    'epoch_idx': i,
                    'from_stage': stages[i-1],
                    'time_seconds': i * 30,
                    'type': label,
                    'w_run_epochs': run,
                })
                i = j
            else:
                i += 1
        
        return transitions


# ============================================================================
# ANALIZADOR PRINCIPAL
# ============================================================================

class AwakeningCascadeAnalyzer:
    def __init__(self, data_path, output_dir):
        self.loader = SleepEDFLoader(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.pre_s = WINDOW_CONFIG['pre_seconds']
        self.post_s = WINDOW_CONFIG['post_seconds']
        self.win_size = WINDOW_CONFIG['window_size_s']
        self.win_step = WINDOW_CONFIG['window_step_s']
        
        self.nan_report = []
        self.confirmation_report = []
        
        print(f"\n{'='*70}")
        print(f"🔬 CASCADA DE DESPERTAR {__version__}")
        print(f"{'='*70}")
        print(f"Ventana: [{-self.pre_s//60} min, +{self.post_s//60} min]")
        print(f"Baseline: t < {ONSET_CONFIG['baseline_end_s']//60} min")
        print(f"{'='*70}\n")
    
    def extract_window(self, signal, fs, transition_time):
        start = int((transition_time - self.pre_s) * fs)
        end = int((transition_time + self.post_s) * fs)
        if start < 0 or end > len(signal):
            return None
        return signal[start:end]
    
    def segment_window_with_stages(self, window, fs, stages, transition_epoch_idx):
        win_samples = int(self.win_size * fs)
        step_samples = int(self.win_step * fs)
        
        segments = []
        times = []
        window_stages = []
        
        start_epoch = transition_epoch_idx - int(self.pre_s / 30)
        
        i = 0
        win_idx = 0
        while i + win_samples <= len(window):
            seg = window[i:i+win_samples]
            center_time = (i + win_samples/2) / fs - self.pre_s
            
            epoch_idx = start_epoch + win_idx
            
            if 0 <= epoch_idx < len(stages):
                stage = stages[epoch_idx]
            else:
                stage = '?'
            
            segments.append(seg)
            times.append(center_time)
            window_stages.append(stage)
            
            i += step_samples
            win_idx += 1
        
        return segments, times, window_stages
    
    def compute_timeseries_stratified(self, segments, times, window_stages, fs, 
                                       from_stage, signal_scale_uv=1.0, session_trans_seed=None):
        baseline_end = ONSET_CONFIG['baseline_end_s']
        
        baseline_stage_counts = Counter()
        for t, stage in zip(times, window_stages):
            if t < baseline_end:
                baseline_stage_counts[stage] += 1
        
        n_baseline_total = sum(baseline_stage_counts.values())
        n_baseline_same_stage = baseline_stage_counts.get(from_stage, 0)
        pct_baseline_same_stage = n_baseline_same_stage / n_baseline_total if n_baseline_total > 0 else 0
        
        baseline_segments = []
        for seg, t, stage in zip(segments, times, window_stages):
            if t < baseline_end and stage == from_stage:
                baseline_segments.append(seg)
        
        if n_baseline_same_stage < ONSET_CONFIG['min_baseline_windows']:
            return None, None, {}, [], {
                'status': 'insufficient_baseline_same_stage',
                'n_baseline_total': n_baseline_total,
                'n_baseline_same_stage': n_baseline_same_stage,
                'pct_baseline_same_stage': pct_baseline_same_stage,
                'baseline_stage_counts': dict(baseline_stage_counts),
                'from_stage': from_stage,
            }
        
        baseline_stats = compute_baseline_stats_clean(baseline_segments, fs, signal_scale_uv)
        
        if baseline_stats.get('n_clean_baseline', 0) < ONSET_CONFIG['min_baseline_windows']:
            return None, None, {}, [], {
                'status': 'insufficient_clean_baseline',
                'n_clean_baseline': baseline_stats.get('n_clean_baseline', 0),
                'pct_baseline_same_stage': pct_baseline_same_stage,
            }
        
        results = []
        valid_times = []
        artifact_count = {'clean': 0, 'rejected': 0, 'by_type': {}}
        window_artifact_log = []
        
        for win_idx, (seg, t, stage) in enumerate(zip(segments, times, window_stages)):
            is_clean, artifact_type = check_artifacts_full(seg, fs, baseline_stats, signal_scale_uv)
            
            window_artifact_log.append({'time': t, 'is_clean': is_clean, 'artifact_type': artifact_type, 'stage': stage})
            
            if not is_clean:
                artifact_count['rejected'] += 1
                artifact_count['by_type'][artifact_type] = artifact_count['by_type'].get(artifact_type, 0) + 1
                continue
            
            artifact_count['clean'] += 1
            window_seed = (session_trans_seed * 1000 + win_idx) if (session_trans_seed is not None) else None
            metrics = compute_all_metrics(seg, fs, window_seed)
            results.append(metrics)
            valid_times.append(t)
        
        if not results:
            return None, None, artifact_count, window_artifact_log, {'status': 'no_clean_windows'}
        
        df = pd.DataFrame(results)
        df['time'] = valid_times
        
        return df, valid_times, artifact_count, window_artifact_log, {
            'status': 'ok',
            'n_baseline_same_stage': n_baseline_same_stage,
            'n_clean_baseline': baseline_stats.get('n_clean_baseline', 0),
        }
    
    def normalize_timeseries(self, df):
        baseline_mask = df['time'].values < ONSET_CONFIG['baseline_end_s']
        if baseline_mask.sum() < ONSET_CONFIG['min_baseline_windows']:
            return None, 'insufficient_baseline'
        
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
            normalized[col] = (df[col] - median) / mad if mad > 1e-10 else df[col] - median
        
        return normalized, 'ok'
    
    def detect_onset_confirmed(self, values, times, threshold_sd=None, k=None, dir_k=None, confirm_threshold=None):
        if threshold_sd is None:
            threshold_sd = ONSET_CONFIG['threshold_sd']
        if k is None:
            k = ONSET_CONFIG['consecutive_windows']
        if dir_k is None:
            dir_k = ONSET_CONFIG['direction_windows']
        if confirm_threshold is None:
            confirm_threshold = ONSET_CONFIG['confirmation_threshold']
        
        values = np.array(values)
        times = np.array(times)
        n = len(values)
        
        above = (~np.isnan(values)) & (np.abs(values) > threshold_sd)
        run = 0
        onset_idx = None
        
        for i, ok in enumerate(above):
            if ok:
                run += 1
                if run >= k:
                    onset_idx = i - k + 1
                    break
            else:
                run = 0
        
        if onset_idx is None:
            return np.nan, 'none', False, {'screening_passed': False}
        
        onset_time = times[onset_idx]
        
        confirm_start = onset_idx + k
        confirm_end = confirm_start + dir_k
        
        if confirm_end > n:
            return onset_time, 'insufficient_post_run', False, {
                'screening_passed': True, 'confirm_data_available': False
            }
        
        confirm_values = values[confirm_start:confirm_end]
        valid_mask = ~np.isnan(confirm_values)
        n_valid = np.sum(valid_mask)
        
        if n_valid < dir_k:
            return onset_time, 'too_many_nans', False, {
                'screening_passed': True, 'confirm_data_available': True,
                'n_valid_confirm': int(n_valid),
                'dir_k': int(dir_k),
            }
        
        valid_values = confirm_values[valid_mask]
        signs = np.sign(valid_values)
        n_positive = np.sum(signs > 0)
        n_negative = np.sum(signs < 0)
        mean_val = np.mean(valid_values)
        
        n_nonzero = n_positive + n_negative
        
        if n_nonzero < dir_k:
            return onset_time, 'insufficient_nonzero', False, {
                'screening_passed': True, 'confirm_data_available': True,
                'n_valid_confirm': int(n_valid), 
                'n_nonzero': int(n_nonzero),
                'dir_k': int(dir_k),
            }
        
        pos_frac = n_positive / n_nonzero
        neg_frac = n_negative / n_nonzero
        
        if pos_frac >= confirm_threshold:
            direction = 'increase'
            is_confirmed = True
        elif neg_frac >= confirm_threshold:
            direction = 'decrease'
            is_confirmed = True
        else:
            osc_threshold = ONSET_CONFIG['oscillating_mean_abs_max']
            if abs(mean_val) < osc_threshold:
                direction = 'oscillating'
            elif mean_val > 0:
                direction = 'increase_unstable'
            else:
                direction = 'decrease_unstable'
            is_confirmed = False
        
        return onset_time, direction, is_confirmed, {
            'screening_passed': True, 'confirm_data_available': True,
            'n_valid_confirm': int(n_valid), 
            'n_nonzero': n_nonzero,
            'sign_consistency': max(pos_frac, neg_frac),
            'confirm_threshold_used': confirm_threshold,
        }
    
    def analyze_transition(self, df_norm, metrics_list):
        times = df_norm['time'].values
        results = {}
        
        for col in metrics_list:
            if col not in df_norm.columns:
                continue
            values = df_norm[col].values
            onset_time, direction, is_confirmed, details = self.detect_onset_confirmed(values, times)
            
            nan_rate = np.isnan(values).mean()
            
            results[col] = {
                'onset_time': onset_time,
                'direction': direction,
                'is_confirmed': is_confirmed,
                'nan_rate': nan_rate,
                **details
            }
        
        return results
    
    def run(self):
        print("📂 Buscando archivos...")
        files = self.loader.find_psg_files()
        print(f"   Encontrados: {len(files)} sesiones")
        
        pd.DataFrame(self.loader.pairing_log).to_csv(self.output_dir / "psg_hyp_pairing.csv", index=False)
        
        all_subjects = sorted({f['subject_id'] for f in files})
        rng = np.random.default_rng(SACRED_SEED)
        rng.shuffle(all_subjects)
        n_dev = int(len(all_subjects) * DEVELOPMENT_RATIO)
        dev_subjects = set(all_subjects[:n_dev])
        test_subjects = set(all_subjects[n_dev:])
        dev_files = [f for f in files if f['subject_id'] in dev_subjects]
        
        print(f"   Development: {len(dev_subjects)} sujetos, {len(dev_files)} sesiones")
        print(f"   Holdout (no analizado): {len(test_subjects)} sujetos")
        
        session_onsets = {}
        subject_sessions = {}
        transition_counts = Counter()
        baseline_quality_log = []
        
        for i, file_info in enumerate(dev_files):
            session_id = file_info['session_id']
            subject_id = file_info['subject_id']
            
            print(f"\r[{i+1}/{len(dev_files)}] {session_id[:30]}...", end="", flush=True)
            
            data = self.loader.load_session(file_info['psg_path'], file_info['hyp_path'], session_id)
            if data is None:
                continue
            
            transitions = self.loader.find_transitions(data['stages'])
            if not transitions:
                continue
            
            session_results = []
            
            for trans_idx, trans in enumerate(transitions):
                transition_counts[trans['type']] += 1
                transition_counts[f"{trans['from_stage']}_{trans['type']}"] += 1
                
                window = self.extract_window(data['signal'], data['fs'], trans['time_seconds'])
                if window is None:
                    continue
                
                segments, times, window_stages = self.segment_window_with_stages(
                    window, data['fs'], data['stages'], trans['epoch_idx']
                )
                
                session_trans_seed = zlib.crc32(session_id.encode()) + trans_idx
                
                df_metrics, valid_times, artifact_count, window_log, quality_info = \
                    self.compute_timeseries_stratified(
                        segments, times, window_stages, data['fs'],
                        from_stage=trans['from_stage'],
                        signal_scale_uv=data.get('signal_scale_uv', 1.0),
                        session_trans_seed=session_trans_seed
                    )
                
                n_total_windows = len(segments)
                n_clean_windows = artifact_count.get('clean', 0)
                n_rejected_windows = artifact_count.get('rejected', 0)
                window_rejection_rate = n_rejected_windows / n_total_windows if n_total_windows > 0 else 0
                
                n_pre = sum(1 for w in window_log if w['time'] < 0)
                n_post = sum(1 for w in window_log if w['time'] >= 0)
                n_rejected_pre = sum(1 for w in window_log if w['time'] < 0 and not w['is_clean'])
                n_rejected_post = sum(1 for w in window_log if w['time'] >= 0 and not w['is_clean'])
                rejection_rate_pre = n_rejected_pre / n_pre if n_pre > 0 else np.nan
                rejection_rate_post = n_rejected_post / n_post if n_post > 0 else np.nan
                
                baseline_quality_log.append({
                    'session_id': session_id,
                    'subject_id': subject_id,
                    'trans_idx': trans_idx,
                    'trans_type': trans['type'],
                    'from_stage': trans['from_stage'],
                    'n_total_windows': n_total_windows,
                    'n_clean_windows': n_clean_windows,
                    'window_rejection_rate': window_rejection_rate,
                    'rejection_rate_pre': rejection_rate_pre,
                    'rejection_rate_post': rejection_rate_post,
                    'n_windows_pre': n_pre,
                    'n_windows_post': n_post,
                    **quality_info
                })
                
                if df_metrics is None:
                    continue
                
                df_norm, norm_status = self.normalize_timeseries(df_metrics)
                if df_norm is None:
                    continue
                
                metrics_list = [c for c in df_norm.columns if c != 'time']
                
                trans_results = self.analyze_transition(df_norm, metrics_list)
                
                trans_id = f"{session_id}_t{trans_idx}"
                series_dir = self.output_dir / "series"
                series_dir.mkdir(exist_ok=True)
                series_path = series_dir / f"{trans_id}.npz"
                
                series_data = {'times': df_norm['time'].values}
                for m in metrics_list:
                    series_data[m] = df_norm[m].values
                
                np.savez_compressed(series_path, **series_data)
                
                trans_results['_series_path'] = str(series_path.relative_to(self.output_dir))
                
                trans_results['_meta'] = {
                    'session_id': session_id,
                    'subject_id': subject_id,
                    'from_stage': trans['from_stage'],
                    'transition_type': trans['type'],
                    'n_clean_windows': n_clean_windows,
                    'window_rejection_rate': window_rejection_rate,
                    'trans_id': trans_id,
                }
                
                for metric, res in trans_results.items():
                    if metric.startswith('_'):
                        continue
                    
                    metric_nan_rate = res.get('nan_rate', np.nan)
                    
                    self.nan_report.append({
                        'session_id': session_id,
                        'trans_id': trans_id,
                        'metric': metric,
                        'window_rejection_rate': window_rejection_rate,
                        'metric_nan_rate_given_clean': metric_nan_rate,
                        'from_stage': trans['from_stage'],
                    })
                    
                    if res.get('screening_passed', False):
                        self.confirmation_report.append({
                            'session_id': session_id,
                            'trans_id': trans_id,
                            'metric': metric,
                            'onset_time': res.get('onset_time', np.nan),
                            'is_confirmed': res['is_confirmed'],
                            'direction': res['direction'],
                            'confirm_data_available': res.get('confirm_data_available', False),
                            'n_valid_confirm': res.get('n_valid_confirm', 0),
                            'n_nonzero': res.get('n_nonzero', 0),
                            'sign_consistency': res.get('sign_consistency', np.nan),
                            'dir_k': ONSET_CONFIG['direction_windows'],
                            'from_stage': trans['from_stage'],
                        })
                
                session_results.append(trans_results)
            
            if session_results:
                session_onsets[session_id] = session_results
                if subject_id not in subject_sessions:
                    subject_sessions[subject_id] = []
                subject_sessions[subject_id].append(session_id)
        
        print(f"\n\n   ✅ Sesiones: {len(session_onsets)}, Sujetos: {len(subject_sessions)}")
        print(f"   📊 Transiciones: {dict(transition_counts)}")
        
        if not session_onsets:
            print("❌ No hay datos")
            return None
        
        pd.DataFrame(baseline_quality_log).to_csv(self.output_dir / "baseline_quality.csv", index=False)
        pd.DataFrame(self.nan_report).to_csv(self.output_dir / "nan_report.csv", index=False)
        pd.DataFrame(self.confirmation_report).to_csv(self.output_dir / "confirmation_report.csv", index=False)
        
        if self.loader.channel_log:
            pd.DataFrame(self.loader.channel_log).to_csv(self.output_dir / "channel_log.csv", index=False)
        if self.loader.duration_mismatches:
            pd.DataFrame(self.loader.duration_mismatches).to_csv(self.output_dir / "duration_mismatches.csv", index=False)
        
        sample_results = list(session_onsets.values())[0][0]
        metrics = [k for k in sample_results.keys() if not k.startswith('_')]
        
        min_subj = RANKING_CONFIG['min_subjects']
        
        def compute_ranking(session_onsets, subject_sessions, metrics, filter_func=None):
            subject_onsets = []
            
            for subject_id, session_ids in subject_sessions.items():
                all_trans = []
                for sid in session_ids:
                    if sid in session_onsets:
                        for t in session_onsets[sid]:
                            if filter_func is None or filter_func(t):
                                all_trans.append(t)
                
                if not all_trans:
                    continue
                
                row = {'subject_id': subject_id, 'n_transitions': len(all_trans)}
                
                for metric in metrics:
                    onsets = [t[metric]['onset_time'] for t in all_trans 
                             if metric in t and not np.isnan(t[metric]['onset_time']) 
                             and t[metric].get('is_confirmed', False)]
                    
                    all_onsets = [t[metric]['onset_time'] for t in all_trans 
                                 if metric in t and not np.isnan(t[metric]['onset_time'])]
                    
                    if onsets:
                        row[f'{metric}_onset_median'] = np.median(onsets)
                        row[f'{metric}_n_confirmed'] = len(onsets)
                    else:
                        row[f'{metric}_onset_median'] = np.nan
                        row[f'{metric}_n_confirmed'] = 0
                    row[f'{metric}_n_all'] = len(all_onsets)
                
                subject_onsets.append(row)
            
            if not subject_onsets:
                return pd.DataFrame()
            
            df_subject = pd.DataFrame(subject_onsets)
            
            stats_list = []
            for metric in metrics:
                col = f'{metric}_onset_median'
                if col not in df_subject.columns:
                    continue
                values = df_subject[col].dropna()
                if len(values) < min_subj:
                    continue
                
                n_conf = df_subject[f'{metric}_n_confirmed'].sum()
                n_all = df_subject[f'{metric}_n_all'].sum()
                
                stats_list.append({
                    'metric': metric,
                    'median_onset_s': np.median(values),
                    'iqr_s': np.percentile(values, 75) - np.percentile(values, 25),
                    'n_subjects': len(values),
                    'pct_before_t0': 100 * (values < 0).sum() / len(values),
                    'pct_conf_det': 100 * n_conf / n_all if n_all > 0 else 0,
                })
            
            return pd.DataFrame(stats_list).sort_values('median_onset_s') if stats_list else pd.DataFrame()
        
        print(f"\n{'='*70}")
        print("CALCULANDO RANKINGS")
        print(f"{'='*70}")
        
        df_all = compute_ranking(session_onsets, subject_sessions, metrics)
        df_stable = compute_ranking(session_onsets, subject_sessions, metrics, 
                                    lambda t: t.get('_meta', {}).get('transition_type') == 'stable')
        df_transient = compute_ranking(session_onsets, subject_sessions, metrics,
                                       lambda t: t.get('_meta', {}).get('transition_type') == 'transient')
        df_n2 = compute_ranking(session_onsets, subject_sessions, metrics,
                                lambda t: t.get('_meta', {}).get('from_stage') == 'N2')
        df_n3 = compute_ranking(session_onsets, subject_sessions, metrics,
                                lambda t: t.get('_meta', {}).get('from_stage') == 'N3')
        
        df_all.to_csv(self.output_dir / "ranking_all.csv", index=False)
        if len(df_stable) > 0:
            df_stable.to_csv(self.output_dir / "ranking_stable.csv", index=False)
        if len(df_transient) > 0:
            df_transient.to_csv(self.output_dir / "ranking_transient.csv", index=False)
        if len(df_n2) > 0:
            df_n2.to_csv(self.output_dir / "ranking_N2.csv", index=False)
        if len(df_n3) > 0:
            df_n3.to_csv(self.output_dir / "ranking_N3.csv", index=False)
        
        print(f"\n   RANKING GLOBAL (top 15):")
        print(f"   {'─'*90}")
        print(f"   {'Métrica':<30} {'Mediana':>12} {'IQR':>10} {'%<t=0':>8} {'%Conf|Det':>10}")
        print(f"   {'─'*90}")
        for _, row in df_all.head(15).iterrows():
            print(f"   {row['metric']:<30} {row['median_onset_s']:>+10.0f}s "
                  f"{row['iqr_s']:>9.0f}s {row['pct_before_t0']:>7.0f}% "
                  f"{row['pct_conf_det']:>9.1f}%")
        
        import hashlib
        import inspect
        
        code_hash = 'unknown'
        code_hash_source = 'unknown'
        
        try:
            script_path = Path(__file__)
            code_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()[:16]
            code_hash_source = 'file'
        except:
            try:
                if sys.argv and sys.argv[0]:
                    script_path = Path(sys.argv[0])
                    if script_path.exists():
                        code_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()[:16]
                        code_hash_source = 'argv'
            except:
                pass
        
        dev_subjects_list = sorted(list(dev_subjects))
        dev_subjects_hash = hashlib.sha256(','.join(dev_subjects_list).encode()).hexdigest()[:16]
        test_subjects_list = sorted(list(test_subjects))
        test_subjects_hash = hashlib.sha256(','.join(test_subjects_list).encode()).hexdigest()[:16]
        
        summary = {
            'version': __version__,
            'code_hash': code_hash,
            'code_hash_source': code_hash_source,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'config': get_full_config(),
            'runtime': get_runtime_versions(),
            'data': {
                'n_subjects': len(subject_sessions),
                'n_sessions': len(session_onsets),
                'transition_counts': dict(transition_counts),
            },
            'split': {
                'dev_subjects_hash': dev_subjects_hash,
                'dev_subjects_n': len(dev_subjects),
                'test_subjects_hash': test_subjects_hash,
                'test_subjects_n': len(test_subjects),
            },
            'changes_v36': [
                'max_duration_mismatch_s: 30 → 36000 (permite PSG/HYP desincronizados)',
            ],
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if SENSITIVITY_CONFIG['enabled']:
            print(f"\n{'='*70}")
            print("ANÁLISIS DE SENSIBILIDAD")
            print(f"{'='*70}")
            self.run_sensitivity_analysis(session_onsets, subject_sessions, metrics, df_all)
        
        print(f"\n   📁 Resultados en: {self.output_dir}")
        
        return {
            'ranking_all': df_all, 
            'ranking_stable': df_stable, 
            'ranking_n2': df_n2, 
            'ranking_n3': df_n3,
            'ranking_transient': df_transient,
            'total_subjects': len(subject_sessions),
            'total_sessions': len(session_onsets),
        }
    
    def run_sensitivity_analysis(self, session_onsets, subject_sessions, metrics, df_baseline):
        from scipy.stats import spearmanr
        
        sensitivity_results = []
        baseline_order = df_baseline['metric'].tolist() if len(df_baseline) > 0 else []
        
        series_paths = []
        for session_id, trans_list in session_onsets.items():
            for trans in trans_list:
                if '_series_path' in trans and '_meta' in trans:
                    series_paths.append({
                        'path': self.output_dir / trans['_series_path'],
                        'meta': trans['_meta'],
                    })
        
        if not series_paths:
            print("   ⚠️ No hay series guardadas")
            return
        
        print(f"   Series: {len(series_paths)}")
        
        n_combinations = (len(SENSITIVITY_CONFIG['threshold_sd_values']) * 
                         len(SENSITIVITY_CONFIG['consecutive_windows_values']) * 
                         len(SENSITIVITY_CONFIG['confirmation_thresholds']))
        combo_idx = 0
        
        for thresh in SENSITIVITY_CONFIG['threshold_sd_values']:
            for k in SENSITIVITY_CONFIG['consecutive_windows_values']:
                for conf_thresh in SENSITIVITY_CONFIG['confirmation_thresholds']:
                    combo_idx += 1
                    print(f"\r   [{combo_idx}/{n_combinations}] thresh={thresh}, k={k}, conf={conf_thresh:.2f}", 
                          end="", flush=True)
                    
                    subject_results = {}
                    
                    for item in series_paths:
                        meta = item['meta']
                        subject_id = meta['subject_id']
                        
                        try:
                            with np.load(item['path'], allow_pickle=False) as data:
                                times = data['times']
                                
                                if subject_id not in subject_results:
                                    subject_results[subject_id] = {m: [] for m in metrics}
                                
                                for metric in metrics:
                                    if metric not in data.files:
                                        continue
                                    
                                    values = data[metric]
                                    onset_time, direction, is_confirmed, details = \
                                        self.detect_onset_confirmed(values, times, thresh, k,
                                                                    ONSET_CONFIG['direction_windows'], conf_thresh)
                                    
                                    if is_confirmed and not np.isnan(onset_time):
                                        subject_results[subject_id][metric].append(onset_time)
                        except:
                            continue
                    
                    subject_onsets = []
                    for subject_id, metric_onsets in subject_results.items():
                        row = {'subject_id': subject_id}
                        for metric, onsets in metric_onsets.items():
                            row[f'{metric}_onset_median'] = np.median(onsets) if onsets else np.nan
                        subject_onsets.append(row)
                    
                    if not subject_onsets:
                        continue
                    
                    df_subj = pd.DataFrame(subject_onsets)
                    
                    ranking = []
                    for metric in metrics:
                        col = f'{metric}_onset_median'
                        if col in df_subj.columns:
                            values = df_subj[col].dropna()
                            if len(values) >= RANKING_CONFIG['min_subjects']:
                                ranking.append((metric, np.median(values)))
                    
                    ranking.sort(key=lambda x: x[1])
                    new_order = [r[0] for r in ranking]
                    
                    rho, pval = np.nan, np.nan
                    if baseline_order and new_order:
                        common = [m for m in baseline_order if m in set(new_order)]
                        if len(common) >= RANKING_CONFIG['min_subjects']:
                            base_ranks = [baseline_order.index(m) for m in common]
                            new_ranks = [new_order.index(m) for m in common]
                            rho, pval = spearmanr(base_ranks, new_ranks)
                    
                    total_confirmed = sum(len(o) for s in subject_results.values() for o in s.values())
                    
                    sensitivity_results.append({
                        'threshold_sd': thresh,
                        'consecutive_windows': k,
                        'confirmation_threshold': conf_thresh,
                        'n_metrics_ranked': len(new_order),
                        'total_confirmed': total_confirmed,
                        'spearman_rho': rho,
                        'spearman_pval': pval,
                    })
        
        print()
        
        df_sens = pd.DataFrame(sensitivity_results)
        df_sens.to_csv(self.output_dir / "sensitivity_analysis.csv", index=False)
        print(f"   Guardado: sensitivity_analysis.csv ({len(df_sens)} combinaciones)")
    
    def generate_summary_report(self, results):
        report_dir = self.output_dir / "REPORT"
        report_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*70}")
        print("📊 GENERANDO REPORTE")
        print(f"{'='*70}")
        
        df_all = results.get('ranking_all')
        
        if df_all is not None and len(df_all) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            top_n = min(20, len(df_all))
            df_top = df_all.head(top_n)
            
            colors = ['#2ecc71' if x < 0 else '#e74c3c' for x in df_top['median_onset_s']]
            y_pos = np.arange(top_n)
            ax.barh(y_pos, df_top['median_onset_s'], color=colors, alpha=0.8)
            ax.errorbar(df_top['median_onset_s'], y_pos, xerr=df_top['iqr_s']/2, 
                       fmt='none', color='black', alpha=0.5, capsize=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df_top['metric'], fontsize=9)
            ax.invert_yaxis()
            ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
            ax.set_xlabel('Onset (s)')
            ax.set_title(f'Top {top_n} Metrics (n={results["total_subjects"]} subjects)')
            
            plt.tight_layout()
            fig.savefig(report_dir / "fig1_ranking.png", dpi=150)
            fig.savefig(report_dir / "fig1_ranking.pdf")
            plt.close(fig)
            print("   ✅ fig1_ranking.png/pdf")
        
        import shutil
        for f in ['ranking_all.csv', 'summary.json']:
            src = self.output_dir / f
            if src.exists():
                shutil.copy(src, report_dir / f)
        
        print(f"   📁 {report_dir}")
        return report_dir


def find_data_path():
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and list(p.rglob("*PSG*.edf")):
            return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sleep-edf-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--no-sensitivity', action='store_true')
    parser.add_argument('--no-report', action='store_true')
    args = parser.parse_args()
    
    if args.no_sensitivity:
        SENSITIVITY_CONFIG['enabled'] = False
    
    data_path = Path(args.sleep_edf_path) if args.sleep_edf_path else find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontraron datos")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    analyzer = AwakeningCascadeAnalyzer(str(data_path), args.output_dir)
    results = analyzer.run()
    
    if results and not args.no_report:
        analyzer.generate_summary_report(results)
    
    print("\n" + "="*70)
    print("✅ COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main() 