#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFH-BETA PILOT v1.2 - Sleep-EDF Only
=====================================

CORRECCIÓN CRÍTICA: Pairing por primeros 6 caracteres
PSG: SC4001E0-PSG.edf → HYP: SC4001EC-Hypnogram.edf
Match: SC4001 (primeros 6 chars)

Version: 1.2.0
Author: Dr. Camilo Alejandro Sjöberg Tala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import warnings
from pathlib import Path
from scipy.signal import butter, filtfilt, resample, welch, hilbert, correlate
from scipy.stats import entropy, zscore
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SLEEP_DIR = Path(r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette")
OUTPUT_DIR = Path(r"G:\Mi unidad\AFH\PILOT_BETA")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FS_TARGET = 100
BANDPASS = (1, 40)
MAX_AMPLITUDE = 200
MIN_EPOCHS_PER_STATE = 10  # Reducido de 20 a 10

DEV_SUBJECTS = [
    'SC4001', 'SC4002', 'SC4011', 'SC4012', 'SC4021',
    'SC4022', 'SC4031', 'SC4032', 'SC4041', 'SC4042'
]

STAGE_MAPPING = {
    'Sleep stage W': 'W',
    'Sleep stage 1': 'N1',
    'Sleep stage 2': 'N2',
    'Sleep stage 3': 'N3',
    'Sleep stage 4': 'N4',
    'Sleep stage R': 'R',
    'Sleep stage ?': '?',
    'Movement time': 'M'
}

# ============================================================================
# UTILIDADES
# ============================================================================

def apply_bandpass(data, fs, lowcut, highcut):
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data)
    except:
        return None

def resample_to_target(data, fs_original, fs_target):
    n_samples_new = int(len(data) * fs_target / fs_original)
    return resample(data, n_samples_new)

def compute_band_power(signal, fs, band):
    try:
        freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        if len(idx) == 0:
            return np.nan
        return np.trapz(psd[idx], freqs[idx])
    except:
        return np.nan

# ============================================================================
# PAIRING CORREGIDO
# ============================================================================

def find_psg_pairs():
    """
    Pairing correcto por primeros 6 caracteres
    
    Patrón Sleep-EDF:
    - PSG: SC4001E0-PSG.edf
    - HYP: SC4001EC-Hypnogram.edf
    - Match: SC4001 (primeros 6 chars)
    """
    psg_files = list(SLEEP_DIR.rglob("*PSG*.edf"))
    hyp_files = list(SLEEP_DIR.rglob("*Hypnogram*.edf"))
    
    print(f"\n📂 Archivos encontrados:")
    print(f"  PSG: {len(psg_files)}")
    print(f"  Hypnogram: {len(hyp_files)}")
    
    # Indexar hypnograms por subject ID (primeros 6 chars)
    hyp_by_subject = {}
    for hyp in hyp_files:
        subject_id = hyp.stem[:6]  # SC4001, SC4002, etc.
        if subject_id not in hyp_by_subject:
            hyp_by_subject[subject_id] = []
        hyp_by_subject[subject_id].append(hyp)
    
    # Parear
    pairs = []
    
    for psg in psg_files:
        subject_id = psg.stem[:6]
        
        if subject_id in hyp_by_subject:
            hyp_candidates = hyp_by_subject[subject_id]
            
            if len(hyp_candidates) == 1:
                pairs.append({
                    'subject_id': subject_id,
                    'psg': psg,
                    'hyp': hyp_candidates[0]
                })
            elif len(hyp_candidates) > 1:
                # Múltiples hypnograms, tomar primero
                pairs.append({
                    'subject_id': subject_id,
                    'psg': psg,
                    'hyp': hyp_candidates[0]
                })
    
    print(f"  Pares válidos: {len(pairs)}")
    
    # Filtrar por DEV_SUBJECTS
    dev_pairs = [p for p in pairs if p['subject_id'] in DEV_SUBJECTS]
    print(f"  Pares en DEV: {len(dev_pairs)}")
    
    return dev_pairs

# ============================================================================
# EXTRACCIÓN
# ============================================================================

def extract_subject(pair):
    """Extrae Wake y N3 de un sujeto"""
    
    subject_id = pair['subject_id']
    psg_file = pair['psg']
    hyp_file = pair['hyp']
    
    try:
        # Cargar PSG
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
        
        # Buscar canal
        channel = None
        for ch in raw.ch_names:
            if 'FPZ' in ch.upper() and 'CZ' in ch.upper():
                channel = ch
                break
        
        if channel is None:
            for ch in raw.ch_names:
                if 'EEG' in ch.upper():
                    channel = ch
                    break
        
        if channel is None:
            return None
        
        # Señal
        fs = raw.info['sfreq']
        signal = raw.copy().pick_channels([channel]).get_data()[0]
        
        # Hypnogram
        annotations = mne.read_annotations(str(hyp_file))
        
        n_epochs = int(len(signal) / fs // 30)
        stages = ['?'] * n_epochs
        
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            stage = STAGE_MAPPING.get(desc, '?')
            start_epoch = int(np.floor(onset / 30))
            end_epoch = int(np.ceil((onset + duration) / 30))
            
            for e in range(start_epoch, min(end_epoch, n_epochs)):
                stages[e] = stage
        
        # Extraer epochs
        samples_per_epoch = int(30 * fs)
        
        wake_epochs = []
        n3_epochs = []
        
        for i, stage in enumerate(stages):
            start_sample = i * samples_per_epoch
            end_sample = start_sample + samples_per_epoch
            
            if end_sample > len(signal):
                break
            
            epoch = signal[start_sample:end_sample]
            
            # Resample
            if fs != FS_TARGET:
                epoch = resample_to_target(epoch, fs, FS_TARGET)
            
            # Bandpass
            epoch = apply_bandpass(epoch, FS_TARGET, BANDPASS[0], BANDPASS[1])
            
            if epoch is None:
                continue
            
            # Clasificar
            if stage == 'W':
                wake_epochs.append(epoch)
            elif stage in ['N3', 'N4']:
                n3_epochs.append(epoch)
        
        # Arrays
        wake_epochs = np.array(wake_epochs) if wake_epochs else np.array([])
        n3_epochs = np.array(n3_epochs) if n3_epochs else np.array([])
        
        # Artifact rejection
        if len(wake_epochs) > 0:
            wake_valid = np.max(np.abs(wake_epochs), axis=1) < MAX_AMPLITUDE
            wake_epochs = wake_epochs[wake_valid]
        
        if len(n3_epochs) > 0:
            n3_valid = np.max(np.abs(n3_epochs), axis=1) < MAX_AMPLITUDE
            n3_epochs = n3_epochs[n3_valid]
        
        # Verificar mínimos
        if len(wake_epochs) < MIN_EPOCHS_PER_STATE or len(n3_epochs) < MIN_EPOCHS_PER_STATE:
            return None
        
        return {
            'subject_id': subject_id,
            'wake': wake_epochs,
            'n3': n3_epochs
        }
        
    except Exception as e:
        return None

def normalize_by_state(data):
    """Normaliza cada estado por sí mismo"""
    wake_mean = np.mean(data['wake'])
    wake_std = np.std(data['wake'])
    
    n3_mean = np.mean(data['n3'])
    n3_std = np.std(data['n3'])
    
    wake_norm = (data['wake'] - wake_mean) / (wake_std + 1e-10)
    n3_norm = (data['n3'] - n3_mean) / (n3_std + 1e-10)
    
    return {'wake': wake_norm, 'n3': n3_norm}

# ============================================================================
# MÉTRICAS
# ============================================================================

def compute_lz(epoch):
    """Lempel-Ziv Complexity"""
    try:
        binary = (epoch > np.median(epoch)).astype(int)
        binary_str = ''.join(map(str, binary))
        
        i, k, l = 0, 1, 1
        c, n = 1, len(binary_str)
        k_max = 1
        
        while True:
            if binary_str[i + k - 1] == binary_str[l + k - 1]:
                k += 1
                if l + k > n:
                    c += 1
                    break
            else:
                k_max = max(k, k_max)
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > n:
                        break
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
        
        lz = c * np.log2(n) / n
        return lz
    except:
        return np.nan

def compute_spectral_entropy(epoch, fs=100):
    try:
        freqs, psd = welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
        psd_norm = psd / (psd.sum() + 1e-10)
        return entropy(psd_norm + 1e-10)
    except:
        return np.nan

def compute_aperiodic_slope(epoch, fs=100):
    try:
        freqs, psd = welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
        mask = (freqs >= 2) & (freqs <= 20)
        if mask.sum() < 3:
            return np.nan
        log_freqs = np.log10(freqs[mask])
        log_psd = np.log10(psd[mask] + 1e-10)
        slope = np.polyfit(log_freqs, log_psd, 1)[0]
        return slope
    except:
        return np.nan

def compute_slow_osc_power(epoch, fs=100):
    try:
        b, a = butter(4, [0.1, 1], btype='band', fs=fs)
        slow_osc = filtfilt(b, a, epoch)
        return np.var(slow_osc)
    except:
        return np.nan

def compute_delta_theta_ratio(epoch, fs=100):
    try:
        delta = compute_band_power(epoch, fs, (0.5, 4))
        theta = compute_band_power(epoch, fs, (4, 8))
        if np.isnan(delta) or np.isnan(theta):
            return np.nan
        return delta / (theta + 1e-10)
    except:
        return np.nan

def compute_autocorr_tau(epoch, fs=100):
    try:
        acf = correlate(epoch, epoch, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / (acf[0] + 1e-10)
        
        max_lag = min(int(5 * fs), len(acf))
        lags = np.arange(max_lag)
        acf_segment = acf[:max_lag]
        
        mask = (acf_segment > 0.1) & (acf_segment < 1.0)
        
        if mask.sum() < 10:
            return np.nan
        
        log_acf = np.log(acf_segment[mask] + 1e-10)
        slope = np.polyfit(lags[mask], log_acf, 1)[0]
        tau = -1 / slope if slope < 0 else np.nan
        
        return tau
    except:
        return np.nan

def compute_metrics(epochs, subject_id, state):
    """Calcula todas las métricas"""
    n = len(epochs)
    
    metrics = {
        'A1_lz': np.zeros(n),
        'A2_spec_ent': np.zeros(n),
        'A3_slope': np.zeros(n),
        'B1_slow_osc': np.zeros(n),
        'B2_dt_ratio': np.zeros(n),
        'B3_tau': np.zeros(n)
    }
    
    for i, epoch in enumerate(epochs):
        metrics['A1_lz'][i] = compute_lz(epoch)
        metrics['A2_spec_ent'][i] = compute_spectral_entropy(epoch, FS_TARGET)
        metrics['A3_slope'][i] = compute_aperiodic_slope(epoch, FS_TARGET)
        metrics['B1_slow_osc'][i] = compute_slow_osc_power(epoch, FS_TARGET)
        metrics['B2_dt_ratio'][i] = compute_delta_theta_ratio(epoch, FS_TARGET)
        metrics['B3_tau'][i] = compute_autocorr_tau(epoch, FS_TARGET)
    
    result = {'subject_id': subject_id, 'state': state, 'n_epochs': n}
    
    for name, values in metrics.items():
        nan_rate = np.isnan(values).mean()
        if nan_rate > 0.20:
            result[name] = np.nan
        else:
            result[name] = np.nanmedian(values)
    
    return result

# ============================================================================
# PIPELINE
# ============================================================================

def run_pilot_sleep_edf_only():
    """Pipeline completo Sleep-EDF"""
    
    print("\n" + "="*80)
    print("AFH-BETA PILOT v1.2 (Sleep-EDF only)")
    print("="*80)
    
    # Pairing
    pairs = find_psg_pairs()
    
    if len(pairs) < 3:
        raise RuntimeError("Muy pocos pares. Revise DEV_SUBJECTS.")
    
    # Extracción
    subjects_data = {}
    
    for pair in tqdm(pairs, desc="Extracting sessions"):
        result = extract_subject(pair)
        if result:
            subject_id = result['subject_id']
            normalized = normalize_by_state(result)
            subjects_data[subject_id] = normalized
    
    print(f"\n✓ Sujetos con Wake+N3 válidos: {len(subjects_data)}")
    
    if len(subjects_data) < 3:
        raise RuntimeError("Muy pocos sujetos válidos. Revise paths, nombres de archivos y/o MIN_EPOCHS_PER_STATE.")
    
    # Calcular métricas
    print("\n📊 Calculando métricas...")
    
    results = []
    for subject_id, data in tqdm(subjects_data.items(), desc="Computing metrics"):
        wake_result = compute_metrics(data['wake'], subject_id, 'wake')
        n3_result = compute_metrics(data['n3'], subject_id, 'n3')
        results.extend([wake_result, n3_result])
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'pilot_results_v1.2.csv', index=False)
    
    print(f"✓ {len(df)} filas guardadas")
    
    # Análisis
    print("\n📈 Análisis exploratorio...")
    
    metrics_A = ['A1_lz', 'A2_spec_ent', 'A3_slope']
    metrics_B = ['B1_slow_osc', 'B2_dt_ratio', 'B3_tau']
    
    for metric in metrics_A + metrics_B:
        df[f'{metric}_z'] = zscore(df[metric], nan_policy='omit')
    
    df['score_A'] = df[[f'{m}_z' for m in metrics_A]].mean(axis=1)
    df['score_B'] = df[[f'{m}_z' for m in metrics_B]].mean(axis=1)
    
    # Criterios
    df_n3 = df[df['state'] == 'n3']
    sd_B_n3 = float(np.std(df_n3['score_B'].dropna()))
    corr_AB = float(df[['score_A', 'score_B']].corr().iloc[0, 1])
    
    print(f"\n{'='*80}")
    print("RESULTADOS")
    print("="*80)
    print(f"Sujetos: {len(subjects_data)}")
    print(f"SD(score_B) en N3: {sd_B_n3:.3f} (criterio: ≥0.5)")
    print(f"Corr(A,B): {corr_AB:.3f} (criterio: <0.8)")
    
    criterio1 = sd_B_n3 >= 0.5
    criterio2 = abs(corr_AB) < 0.8
    
    print(f"\nCriterio 1: {'✓ PASS' if criterio1 else '✗ FAIL'}")
    print(f"Criterio 2: {'✓ PASS' if criterio2 else '✗ FAIL'}")
    
    # Visualización
    figures_dir = OUTPUT_DIR / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Distribuciones B
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    for i, metric in enumerate(metrics_B):
        wake_vals = df[df['state'] == 'wake'][metric].dropna()
        n3_vals = df[df['state'] == 'n3'][metric].dropna()
        
        axes[i, 0].hist(wake_vals, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i, 0].set_title(f'{metric} - Wake\nSD={np.std(wake_vals):.3f}')
        
        axes[i, 1].hist(n3_vals, bins=15, alpha=0.7, color='darkorange', edgecolor='black')
        axes[i, 1].set_title(f'{metric} - N3\nSD={np.std(n3_vals):.3f}')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'metrics_B_distributions_v1.2.png', dpi=300)
    plt.close()
    
    print(f"\n✓ Figuras: {figures_dir}")
    
    return df, {'sd_B_n3': sd_B_n3, 'corr_AB': corr_AB}

def main():
    try:
        df, metrics = run_pilot_sleep_edf_only()
        
        print("\n" + "="*80)
        print("PILOT COMPLETO")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()