#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
HOLDOUT VALIDATION - AFH BETA PILOT v2.2.3 (OPCIÓN C - FINAL)
═══════════════════════════════════════════════════════════════════════════════

CRITICAL: Este análisis está PRE-REGISTRADO
SHA-256: 787F42B994427AEAC8CE4B1EB3241F15DF281D0A902516570FF020E397DC177C
Timestamp: 2026-01-06 21:51:56

CORRECCIONES v2.2.3 (FINAL):
1. ✅ Validación offset hipnograma-PSG (exclusión + log)
2. ✅ Channel picking robusto (match exacto + triple fallback)
3. ✅ pearsonr con dropna (crash-proof, no sesgos)
4. ✅ Bootstrap con RNG local (seeds diferenciados)
5. ✅ Logging completo de exclusiones
6. ✅ FILTRADO CONSERVADOR: Solo epochs dentro del rango PSG
7. ✅ JSON serialization fix (numpy bool → Python bool)
8. ✅ Print format fix (manejo correcto de None values)

Author: Dr. Camilo Alejandro Sjöberg Tala, M.D.
ORCID: 0009-0009-6052-0212
Date: January 2026
Version: 2.2.3 (FINAL - DEFENDIBLE)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr, entropy
from scipy.signal import hilbert
import json
import hashlib
import warnings
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN PRE-REGISTRADA
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette")
OUTPUT_DIR = Path(r"G:\Mi unidad\AFH\PILOT_BETA\HOLDOUT")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# LOGGING DE EXCLUSIONES
EXCLUSION_LOG = OUTPUT_DIR / "exclusions_log.csv"
with open(EXCLUSION_LOG, "w", encoding="utf-8") as f:
    f.write("subject_id,reason,psg_t_min,psg_t_max,h_min,h_max,n_valid_epochs,details\n")

DEVELOPMENT_SUBJECTS = [
    'SC4001E0', 'SC4002E0', 'SC4011E0', 'SC4012E0', 'SC4021E0',
    'SC4022E0', 'SC4031E0', 'SC4032E0', 'SC4041E0', 'SC4042E0'
]

FS_TARGET = 100
EPOCH_DURATION = 30
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 40.0

CONFIGS = {
    'BASE': {
        'artifact_threshold': 300,
        'normalization': 'by_state'
    },
    'QC_STRICT': {
        'artifact_threshold': 200,
        'normalization': 'by_state'
    },
    'NORM_GLOBAL': {
        'artifact_threshold': 300,
        'normalization': 'global'
    }
}

THRESHOLD_SD_B = 0.5
THRESHOLD_CORR_GLOBAL = 0.8
THRESHOLD_CORR_N3 = 0.3

BOOTSTRAP_SEED = 2025
N_BOOTSTRAP = 10000

print("="*80)
print("HOLDOUT VALIDATION - AFH BETA PILOT v2.2.3 (FINAL)")
print("="*80)
print(f"\nPRE-REGISTRATION:")
print(f"  Hash: 787F42B994427AEAC8CE4B1EB3241F15DF281D0A902516570FF020E397DC177C")
print(f"  Timestamp: 2026-01-06 21:51:56")
print(f"\nCORRECTIONS v2.2.3:")
print(f"  ✅ Hypnogram-PSG alignment: CONSERVATIVE FILTERING")
print(f"  ✅ Only epochs within PSG time range are used")
print(f"  ✅ Robust channel picking (exact + triple fallback)")
print(f"  ✅ pearsonr with dropna (crash-proof)")
print(f"  ✅ Bootstrap with local RNG (diff seeds)")
print(f"  ✅ Complete logging of exclusions")
print(f"  ✅ JSON serialization fix (numpy bool → Python bool)")
print(f"  ✅ Print format fix (None value handling)")
print(f"\nBASE_DIR: {BASE_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"EXCLUSION_LOG: {EXCLUSION_LOG}")
print("="*80)

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE MÉTRICAS (NO MODIFICAR)
# ═══════════════════════════════════════════════════════════════════════════

def lempel_ziv_complexity(signal_data):
    """A1: Lempel-Ziv Complexity"""
    try:
        threshold = np.median(signal_data)
        binary = (signal_data > threshold).astype(int)
        
        n = len(binary)
        c = 1
        l = 1
        i = 0
        k = 1
        k_max = 1
        
        while True:
            if i + k >= n:
                c += 1
                break
            
            if binary[i + k] != binary[l + k - 1]:
                c += 1
                l = i + 1
                i = i + k
                k = 1
                k_max = 1
            else:
                k += 1
                if k > k_max:
                    k_max = k
                if l + k - 1 >= i:
                    c += 1
                    l = i + 1
                    i = i + k
                    k = 1
                    k_max = 1
        
        lz_complexity = c * np.log2(n) / n if n > 0 else 0
        return lz_complexity
    except:
        return np.nan

def spectral_entropy(signal_data, fs=100):
    """A2: Spectral Entropy"""
    try:
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
        psd_norm = psd / np.sum(psd)
        spec_ent = entropy(psd_norm, base=2)
        return spec_ent
    except:
        return np.nan

def aperiodic_slope(signal_data, fs=100):
    """A3: Aperiodic Slope"""
    try:
        freqs, psd = signal.welch(signal_data, fs=fs, nperseg=min(256, len(signal_data)))
        
        mask = (freqs >= 2) & (freqs <= 20)
        freqs_fit = freqs[mask]
        psd_fit = psd[mask]
        
        if len(freqs_fit) < 2:
            return np.nan
        
        log_freqs = np.log10(freqs_fit)
        log_psd = np.log10(psd_fit + 1e-10)
        
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        return slope
    except:
        return np.nan

def temporal_variability(signal_data, fs=100):
    """B1: Temporal Variability"""
    try:
        sos = signal.butter(4, [0.1, 1.0], btype='band', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, signal_data)
        
        window_size = 5 * fs
        n_windows = len(filtered) // window_size
        
        if n_windows < 2:
            return np.nan
        
        variances = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_var = np.var(filtered[start:end])
            variances.append(window_var)
        
        variances = np.array(variances)
        mean_var = np.mean(variances)
        std_var = np.std(variances)
        
        if mean_var == 0:
            return np.nan
        
        cv = std_var / mean_var
        return cv
    except:
        return np.nan

def phase_concentration_stability(signal_data, fs=100):
    """B2: Phase Stability"""
    try:
        sos = signal.butter(4, [0.5, 4.0], btype='band', fs=fs, output='sos')
        filtered = signal.sosfiltfilt(sos, signal_data)
        
        analytic_signal = hilbert(filtered)
        phase = np.angle(analytic_signal)
        
        window_size = 5 * fs
        n_windows = len(phase) // window_size
        
        if n_windows < 2:
            return np.nan
        
        concentrations = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_phase = phase[start:end]
            
            complex_phase = np.exp(1j * window_phase)
            concentration = np.abs(np.mean(complex_phase))
            concentrations.append(concentration)
        
        concentrations = np.array(concentrations)
        mean_conc = np.mean(concentrations)
        std_conc = np.std(concentrations)
        
        if std_conc == 0:
            return 50.0
        
        cv = std_conc / mean_conc
        stability = 1 / cv
        stability = np.clip(stability, 0, 50)
        return stability
    except:
        return np.nan

def spectral_variability(signal_data, fs=100):
    """B3: Spectral Variability"""
    try:
        window_size = 5 * fs
        n_windows = len(signal_data) // window_size
        
        if n_windows < 2:
            return np.nan
        
        psds = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_data = signal_data[start:end]
            
            freqs, psd = signal.welch(window_data, fs=fs, nperseg=min(256, len(window_data)))
            mask = (freqs >= 1) & (freqs <= 10)
            psd_range = psd[mask]
            
            psd_norm = psd_range / (np.sum(psd_range) + 1e-10)
            psds.append(psd_norm)
        
        distances = []
        for i in range(len(psds) - 1):
            min_len = min(len(psds[i]), len(psds[i+1]))
            dist = np.linalg.norm(psds[i][:min_len] - psds[i+1][:min_len])
            distances.append(dist)
        
        mean_dist = np.mean(distances)
        return mean_dist
    except:
        return np.nan

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def compute_hash(data_str):
    """SHA-256 hash"""
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

def find_holdout_subjects(base_dir, development_subjects):
    """Encuentra holdout con manifest hasheado"""
    print("\n[1/7] Identificando sujetos holdout...")
    
    psg_files = list(base_dir.glob("*-PSG.edf"))
    
    holdout_manifest = []
    
    for psg in psg_files:
        code = psg.stem.replace("-PSG", "")
        
        if code in development_subjects:
            continue
        
        hypno_code = code[:-1]
        hypno_pattern = f"{hypno_code}*-Hypnogram.edf"
        hypno_files = sorted(base_dir.glob(hypno_pattern))
        
        if len(hypno_files) > 0:
            holdout_manifest.append({
                'subject_id': code,
                'psg_path': str(psg),
                'hypno_path': str(hypno_files[0])
            })
    
    holdout_manifest = sorted(holdout_manifest, key=lambda x: x['subject_id'])
    
    print(f"  ✓ Sujetos holdout encontrados: {len(holdout_manifest)}")
    
    if len(holdout_manifest) != 43:
        print(f"  ⚠ WARNING: Expected 43 subjects, found {len(holdout_manifest)}")
    
    manifest_file = OUTPUT_DIR / "holdout_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(holdout_manifest, f, indent=2)
    
    manifest_str = json.dumps(holdout_manifest, sort_keys=True)
    manifest_hash = compute_hash(manifest_str)
    
    print(f"  ✓ Manifest guardado: {manifest_file}")
    print(f"  ✓ Manifest SHA-256: {manifest_hash}")
    
    hash_file = OUTPUT_DIR / "holdout_manifest_hash.txt"
    with open(hash_file, 'w') as f:
        f.write(f"Holdout Manifest Hash\n")
        f.write(f"=====================\n")
        f.write(f"SHA-256: {manifest_hash}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"N subjects: {len(holdout_manifest)}\n")
    
    print(f"  ✓ Hash guardado: {hash_file}")
    
    holdout_tuples = [
        (Path(item['psg_path']), Path(item['hypno_path']), item['subject_id'])
        for item in holdout_manifest
    ]
    
    return holdout_tuples, manifest_hash

def load_hypnogram(hypno_path):
    """Carga hypnogram con t_start real"""
    annotations = mne.read_annotations(str(hypno_path))
    
    STAGE_MAP = {
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
        "Sleep stage 3": 3, "Sleep stage 4": 4, "Sleep stage R": 5,
        "W": 0, "1": 1, "2": 2, "3": 3, "4": 4, "R": 5,
        "?": -1
    }
    
    epochs_info = []
    
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        stage = STAGE_MAP.get(description, -1)
        n_epochs = max(1, round(duration / EPOCH_DURATION))
        
        for i in range(n_epochs):
            t_start_sec = onset + i * EPOCH_DURATION
            epochs_info.append({
                't_start': t_start_sec,
                'stage': stage
            })
    
    return pd.DataFrame(epochs_info)

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: extract_epoch (BLINDADA v2.2.3)
# ═══════════════════════════════════════════════════════════════════════════

def extract_epoch(raw, t_start_sec, fs_target):
    """
    Extrae época de 30s usando t_start REAL del hypnograma.
    
    Corrección v2.2.3:
    - Channel picking robusto: match exacto + triple fallback
    - Chequeo de límites robusto
    """
    sfreq = float(raw.info['sfreq'])

    start_sample = int(round(float(t_start_sec) * sfreq))
    end_sample = int(round((float(t_start_sec) + EPOCH_DURATION) * sfreq))

    if start_sample < 0 or end_sample > raw.n_times or end_sample <= start_sample:
        return None

    # 1) Match exacto (robusto)
    target_names = {"eeg fpz-cz", "fpz-cz", "eeg fpz-cz (ref)", "eeg fpz-cz ref"}
    ch_names_lower = [ch.lower().strip() for ch in raw.ch_names]

    fpz_cz_idx = None
    for i, name in enumerate(ch_names_lower):
        if name in target_names:
            fpz_cz_idx = i
            break

    # 2) Fallback heurístico con EEG
    if fpz_cz_idx is None:
        for i, name in enumerate(ch_names_lower):
            if ('fpz' in name) and ('cz' in name) and ('eeg' in name or 'eeg ' in name):
                fpz_cz_idx = i
                break

    # 3) Último fallback: solo fpz + cz
    if fpz_cz_idx is None:
        for i, name in enumerate(ch_names_lower):
            if ('fpz' in name) and ('cz' in name):
                fpz_cz_idx = i
                break

    if fpz_cz_idx is None:
        return None

    data = raw.get_data(start=start_sample, stop=end_sample)[fpz_cz_idx]

    # Resample
    if int(round(sfreq)) != int(fs_target):
        n_samples_new = int(EPOCH_DURATION * fs_target)
        if n_samples_new <= 0:
            return None
        data = signal.resample(data, n_samples_new)

    return data

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: process_subject (OPCIÓN C v2.2.3)
# ═══════════════════════════════════════════════════════════════════════════

def process_subject(psg_path, hypno_path, subject_id, config):
    """
    Procesa un sujeto completo.
    
    Corrección v2.2.3 (OPCIÓN C):
    - FILTRADO CONSERVADOR: Solo usa epochs dentro del rango PSG
    - Log completo de exclusiones
    """
    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        hypno_df = load_hypnogram(hypno_path)

        # Detección de escala
        data_sample = raw.get_data()[:, :1000].flatten()
        if np.percentile(np.abs(data_sample), 99) < 1.0:
            raw.apply_function(lambda x: x * 1e6, picks='eeg')

        # Filtrado
        raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design='firwin', verbose=False)

        # ── OPCIÓN C: FILTRADO CONSERVADOR ────────────────────
        psg_t_min = float(raw.times[0])
        psg_t_max = float(raw.times[-1])

        h_min_orig = float(hypno_df['t_start'].min()) if len(hypno_df) else np.nan
        h_max_orig = float(hypno_df['t_start'].max()) if len(hypno_df) else np.nan

        # Filtrar epochs que caen COMPLETAMENTE dentro del PSG
        valid_mask = (
            (hypno_df['t_start'] >= psg_t_min) &
            (hypno_df['t_start'] + EPOCH_DURATION <= psg_t_max)
        )
        hypno_df = hypno_df[valid_mask].copy()

        n_valid_epochs = len(hypno_df)

        if n_valid_epochs == 0:
            with open(EXCLUSION_LOG, "a", encoding="utf-8") as f:
                f.write(f"{subject_id},no_valid_epochs,{psg_t_min:.3f},{psg_t_max:.3f},{h_min_orig:.3f},{h_max_orig:.3f},{n_valid_epochs},all_epochs_out_of_bounds\n")
            print(f"  ⚠ {subject_id}: no valid epochs within PSG range -> EXCLUDED")
            return None
        # ───────────────────────────────────────────────────────

        results = {'wake': [], 'n3': []}

        # Wake
        wake_rows = hypno_df[hypno_df['stage'] == 0]
        for _, row in wake_rows.iterrows():
            t_start = float(row['t_start'])
            epoch_data = extract_epoch(raw, t_start, FS_TARGET)
            if epoch_data is None:
                continue

            if np.max(np.abs(epoch_data)) > config['artifact_threshold']:
                continue

            results['wake'].append({
                'A1_lz': lempel_ziv_complexity(epoch_data),
                'A2_spec_ent': spectral_entropy(epoch_data, FS_TARGET),
                'A3_slope': aperiodic_slope(epoch_data, FS_TARGET),
                'B1_temp_var': temporal_variability(epoch_data, FS_TARGET),
                'B2_phase_stab': phase_concentration_stability(epoch_data, FS_TARGET),
                'B3_spec_var': spectral_variability(epoch_data, FS_TARGET)
            })

        # N3
        n3_rows = hypno_df[hypno_df['stage'].isin([3, 4])]
        for _, row in n3_rows.iterrows():
            t_start = float(row['t_start'])
            epoch_data = extract_epoch(raw, t_start, FS_TARGET)
            if epoch_data is None:
                continue

            if np.max(np.abs(epoch_data)) > config['artifact_threshold']:
                continue

            results['n3'].append({
                'A1_lz': lempel_ziv_complexity(epoch_data),
                'A2_spec_ent': spectral_entropy(epoch_data, FS_TARGET),
                'A3_slope': aperiodic_slope(epoch_data, FS_TARGET),
                'B1_temp_var': temporal_variability(epoch_data, FS_TARGET),
                'B2_phase_stab': phase_concentration_stability(epoch_data, FS_TARGET),
                'B3_spec_var': spectral_variability(epoch_data, FS_TARGET)
            })

        # Insuficientes épocas
        if len(results['wake']) < 10 or len(results['n3']) < 10:
            with open(EXCLUSION_LOG, "a", encoding="utf-8") as f:
                f.write(f"{subject_id},insufficient_epochs,{psg_t_min:.3f},{psg_t_max:.3f},{h_min_orig:.3f},{h_max_orig:.3f},{n_valid_epochs},wake={len(results['wake'])}_n3={len(results['n3'])}\n")
            return None

        return results

    except Exception as e:
        with open(EXCLUSION_LOG, "a", encoding="utf-8") as f:
            f.write(f"{subject_id},exception,nan,nan,nan,nan,0,{str(e)[:100]}\n")
        print(f"  ⚠ Error en {subject_id}: {str(e)[:120]}")
        return None

def aggregate_subject(results_epochs, normalization='by_state'):
    """Agrega épocas: mediana, rechazo NaN>20%"""
    aggregated = {}
    
    for state in ['wake', 'n3']:
        if len(results_epochs[state]) == 0:
            continue
        
        df_epochs = pd.DataFrame(results_epochs[state])
        
        valid_metrics = {}
        for metric in df_epochs.columns:
            nan_rate = df_epochs[metric].isna().sum() / len(df_epochs)
            if nan_rate <= 0.2:
                valid_metrics[metric] = df_epochs[metric].median()
        
        aggregated[state] = valid_metrics
    
    return aggregated

def compute_composite_scores(df, normalization='by_state'):
    """Z-score y composite scores (blindado v2.2.3)"""
    metrics = ['A1_lz', 'A2_spec_ent', 'A3_slope', 
               'B1_temp_var', 'B2_phase_stab', 'B3_spec_var']
    
    for metric in metrics:
        df[f'{metric}_z'] = np.nan
    
    if normalization == 'by_state':
        for state in ['wake', 'n3']:
            mask = df['state'] == state
            for metric in metrics:
                if metric in df.columns:
                    values = df.loc[mask, metric]
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        df.loc[mask, f'{metric}_z'] = (values - mean_val) / std_val
                    else:
                        df.loc[mask, f'{metric}_z'] = 0
    else:
        for metric in metrics:
            if metric in df.columns:
                values = df[metric]
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    df[f'{metric}_z'] = (values - mean_val) / std_val
                else:
                    df[f'{metric}_z'] = 0
    
    df['score_A'] = df[['A1_lz_z', 'A2_spec_ent_z', 'A3_slope_z']].mean(axis=1, skipna=True)
    df['score_B'] = df[['B1_temp_var_z', 'B2_phase_stab_z', 'B3_spec_var_z']].mean(axis=1, skipna=True)
    
    df['n_metrics_A'] = df[['A1_lz_z', 'A2_spec_ent_z', 'A3_slope_z']].notna().sum(axis=1)
    df['n_metrics_B'] = df[['B1_temp_var_z', 'B2_phase_stab_z', 'B3_spec_var_z']].notna().sum(axis=1)
    
    return df

# ═══════════════════════════════════════════════════════════════════════════
# FUNCIÓN: test_hypotheses (BLINDADA v2.2.3 + JSON FIX)
# ═══════════════════════════════════════════════════════════════════════════

def test_hypotheses(df):
    """
    Tests determinísticos.
    
    Corrección v2.2.3:
    - pearsonr con dropna (crash-proof)
    - Bootstrap con RNG local y seeds diferenciados
    - JSON fix: numpy bool → Python bool
    """
    results = {}
    
    # H1
    n3_data = df[df['state'] == 'n3']
    sd_b_n3 = n3_data['score_B'].std()
    
    results['H1'] = {
        'SD_B_N3': float(sd_b_n3),
        'threshold': THRESHOLD_SD_B,
        'passed': bool(sd_b_n3 >= THRESHOLD_SD_B)  # JSON FIX
    }
    
    # Helper: correlación segura
    def safe_pearsonr(x, y):
        tmp = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(tmp) < 3:
            return np.nan
        r, _ = pearsonr(tmp['x'].values, tmp['y'].values)
        return r
    
    wake_data = df[df['state'] == 'wake']
    
    r_global = safe_pearsonr(df['score_A'], df['score_B'])
    r_wake = safe_pearsonr(wake_data['score_A'], wake_data['score_B'])
    r_n3 = safe_pearsonr(n3_data['score_A'], n3_data['score_B'])
    
    # Bootstrap CI con RNG local
    def bootstrap_ci(x, y, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
        tmp = pd.DataFrame({'x': x, 'y': y}).dropna()
        n = len(tmp)
        if n < 3:
            return np.array([np.nan, np.nan])
        
        rng = np.random.default_rng(seed)
        xv = tmp['x'].values
        yv = tmp['y'].values
        
        boot_rs = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            r, _ = pearsonr(xv[idx], yv[idx])
            boot_rs[i] = r
        
        return np.percentile(boot_rs, [2.5, 97.5])
    
    ci_global = bootstrap_ci(df['score_A'], df['score_B'], seed=BOOTSTRAP_SEED)
    ci_wake = bootstrap_ci(wake_data['score_A'], wake_data['score_B'], seed=BOOTSTRAP_SEED + 1)
    ci_n3 = bootstrap_ci(n3_data['score_A'], n3_data['score_B'], seed=BOOTSTRAP_SEED + 2)
    
    results['H2'] = {
        'r_global': float(r_global) if np.isfinite(r_global) else None,
        'ci_global': [float(ci_global[0]), float(ci_global[1])],
        'r_wake': float(r_wake) if np.isfinite(r_wake) else None,
        'ci_wake': [float(ci_wake[0]), float(ci_wake[1])],
        'r_n3': float(r_n3) if np.isfinite(r_n3) else None,
        'ci_n3': [float(ci_n3[0]), float(ci_n3[1])],
        'passed_global': bool(np.isfinite(r_global) and abs(r_global) < THRESHOLD_CORR_GLOBAL),  # JSON FIX
        'passed_n3': bool(np.isfinite(r_n3) and abs(r_n3) < THRESHOLD_CORR_N3)  # JSON FIX
    }
    
    results['GO_DECISION'] = bool(
        results['H1']['passed'] and
        results['H2']['passed_global'] and
        results['H2']['passed_n3']
    )  # JSON FIX
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL (v2.2.3 - PRINT FIX)
# ═══════════════════════════════════════════════════════════════════════════

def run_holdout_validation():
    """Pipeline completo v2.2.3"""
    timestamp_start = datetime.now().isoformat()
    
    print("\n" + "="*80)
    print("INICIANDO HOLDOUT VALIDATION v2.2.3")
    print("="*80)
    
    holdout_subjects, manifest_hash = find_holdout_subjects(BASE_DIR, DEVELOPMENT_SUBJECTS)
    
    all_results = {}
    
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"CONFIGURACIÓN: {config_name}")
        print(f"{'='*80}")
        print(f"  Artifact threshold: {config['artifact_threshold']} µV (max|x|)")
        print(f"  Normalization: {config['normalization']}")
        
        print(f"\n[2/7] Procesando {len(holdout_subjects)} sujetos...")
        
        all_subject_data = []
        
        for psg_path, hypno_path, subject_id in tqdm(holdout_subjects, desc=f"  {config_name}"):
            results = process_subject(psg_path, hypno_path, subject_id, config)
            
            if results is None:
                continue
            
            agg = aggregate_subject(results, config['normalization'])
            
            for state in ['wake', 'n3']:
                if state in agg:
                    row = agg[state].copy()
                    row['subject'] = subject_id
                    row['state'] = state
                    all_subject_data.append(row)
        
        df = pd.DataFrame(all_subject_data)
        
        print(f"\n[3/7] Sujetos incluidos: {len(df['subject'].unique())}")
        print(f"  Wake observations: {len(df[df['state']=='wake'])}")
        print(f"  N3 observations: {len(df[df['state']=='n3'])}")
        
        if len(df) == 0:
            print("  ⚠ ERROR: No data collected")
            continue
        
        print(f"\n[4/7] Calculando composite scores...")
        df = compute_composite_scores(df, config['normalization'])
        
        print(f"\n[5/7] Testing pre-registered hypotheses...")
        hypothesis_results = test_hypotheses(df)
        
        print(f"\n{'─'*80}")
        print(f"RESULTADOS {config_name}:")
        print(f"{'─'*80}")
        
        print(f"\nH1: Variance in N3")
        print(f"  SD(score_B | N3) = {hypothesis_results['H1']['SD_B_N3']:.3f}")
        print(f"  Threshold: ≥ {THRESHOLD_SD_B}")
        print(f"  Result: {'✓ PASS' if hypothesis_results['H1']['passed'] else '✗ FAIL'}")
        
        print(f"\nH2: State-Dependent Correlation")
        
        # PRINT FIX: Manejo correcto de None values
        r_g = hypothesis_results['H2']['r_global']
        r_w = hypothesis_results['H2']['r_wake']
        r_n = hypothesis_results['H2']['r_n3']
        
        ci_g = hypothesis_results['H2']['ci_global']
        ci_w = hypothesis_results['H2']['ci_wake']
        ci_n = hypothesis_results['H2']['ci_n3']
        
        def fmt_r(r):
            return f"{r:.3f}" if r is not None else "nan"
        
        def fmt_ci(ci):
            return f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        
        print(f"  Global: r = {fmt_r(r_g)}, 95% CI {fmt_ci(ci_g)}")
        print(f"  Wake: r = {fmt_r(r_w)}, 95% CI {fmt_ci(ci_w)}")
        print(f"  N3: r = {fmt_r(r_n)}, 95% CI {fmt_ci(ci_n)}")
        
        print(f"\n  |r_global| < {THRESHOLD_CORR_GLOBAL}: {'✓ PASS' if hypothesis_results['H2']['passed_global'] else '✗ FAIL'}")
        print(f"  |r_N3| < {THRESHOLD_CORR_N3}: {'✓ PASS' if hypothesis_results['H2']['passed_n3'] else '✗ FAIL'}")
        
        print(f"\n{'─'*80}")
        print(f"GO/NO-GO DECISION: {'✓ GO' if hypothesis_results['GO_DECISION'] else '✗ NO-GO'}")
        print(f"{'─'*80}")
        
        all_results[config_name] = {
            'data': df,
            'hypotheses': hypothesis_results,
            'n_subjects': len(df['subject'].unique())
        }
        
        output_file = OUTPUT_DIR / f"holdout_results_{config_name.lower()}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n[6/7] Datos guardados: {output_file}")
    
    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN FINAL HOLDOUT VALIDATION")
    print(f"{'='*80}")
    
    for config_name in CONFIGS.keys():
        if config_name in all_results:
            go = all_results[config_name]['hypotheses']['GO_DECISION']
            print(f"  {config_name}: {'✓ GO' if go else '✗ NO-GO'}")
    
    # H3
    go_count = sum(1 for cfg in all_results.values() if cfg['hypotheses']['GO_DECISION'])
    h3_passed = go_count >= 2
    
    print(f"\nH3: Robustness Across Configurations")
    print(f"  GO decisions: {go_count}/3")
    print(f"  Threshold: ≥ 2/3")
    print(f"  Result: {'✓ PASS' if h3_passed else '✗ FAIL'}")
    
    # Análisis de exclusiones
    print(f"\n{'='*80}")
    print("ANÁLISIS DE EXCLUSIONES")
    print(f"{'='*80}")
    
    try:
        df_excl = pd.read_csv(EXCLUSION_LOG)
        total_excluded = len(df_excl['subject_id'].unique())
        print(f"\nSujetos únicos excluidos: {total_excluded}")
        print(f"\nPor razón:")
        for reason, count in df_excl['reason'].value_counts().items():
            print(f"  {reason}: {count}")
    except:
        print("\n  (No se pudo leer log de exclusiones)")
    
    # VEREDICTO
    print(f"\n{'='*80}")
    if h3_passed:
        print("✓ HOLDOUT VALIDATION SUCCESSFUL")
        print("  Hypotheses H1, H2, H3 replicated in independent cohort")
    else:
        print("✗ HOLDOUT VALIDATION FAILED")
        print("  Development results were sample-specific")
    print(f"{'='*80}")
    
    # Summary JSON
    timestamp_end = datetime.now().isoformat()
    
    summary = {
        'preregistration': {
            'hash': '787F42B994427AEAC8CE4B1EB3241F15DF281D0A902516570FF020E397DC177C',
            'timestamp': '2026-01-06 21:51:56'
        },
        'holdout_manifest': {
            'hash': manifest_hash,
            'n_subjects': len(holdout_subjects)
        },
        'execution': {
            'start': timestamp_start,
            'end': timestamp_end,
            'version': '2.2.3'
        },
        'configurations': {
            cfg: {
                'n_subjects': res['n_subjects'],
                'go_decision': res['hypotheses']['GO_DECISION'],
                'H1': res['hypotheses']['H1'],
                'H2': res['hypotheses']['H2']
            }
            for cfg, res in all_results.items()
        },
        'final_verdict': {
            'H3_passed': bool(h3_passed),
            'validation_successful': bool(h3_passed)
        },
        'corrections_v2.2.3': {
            'epoch_filtering': 'OPCIÓN C - Conservative: only epochs within PSG range',
            'channel_picking': 'exact match + triple fallback (robust)',
            'pearsonr': 'dropna before correlation (crash-proof)',
            'bootstrap': 'local RNG with differentiated seeds',
            'artifact_threshold': 'max|x|, NOT peak-to-peak',
            'logging': 'complete exclusion log with reasons and details',
            'json_fix': 'numpy bool converted to Python bool',
            'print_fix': 'format string handling for None values'
        }
    }
    
    summary_file = OUTPUT_DIR / "holdout_validation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[7/7] Resumen guardado: {summary_file}")
    print(f"       Exclusion log: {EXCLUSION_LOG}")
    print("="*80)
    
    return all_results

# ═══════════════════════════════════════════════════════════════════════════
# EJECUTAR
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_holdout_validation()