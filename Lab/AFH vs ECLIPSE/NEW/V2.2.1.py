#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFH-BETA PILOT v1.3.3 - CON ANÁLISIS DE ROBUSTEZ
================================================

Novedades v1.3.3:
1) Análisis de correlación DENTRO de estado (crítico)
2) Bootstrap CI para correlación A-B
3) Análisis de sensibilidad automático:
   - Configuración BASE (MAX_AMP=300)
   - Sensibilidad QC (MAX_AMP=200)
   - Sensibilidad normalización (global, no por estado)
4) Reporte completo de robustez

Author: Dr. Camilo Alejandro Sjöberg Tala
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from scipy.signal import butter, filtfilt, resample, welch, hilbert
from scipy.stats import entropy, zscore
from tqdm import tqdm

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

# =============================================================================
# CONFIG
# =============================================================================

SLEEP_DIR = Path(r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette")
OUTPUT_DIR = Path(r"G:\Mi unidad\AFH\PILOT_BETA")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FS_TARGET = 100
BANDPASS = (1, 40)
MIN_EPOCHS_PER_STATE = 10

DEV_SUBJECTS = [
    "SC4001", "SC4002", "SC4011", "SC4012", "SC4021",
    "SC4022", "SC4031", "SC4032", "SC4041", "SC4042"
]

STAGE_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N4",
    "Sleep stage R": "R",
    "Sleep stage ?": "?",
    "Movement time": "M",
    "Sleep stage N1": "N1",
    "Sleep stage N2": "N2",
    "Sleep stage N3": "N3",
    "Sleep stage REM": "R",
    "Sleep stage S1": "N1",
    "Sleep stage S2": "N2",
    "Sleep stage S3": "N3",
    "Sleep stage S4": "N4",
}

# CONFIGURACIONES DE SENSIBILIDAD
SENSITIVITY_CONFIGS = {
    'base': {'max_amp': 300, 'norm_mode': 'by_state', 'label': 'BASE (300µV, norm_by_state)'},
    'qc_strict': {'max_amp': 200, 'norm_mode': 'by_state', 'label': 'QC_STRICT (200µV)'},
    'norm_global': {'max_amp': 300, 'norm_mode': 'global', 'label': 'NORM_GLOBAL (across_states)'},
}

# =============================================================================
# UTILS
# =============================================================================

def apply_bandpass(x, fs, lowcut, highcut):
    try:
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, x)
    except Exception:
        return None

def resample_to_target(x, fs_in, fs_out):
    n_samples_new = int(len(x) * fs_out / fs_in)
    return resample(x, n_samples_new)

def subject_id_from_name(name: str):
    up = name.upper()
    m = re.search(r"(SC\d{4})", up)
    return m.group(1) if m else None

def pick_eeg_channel(raw):
    def norm(s): 
        return re.sub(r"[^A-Z0-9]", "", s.upper())

    target = norm("Fpz-Cz")
    for ch in raw.ch_names:
        if target == norm(ch):
            return ch

    for ch in raw.ch_names:
        up = ch.upper()
        if "EEG" in up and ("EOG" not in up) and ("EMG" not in up) and ("ECG" not in up):
            return ch

    for ch in raw.ch_names:
        if "EEG" in ch.upper():
            return ch

    return None

# =============================================================================
# PAIRING
# =============================================================================

def looks_like_hypnogram(edf_path: Path) -> bool:
    name = edf_path.name.upper()
    if "HYP" in name or "HYPNOGRAM" in name:
        return True
    
    try:
        ann = mne.read_annotations(str(edf_path))
        if len(ann) == 0:
            return False
        
        sample = " ".join(list(map(str, ann.description[:10]))).upper()
        return ("SLEEP STAGE" in sample) or ("MOVEMENT" in sample)
    except Exception:
        return False

def find_sleep_edf_pairs(root):
    edf_all = sorted(root.rglob("*.edf"))
    if not edf_all:
        raise FileNotFoundError(f"No EDF encontrados en: {root}")

    hyp_files = [p for p in edf_all if looks_like_hypnogram(p)]
    psg_files = [p for p in edf_all if p not in hyp_files]

    hyp_index = {}
    for h in hyp_files:
        sid = subject_id_from_name(h.name)
        if sid is None:
            continue
        key = (h.parent, sid)
        if key not in hyp_index:
            hyp_index[key] = []
        hyp_index[key].append(h)

    pairs = []
    log = []

    for p in psg_files:
        sid = subject_id_from_name(p.name)
        if sid is None:
            continue

        cands = hyp_index.get((p.parent, sid), [])
        if len(cands) == 1:
            pairs.append({"subject_id": sid, "psg": p, "hyp": cands[0]})
            log.append({"subject_id": sid, "psg": str(p), "hyp": str(cands[0]), "status": "ok"})
        elif len(cands) == 0:
            log.append({"subject_id": sid, "psg": str(p), "hyp": None, "status": "no_hyp_in_folder"})
        else:
            log.append({"subject_id": sid, "psg": str(p), "hyp": None, "status": "ambiguous_multiple_hyp"})

    pd.DataFrame(log).to_csv(OUTPUT_DIR / "pairing_log.csv", index=False)

    dev_pairs = [p for p in pairs if p["subject_id"] in DEV_SUBJECTS]
    
    return dev_pairs

# =============================================================================
# EXTRACTION
# =============================================================================

def extract_subject(pair, max_amplitude):
    sid = pair["subject_id"]
    psg_file = pair["psg"]
    hyp_file = pair["hyp"]

    try:
        raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
        ch = pick_eeg_channel(raw)
        if ch is None:
            return None

        fs_in = float(raw.info["sfreq"])
        signal = raw.copy().pick_channels([ch]).get_data()[0]
        
        # Conversión automática V → µV
        p99 = np.percentile(np.abs(signal), 99)
        if p99 < 1:
            signal = signal * 1e6

        annotations = mne.read_annotations(str(hyp_file))
        if len(annotations) == 0:
            return None

        n_epochs = int((len(signal) / fs_in) // 30)
        if n_epochs <= 0:
            return None

        stages = np.array(["?"] * n_epochs, dtype=object)
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            stage = STAGE_MAPPING.get(desc, "?")
            start_epoch = int(np.floor(onset / 30))
            end_epoch = int(np.ceil((onset + duration) / 30))
            if end_epoch <= 0:
                continue
            for e in range(max(0, start_epoch), min(end_epoch, n_epochs)):
                stages[e] = stage

        samples_per_epoch = int(30 * fs_in)
        wake_epochs = []
        n3_epochs = []

        for i, st in enumerate(stages):
            start = i * samples_per_epoch
            end = start + samples_per_epoch
            if end > len(signal):
                break

            epoch = signal[start:end]

            if fs_in != FS_TARGET:
                epoch = resample_to_target(epoch, fs_in, FS_TARGET)

            epoch = apply_bandpass(epoch, FS_TARGET, BANDPASS[0], BANDPASS[1])
            if epoch is None:
                continue

            if st == "W":
                wake_epochs.append(epoch)
            elif st in ("N3", "N4"):
                n3_epochs.append(epoch)

        wake_epochs = np.asarray(wake_epochs) if wake_epochs else np.asarray([])
        n3_epochs = np.asarray(n3_epochs) if n3_epochs else np.asarray([])

        # QC con max_amplitude variable
        if len(wake_epochs) > 0:
            wake_epochs = wake_epochs[np.max(np.abs(wake_epochs), axis=1) < max_amplitude]
        if len(n3_epochs) > 0:
            n3_epochs = n3_epochs[np.max(np.abs(n3_epochs), axis=1) < max_amplitude]

        if len(wake_epochs) < MIN_EPOCHS_PER_STATE or len(n3_epochs) < MIN_EPOCHS_PER_STATE:
            return None

        return {
            "subject_id": sid,
            "status": "accepted",
            "wake": wake_epochs,
            "n3": n3_epochs,
        }

    except Exception:
        return None

def normalize_data(subjects_data, mode='by_state'):
    """
    Normaliza datos según modo:
    - by_state: cada estado por sí mismo
    - global: todo junto por sujeto
    """
    normalized = {}
    
    for sid, data in subjects_data.items():
        if mode == 'by_state':
            w_mean, w_std = np.mean(data['wake']), np.std(data['wake'])
            n_mean, n_std = np.mean(data['n3']), np.std(data['n3'])
            
            wake_norm = (data['wake'] - w_mean) / (w_std + 1e-10)
            n3_norm = (data['n3'] - n_mean) / (n_std + 1e-10)
            
        elif mode == 'global':
            all_data = np.concatenate([data['wake'].flatten(), data['n3'].flatten()])
            g_mean, g_std = np.mean(all_data), np.std(all_data)
            
            wake_norm = (data['wake'] - g_mean) / (g_std + 1e-10)
            n3_norm = (data['n3'] - g_mean) / (g_std + 1e-10)
        
        normalized[sid] = {'wake': wake_norm, 'n3': n3_norm}
    
    return normalized

# =============================================================================
# METRICS (sin cambios)
# =============================================================================

def compute_lz(epoch):
    try:
        binary = (epoch > np.median(epoch)).astype(int)
        s = "".join(map(str, binary))
        i, k, l = 0, 1, 1
        c, n = 1, len(s)
        k_max = 1
        while True:
            if s[i + k - 1] == s[l + k - 1]:
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
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        return c * np.log2(n) / n
    except Exception:
        return np.nan

def compute_spectral_entropy(epoch, fs=100):
    try:
        freqs, psd = welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
        psd_norm = psd / (psd.sum() + 1e-10)
        return entropy(psd_norm + 1e-10)
    except Exception:
        return np.nan

def compute_aperiodic_slope(epoch, fs=100):
    try:
        freqs, psd = welch(epoch, fs=fs, nperseg=min(256, len(epoch)))
        mask = (freqs >= 2) & (freqs <= 20)
        if mask.sum() < 3:
            return np.nan
        log_f = np.log10(freqs[mask])
        log_p = np.log10(psd[mask] + 1e-10)
        return np.polyfit(log_f, log_p, 1)[0]
    except Exception:
        return np.nan

def compute_temporal_variability(epoch, fs=100):
    try:
        b, a = butter(4, [0.1, 1], btype="band", fs=fs)
        slow = filtfilt(b, a, epoch)
        win = int(5 * fs)
        n_w = len(slow) // win
        if n_w < 2:
            return np.nan
        vals = []
        for i in range(n_w):
            seg = slow[i*win:(i+1)*win]
            vals.append(np.var(seg))
        vals = np.asarray(vals)
        m = np.mean(vals)
        s = np.std(vals)
        if m < 1e-12:
            return np.nan
        return s / m
    except Exception:
        return np.nan

def compute_phase_concentration_stability(epoch, fs=100):
    try:
        b, a = butter(4, [0.5, 4], btype="band", fs=fs)
        delta = filtfilt(b, a, epoch)
        phase = np.angle(hilbert(delta))
        win = int(5 * fs)
        n_w = len(phase) // win
        if n_w < 2:
            return np.nan
        conc = []
        for i in range(n_w):
            ph = phase[i*win:(i+1)*win]
            conc.append(np.abs(np.mean(np.exp(1j * ph))))
        conc = np.asarray(conc)
        m = np.mean(conc)
        s = np.std(conc)
        cv = s / (m + 1e-10)
        return float(np.clip(1.0 / (cv + 1e-6), 0.0, 50.0))
    except Exception:
        return np.nan

def compute_spectral_variability(epoch, fs=100):
    try:
        win = int(5 * fs)
        n_w = len(epoch) // win
        if n_w < 2:
            return np.nan
        psds = []
        for i in range(n_w):
            seg = epoch[i*win:(i+1)*win]
            freqs, psd = welch(seg, fs=fs, nperseg=min(256, len(seg)))
            mask = (freqs >= 1) & (freqs <= 10)
            band = psd[mask]
            band = band / (band.sum() + 1e-10)
            psds.append(band)
        if len(psds) < 2:
            return np.nan
        dists = []
        for i in range(len(psds) - 1):
            dists.append(np.linalg.norm(psds[i] - psds[i+1]))
        return float(np.mean(dists))
    except Exception:
        return np.nan

def compute_metrics(epochs, subject_id, state):
    n = len(epochs)
    A1 = np.zeros(n)
    A2 = np.zeros(n)
    A3 = np.zeros(n)
    B1 = np.zeros(n)
    B2 = np.zeros(n)
    B3 = np.zeros(n)

    for i, ep in enumerate(epochs):
        A1[i] = compute_lz(ep)
        A2[i] = compute_spectral_entropy(ep, FS_TARGET)
        A3[i] = compute_aperiodic_slope(ep, FS_TARGET)
        B1[i] = compute_temporal_variability(ep, FS_TARGET)
        B2[i] = compute_phase_concentration_stability(ep, FS_TARGET)
        B3[i] = compute_spectral_variability(ep, FS_TARGET)

    def agg(x):
        nan_rate = np.isnan(x).mean()
        return np.nan if nan_rate > 0.20 else float(np.nanmedian(x))

    return {
        "subject_id": subject_id,
        "state": state,
        "n_epochs": int(n),
        "A1_lz": agg(A1),
        "A2_spec_ent": agg(A2),
        "A3_slope": agg(A3),
        "B1_temp_var": agg(B1),
        "B2_phase_stab": agg(B2),
        "B3_spec_var": agg(B3),
    }

# =============================================================================
# ANÁLISIS DE CORRELACIÓN MEJORADO
# =============================================================================

def bootstrap_correlation(x, y, n_boot=10000):
    """Bootstrap CI para correlación"""
    correlations = []
    n = len(x)
    
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        r = np.corrcoef(x[idx], y[idx])[0, 1]
        correlations.append(r)
    
    correlations = np.array(correlations)
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)
    
    return ci_lower, ci_upper

def analyze_correlations(df):
    """Análisis completo de correlaciones"""
    
    print("\n" + "="*80)
    print("ANÁLISIS DE CORRELACIONES")
    print("="*80)
    
    # Global
    corr_global = df[['score_A', 'score_B']].corr().iloc[0, 1]
    ci_low, ci_high = bootstrap_correlation(
        df['score_A'].values, 
        df['score_B'].values
    )
    
    print(f"\nCORRELACIÓN GLOBAL:")
    print(f"  r = {corr_global:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Por estado
    df_wake = df[df['state'] == 'wake']
    df_n3 = df[df['state'] == 'n3']
    
    if len(df_wake) >= 3:
        corr_wake = df_wake[['score_A', 'score_B']].corr().iloc[0, 1]
        print(f"\nCORRELACIÓN DENTRO DE WAKE:")
        print(f"  r = {corr_wake:.3f} (n={len(df_wake)})")
    
    if len(df_n3) >= 3:
        corr_n3 = df_n3[['score_A', 'score_B']].corr().iloc[0, 1]
        print(f"\nCORRELACIÓN DENTRO DE N3:")
        print(f"  r = {corr_n3:.3f} (n={len(df_n3)})")
    
    print(f"\n💡 INTERPRETACIÓN:")
    if abs(corr_wake) < 0.3 and abs(corr_n3) < 0.3:
        print("  ✅ Correlación global es principalmente mezcla de estados")
        print("  ✅ Dentro de cada estado, A y B son independientes")
    else:
        print("  ⚠ Correlación persiste dentro de estados")
        print("  ⚠ A y B tienen dependencia genuina")
    
    return {
        'global': corr_global,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'wake': corr_wake if len(df_wake) >= 3 else np.nan,
        'n3': corr_n3 if len(df_n3) >= 3 else np.nan
    }

# =============================================================================
# PIPELINE CON SENSIBILIDAD
# =============================================================================

def run_single_configuration(pairs, config_name, config):
    """Ejecuta pipeline con una configuración específica"""
    
    max_amp = config['max_amp']
    norm_mode = config['norm_mode']
    
    print(f"\n{'='*80}")
    print(f"CONFIGURACIÓN: {config['label']}")
    print(f"{'='*80}")
    
    # Extracción
    subjects_raw = {}
    
    for pair in tqdm(pairs, desc=f"  Extracting ({config_name})"):
        result = extract_subject(pair, max_amp)
        if result and result.get('status') == 'accepted':
            sid = result['subject_id']
            subjects_raw[sid] = {
                'wake': result['wake'],
                'n3': result['n3']
            }
    
    if len(subjects_raw) < 3:
        print(f"  ⚠ Solo {len(subjects_raw)} sujetos - SKIP")
        return None
    
    # Normalización
    subjects_norm = normalize_data(subjects_raw, mode=norm_mode)
    
    # Métricas
    results = []
    for sid, data in subjects_norm.items():
        results.append(compute_metrics(data['wake'], sid, 'wake'))
        results.append(compute_metrics(data['n3'], sid, 'n3'))
    
    df = pd.DataFrame(results)
    
    # Scores
    metrics_A = ["A1_lz", "A2_spec_ent", "A3_slope"]
    metrics_B = ["B1_temp_var", "B2_phase_stab", "B3_spec_var"]
    
    for m in metrics_A + metrics_B:
        df[f"{m}_z"] = zscore(df[m], nan_policy="omit")
    
    df['score_A'] = df[[f"{m}_z" for m in metrics_A]].mean(axis=1)
    df['score_B'] = df[[f"{m}_z" for m in metrics_B]].mean(axis=1)
    
    # Análisis
    df_n3 = df[df['state'] == 'n3']
    sd_B_n3 = float(np.std(df_n3['score_B'].dropna()))
    corr_AB = float(df[['score_A', 'score_B']].corr().iloc[0, 1])
    
    go_decision = (sd_B_n3 >= 0.5 and abs(corr_AB) < 0.8)
    
    print(f"\n  Sujetos: {len(subjects_raw)}")
    print(f"  SD(score_B|N3): {sd_B_n3:.3f}")
    print(f"  Corr(A,B): {corr_AB:.3f}")
    print(f"  Decisión: {'GO ✅' if go_decision else 'NO-GO ❌'}")
    
    return {
        'config': config_name,
        'n_subjects': len(subjects_raw),
        'sd_B_n3': sd_B_n3,
        'corr_AB': corr_AB,
        'go': go_decision,
        'df': df
    }

def run():
    print("\n" + "="*80)
    print("AFH-BETA PILOT v1.3.3 - ANÁLISIS DE ROBUSTEZ")
    print("="*80)

    print(f"[DIAG] SLEEP_DIR exists: {SLEEP_DIR.exists()}")
    edf_all = sorted(SLEEP_DIR.rglob("*.edf"))
    print(f"[DIAG] total EDF: {len(edf_all)}")

    # Pairing (una sola vez)
    print("\n📂 Pairing:")
    pairs = find_sleep_edf_pairs(SLEEP_DIR)
    
    if len(pairs) < 3:
        raise RuntimeError("Muy pocos pares DEV")
    
    print(f"  Pares DEV: {len(pairs)}")

    # Ejecutar todas las configuraciones
    sensitivity_results = {}
    
    for config_name, config in SENSITIVITY_CONFIGS.items():
        result = run_single_configuration(pairs, config_name, config)
        if result:
            sensitivity_results[config_name] = result

    # Reporte de robustez
    print("\n" + "="*80)
    print("REPORTE DE ROBUSTEZ")
    print("="*80)
    
    robustness_table = []
    for config_name, result in sensitivity_results.items():
        robustness_table.append({
            'Configuration': SENSITIVITY_CONFIGS[config_name]['label'],
            'N_subjects': result['n_subjects'],
            'SD(B|N3)': f"{result['sd_B_n3']:.3f}",
            'Corr(A,B)': f"{result['corr_AB']:.3f}",
            'Decision': 'GO ✅' if result['go'] else 'NO-GO ❌'
        })
    
    df_robustness = pd.DataFrame(robustness_table)
    print("\n" + df_robustness.to_string(index=False))
    
    # Guardar reporte
    df_robustness.to_csv(OUTPUT_DIR / 'robustness_report_v1.3.3.csv', index=False)
    
    # Análisis de correlaciones detallado (configuración BASE)
    if 'base' in sensitivity_results:
        df_base = sensitivity_results['base']['df']
        corr_analysis = analyze_correlations(df_base)
        
        # Guardar CSV base
        df_base.to_csv(OUTPUT_DIR / 'pilot_results_v1.3.3_base.csv', index=False)
    
    # Conclusión
    n_go = sum(1 for r in sensitivity_results.values() if r['go'])
    n_total = len(sensitivity_results)
    
    print(f"\n" + "="*80)
    print(f"CONCLUSIÓN: {n_go}/{n_total} configuraciones pasan GO")
    print("="*80)
    
    if n_go == n_total:
        print("✅ ROBUSTEZ CONFIRMADA: GO en todas las configuraciones")
    elif n_go >= n_total / 2:
        print("⚠ ROBUSTEZ PARCIAL: GO en mayoría de configuraciones")
    else:
        print("❌ ROBUSTEZ DÉBIL: GO solo en minoría de configuraciones")
    
    print(f"\n✓ Reportes: {OUTPUT_DIR}")
    print(f"  - robustness_report_v1.3.3.csv")
    print(f"  - pilot_results_v1.3.3_base.csv")

if __name__ == "__main__":
    run()