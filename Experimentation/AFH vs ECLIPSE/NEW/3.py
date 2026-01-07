#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFH-BETA PILOT v1.3.2 (Sleep-EDF only) - FULLY ROBUST
=====================================================

Mejoras v1.3.2:
1) Detecta hypnogram aunque no tenga "HYP" en nombre (lee annotations)
2) Convierte automáticamente Volts → µV para QC correcto
3) Diagnóstico automático de rechazos con recomendaciones

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
MAX_AMPLITUDE = 300  # µV
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
# PAIRING ROBUSTO (PARCHE 1: detecta hypnogram sin "HYP" en nombre)
# =============================================================================

def looks_like_hypnogram(edf_path: Path) -> bool:
    """Detecta hypnogram por nombre O por contenido de annotations"""
    name = edf_path.name.upper()
    if "HYP" in name or "HYPNOGRAM" in name:
        return True
    
    # Fallback: leer annotations rápido
    try:
        ann = mne.read_annotations(str(edf_path))
        if len(ann) == 0:
            return False
        
        # Si hay descriptores tipo "Sleep stage ..."
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
    print("\n📂 Pairing:")
    print(f"  EDF totales: {len(edf_all)} | PSG cand: {len(psg_files)} | Hyp cand: {len(hyp_files)}")
    print(f"  Pares OK: {len(pairs)} | Pares DEV: {len(dev_pairs)}")
    print(f"  Log: {OUTPUT_DIR / 'pairing_log.csv'}")

    return dev_pairs

# =============================================================================
# EXTRACTION (PARCHE 2: conversión automática V → µV)
# =============================================================================

def extract_subject(pair):
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
        
        # PARCHE 2: Detectar escala y convertir a µV
        p99 = np.percentile(np.abs(signal), 99)
        if p99 < 1:  # Probablemente en Volts
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

        if len(wake_epochs) > 0:
            wake_epochs = wake_epochs[np.max(np.abs(wake_epochs), axis=1) < MAX_AMPLITUDE]
        if len(n3_epochs) > 0:
            n3_epochs = n3_epochs[np.max(np.abs(n3_epochs), axis=1) < MAX_AMPLITUDE]

        if len(wake_epochs) < MIN_EPOCHS_PER_STATE or len(n3_epochs) < MIN_EPOCHS_PER_STATE:
            return {
                "subject_id": sid,
                "status": "rejected_min_epochs",
                "picked_channel": ch,
                "fs_in": fs_in,
                "n_epochs_total": int(n_epochs),
                "n_stage_W": int(np.sum(stages == "W")),
                "n_stage_N3": int(np.sum(np.isin(stages, ["N3", "N4"]))),
                "wake_postQC": int(len(wake_epochs)),
                "n3_postQC": int(len(n3_epochs)),
            }

        return {
            "subject_id": sid,
            "status": "accepted",
            "picked_channel": ch,
            "fs_in": fs_in,
            "wake": wake_epochs,
            "n3": n3_epochs,
            "n_epochs_total": int(n_epochs),
            "n_stage_W": int(np.sum(stages == "W")),
            "n_stage_N3": int(np.sum(np.isin(stages, ["N3", "N4"]))),
            "wake_postQC": int(len(wake_epochs)),
            "n3_postQC": int(len(n3_epochs)),
        }

    except Exception:
        return None

def normalize_by_state(wake_epochs, n3_epochs):
    w = (wake_epochs - np.mean(wake_epochs)) / (np.std(wake_epochs) + 1e-10)
    n = (n3_epochs - np.mean(n3_epochs)) / (np.std(n3_epochs) + 1e-10)
    return w, n

# =============================================================================
# METRICS A
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

# =============================================================================
# METRICS B
# =============================================================================

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
# DIAGNÓSTICO AUTOMÁTICO
# =============================================================================

def diagnose_rejections(debug_df):
    """Analiza rechazos y genera recomendaciones"""
    print("\n" + "="*80)
    print("DIAGNÓSTICO DE RECHAZOS")
    print("="*80)
    
    if len(debug_df) == 0:
        print("⚠ Sin datos de debug")
        return
    
    # Conteo por status
    status_counts = debug_df['status'].value_counts()
    print("\nStatus counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Análisis específico
    rejected = debug_df[debug_df['status'] == 'rejected_min_epochs']
    
    if len(rejected) > 0:
        print(f"\n⚠ {len(rejected)} sujetos rechazados por MIN_EPOCHS")
        print("\nEstadísticas de stages:")
        print(f"  n_stage_W promedio: {rejected['n_stage_W'].mean():.1f}")
        print(f"  n_stage_N3 promedio: {rejected['n_stage_N3'].mean():.1f}")
        print(f"  wake_postQC promedio: {rejected['wake_postQC'].mean():.1f}")
        print(f"  n3_postQC promedio: {rejected['n3_postQC'].mean():.1f}")
        
        # Recomendaciones
        if rejected['n_stage_N3'].mean() < 5:
            print("\n💡 RECOMENDACIÓN: Problema de etiquetado N3")
            print("   → Revisar STAGE_MAPPING")
            print("   → Verificar hypnogram contiene 'Sleep stage 3' o 'N3'")
        
        if rejected['wake_postQC'].mean() / (rejected['n_stage_W'].mean() + 1) < 0.5:
            print("\n💡 RECOMENDACIÓN: QC demasiado estricto en Wake")
            print(f"   → Subir MAX_AMPLITUDE de {MAX_AMPLITUDE} a 400-500")
        
        if rejected['n3_postQC'].mean() / (rejected['n_stage_N3'].mean() + 1) < 0.5:
            print("\n💡 RECOMENDACIÓN: QC demasiado estricto en N3")
            print(f"   → Subir MAX_AMPLITUDE de {MAX_AMPLITUDE} a 400-500")

# =============================================================================
# PIPELINE
# =============================================================================

def run():
    print("\n" + "="*80)
    print("AFH-BETA PILOT v1.3.2 - FULLY ROBUST")
    print("="*80)

    print(f"[DIAG] SLEEP_DIR exists: {SLEEP_DIR.exists()}")
    edf_all = sorted(SLEEP_DIR.rglob("*.edf"))
    print(f"[DIAG] total EDF: {len(edf_all)}")
    if edf_all:
        print(f"[DIAG] example EDF: {edf_all[0].name}")

    pairs = find_sleep_edf_pairs(SLEEP_DIR)
    if len(pairs) < 3:
        raise RuntimeError("Muy pocos pares DEV. Revise pairing_log.csv")

    subjects = {}
    debug_rows = []

    for pair in tqdm(pairs, desc="Extracting sessions"):
        out = extract_subject(pair)
        if out is None:
            debug_rows.append({"subject_id": pair["subject_id"], "status": "extract_exception"})
            continue

        if out.get("status") != "accepted":
            debug_rows.append(out)
            continue

        sid = out["subject_id"]
        wake, n3 = normalize_by_state(out["wake"], out["n3"])
        subjects[sid] = {"wake": wake, "n3": n3}

        debug_rows.append({
            "subject_id": sid,
            "status": "accepted",
            "picked_channel": out["picked_channel"],
            "fs_in": out["fs_in"],
            "n_epochs_total": out["n_epochs_total"],
            "n_stage_W": out["n_stage_W"],
            "n_stage_N3": out["n_stage_N3"],
            "wake_postQC": out["wake_postQC"],
            "n3_postQC": out["n3_postQC"],
        })

    debug_df = pd.DataFrame(debug_rows)
    debug_df.to_csv(OUTPUT_DIR / "session_debug.csv", index=False)
    
    print(f"\n[DIAG] session_debug.csv: {OUTPUT_DIR / 'session_debug.csv'}")
    print(f"✓ Sujetos aceptados: {len(subjects)}")
    
    # Diagnóstico automático
    diagnose_rejections(debug_df)

    if len(subjects) < 3:
        raise RuntimeError("Muy pocos sujetos aceptados. Ver diagnóstico arriba.")

    # Metrics
    results = []
    for sid, data in tqdm(subjects.items(), desc="Computing metrics"):
        results.append(compute_metrics(data["wake"], sid, "wake"))
        results.append(compute_metrics(data["n3"], sid, "n3"))

    df = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "pilot_results_v1.3.2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ Resultados: {out_csv}")

    metrics_A = ["A1_lz", "A2_spec_ent", "A3_slope"]
    metrics_B = ["B1_temp_var", "B2_phase_stab", "B3_spec_var"]

    for m in metrics_A + metrics_B:
        df[f"{m}_z"] = zscore(df[m], nan_policy="omit")

    df["score_A"] = df[[f"{m}_z" for m in metrics_A]].mean(axis=1)
    df["score_B"] = df[[f"{m}_z" for m in metrics_B]].mean(axis=1)

    df_n3 = df[df["state"] == "n3"]
    sd_B_n3 = float(np.std(df_n3["score_B"].dropna()))
    corr_AB = float(df[["score_A", "score_B"]].corr().iloc[0, 1])

    print("\n" + "="*80)
    print("RESULTADOS v1.3.2")
    print("="*80)
    print(f"Sujetos aceptados: {len(subjects)}")
    print(f"SD(score_B) en N3: {sd_B_n3:.3f}  (umbral: ≥ 0.5)")
    print(f"Corr(score_A, score_B): {corr_AB:.3f} (umbral: |r| < 0.8)")
    print("="*80)
    
    go_decision = (sd_B_n3 >= 0.5 and abs(corr_AB) < 0.8)
    print("GO ✅" if go_decision else "NO-GO ❌")
    print("="*80)

    # Figures
    figures_dir = OUTPUT_DIR / "figures_v1.3.2"
    figures_dir.mkdir(exist_ok=True)

    corr_matrix = df[metrics_A + metrics_B].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Matriz de Correlaciones v1.3.2")
    plt.tight_layout()
    plt.savefig(figures_dir / "heatmap_correlations_v1.3.2.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    for st in df["state"].unique():
        sub = df[df["state"] == st]
        plt.scatter(sub["score_A"], sub["score_B"], label=st, alpha=0.75, s=80)
    plt.xlabel("Score A (diferenciación)")
    plt.ylabel("Score B (variabilidad/estabilidad)")
    plt.title(f"Scores A vs B v1.3.2 (r={corr_AB:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "scatter_A_vs_B_v1.3.2.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, m in enumerate(metrics_B):
        w = df[df["state"] == "wake"][m].dropna()
        n = df[df["state"] == "n3"][m].dropna()
        axes[i, 0].hist(w, bins=15, alpha=0.75)
        axes[i, 0].set_title(f"{m} Wake (SD={np.std(w):.3f})")
        axes[i, 0].grid(alpha=0.3)
        axes[i, 1].hist(n, bins=15, alpha=0.75)
        axes[i, 1].set_title(f"{m} N3 (SD={np.std(n):.3f})")
        axes[i, 1].grid(alpha=0.3)
    plt.suptitle("Distribuciones métricas B v1.3.2")
    plt.tight_layout()
    plt.savefig(figures_dir / "metrics_B_distributions_v1.3.2.png", dpi=300)
    plt.close()

    print(f"\n✓ Figuras: {figures_dir}")
    print(f"✓ Logs: pairing_log.csv | session_debug.csv")

if __name__ == "__main__":
    run()