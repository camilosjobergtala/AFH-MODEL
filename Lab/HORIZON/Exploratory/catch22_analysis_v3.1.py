#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CATCH22 + EEG FEATURES FOR CONSCIOUSNESS DISCRIMINATION
================================================================================

Versión: 3.1 (subject-level, optimized bootstrap)

CAMBIOS RESPECTO A v3.0:
- Bootstrap OPTIMIZADO con numpy vectorizado (~50x más rápido)
- N_BOOTSTRAP = 1000 por defecto
- Progress indicators

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
from scipy.signal import welch
from collections import defaultdict
import hashlib

warnings.filterwarnings('ignore')

# catch22 es REQUERIDO
try:
    import pycatch22
    CATCH22_AVAILABLE = True
    print("✅ pycatch22 disponible")
except ImportError:
    CATCH22_AVAILABLE = False
    print("❌ pycatch22 NO disponible")
    print("   Instalar con: pip install pycatch22")
    sys.exit(1)

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("❌ MNE no disponible")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

__version__ = "catch22_v3.1_subject_level_optimized"

SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.7
N_BOOTSTRAP = 1000  # Reducido de 10000 para velocidad
REDUNDANCY_THRESHOLD = 0.90

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
    r"/mnt/data/sleep-edf",
    r"./sleep-cassette",
]

DEFAULT_OUTPUT = r"./catch22_subject_level_output"

EPOCH_CONFIG = {
    'epoch_duration_s': 30,
    'min_epochs_per_state': 10,
    'max_epochs_per_state': 200,
}

ARTIFACT_CONFIG = {
    'max_amplitude_uv': 200,
}


# ============================================================================
# FEATURE CATEGORIES
# ============================================================================

def get_category(feature_name):
    """Asigna categoría a una característica."""
    CATCH22_CATEGORIES = {
        'DN_HistogramMode_5': 'distribution',
        'DN_HistogramMode_10': 'distribution',
        'DN_OutlierInclude_p_001_mdrmd': 'distribution',
        'DN_OutlierInclude_n_001_mdrmd': 'distribution',
        'CO_f1ecac': 'temporal_autocorr',
        'CO_FirstMin_ac': 'temporal_autocorr',
        'CO_HistogramAMI_even_2_5': 'temporal_autocorr',
        'CO_trev_1_num': 'temporal_autocorr',
        'IN_AutoMutualInfoStats_40_gaussian_fmmi': 'temporal_autocorr',
        'FC_LocalSimple_mean1_tauresrat': 'temporal_autocorr',
        'FC_LocalSimple_mean3_stderr': 'temporal_dynamics',
        'SB_BinaryStats_diff_longstretch0': 'temporal_dynamics',
        'SB_BinaryStats_mean_longstretch1': 'temporal_dynamics',
        'SB_MotifThree_quantile_hh': 'temporal_dynamics',
        'SB_TransitionMatrix_3ac_sumdiagcov': 'temporal_dynamics',
        'MD_hrv_classic_pnn40': 'temporal_dynamics',
        'PD_PeriodicityWang_th0_01': 'nonlinear',
        'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': 'nonlinear',
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1': 'nonlinear',
        'SP_Summaries_welch_rect_area_5_1': 'spectral',
        'SP_Summaries_welch_rect_centroid': 'spectral',
        'DN_Mean': 'distribution',
        'CO_Embed2_Dist_tau_d_expfit_meandiff': 'temporal_autocorr',
    }
    
    if feature_name.startswith('C22_'):
        base_name = feature_name[4:]
        return CATCH22_CATEGORIES.get(base_name, 'catch22_other')
    
    if 'power_' in feature_name or 'relpower_' in feature_name:
        return 'eeg_spectral_power'
    if 'ratio_' in feature_name:
        return 'eeg_spectral_ratio'
    if 'entropy' in feature_name:
        return 'eeg_entropy'
    if 'slope' in feature_name:
        return 'eeg_aperiodic'
    
    return 'other'


def is_temporal_category(category):
    """Determina si una categoría es temporal."""
    return category in ['temporal_autocorr', 'temporal_dynamics', 'nonlinear']


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """Extractor catch22 + EEG features."""
    
    def __init__(self, fs=100):
        self.fs = fs
    
    def extract_all(self, signal):
        features = {}
        signal = np.asarray(signal, dtype=np.float64)
        
        if len(signal) < 100:
            return features
        
        # catch22
        try:
            result = pycatch22.catch22_all(signal)
            for name, value in zip(result['names'], result['values']):
                features[f'C22_{name}'] = value
        except Exception as e:
            pass
        
        # EEG spectral
        features.update(self._eeg_spectral_features(signal))
        
        return features
    
    def _eeg_spectral_features(self, signal):
        features = {}
        try:
            nperseg = min(256, len(signal) // 2)
            if nperseg < 32:
                return features
            
            freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
            
            bands = {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 45),
            }
            
            total_power = np.trapz(psd, freqs)
            
            for band_name, (low, high) in bands.items():
                if high < self.fs / 2:
                    mask = (freqs >= low) & (freqs <= high)
                    band_power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0
                    features[f'EEG_power_{band_name}'] = band_power
                    features[f'EEG_relpower_{band_name}'] = band_power / total_power if total_power > 0 else np.nan
            
            if features.get('EEG_power_delta', 0) > 0:
                features['EEG_ratio_theta_delta'] = features.get('EEG_power_theta', 0) / features['EEG_power_delta']
                features['EEG_ratio_alpha_delta'] = features.get('EEG_power_alpha', 0) / features['EEG_power_delta']
                features['EEG_ratio_beta_delta'] = features.get('EEG_power_beta', 0) / features['EEG_power_delta']
            
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            features['EEG_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            mask = (freqs >= 2) & (freqs <= 20)
            if np.sum(mask) >= 5:
                log_freqs = np.log10(freqs[mask])
                log_psd = np.log10(psd[mask] + 1e-10)
                slope, _ = np.polyfit(log_freqs, log_psd, 1)
                features['EEG_slope_1f'] = slope
            
        except:
            pass
        
        return features


# ============================================================================
# SLEEP-EDF LOADER
# ============================================================================

class SleepEDFLoader:
    """Carga datos Sleep-EDF."""
    
    def __init__(self, base_path, target_fs=100):
        self.base_path = Path(base_path)
        self.target_fs = target_fs
    
    def find_psg_files(self):
        psg_files = list(self.base_path.rglob("*PSG*.edf"))
        
        sessions = []
        for psg_path in psg_files:
            hyp_path = self._find_hypnogram(psg_path)
            if hyp_path:
                fname = psg_path.stem
                subject_id = fname[:6] if len(fname) >= 6 else fname
                sessions.append({
                    'psg_path': psg_path,
                    'hypnogram_path': hyp_path,
                    'subject_id': subject_id,
                    'session_id': fname,
                })
        
        return sessions
    
    def _find_hypnogram(self, psg_path):
        psg_name = psg_path.stem
        folder = psg_path.parent
        
        hyp_name = psg_name.replace("PSG", "Hypnogram")
        hyp_path = folder / f"{hyp_name}.edf"
        if hyp_path.exists():
            return hyp_path
        
        token = psg_name.split('-')[0] if '-' in psg_name else psg_name[:6]
        for f in folder.glob("*Hypnogram*.edf"):
            if token in f.stem:
                return f
        
        subject_id = psg_name[:6]
        for f in folder.glob("*Hypnogram*.edf"):
            if f.stem[:6] == subject_id:
                return f
        
        return None
    
    def load_session(self, file_info):
        try:
            raw = mne.io.read_raw_edf(str(file_info['psg_path']), preload=True, verbose=False)
            
            channel = None
            for ch in ['EEG Fpz-Cz', 'EEG FPZ-CZ', 'FPZ-CZ', 'Fpz-Cz']:
                if ch in raw.ch_names:
                    channel = ch
                    break
            
            if channel is None:
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        channel = ch
                        break
            
            if channel is None:
                return None
            
            raw.pick([channel])
            
            if raw.info['sfreq'] != self.target_fs:
                raw.resample(self.target_fs, verbose=False)
            
            hyp = mne.read_annotations(str(file_info['hypnogram_path']))
            
            return {
                'raw': raw,
                'annotations': hyp,
                'fs': self.target_fs,
                'channel': channel,
            }
        except Exception as e:
            return None
    
    def extract_epochs_by_stage(self, session_data, epoch_duration=30):
        raw = session_data['raw']
        annotations = session_data['annotations']
        fs = session_data['fs']
        
        data = raw.get_data()[0]
        
        percentile_range = np.percentile(np.abs(data), 99)
        if percentile_range > 1e-3:
            signal_scale = 1
        elif percentile_range > 1e-6:
            signal_scale = 1e6
        else:
            signal_scale = 1
        
        data = data * signal_scale
        
        stage_map = {
            'Sleep stage W': 'W',
            'Sleep stage 1': 'N1',
            'Sleep stage 2': 'N2',
            'Sleep stage 3': 'N3',
            'Sleep stage 4': 'N3',
            'Sleep stage R': 'R',
            'Sleep stage ?': 'Unknown',
            'Movement time': 'Movement',
        }
        
        epochs_by_stage = defaultdict(list)
        samples_per_epoch = int(epoch_duration * fs)
        
        for ann in annotations:
            stage = stage_map.get(ann['description'], 'Unknown')
            if stage in ['Unknown', 'Movement']:
                continue
            
            onset_sample = int(ann['onset'] * fs)
            duration_samples = int(ann['duration'] * fs)
            
            n_epochs_in_ann = duration_samples // samples_per_epoch
            
            for i in range(n_epochs_in_ann):
                start = onset_sample + i * samples_per_epoch
                end = start + samples_per_epoch
                
                if end > len(data):
                    break
                
                epoch_data = data[start:end]
                
                if np.max(np.abs(epoch_data)) > ARTIFACT_CONFIG['max_amplitude_uv']:
                    continue
                
                epochs_by_stage[stage].append({
                    'data': epoch_data,
                    'start_sample': start,
                    'epoch_idx': len(epochs_by_stage[stage]),
                })
        
        return dict(epochs_by_stage)


# ============================================================================
# SUBJECT-LEVEL ANALYSIS
# ============================================================================

def compute_subject_level_stats(df, feature_cols):
    """
    Agrega a nivel sujeto y calcula paired Cohen's d.
    """
    
    df_subject = df.groupby(['subject_id', 'state'])[feature_cols].median()
    
    subjects_with_both = []
    for subj in df['subject_id'].unique():
        subj_states = df[df['subject_id'] == subj]['state'].unique()
        if 'W' in subj_states and 'N3' in subj_states:
            subjects_with_both.append(subj)
    
    print(f"   Sujetos con Wake y N3: {len(subjects_with_both)}")
    
    effect_sizes = {}
    
    for col in feature_cols:
        wake_vals = []
        n3_vals = []
        
        for subj in subjects_with_both:
            try:
                w = df_subject.loc[(subj, 'W'), col]
                n = df_subject.loc[(subj, 'N3'), col]
                if pd.notna(w) and pd.notna(n):
                    wake_vals.append(w)
                    n3_vals.append(n)
            except:
                continue
        
        if len(wake_vals) < 10:
            effect_sizes[col] = {'d': np.nan, 'n': 0}
            continue
        
        wake_arr = np.array(wake_vals)
        n3_arr = np.array(n3_vals)
        diff = wake_arr - n3_arr
        
        d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
        
        effect_sizes[col] = {
            'd': d,
            'n': len(wake_vals),
            'wake_mean': np.mean(wake_arr),
            'n3_mean': np.mean(n3_arr),
            'diff_mean': np.mean(diff),
            'diff_std': np.std(diff, ddof=1),
        }
    
    return df_subject, effect_sizes, subjects_with_both


def prune_redundant_features(df, feature_cols, threshold=REDUNDANCY_THRESHOLD, effect_sizes=None):
    """
    Elimina features redundantes (|r| > threshold).
    Mantiene el que tiene mayor |d|.
    """
    
    corr_matrix = df[feature_cols].corr(method='spearman')
    
    to_remove = set()
    
    for i, f1 in enumerate(feature_cols):
        if f1 in to_remove:
            continue
        for f2 in feature_cols[i+1:]:
            if f2 in to_remove:
                continue
            
            r = corr_matrix.loc[f1, f2]
            if abs(r) > threshold:
                d1 = abs(effect_sizes.get(f1, {}).get('d', 0) or 0)
                d2 = abs(effect_sizes.get(f2, {}).get('d', 0) or 0)
                
                if d1 < d2:
                    to_remove.add(f1)
                else:
                    to_remove.add(f2)
    
    pruned_cols = [f for f in feature_cols if f not in to_remove]
    
    print(f"   Redundancy pruning: {len(feature_cols)} → {len(pruned_cols)} features")
    print(f"   Removidos: {len(to_remove)} features con |r| > {threshold}")
    
    return pruned_cols, to_remove


def bootstrap_proportion_up(df_subject, feature_cols, effect_sizes, subjects_with_both, 
                            n_top=10, n_boot=N_BOOTSTRAP):
    """
    Bootstrap CI para proporción de features UP (d < 0) en top N.
    OPTIMIZED: Pre-computed difference matrix + vectorized operations.
    """
    
    print(f"   🔄 Bootstrap ({n_boot} iteraciones)...")
    
    # Pre-compute difference matrix (subjects x features)
    n_subjects = len(subjects_with_both)
    n_features = len(feature_cols)
    
    D = np.full((n_subjects, n_features), np.nan)
    
    for i, subj in enumerate(subjects_with_both):
        for j, col in enumerate(feature_cols):
            try:
                w = df_subject.loc[(subj, 'W'), col]
                n = df_subject.loc[(subj, 'N3'), col]
                if pd.notna(w) and pd.notna(n):
                    D[i, j] = w - n
            except:
                pass
    
    # Find valid features (enough non-nan subjects)
    valid_counts = np.sum(~np.isnan(D), axis=0)
    valid_mask = valid_counts >= 10
    
    if not np.any(valid_mask):
        return np.nan, np.nan
    
    rng = np.random.default_rng(SACRED_SEED + 1)
    proportions = []
    
    for b in range(n_boot):
        if (b + 1) % 200 == 0:
            print(f"      [{b+1}/{n_boot}]...")
        
        # Resample rows (subjects) with replacement
        boot_idx = rng.integers(0, n_subjects, size=n_subjects)
        D_boot = D[boot_idx, :]
        
        # Compute d for each feature (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            means = np.nanmean(D_boot, axis=0)
            stds = np.nanstd(D_boot, axis=0, ddof=1)
            d_vals = np.where(stds > 0, means / stds, 0)
        
        # Only consider features with enough data
        d_valid = [(feature_cols[j], d_vals[j]) 
                   for j in range(n_features) 
                   if valid_mask[j] and np.isfinite(d_vals[j])]
        
        if len(d_valid) < n_top:
            continue
        
        # Rank by |d| and count UP in top N
        ranked = sorted(d_valid, key=lambda x: abs(x[1]), reverse=True)
        top_n = ranked[:n_top]
        n_up = sum(1 for _, d in top_n if d < 0)
        proportions.append(n_up / n_top)
    
    if not proportions:
        return np.nan, np.nan
    
    print(f"   ✅ Bootstrap completado ({len(proportions)} muestras válidas)")
    
    return np.percentile(proportions, 2.5), np.percentile(proportions, 97.5)


# ============================================================================
# HYPOTHESIS EVALUATION
# ============================================================================

def evaluate_hypotheses(df_disc_subject, pruned_features, effect_sizes, 
                        df_subject, subjects_with_both):
    """
    Evalúa las hipótesis pre-registradas.
    """
    
    results = {}
    
    # Preparar ranking
    ranking = []
    for f in pruned_features:
        es = effect_sizes.get(f, {})
        d = es.get('d', np.nan)
        if not np.isnan(d):
            ranking.append({
                'feature': f,
                'd': d,
                'abs_d': abs(d),
                'category': get_category(f),
                'is_temporal': is_temporal_category(get_category(f)),
            })
    
    ranking = sorted(ranking, key=lambda x: x['abs_d'], reverse=True)
    
    # H1: ≥3/10 con d < 0
    top10 = ranking[:10]
    n_up_top10 = sum(1 for r in top10 if r['d'] < 0)
    
    # Bootstrap CI
    ci_lower, ci_upper = bootstrap_proportion_up(
        df_subject, pruned_features, effect_sizes, subjects_with_both, n_top=10
    )
    
    h1_count = n_up_top10 >= 3
    h1_ci = ci_lower > 0.10 if not np.isnan(ci_lower) else False
    h1_pass = h1_count and h1_ci
    
    results['H1'] = {
        'pass': h1_pass,
        'n_up_top10': n_up_top10,
        'prop_up': n_up_top10 / 10,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'criterion': '≥3/10 AND CI_lower > 0.10',
    }
    
    # H2: ≥3/5 temporal en top 5
    top5 = ranking[:5]
    n_temporal_top5 = sum(1 for r in top5 if r['is_temporal'])
    h2_pass = n_temporal_top5 >= 3
    
    results['H2'] = {
        'pass': h2_pass,
        'n_temporal_top5': n_temporal_top5,
        'top5_features': [(r['feature'], r['category'], r['d']) for r in top5],
        'criterion': '≥3/5 temporal',
    }
    
    # H3: |d| ≥ 1.5 para CO_Embed2
    embed2_features = [r for r in ranking if 'CO_Embed2' in r['feature']]
    if embed2_features:
        d_embed2 = embed2_features[0]['d']
        h3_pass = abs(d_embed2) >= 1.5
    else:
        d_embed2 = np.nan
        h3_pass = False
    
    results['H3'] = {
        'pass': h3_pass,
        'd_CO_Embed2': d_embed2,
        'criterion': '|d| ≥ 1.5',
    }
    
    # H4: 3 key features maintain d < 0
    key_features = ['CO_Embed2', 'SB_BinaryStats_diff', 'CO_HistogramAMI']
    key_directions = {}
    for kf in key_features:
        matches = [r for r in ranking if kf in r['feature']]
        if matches:
            key_directions[kf] = matches[0]['d'] < 0
        else:
            key_directions[kf] = None
    
    h4_pass = all(v == True for v in key_directions.values() if v is not None)
    
    results['H4'] = {
        'pass': h4_pass,
        'directions': key_directions,
        'criterion': 'All 3 maintain d < 0',
    }
    
    # H0: <2/10 con d < 0
    h0_pass = n_up_top10 < 2 and (ci_upper < 0.20 if not np.isnan(ci_upper) else False)
    
    results['H0'] = {
        'pass': h0_pass,
        'criterion': '<2/10 AND CI_upper < 0.20',
    }
    
    # Overall verdict
    n_afh_pass = sum(1 for h in ['H1', 'H2', 'H3'] if results[h]['pass'])
    
    if n_afh_pass >= 3 and results['H4']['pass']:
        verdict = "AFH_STRONG"
    elif n_afh_pass >= 2:
        verdict = "AFH_MODERATE"
    elif results['H0']['pass']:
        verdict = "QUANTITATIVE"
    else:
        verdict = "MIXED"
    
    results['verdict'] = verdict
    results['n_afh_hypotheses_passed'] = n_afh_pass
    results['ranking'] = ranking
    
    return results


# ============================================================================
# MAIN ANALYZER
# ============================================================================

class SubjectLevelAnalyzer:
    """Análisis a nivel sujeto con catch22."""
    
    def __init__(self, sleep_edf_path, output_dir=DEFAULT_OUTPUT):
        self.sleep_edf_path = Path(sleep_edf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, development_only=True):
        """Ejecuta el análisis completo."""
        
        print(f"\n{'='*70}")
        print("🔬 CATCH22 SUBJECT-LEVEL ANALYSIS (OPTIMIZED)")
        print(f"{'='*70}")
        print(f"Version: {__version__}")
        print(f"Output: {self.output_dir}")
        print(f"Bootstrap iterations: {N_BOOTSTRAP}")
        print(f"Redundancy threshold: {REDUNDANCY_THRESHOLD}")
        print(f"{'='*70}")
        
        # 1. Cargar archivos
        loader = SleepEDFLoader(self.sleep_edf_path)
        sessions = loader.find_psg_files()
        
        print(f"\n📂 Encontradas: {len(sessions)} sesiones")
        
        if not sessions:
            print("❌ No se encontraron archivos")
            return None
        
        # 2. Split sujetos (DETERMINISTIC)
        rng = np.random.default_rng(SACRED_SEED)
        all_subjects = sorted(list(set(s['subject_id'] for s in sessions)))
        shuffled_subjects = rng.permutation(all_subjects).tolist()
        
        n_dev = int(len(shuffled_subjects) * DEVELOPMENT_RATIO)
        dev_subjects = set(shuffled_subjects[:n_dev])
        test_subjects = set(shuffled_subjects[n_dev:])
        
        print(f"   Development: {len(dev_subjects)} sujetos")
        print(f"   Holdout: {len(test_subjects)} sujetos (reservados)")
        
        if development_only:
            sessions = [s for s in sessions if s['subject_id'] in dev_subjects]
            print(f"   Analizando: {len(sessions)} sesiones (development)")
        
        # 3. Extraer features
        print(f"\n📊 Extrayendo características...")
        
        extractor = FeatureExtractor(fs=100)
        all_features = []
        subject_epoch_counts = defaultdict(lambda: {'W': 0, 'N3': 0})
        
        for i, file_info in enumerate(sessions, 1):
            if i % 20 == 0 or i == len(sessions):
                print(f"   [{i}/{len(sessions)}] {file_info['session_id']}...")
            
            session_data = loader.load_session(file_info)
            if session_data is None:
                continue
            
            epochs = loader.extract_epochs_by_stage(session_data)
            
            wake_epochs = epochs.get('W', [])[:EPOCH_CONFIG['max_epochs_per_state']]
            n3_epochs = epochs.get('N3', [])[:EPOCH_CONFIG['max_epochs_per_state']]
            
            if len(wake_epochs) < EPOCH_CONFIG['min_epochs_per_state']:
                continue
            if len(n3_epochs) < EPOCH_CONFIG['min_epochs_per_state']:
                continue
            
            for epoch in wake_epochs:
                features = extractor.extract_all(epoch['data'])
                features['state'] = 'W'
                features['subject_id'] = file_info['subject_id']
                features['session_id'] = file_info['session_id']
                features['epoch_idx'] = epoch['epoch_idx']
                all_features.append(features)
                subject_epoch_counts[file_info['subject_id']]['W'] += 1
            
            for epoch in n3_epochs:
                features = extractor.extract_all(epoch['data'])
                features['state'] = 'N3'
                features['subject_id'] = file_info['subject_id']
                features['session_id'] = file_info['session_id']
                features['epoch_idx'] = epoch['epoch_idx']
                all_features.append(features)
                subject_epoch_counts[file_info['subject_id']]['N3'] += 1
        
        if not all_features:
            print("❌ No se extrajeron features")
            return None
        
        df = pd.DataFrame(all_features)
        
        meta_cols = ['state', 'subject_id', 'session_id', 'epoch_idx']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"   Total épocas: {len(df)}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Sujetos con datos: {len(subject_epoch_counts)}")
        
        # 4. EPOCH-LEVEL ANALYSIS (para comparación)
        print(f"\n{'='*70}")
        print("📊 ANÁLISIS EPOCH-LEVEL (para comparación)")
        print(f"{'='*70}")
        
        epoch_results = []
        wake_mask = df['state'] == 'W'
        n3_mask = df['state'] == 'N3'
        
        for col in feature_cols:
            wake_vals = df.loc[wake_mask, col].dropna()
            n3_vals = df.loc[n3_mask, col].dropna()
            
            if len(wake_vals) < 30 or len(n3_vals) < 30:
                continue
            
            pooled_std = np.sqrt(((len(wake_vals)-1)*np.var(wake_vals) + 
                                  (len(n3_vals)-1)*np.var(n3_vals)) / 
                                 (len(wake_vals) + len(n3_vals) - 2))
            
            d = (np.mean(wake_vals) - np.mean(n3_vals)) / pooled_std if pooled_std > 0 else 0
            
            epoch_results.append({
                'feature': col,
                'cohens_d_epoch': d,
                'abs_d_epoch': abs(d),
                'n_wake': len(wake_vals),
                'n_n3': len(n3_vals),
            })
        
        df_epoch = pd.DataFrame(epoch_results).sort_values('abs_d_epoch', ascending=False)
        
        print(f"\n   TOP 5 (epoch-level):")
        for i, row in enumerate(df_epoch.head(5).itertuples(), 1):
            print(f"   {i}. {row.feature[:45]}: d = {row.cohens_d_epoch:+.2f}")
        
        # 5. SUBJECT-LEVEL ANALYSIS
        print(f"\n{'='*70}")
        print("📊 ANÁLISIS SUBJECT-LEVEL (primario)")
        print(f"{'='*70}")
        
        df_subject, effect_sizes, subjects_with_both = compute_subject_level_stats(
            df, feature_cols
        )
        
        # 6. Redundancy pruning
        print(f"\n🔧 Redundancy pruning...")
        pruned_cols, removed_cols = prune_redundant_features(
            df, feature_cols, REDUNDANCY_THRESHOLD, effect_sizes
        )
        
        # 7. Create subject-level results
        subject_results = []
        for col in pruned_cols:
            es = effect_sizes.get(col, {})
            d = es.get('d', np.nan)
            if not np.isnan(d):
                subject_results.append({
                    'feature': col,
                    'category': get_category(col),
                    'cohens_d_subject': d,
                    'abs_d_subject': abs(d),
                    'n_subjects': es.get('n', 0),
                    'direction': 'W>N3' if d > 0 else 'N3>W',
                    'is_temporal': is_temporal_category(get_category(col)),
                })
        
        df_subject_disc = pd.DataFrame(subject_results).sort_values('abs_d_subject', ascending=False)
        
        # 8. Print results
        print(f"\n🔝 TOP 20 (SUBJECT-LEVEL, n={len(subjects_with_both)} sujetos):")
        print(f"{'─'*90}")
        print(f"{'Rank':<5} {'Feature':<40} {'Category':<20} {'d':>10} {'Dir':>8}")
        print(f"{'─'*90}")
        
        for rank, row in enumerate(df_subject_disc.head(20).itertuples(), 1):
            print(f"{rank:<5} {row.feature[:38]:<40} {row.category:<20} "
                  f"{row.cohens_d_subject:>+10.2f} {row.direction:>8}")
        
        # 9. Evaluate hypotheses
        print(f"\n{'='*70}")
        print("🧪 EVALUACIÓN DE HIPÓTESIS (Registered Report)")
        print(f"{'='*70}")
        
        hyp_results = evaluate_hypotheses(
            df_subject_disc, pruned_cols, effect_sizes, 
            df_subject, subjects_with_both
        )
        
        print(f"\n📌 H1 (Heterogeneidad): {'✅ PASS' if hyp_results['H1']['pass'] else '❌ FAIL'}")
        print(f"   N UP en top 10: {hyp_results['H1']['n_up_top10']}/10")
        print(f"   Bootstrap CI: [{hyp_results['H1']['ci_lower']:.3f}, {hyp_results['H1']['ci_upper']:.3f}]")
        print(f"   Criterio: {hyp_results['H1']['criterion']}")
        
        print(f"\n📌 H2 (Dominancia temporal): {'✅ PASS' if hyp_results['H2']['pass'] else '❌ FAIL'}")
        print(f"   N temporal en top 5: {hyp_results['H2']['n_temporal_top5']}/5")
        print(f"   Top 5:")
        for f, cat, d in hyp_results['H2']['top5_features']:
            temp_mark = "⏱️" if is_temporal_category(cat) else "  "
            print(f"      {temp_mark} {f[:35]}: {cat} (d={d:+.2f})")
        
        print(f"\n📌 H3 (Replicación CO_Embed2): {'✅ PASS' if hyp_results['H3']['pass'] else '❌ FAIL'}")
        print(f"   d(CO_Embed2) = {hyp_results['H3']['d_CO_Embed2']:.2f}")
        print(f"   Criterio: {hyp_results['H3']['criterion']}")
        
        print(f"\n📌 H4 (Consistencia direccional): {'✅ PASS' if hyp_results['H4']['pass'] else '❌ FAIL'}")
        for kf, dir_ok in hyp_results['H4']['directions'].items():
            status = "✅ d<0" if dir_ok else ("❌ d>0" if dir_ok == False else "⚠️ N/A")
            print(f"   {kf}: {status}")
        
        print(f"\n📌 H0 (Null cuantitativo): {'✅ PASS' if hyp_results['H0']['pass'] else '❌ FAIL'}")
        
        print(f"\n{'='*70}")
        print(f"🏆 VEREDICTO: {hyp_results['verdict']}")
        print(f"   Hipótesis AFH pasadas: {hyp_results['n_afh_hypotheses_passed']}/3")
        if hyp_results['H4']['pass']:
            print(f"   + H4 direccional: ✅")
        print(f"{'='*70}")
        
        # 10. Comparación epoch vs subject
        print(f"\n📊 COMPARACIÓN EPOCH-LEVEL vs SUBJECT-LEVEL:")
        print(f"{'─'*70}")
        
        df_compare = df_epoch.merge(
            df_subject_disc[['feature', 'cohens_d_subject', 'abs_d_subject']],
            on='feature', how='outer'
        ).sort_values('abs_d_subject', ascending=False, na_position='last')
        
        print(f"{'Feature':<35} {'d(epoch)':>12} {'d(subject)':>12} {'Shrink%':>10}")
        print(f"{'─'*70}")
        for row in df_compare.head(15).itertuples():
            d_e = row.cohens_d_epoch if pd.notna(row.cohens_d_epoch) else np.nan
            d_s = row.cohens_d_subject if pd.notna(row.cohens_d_subject) else np.nan
            if pd.notna(d_e) and pd.notna(d_s) and abs(d_e) > 0.01:
                shrink = (1 - abs(d_s)/abs(d_e)) * 100
                shrink_str = f"{shrink:+.0f}%"
            else:
                shrink_str = "N/A"
            print(f"{row.feature[:33]:<35} {d_e:>+12.2f} {d_s:>+12.2f} {shrink_str:>10}")
        
        # 11. Save results
        print(f"\n💾 Guardando resultados...")
        
        df.to_csv(self.output_dir / "all_epochs.csv", index=False)
        df_epoch.to_csv(self.output_dir / "epoch_level_results.csv", index=False)
        df_subject_disc.to_csv(self.output_dir / "subject_level_results.csv", index=False)
        df_compare.to_csv(self.output_dir / "epoch_vs_subject_comparison.csv", index=False)
        
        # Save hypothesis results
        hyp_export = {
            'version': __version__,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_subjects': len(subjects_with_both),
            'n_epochs': len(df),
            'n_features_original': len(feature_cols),
            'n_features_pruned': len(pruned_cols),
            'hypotheses': {
                'H1': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                       for k, v in hyp_results['H1'].items()},
                'H2': {k: (float(v) if isinstance(v, (np.floating, float)) else 
                          [(str(a), str(b), float(c)) for a,b,c in v] if isinstance(v, list) else v)
                       for k, v in hyp_results['H2'].items()},
                'H3': {k: (float(v) if isinstance(v, (np.floating, float)) else v) 
                       for k, v in hyp_results['H3'].items()},
                'H4': hyp_results['H4'],
                'H0': hyp_results['H0'],
            },
            'verdict': hyp_results['verdict'],
            'n_afh_passed': hyp_results['n_afh_hypotheses_passed'],
        }
        
        with open(self.output_dir / "hypothesis_evaluation.json", 'w') as f:
            json.dump(hyp_export, f, indent=2, default=str)
        
        print(f"\n📁 Resultados guardados en: {self.output_dir}")
        
        return {
            'df': df,
            'df_epoch': df_epoch,
            'df_subject': df_subject_disc,
            'hypotheses': hyp_results,
        }


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and list(p.rglob("*PSG*.edf")):
            return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Catch22 Subject-Level Analysis ({__version__})")
    parser.add_argument('--sleep-edf-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--include-holdout', action='store_true', 
                        help='Include holdout subjects (for final validation only!)')
    args = parser.parse_args()
    
    data_path = Path(args.sleep_edf_path) if args.sleep_edf_path else find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontraron datos Sleep-EDF")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    if args.include_holdout:
        print("\n⚠️  ADVERTENCIA: Incluyendo holdout subjects!")
        print("   Esto debe hacerse SOLO para validación final post-IPA")
        confirm = input("   Escribir 'CONFIRMO VALIDACION FINAL': ")
        if confirm != 'CONFIRMO VALIDACION FINAL':
            print("   Cancelado.")
            sys.exit(0)
    
    analyzer = SubjectLevelAnalyzer(str(data_path), args.output_dir)
    results = analyzer.run(development_only=not args.include_holdout)
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()