#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CATCH22 + EEG FEATURES FOR CONSCIOUSNESS DISCRIMINATION
================================================================================

Sigue la línea Tsuchiya/Fulcher:
- catch22: 22 características canónicas de HCTSA (Lubba et al., 2019)
- Seleccionadas por máximo poder discriminativo + mínima redundancia
- Complementadas con características EEG-específicas (bandas, ratios)

Objetivo:
- Identificar qué características discriminan Wake vs N3 sin sesgo de selección
- Categorizar las ganadoras por dominio (temporal, espectral, distribución)
- Evaluar si características temporales dominan (compatible con AFH)

Predicción AFH operacionalizable:
- Si características relacionadas con estructura temporal dominan el top 20,
  eso sugiere que la organización temporal es clave para consciencia.

REQUIERE: pip install pycatch22 mne numpy scipy pandas matplotlib

================================================================================
Versión: 2.0 (catch22)
Basado en: Lubba et al. (2019) - catch22: CAnonical Time-series CHaracteristics
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
from scipy.signal import welch, butter, filtfilt, hilbert, detrend
from scipy.fft import fft, fftfreq
from collections import Counter, defaultdict
import hashlib

warnings.filterwarnings('ignore')

# catch22 es REQUERIDO para seguir línea Tsuchiya/Fulcher
try:
    import pycatch22
    CATCH22_AVAILABLE = True
    print("✅ pycatch22 disponible")
except ImportError:
    CATCH22_AVAILABLE = False
    print("❌ pycatch22 NO disponible")
    print("   Instalar con: pip install pycatch22")
    print("   Este análisis REQUIERE catch22 para seguir la línea Tsuchiya/Fulcher")
    sys.exit(1)

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("❌ MNE no disponible - necesario para cargar Sleep-EDF")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

__version__ = "catch22_v2.0"

SACRED_SEED = 2025
DEVELOPMENT_RATIO = 0.7

# Rutas por defecto (ajustar según tu sistema)
DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
    r"/mnt/data/sleep-edf",
    r"./sleep-cassette",
]

DEFAULT_OUTPUT = r"./hctsa_analysis_output"

# Configuración de épocas
EPOCH_CONFIG = {
    'epoch_duration_s': 30,
    'min_epochs_per_state': 50,  # mínimo por sujeto para incluirlo
    'max_epochs_per_state': 200,  # máximo para balancear
}

# Configuración de artefactos
ARTIFACT_CONFIG = {
    'max_amplitude_uv': 200,
    'max_kurtosis': 5,
}

# Categorías ahora definidas en get_category() basadas en catch22


# ============================================================================
# UTILIDADES
# ============================================================================

def get_category(feature_name):
    """Asigna una categoría a una característica."""
    # catch22 categories
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
    }
    
    # catch22
    if feature_name.startswith('C22_'):
        base_name = feature_name[4:]
        return CATCH22_CATEGORIES.get(base_name, 'catch22_other')
    
    # EEG-específicas
    if 'power_' in feature_name or 'relpower_' in feature_name:
        return 'eeg_spectral_power'
    if 'ratio_' in feature_name:
        return 'eeg_spectral_ratio'
    if 'entropy' in feature_name:
        return 'eeg_entropy'
    if 'slope' in feature_name:
        return 'eeg_aperiodic'
    
    return 'other'


def estimate_signal_scale(signal):
    """Estima si la señal está en uV o V."""
    p99 = np.percentile(np.abs(signal), 99)
    if p99 > 10:
        return 1.0  # Ya en uV
    else:
        return 1e6  # Convertir de V a uV


def check_epoch_quality(epoch, signal_scale=1.0):
    """Verifica si una época es de buena calidad."""
    amplitude_uv = np.ptp(epoch) * signal_scale
    if amplitude_uv > ARTIFACT_CONFIG['max_amplitude_uv']:
        return False, 'high_amplitude'
    
    kurt = stats.kurtosis(epoch, fisher=True, bias=False)
    if abs(kurt) > ARTIFACT_CONFIG['max_kurtosis']:
        return False, 'abnormal_kurtosis'
    
    if np.any(np.isnan(epoch)) or np.any(np.isinf(epoch)):
        return False, 'invalid_values'
    
    return True, 'clean'


# ============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS - IMPLEMENTACIONES PROPIAS
# ============================================================================

class FeatureExtractor:
    """
    Extractor de características basado en catch22 + características EEG-específicas.
    
    catch22: 22 características canónicas de HCTSA (Lubba et al., 2019)
    - Seleccionadas por máximo poder discriminativo y mínima redundancia
    - Es el estándar recomendado por Tsuchiya/Fulcher
    
    EEG-específicas: Características que catch22 no cubre pero son relevantes para EEG
    - Potencia espectral por bandas
    - Ratios espectrales
    - Entropía espectral
    """
    
    # Categorías de las 22 características de catch22
    CATCH22_CATEGORIES = {
        # Distribución
        'DN_HistogramMode_5': 'distribution',
        'DN_HistogramMode_10': 'distribution',
        'DN_OutlierInclude_p_001_mdrmd': 'distribution',
        'DN_OutlierInclude_n_001_mdrmd': 'distribution',
        
        # Autocorrelación / Temporal
        'CO_f1ecac': 'temporal_autocorr',
        'CO_FirstMin_ac': 'temporal_autocorr',
        'CO_HistogramAMI_even_2_5': 'temporal_autocorr',
        'CO_trev_1_num': 'temporal_autocorr',
        'IN_AutoMutualInfoStats_40_gaussian_fmmi': 'temporal_autocorr',
        'FC_LocalSimple_mean1_tauresrat': 'temporal_autocorr',
        'FC_LocalSimple_mean3_stderr': 'temporal_dynamics',
        
        # Dinámica / No lineal
        'SB_BinaryStats_diff_longstretch0': 'temporal_dynamics',
        'SB_BinaryStats_mean_longstretch1': 'temporal_dynamics',
        'SB_MotifThree_quantile_hh': 'temporal_dynamics',
        'SB_TransitionMatrix_3ac_sumdiagcov': 'temporal_dynamics',
        'MD_hrv_classic_pnn40': 'temporal_dynamics',
        
        # Predicción / Complejidad
        'PD_PeriodicityWang_th0_01': 'nonlinear',
        'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1': 'nonlinear',
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1': 'nonlinear',
        'SP_Summaries_welch_rect_area_5_1': 'spectral',
        'SP_Summaries_welch_rect_centroid': 'spectral',
        
        # Estadístico
        'DN_Mean': 'distribution',
    }
    
    def __init__(self, fs=100):
        self.fs = fs
    
    def extract_all(self, signal):
        """Extrae catch22 + características EEG-específicas."""
        features = {}
        
        signal = np.asarray(signal, dtype=np.float64)
        if len(signal) < 100:
            return features
        
        # 1. CATCH22 - El núcleo (22 características canónicas)
        try:
            catch22_result = pycatch22.catch22_all(signal)
            for name, value in zip(catch22_result['names'], catch22_result['values']):
                features[f'C22_{name}'] = value
        except Exception as e:
            print(f"      ⚠️ catch22 error: {e}")
        
        # 2. Características EEG-específicas (que catch22 no cubre)
        features.update(self._eeg_spectral_features(signal))
        
        return features
    
    def _eeg_spectral_features(self, signal):
        """Características espectrales específicas de EEG."""
        features = {}
        
        try:
            nperseg = min(256, len(signal) // 2)
            if nperseg < 32:
                return features
            
            freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)
            
            # Potencia por bandas EEG clásicas
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
            
            # Ratios espectrales (biomarcadores clásicos de consciencia)
            if features.get('EEG_power_delta', 0) > 0:
                features['EEG_ratio_theta_delta'] = features.get('EEG_power_theta', 0) / features['EEG_power_delta']
                features['EEG_ratio_alpha_delta'] = features.get('EEG_power_alpha', 0) / features['EEG_power_delta']
                features['EEG_ratio_beta_delta'] = features.get('EEG_power_beta', 0) / features['EEG_power_delta']
            
            # Entropía espectral
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]
            features['EEG_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Pendiente 1/f (aperiodic component)
            mask = (freqs >= 2) & (freqs <= 20)
            if np.sum(mask) >= 5:
                log_freqs = np.log10(freqs[mask])
                log_psd = np.log10(psd[mask] + 1e-10)
                slope, _ = np.polyfit(log_freqs, log_psd, 1)
                features['EEG_slope_1f'] = slope
            
        except Exception as e:
            pass
        
        return features
    
    def get_category(self, feature_name):
        """Asigna categoría a una característica."""
        # catch22
        if feature_name.startswith('C22_'):
            base_name = feature_name[4:]  # Quitar 'C22_'
            return self.CATCH22_CATEGORIES.get(base_name, 'catch22_other')
        
        # EEG-específicas
        if 'power_' in feature_name or 'relpower_' in feature_name:
            return 'eeg_spectral_power'
        if 'ratio_' in feature_name:
            return 'eeg_spectral_ratio'
        if 'entropy' in feature_name:
            return 'eeg_entropy'
        if 'slope' in feature_name:
            return 'eeg_aperiodic'
        
        return 'other'


# catch22 ahora integrado en FeatureExtractor.extract_all()


# ============================================================================
# SLEEP-EDF LOADER
# ============================================================================

class SleepEDFLoader:
    """Cargador de datos Sleep-EDF (copiado de cascade v3.5)."""
    
    STAGE_MAPPING = {
        'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3', 'Sleep stage 4': 'N4', 'Sleep stage R': 'R',
        'Sleep stage ?': '?', 'Movement time': 'M'
    }
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.pairing_log = []
    
    def find_psg_files(self):
        """Encuentra pares PSG-Hypnogram (lógica exacta de cascade v3.5)."""
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
            
            # Método 1: Nombre exacto
            expected_hyp_name = psg_name.replace('PSG', 'Hypnogram')
            if expected_hyp_name in hyp_by_stem:
                candidates = [h for h in hyp_by_stem[expected_hyp_name] if str(h.parent) == psg_folder]
                if len(candidates) == 1:
                    hyp_file = candidates[0]
                    match_method = 'exact_name'
                elif len(candidates) > 1:
                    continue
            
            # Método 2: Mismo token
            if hyp_file is None and psg_folder in hyp_by_folder:
                matching = [h for h in hyp_by_folder[psg_folder] if h.stem.split('-')[0] == psg_token]
                if len(matching) == 1:
                    hyp_file = matching[0]
                    match_method = 'same_token'
                elif len(matching) > 1:
                    continue
            
            # Método 3: Mismo subject
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
                elif len(matching) > 1:
                    continue
            
            if hyp_file is None:
                continue
            
            files.append({
                'session_id': session_id,
                'subject_id': subject_id,
                'psg_path': psg_file,
                'hyp_path': hyp_file,
            })
        
        # Deduplicar
        seen = set()
        unique = []
        for f in files:
            if str(f['psg_path']) not in seen:
                seen.add(str(f['psg_path']))
                unique.append(f)
        return unique
    
    def load_session(self, psg_path, hyp_path, target_channel='Fpz-Cz'):
        """Carga una sesión completa."""
        try:
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
            
            fs = raw.info['sfreq']
            signal = raw.copy().pick_channels([channel]).get_data()[0]
            signal_scale = estimate_signal_scale(signal)
            
            annotations = mne.read_annotations(str(hyp_path))
            if len(annotations) == 0:
                return None
            
            psg_duration_s = len(signal) / fs
            n_epochs = int(psg_duration_s // 30)
            stages = ['?'] * n_epochs
            
            for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
                stage = self.STAGE_MAPPING.get(desc, '?')
                start_epoch = int(np.floor(onset / 30))
                end_epoch = int(np.ceil((onset + duration) / 30))
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            return {
                'signal': signal,
                'fs': fs,
                'stages': stages,
                'channel': channel,
                'signal_scale': signal_scale,
            }
        except:
            return None
    
    def extract_epochs_by_stage(self, signal, fs, stages, target_stages, signal_scale=1.0):
        """Extrae épocas de estadios específicos."""
        epoch_samples = int(30 * fs)
        epochs = []
        
        for i, stage in enumerate(stages):
            if stage not in target_stages:
                continue
            
            # N4 se agrupa con N3
            effective_stage = 'N3' if stage == 'N4' else stage
            if effective_stage not in target_stages and stage not in target_stages:
                continue
            
            start = i * epoch_samples
            end = start + epoch_samples
            
            if end > len(signal):
                continue
            
            epoch = signal[start:end]
            is_clean, _ = check_epoch_quality(epoch, signal_scale)
            
            if is_clean:
                epochs.append({
                    'data': epoch,
                    'stage': stage,
                    'epoch_idx': i,
                })
        
        return epochs


# ============================================================================
# ANÁLISIS PRINCIPAL
# ============================================================================

class HCTSAAnalyzer:
    """Analizador principal estilo HCTSA."""
    
    def __init__(self, data_path, output_dir):
        self.loader = SleepEDFLoader(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"🔬 HCTSA-STYLE ANALYSIS FOR CONSCIOUSNESS DISCRIMINATION")
        print(f"{'='*70}")
        print(f"Version: {__version__}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def run(self):
        """Ejecuta el análisis completo."""
        
        # 1. Encontrar archivos
        print("📂 Buscando archivos Sleep-EDF...")
        files = self.loader.find_psg_files()
        print(f"   Encontrados: {len(files)} sesiones")
        
        if not files:
            print("❌ No se encontraron archivos")
            return None
        
        # 2. Split desarrollo/holdout
        all_subjects = sorted({f['subject_id'] for f in files})
        rng = np.random.default_rng(SACRED_SEED)
        rng.shuffle(all_subjects)
        n_dev = int(len(all_subjects) * DEVELOPMENT_RATIO)
        dev_subjects = set(all_subjects[:n_dev])
        test_subjects = set(all_subjects[n_dev:])
        dev_files = [f for f in files if f['subject_id'] in dev_subjects]
        
        print(f"   Development: {len(dev_subjects)} sujetos, {len(dev_files)} sesiones")
        print(f"   Holdout: {len(test_subjects)} sujetos (reservados)")
        
        # 3. Extraer características
        print(f"\n📊 Extrayendo características...")
        
        all_features = []
        subject_epoch_counts = defaultdict(lambda: {'W': 0, 'N3': 0})
        
        for i, file_info in enumerate(dev_files):
            print(f"\r   [{i+1}/{len(dev_files)}] {file_info['session_id'][:40]}...", end="", flush=True)
            
            data = self.loader.load_session(file_info['psg_path'], file_info['hyp_path'])
            if data is None:
                continue
            
            # Extraer épocas Wake y N3
            wake_epochs = self.loader.extract_epochs_by_stage(
                data['signal'], data['fs'], data['stages'], ['W'], data['signal_scale']
            )
            n3_epochs = self.loader.extract_epochs_by_stage(
                data['signal'], data['fs'], data['stages'], ['N3'], data['signal_scale']
            )
            
            # Verificar mínimos
            if len(wake_epochs) < EPOCH_CONFIG['min_epochs_per_state']:
                continue
            if len(n3_epochs) < EPOCH_CONFIG['min_epochs_per_state']:
                continue
            
            # Limitar máximos para balance
            max_epochs = EPOCH_CONFIG['max_epochs_per_state']
            if len(wake_epochs) > max_epochs:
                rng_local = np.random.default_rng(SACRED_SEED + i)
                indices = rng_local.choice(len(wake_epochs), max_epochs, replace=False)
                wake_epochs = [wake_epochs[j] for j in indices]
            if len(n3_epochs) > max_epochs:
                rng_local = np.random.default_rng(SACRED_SEED + i + 1000)
                indices = rng_local.choice(len(n3_epochs), max_epochs, replace=False)
                n3_epochs = [n3_epochs[j] for j in indices]
            
            # Crear extractor de características
            extractor = FeatureExtractor(fs=data['fs'])
            
            # Procesar épocas Wake
            for epoch in wake_epochs:
                features = extractor.extract_all(epoch['data'])
                
                features['state'] = 'W'
                features['subject_id'] = file_info['subject_id']
                features['session_id'] = file_info['session_id']
                features['epoch_idx'] = epoch['epoch_idx']
                
                all_features.append(features)
                subject_epoch_counts[file_info['subject_id']]['W'] += 1
            
            # Procesar épocas N3
            for epoch in n3_epochs:
                features = extractor.extract_all(epoch['data'])
                
                features['state'] = 'N3'
                features['subject_id'] = file_info['subject_id']
                features['session_id'] = file_info['session_id']
                features['epoch_idx'] = epoch['epoch_idx']
                
                all_features.append(features)
                subject_epoch_counts[file_info['subject_id']]['N3'] += 1
        
        print(f"\n   Total épocas extraídas: {len(all_features)}")
        
        if len(all_features) == 0:
            print("❌ No se extrajeron características")
            return None
        
        # 4. Crear DataFrame
        df = pd.DataFrame(all_features)
        
        # Separar metadatos de características
        meta_cols = ['state', 'subject_id', 'session_id', 'epoch_idx']
        feature_cols = [c for c in df.columns if c not in meta_cols]
        
        print(f"   Características extraídas: {len(feature_cols)}")
        
        # 5. Calcular poder discriminativo
        print(f"\n📈 Calculando poder discriminativo...")
        
        discrimination_results = []
        
        wake_mask = df['state'] == 'W'
        n3_mask = df['state'] == 'N3'
        
        for col in feature_cols:
            wake_vals = df.loc[wake_mask, col].dropna()
            n3_vals = df.loc[n3_mask, col].dropna()
            
            if len(wake_vals) < 30 or len(n3_vals) < 30:
                continue
            
            # Cohen's d
            pooled_std = np.sqrt(((len(wake_vals)-1)*np.var(wake_vals) + (len(n3_vals)-1)*np.var(n3_vals)) / 
                                 (len(wake_vals) + len(n3_vals) - 2))
            if pooled_std > 0:
                cohens_d = (np.mean(wake_vals) - np.mean(n3_vals)) / pooled_std
            else:
                cohens_d = 0
            
            # Mann-Whitney U
            try:
                stat, pval = stats.mannwhitneyu(wake_vals, n3_vals, alternative='two-sided')
                # Effect size r = Z / sqrt(N)
                z = stats.norm.ppf(1 - pval/2) if pval > 0 else np.inf
                effect_r = z / np.sqrt(len(wake_vals) + len(n3_vals))
            except:
                pval = 1.0
                effect_r = 0
            
            # AUC-ROC
            try:
                from sklearn.metrics import roc_auc_score
                y_true = [1] * len(wake_vals) + [0] * len(n3_vals)
                y_score = list(wake_vals) + list(n3_vals)
                auc = roc_auc_score(y_true, y_score)
            except:
                auc = 0.5
            
            # Categoría
            category = get_category(col)
            
            discrimination_results.append({
                'feature': col,
                'category': category,
                'cohens_d': cohens_d,
                'abs_cohens_d': abs(cohens_d),
                'effect_r': effect_r,
                'auc': auc,
                'pval': pval,
                'wake_mean': np.mean(wake_vals),
                'wake_std': np.std(wake_vals),
                'n3_mean': np.mean(n3_vals),
                'n3_std': np.std(n3_vals),
                'n_wake': len(wake_vals),
                'n_n3': len(n3_vals),
                'direction': 'W>N3' if cohens_d > 0 else 'N3>W',
            })
        
        df_disc = pd.DataFrame(discrimination_results)
        df_disc = df_disc.sort_values('abs_cohens_d', ascending=False)
        
        # 6. Análisis por categoría
        print(f"\n📊 Análisis por categoría...")
        
        category_stats = []
        for category in df_disc['category'].unique():
            cat_data = df_disc[df_disc['category'] == category]
            category_stats.append({
                'category': category,
                'n_features': len(cat_data),
                'mean_abs_d': cat_data['abs_cohens_d'].mean(),
                'max_abs_d': cat_data['abs_cohens_d'].max(),
                'n_large_effect': (cat_data['abs_cohens_d'] >= 0.8).sum(),
                'n_medium_effect': ((cat_data['abs_cohens_d'] >= 0.5) & (cat_data['abs_cohens_d'] < 0.8)).sum(),
                'pct_significant': 100 * (cat_data['pval'] < 0.001).mean(),
            })
        
        df_categories = pd.DataFrame(category_stats).sort_values('mean_abs_d', ascending=False)
        
        # 7. Guardar resultados
        print(f"\n💾 Guardando resultados...")
        
        # Features completas
        df.to_csv(self.output_dir / "all_features.csv", index=False)
        
        # Ranking de discriminación
        df_disc.to_csv(self.output_dir / "discrimination_ranking.csv", index=False)
        
        # Estadísticas por categoría
        df_categories.to_csv(self.output_dir / "category_analysis.csv", index=False)
        
        # Conteo por sujeto
        df_counts = pd.DataFrame([
            {'subject_id': s, 'n_wake': c['W'], 'n_n3': c['N3']}
            for s, c in subject_epoch_counts.items()
        ])
        df_counts.to_csv(self.output_dir / "subject_epoch_counts.csv", index=False)
        
        # Resumen JSON
        summary = {
            'version': __version__,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sacred_seed': SACRED_SEED,
            'n_subjects': len(subject_epoch_counts),
            'n_epochs_total': len(df),
            'n_epochs_wake': int(wake_mask.sum()),
            'n_epochs_n3': int(n3_mask.sum()),
            'n_features': len(feature_cols),
            'n_features_analyzed': len(df_disc),
            'top_10_features': df_disc.head(10)[['feature', 'category', 'cohens_d', 'auc']].to_dict('records'),
            'category_summary': df_categories.to_dict('records'),
            'split': {
                'dev_subjects': len(dev_subjects),
                'test_subjects': len(test_subjects),
            }
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 8. Imprimir resultados
        print(f"\n{'='*70}")
        print("📊 RESULTADOS")
        print(f"{'='*70}")
        
        print(f"\n🔝 TOP 20 CARACTERÍSTICAS DISCRIMINATIVAS (Wake vs N3):")
        print(f"{'─'*90}")
        print(f"{'Rank':<5} {'Feature':<40} {'Category':<20} {'Cohen d':>10} {'AUC':>8}")
        print(f"{'─'*90}")
        
        for rank, row in enumerate(df_disc.head(20).itertuples(), 1):
            print(f"{rank:<5} {row.feature[:38]:<40} {row.category:<20} {row.cohens_d:>+10.2f} {row.auc:>8.3f}")
        
        print(f"\n📈 ANÁLISIS POR CATEGORÍA:")
        print(f"{'─'*80}")
        print(f"{'Category':<25} {'N':<6} {'Mean |d|':>10} {'Max |d|':>10} {'Large':>8} {'%Sig':>8}")
        print(f"{'─'*80}")
        
        for row in df_categories.itertuples():
            print(f"{row.category:<25} {row.n_features:<6} {row.mean_abs_d:>10.2f} {row.max_abs_d:>10.2f} "
                  f"{row.n_large_effect:>8} {row.pct_significant:>7.1f}%")
        
        # 9. Interpretación AFH
        print(f"\n{'='*70}")
        print("🧠 INTERPRETACIÓN PARA AFH (basada en catch22)")
        print(f"{'='*70}")
        
        # Contar categorías en top 20
        top20_categories = df_disc.head(20)['category'].value_counts()
        
        print(f"\nDistribución de categorías en TOP 20:")
        for cat, count in top20_categories.items():
            print(f"   {cat}: {count} ({100*count/20:.0f}%)")
        
        # Características temporales en top (catch22 temporal categories)
        temporal_cats = ['temporal_autocorr', 'temporal_dynamics', 'nonlinear']
        temporal_in_top20 = df_disc.head(20)[df_disc.head(20)['category'].isin(temporal_cats)]
        
        print(f"\n📌 Características TEMPORALES (catch22) en TOP 20: {len(temporal_in_top20)}/20 ({100*len(temporal_in_top20)/20:.0f}%)")
        print(f"   (incluye: autocorrelación, dinámica temporal, no-lineal)")
        
        if len(temporal_in_top20) >= 10:
            print("   → FUERTE apoyo para la hipótesis de estructura temporal")
        elif len(temporal_in_top20) >= 5:
            print("   → MODERADO apoyo para la hipótesis de estructura temporal")
        else:
            print("   → DÉBIL apoyo - otras características dominan")
        
        # Características espectrales (catch22 + EEG)
        spectral_cats = ['spectral', 'eeg_spectral_power', 'eeg_spectral_ratio']
        spectral_in_top20 = df_disc.head(20)[df_disc.head(20)['category'].isin(spectral_cats)]
        print(f"\n📌 Características ESPECTRALES en TOP 20: {len(spectral_in_top20)}/20 ({100*len(spectral_in_top20)/20:.0f}%)")
        
        # Distribución
        dist_in_top20 = df_disc.head(20)[df_disc.head(20)['category'] == 'distribution']
        print(f"📌 Características de DISTRIBUCIÓN en TOP 20: {len(dist_in_top20)}/20 ({100*len(dist_in_top20)/20:.0f}%)")
        
        print(f"\n📌 NOTA: catch22 = 22 características canónicas de HCTSA (Lubba et al., 2019)")
        print(f"   Seleccionadas por máximo poder discriminativo + mínima redundancia")
        print(f"   Este análisis sigue la línea Tsuchiya/Fulcher")
        
        print(f"\n📁 Resultados guardados en: {self.output_dir}")
        
        return {
            'df_features': df,
            'df_discrimination': df_disc,
            'df_categories': df_categories,
            'summary': summary,
        }


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def generate_plots(output_dir):
    """Genera visualizaciones de los resultados."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    output_dir = Path(output_dir)
    
    # Cargar datos
    df_disc = pd.read_csv(output_dir / "discrimination_ranking.csv")
    df_cat = pd.read_csv(output_dir / "category_analysis.csv")
    
    # Plot 1: Top 30 features
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top30 = df_disc.head(30)
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_cat)))
    cat_colors = {cat: colors[i] for i, cat in enumerate(df_cat['category'])}
    
    bar_colors = [cat_colors.get(c, 'gray') for c in top30['category']]
    
    y_pos = np.arange(len(top30))
    bars = ax.barh(y_pos, top30['cohens_d'], color=bar_colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:35] for f in top30['feature']], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect (d=0.8)')
    ax.axvline(x=-0.8, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel("Cohen's d (Wake vs N3)")
    ax.set_title("Top 30 Features - Discrimination Power")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_top30_features.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Category comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax = axes[0]
    bars = ax.bar(range(len(df_cat)), df_cat['mean_abs_d'], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(df_cat)))
    ax.set_xticklabels(df_cat['category'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title("Average Discrimination by Category")
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large effect')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
    ax.legend()
    
    ax = axes[1]
    bars = ax.bar(range(len(df_cat)), df_cat['n_large_effect'], color='darkgreen', alpha=0.8)
    ax.set_xticks(range(len(df_cat)))
    ax.set_xticklabels(df_cat['category'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Number of features with |d| ≥ 0.8")
    ax.set_title("Features with Large Effect by Category")
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_category_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 3: Category distribution in top features
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for n_top in [10, 20, 50, 100]:
        if n_top > len(df_disc):
            continue
        top_cats = df_disc.head(n_top)['category'].value_counts()
        for cat in df_cat['category']:
            if cat not in top_cats:
                top_cats[cat] = 0
    
    # Stacked bar para diferentes tops
    tops = [10, 20, 50]
    categories = df_cat['category'].tolist()
    
    data = {}
    for n_top in tops:
        top_cats = df_disc.head(n_top)['category'].value_counts()
        data[f'Top {n_top}'] = [top_cats.get(cat, 0) for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    for i, (label, values) in enumerate(data.items()):
        ax.bar(x + i*width, values, width, label=label, alpha=0.8)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Number of features")
    ax.set_title("Category Distribution in Top N Features")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_category_distribution.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"   ✅ Figuras guardadas en {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    """Busca el directorio de datos."""
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and list(p.rglob("*PSG*.edf")):
            return p
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"HCTSA-style Analysis ({__version__})")
    parser.add_argument('--sleep-edf-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()
    
    data_path = Path(args.sleep_edf_path) if args.sleep_edf_path else find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontraron datos Sleep-EDF")
        print("   Especifica --sleep-edf-path o coloca los datos en una ruta conocida")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    
    analyzer = HCTSAAnalyzer(str(data_path), args.output_dir)
    results = analyzer.run()
    
    if results and not args.no_plots:
        print(f"\n📊 Generando visualizaciones...")
        try:
            generate_plots(args.output_dir)
        except Exception as e:
            print(f"   ⚠️ Error generando plots: {e}")
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == "__main__":
    main()