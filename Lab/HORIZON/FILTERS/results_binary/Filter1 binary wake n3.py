
"""
═══════════════════════════════════════════════════════════════════════════════
FILTER 1: BINARY WAKE vs N3 CLASSIFICATION
═══════════════════════════════════════════════════════════════════════════════

Objetivo: Establecer performance binaria Wake vs N3 con features espectrales
antes de decidir si HCTSA es necesario.

Prerregistro OSF: 10.17605/OSF.IO/GSJNH
- Filter 1: Wake vs N3 discrimination
- Threshold éxito: F1 >= 0.60 (conservador dado que es baseline)

Dataset: Sleep-EDF Database (PhysioNet)
Contraste: Wake (stage 0) vs N3 (stages 3+4)

Author: Camilo Alejandro Sjöberg Tala, M.D.
Date: December 2025
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score, precision_score, recall_score
)
import mne

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('filter1_binary.log')
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Pipeline configuration for binary Wake vs N3"""
    
    project_name: str = "Filter1_Binary_Wake_N3"
    researcher: str = "Camilo Alejandro Sjöberg Tala"
    
    # Paths - ADJUST TO YOUR SYSTEM
    data_path: str = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette"
    output_path: str = r"G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\HORIZON\results_binary"
    checkpoint_path: str = r"G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\HORIZON\results_enhanced\checkpoints"
    
    # Seeds and splits
    sacred_seed: int = 2025
    development_ratio: float = 0.65
    
    # EEG processing
    target_sfreq: float = 100.0
    epoch_duration: float = 30.0
    target_channels: List[str] = field(default_factory=lambda: ['EEG Fpz-Cz', 'EEG Pz-Oz'])
    
    # Filtering
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    
    # Spectral bands
    spectral_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'sigma': (12.0, 16.0),
        'beta': (16.0, 30.0),
    })
    
    # Quality control
    amplitude_threshold_uv: float = 500.0
    flatline_threshold_uv: float = 1.0
    
    # Evaluation
    n_folds: int = 5
    f1_threshold: float = 0.60  # Pre-registered success criterion
    
    # Stage mapping - BINARY
    stage_mapping: Dict[str, int] = field(default_factory=lambda: {
        'Sleep stage W': 0,   # Wake
        'Sleep stage 1': -1,  # Exclude
        'Sleep stage 2': -1,  # Exclude
        'Sleep stage 3': 1,   # N3
        'Sleep stage 4': 1,   # N3 (N4 = deep sleep)
        'Sleep stage R': -1,  # Exclude
        'Sleep stage ?': -1,
        'Movement time': -1,
    })
    
    stage_names: List[str] = field(default_factory=lambda: ['Wake', 'N3'])


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTOR (same as before)
# ═══════════════════════════════════════════════════════════════════════════

class EnhancedFeatureExtractor:
    """Extract comprehensive features from EEG epochs"""
    
    def __init__(self, config):
        self.config = config
        self.sfreq = config.target_sfreq
        
    def compute_psd(self, epoch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nperseg = min(len(epoch), int(self.sfreq * 4))
        freqs, psd = signal.welch(epoch, fs=self.sfreq, nperseg=nperseg)
        return freqs, psd
    
    def compute_band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                           band: Tuple[float, float]) -> float:
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not np.any(idx):
            return 0.0
        return np.trapz(psd[idx], freqs[idx])
    
    def compute_spectral_entropy(self, psd: np.ndarray) -> float:
        psd_sum = np.sum(psd)
        if psd_sum < 1e-10:
            return 0.0
        psd_norm = psd / psd_sum
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    def compute_sef(self, freqs: np.ndarray, psd: np.ndarray, percentile: float = 0.95) -> float:
        cumsum = np.cumsum(psd)
        if cumsum[-1] > 0:
            sef_idx = np.searchsorted(cumsum, percentile * cumsum[-1])
            return freqs[min(sef_idx, len(freqs)-1)]
        return 0.0
    
    def compute_band_ratios(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        eps = 1e-10
        delta = max(band_powers.get('delta', eps), eps)
        theta = max(band_powers.get('theta', eps), eps)
        alpha = max(band_powers.get('alpha', eps), eps)
        sigma = max(band_powers.get('sigma', eps), eps)
        beta = max(band_powers.get('beta', eps), eps)
        
        return {
            'delta_beta_ratio': delta / beta,
            'theta_alpha_ratio': theta / alpha,
            'slowing_ratio': (delta + theta) / (alpha + beta),
            'spindle_ratio': sigma / (delta + theta + alpha + beta),
            'delta_theta_ratio': delta / theta,
            'alpha_theta_ratio': alpha / theta,
            'slow_fast_ratio': (delta + theta + alpha) / (sigma + beta),
        }
    
    def compute_hjorth_parameters(self, epoch: np.ndarray) -> Dict[str, float]:
        activity = np.var(epoch)
        d1 = np.diff(epoch)
        d2 = np.diff(d1)
        
        mobility = np.sqrt(np.var(d1) / activity) if activity > 1e-10 else 0.0
        
        if mobility > 1e-10 and np.var(d1) > 1e-10:
            mobility_d1 = np.sqrt(np.var(d2) / np.var(d1))
            complexity = mobility_d1 / mobility
        else:
            complexity = 0.0
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    def compute_temporal_features(self, epoch: np.ndarray) -> Dict[str, float]:
        features = {}
        features['skewness'] = float(stats.skew(epoch))
        features['kurtosis'] = float(stats.kurtosis(epoch))
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(epoch - np.mean(epoch)))) > 0)
        features['zero_crossing_rate'] = zero_crossings / len(epoch)
        
        features['line_length'] = np.sum(np.abs(np.diff(epoch)))
        features['line_length_norm'] = features['line_length'] / (len(epoch) - 1)
        features['mean_abs_value'] = np.mean(np.abs(epoch))
        features['rms'] = np.sqrt(np.mean(epoch**2))
        features['variance'] = np.var(epoch)
        features['iqr'] = float(stats.iqr(epoch))
        
        return features
    
    def compute_sample_entropy(self, epoch: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        n = len(epoch)
        if n > 1000:
            epoch = epoch[::3]
            n = len(epoch)
        
        if n < 50:
            return 0.0
        
        r = r_factor * np.std(epoch)
        if r < 1e-10:
            return 0.0
        
        def count_matches(template_len):
            count = 0
            templates = np.array([epoch[i:i+template_len] for i in range(n - template_len)])
            for i in range(len(templates)):
                dists = np.max(np.abs(templates - templates[i]), axis=1)
                count += np.sum(dists < r) - 1
            return count
        
        try:
            A = count_matches(m + 1)
            B = count_matches(m)
            if B == 0:
                return 0.0
            return -np.log(A / B) if A > 0 else 0.0
        except Exception:
            return 0.0
    
    def compute_cross_channel_features(self, epoch_ch1: np.ndarray, 
                                         epoch_ch2: np.ndarray) -> Dict[str, float]:
        features = {}
        
        if np.std(epoch_ch1) > 1e-10 and np.std(epoch_ch2) > 1e-10:
            features['cross_correlation'] = float(np.corrcoef(epoch_ch1, epoch_ch2)[0, 1])
        else:
            features['cross_correlation'] = 0.0
        
        try:
            freqs, coh = signal.coherence(epoch_ch1, epoch_ch2, fs=self.sfreq, 
                                          nperseg=min(len(epoch_ch1), int(self.sfreq * 2)))
            
            for band_name, (f_low, f_high) in self.config.spectral_bands.items():
                idx = np.logical_and(freqs >= f_low, freqs <= f_high)
                features[f'coherence_{band_name}'] = float(np.mean(coh[idx])) if np.any(idx) else 0.0
        except Exception:
            for band_name in self.config.spectral_bands.keys():
                features[f'coherence_{band_name}'] = 0.0
        
        try:
            env1 = np.abs(signal.hilbert(epoch_ch1))
            env2 = np.abs(signal.hilbert(epoch_ch2))
            features['envelope_correlation'] = float(np.corrcoef(env1, env2)[0, 1])
        except Exception:
            features['envelope_correlation'] = 0.0
        
        return features
    
    def extract_features(self, epochs: np.ndarray, 
                         channel_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        n_epochs, n_channels, n_samples = epochs.shape
        
        logger.info(f"Extracting features from {n_epochs} epochs...")
        
        all_features = []
        feature_names = None
        
        for i, epoch in enumerate(epochs):
            if i % 5000 == 0 and i > 0:
                logger.info(f"  Processing epoch {i}/{n_epochs}")
            
            epoch_features = {}
            
            for ch_idx in range(n_channels):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
                ch_clean = ch_name.replace(' ', '_').replace('-', '_')
                ch_data = epoch[ch_idx]
                
                freqs, psd = self.compute_psd(ch_data)
                total_power = max(np.trapz(psd, freqs), 1e-20)
                
                band_powers = {}
                for band_name, band_range in self.config.spectral_bands.items():
                    bp = self.compute_band_power(freqs, psd, band_range)
                    band_powers[band_name] = bp
                    epoch_features[f'{ch_clean}_{band_name}_abs'] = bp
                    epoch_features[f'{ch_clean}_{band_name}_rel'] = bp / total_power
                
                epoch_features[f'{ch_clean}_total_power'] = total_power
                epoch_features[f'{ch_clean}_spectral_entropy'] = self.compute_spectral_entropy(psd)
                epoch_features[f'{ch_clean}_peak_freq'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
                epoch_features[f'{ch_clean}_sef95'] = self.compute_sef(freqs, psd, 0.95)
                epoch_features[f'{ch_clean}_sef50'] = self.compute_sef(freqs, psd, 0.50)
                
                for ratio_name, ratio_val in self.compute_band_ratios(band_powers).items():
                    epoch_features[f'{ch_clean}_{ratio_name}'] = ratio_val
                
                for h_name, h_val in self.compute_hjorth_parameters(ch_data).items():
                    epoch_features[f'{ch_clean}_{h_name}'] = h_val
                
                for t_name, t_val in self.compute_temporal_features(ch_data).items():
                    epoch_features[f'{ch_clean}_{t_name}'] = t_val
                
                epoch_features[f'{ch_clean}_sample_entropy'] = self.compute_sample_entropy(ch_data)
            
            if n_channels >= 2:
                for cf_name, cf_val in self.compute_cross_channel_features(epoch[0], epoch[1]).items():
                    epoch_features[f'cross_{cf_name}'] = cf_val
            
            if feature_names is None:
                feature_names = list(epoch_features.keys())
            
            all_features.append([epoch_features[fn] for fn in feature_names])
        
        features_array = np.nan_to_num(np.array(all_features), nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"  Extracted {len(feature_names)} features per epoch")
        
        return features_array, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, namespace: str = "default"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.namespace = namespace
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_processed_subjects(self) -> List[str]:
        processed = []
        for f in self.checkpoint_dir.glob(f"{self.namespace}_subject_*.npz"):
            subject_id = f.stem.replace(f"{self.namespace}_subject_", "")
            processed.append(subject_id)
        return processed
    
    def load_subject_data(self, subject_id: str) -> Optional[Tuple]:
        path = self.checkpoint_dir / f"{self.namespace}_subject_{subject_id}.npz"
        if path.exists():
            try:
                data = np.load(path, allow_pickle=True)
                return data['epochs'], data['labels'], list(data['channels'])
            except Exception:
                return None
        return None


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADER - BINARY VERSION
# ═══════════════════════════════════════════════════════════════════════════

class BinaryDataLoader:
    """Load only Wake and N3 epochs"""
    
    def __init__(self, config: PipelineConfig, checkpoint: CheckpointManager = None):
        self.config = config
        self.data_path = Path(config.data_path)
        self.checkpoint = checkpoint
    
    def get_subject_list(self) -> List[Dict]:
        subjects = []
        psg_files = list(self.data_path.glob("*-PSG.edf"))
        hypno_files = list(self.data_path.glob("*-Hypnogram.edf"))
        
        hypno_map = {}
        for hypno in hypno_files:
            code = hypno.stem.replace("-Hypnogram", "")
            if len(code) >= 2:
                hypno_map[code[:-1]] = hypno
        
        for psg in sorted(psg_files):
            code = psg.stem.replace("-PSG", "")
            if len(code) >= 2:
                base = code[:-1]
                if base in hypno_map:
                    subjects.append({
                        'subject_id': code,
                        'psg_path': str(psg),
                        'hypno_path': str(hypno_map[base]),
                    })
        
        logger.info(f"Found {len(subjects)} complete subjects")
        return subjects
    
    def load_from_checkpoints(self, subject_list: List[Dict]) -> Tuple:
        """Load from existing checkpoints, filtering for Wake/N3 only"""
        
        all_epochs, all_labels, all_subject_ids = [], [], []
        channel_names = None
        
        processed = self.checkpoint.get_processed_subjects() if self.checkpoint else []
        
        for subject in subject_list:
            sid = subject['subject_id']
            if sid not in processed:
                continue
            
            cached = self.checkpoint.load_subject_data(sid)
            if cached is None or len(cached[0]) == 0:
                continue
            
            epochs, labels, channels = cached
            
            # Filter for Wake (0) and N3 (3) only
            # Original labels: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
            mask = (labels == 0) | (labels == 3)
            
            if not np.any(mask):
                continue
            
            filtered_epochs = epochs[mask]
            filtered_labels = labels[mask]
            
            # Convert to binary: Wake=0, N3=1
            binary_labels = (filtered_labels == 3).astype(int)
            
            all_epochs.append(filtered_epochs)
            all_labels.append(binary_labels)
            all_subject_ids.extend([sid] * len(binary_labels))
            
            if channel_names is None:
                channel_names = channels
        
        if not all_epochs:
            raise ValueError("No Wake/N3 epochs found in checkpoints!")
        
        return (
            np.concatenate(all_epochs, axis=0),
            np.concatenate(all_labels),
            np.array(all_subject_ids),
            channel_names
        )


# ═══════════════════════════════════════════════════════════════════════════
# BINARY CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

class BinaryClassifier:
    """Evaluate binary Wake vs N3 classification"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def compute_fisher_ratio(self, X: np.ndarray, y: np.ndarray) -> float:
        classes = np.unique(y)
        if len(classes) < 2:
            return 0.0
        
        grand_mean = np.mean(X)
        between_var, within_var = 0, 0
        
        for c in classes:
            class_data = X[y == c]
            if len(class_data) < 2:
                continue
            between_var += len(class_data) * (np.mean(class_data) - grand_mean) ** 2
            within_var += len(class_data) * np.var(class_data)
        
        return between_var / within_var if within_var > 1e-10 else 0.0
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
                 feature_names: List[str], top_k_list: List[int] = [10, 20, 30, 50, 77]) -> Dict:
        
        logger.info("\n" + "=" * 70)
        logger.info("BINARY CLASSIFICATION: WAKE vs N3")
        logger.info("=" * 70)
        
        # Class distribution
        n_wake = np.sum(y == 0)
        n_n3 = np.sum(y == 1)
        logger.info(f"Class distribution: Wake={n_wake} ({n_wake/len(y)*100:.1f}%), N3={n_n3} ({n_n3/len(y)*100:.1f}%)")
        
        # Compute Fisher ratios
        fisher_scores = []
        for f_idx in range(X.shape[1]):
            score = self.compute_fisher_ratio(X[:, f_idx], y)
            fisher_scores.append((f_idx, score, feature_names[f_idx]))
        
        fisher_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("\nTop 15 features by Fisher ratio (binary):")
        for idx, score, name in fisher_scores[:15]:
            logger.info(f"  {name}: {score:.4f}")
        
        # Cross-validation setup
        unique_subjects = np.unique(subject_ids)
        n_folds = min(self.config.n_folds, len(unique_subjects) // 2)
        n_folds = max(2, n_folds)
        
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, 
                                  random_state=self.config.sacred_seed)
        
        all_results = {}
        
        for top_k in top_k_list:
            top_k = min(top_k, len(fisher_scores))
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating with top {top_k} features")
            logger.info(f"{'='*50}")
            
            top_indices = [f[0] for f in fisher_scores[:top_k]]
            X_top = X[:, top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            fold_results = []
            all_y_true, all_y_pred, all_y_prob = [], [], []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_top, y, subject_ids)):
                X_train, X_val = X_top[train_idx], X_top[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)
                
                # Logistic Regression
                clf_lr = LogisticRegression(
                    penalty='l2', C=1.0, solver='lbfgs',
                    max_iter=1000, random_state=self.config.sacred_seed
                )
                clf_lr.fit(X_train_s, y_train)
                y_pred_lr = clf_lr.predict(X_val_s)
                y_prob_lr = clf_lr.predict_proba(X_val_s)[:, 1]
                
                # Random Forest
                clf_rf = RandomForestClassifier(
                    n_estimators=100, max_depth=10,
                    random_state=self.config.sacred_seed, n_jobs=-1
                )
                clf_rf.fit(X_train_s, y_train)
                y_pred_rf = clf_rf.predict(X_val_s)
                y_prob_rf = clf_rf.predict_proba(X_val_s)[:, 1]
                
                fold_results.append({
                    'fold': fold,
                    'lr_f1': f1_score(y_val, y_pred_lr),
                    'lr_precision': precision_score(y_val, y_pred_lr),
                    'lr_recall': recall_score(y_val, y_pred_lr),
                    'lr_auc': roc_auc_score(y_val, y_prob_lr),
                    'rf_f1': f1_score(y_val, y_pred_rf),
                    'rf_precision': precision_score(y_val, y_pred_rf),
                    'rf_recall': recall_score(y_val, y_pred_rf),
                    'rf_auc': roc_auc_score(y_val, y_prob_rf),
                })
                
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred_rf)
                all_y_prob.extend(y_prob_rf)
            
            results_df = pd.DataFrame(fold_results)
            
            logger.info("\nPer-fold results:")
            logger.info(results_df.to_string())
            
            logger.info("\nMean ± Std:")
            logger.info(f"  Logistic Regression:")
            logger.info(f"    F1:        {results_df['lr_f1'].mean():.3f} ± {results_df['lr_f1'].std():.3f}")
            logger.info(f"    Precision: {results_df['lr_precision'].mean():.3f} ± {results_df['lr_precision'].std():.3f}")
            logger.info(f"    Recall:    {results_df['lr_recall'].mean():.3f} ± {results_df['lr_recall'].std():.3f}")
            logger.info(f"    AUC-ROC:   {results_df['lr_auc'].mean():.3f} ± {results_df['lr_auc'].std():.3f}")
            
            logger.info(f"  Random Forest:")
            logger.info(f"    F1:        {results_df['rf_f1'].mean():.3f} ± {results_df['rf_f1'].std():.3f}")
            logger.info(f"    Precision: {results_df['rf_precision'].mean():.3f} ± {results_df['rf_precision'].std():.3f}")
            logger.info(f"    Recall:    {results_df['rf_recall'].mean():.3f} ± {results_df['rf_recall'].std():.3f}")
            logger.info(f"    AUC-ROC:   {results_df['rf_auc'].mean():.3f} ± {results_df['rf_auc'].std():.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(all_y_true, all_y_pred)
            logger.info(f"\nConfusion Matrix (RF, all folds):")
            logger.info(f"              Pred Wake  Pred N3")
            logger.info(f"  True Wake:    {cm[0,0]:7d}  {cm[0,1]:7d}")
            logger.info(f"  True N3:      {cm[1,0]:7d}  {cm[1,1]:7d}")
            
            # Per-class accuracy
            wake_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
            n3_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            logger.info(f"\nPer-class accuracy:")
            logger.info(f"  Wake: {wake_acc:.3f} ({cm[0,0]}/{cm[0,0]+cm[0,1]})")
            logger.info(f"  N3:   {n3_acc:.3f} ({cm[1,1]}/{cm[1,0]+cm[1,1]})")
            
            all_results[f'top_{top_k}'] = {
                'top_k': top_k,
                'selected_features': top_names[:10],  # First 10
                'lr_f1_mean': float(results_df['lr_f1'].mean()),
                'lr_f1_std': float(results_df['lr_f1'].std()),
                'lr_auc_mean': float(results_df['lr_auc'].mean()),
                'rf_f1_mean': float(results_df['rf_f1'].mean()),
                'rf_f1_std': float(results_df['rf_f1'].std()),
                'rf_auc_mean': float(results_df['rf_auc'].mean()),
                'rf_precision_mean': float(results_df['rf_precision'].mean()),
                'rf_recall_mean': float(results_df['rf_recall'].mean()),
                'confusion_matrix': cm.tolist(),
                'wake_accuracy': float(wake_acc),
                'n3_accuracy': float(n3_acc),
            }
        
        return all_results, fisher_scores


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline():
    """Run binary Wake vs N3 classification"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║     FILTER 1: BINARY WAKE vs N3 CLASSIFICATION                              ║
║     OSF Preregistration: 10.17605/OSF.IO/GSJNH                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    config = PipelineConfig()
    
    # Create directories
    Path(config.output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize
    checkpoint = CheckpointManager(config.checkpoint_path, "enhanced")
    data_loader = BinaryDataLoader(config, checkpoint)
    feature_extractor = EnhancedFeatureExtractor(config)
    classifier = BinaryClassifier(config)
    
    # Get subjects
    subject_list = data_loader.get_subject_list()
    
    # Split (same as 5-class)
    np.random.seed(config.sacred_seed)
    n_dev = int(len(subject_list) * config.development_ratio)
    indices = np.random.permutation(len(subject_list))
    dev_indices = indices[:n_dev]
    dev_subjects = [subject_list[i] for i in dev_indices]
    
    logger.info(f"Development subjects: {len(dev_subjects)}")
    
    # Load data from checkpoints (filtered for Wake/N3)
    logger.info("\n" + "=" * 70)
    logger.info("LOADING DATA (Wake and N3 only)")
    logger.info("=" * 70)
    
    epochs, labels, subject_ids, channel_names = data_loader.load_from_checkpoints(dev_subjects)
    
    logger.info(f"Loaded: {epochs.shape}")
    logger.info(f"Channels: {channel_names}")
    logger.info(f"Wake epochs: {np.sum(labels == 0)}")
    logger.info(f"N3 epochs: {np.sum(labels == 1)}")
    
    # Extract features
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTING FEATURES")
    logger.info("=" * 70)
    
    features, feature_names = feature_extractor.extract_features(epochs, channel_names)
    
    # Evaluate
    results, fisher_scores = classifier.evaluate(
        features, labels, subject_ids, feature_names,
        top_k_list=[10, 20, 30, 50, len(feature_names)]
    )
    
    # Save results
    results_path = Path(config.output_path) / "binary_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_path}")
    
    # Save Fisher scores
    fisher_df = pd.DataFrame([
        {'feature': name, 'fisher_ratio': score}
        for idx, score, name in fisher_scores
    ])
    fisher_path = Path(config.output_path) / "binary_fisher_scores.csv"
    fisher_df.to_csv(fisher_path, index=False)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY - BINARY WAKE vs N3")
    logger.info("=" * 70)
    
    best_rf_f1 = max(r['rf_f1_mean'] for r in results.values())
    best_rf_auc = max(r['rf_auc_mean'] for r in results.values())
    best_config = max(results.items(), key=lambda x: x[1]['rf_f1_mean'])
    
    logger.info(f"\nBest performance (Random Forest):")
    logger.info(f"  Configuration: {best_config[0]}")
    logger.info(f"  F1:      {best_config[1]['rf_f1_mean']:.3f} ± {best_config[1]['rf_f1_std']:.3f}")
    logger.info(f"  AUC-ROC: {best_config[1]['rf_auc_mean']:.3f}")
    logger.info(f"  Wake accuracy: {best_config[1]['wake_accuracy']:.3f}")
    logger.info(f"  N3 accuracy:   {best_config[1]['n3_accuracy']:.3f}")
    
    # Decision based on pre-registered threshold
    logger.info("\n" + "-" * 40)
    logger.info("PRE-REGISTERED CRITERION EVALUATION")
    logger.info("-" * 40)
    logger.info(f"Threshold: F1 >= {config.f1_threshold}")
    logger.info(f"Achieved:  F1 = {best_rf_f1:.3f}")
    
    if best_rf_f1 >= config.f1_threshold:
        logger.info(f"\n✓ CRITERION MET: Spectral features achieve F1 >= {config.f1_threshold}")
        logger.info("  → Filter 1 PASSED with spectral baseline")
        logger.info("  → Question: Is HCTSA necessary for this contrast?")
    else:
        logger.info(f"\n✗ CRITERION NOT MET: F1 < {config.f1_threshold}")
        logger.info("  → Consider HCTSA or additional features")
    
    logger.info("\n" + "=" * 70)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_pipeline()