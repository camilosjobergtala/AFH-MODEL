#!/usr/bin/env python3
"""
===============================================================================
NIVEL 3: TEST DE PRECEDENCIA TEMPORAL H* -> PAC
Autopsychic Fold Hypothesis - Version que extrae desde EDF
===============================================================================

EXTRAE DIRECTAMENTE DESDE LOS ARCHIVOS EDF:
1. Carga EDF + Hipnograma
2. Extrae epocas Wake y N3 para entrenar modelo H*
3. Entrena RandomForest (H* = prob de Wake)
4. Detecta transiciones Wake -> Sleep
5. Analiza precedencia temporal H* vs PAC

Author: Camilo Sjoberg Tala
Date: 2025-12-11
===============================================================================
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal, stats
from scipy.stats import binomtest, wilcoxon
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import sys
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# Suprimir warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
mne.set_log_level('ERROR')

# Logging - sin caracteres especiales para Windows
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nivel3_from_edf.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Tensorpac para PAC
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    logger.warning("tensorpac not available - PAC calculations will use fallback")


# =============================================================================
# CONFIGURACION
# =============================================================================

@dataclass
class Config:
    """Configuracion NIVEL 3"""
    
    # Rutas - AJUSTAR A TU SISTEMA
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path(r'G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\AFH vs ECLIPSE\FOLD\NIVEL 3 PAC\results_nivel3')
    
    # Semilla
    sacred_seed: int = 2025
    
    # Senal
    sampling_rate: float = 100.0
    epoch_duration: float = 30.0  # segundos por epoca
    
    # Ventana de analisis alrededor de transicion
    window_before_transition: float = 300.0  # 5 minutos antes
    window_after_transition: float = 300.0   # 5 minutos despues
    
    # Ventanas deslizantes para analisis temporal
    sliding_window_sec: float = 30.0
    sliding_step_sec: float = 10.0
    
    # PAC (Pliegue)
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 45.0)
    
    # Deteccion de onset
    onset_threshold_sd: float = 1.5
    
    # Canales objetivo
    target_channels: List[str] = field(default_factory=lambda: ['EEG Fpz-Cz', 'EEG Pz-Oz'])
    
    # Bandas espectrales
    spectral_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 12.0),
        'sigma': (12.0, 16.0),
        'beta': (16.0, 30.0),
    })
    
    # Criterios pre-registrados
    criterion_delta_t_threshold: float = 10.0
    criterion_consistency_threshold: float = 0.70
    criterion_p_value_threshold: float = 0.01
    
    # Filtrado
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    
    # Limite de sujetos (None = todos)
    max_subjects: Optional[int] = None


# =============================================================================
# EXTRACTOR DE FEATURES (77 features)
# =============================================================================

class FeatureExtractor:
    """Extrae las 77 features espectrales"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sfreq = config.sampling_rate
        self.feature_names = None
    
    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """PSD con Welch"""
        nperseg = min(len(data), int(self.sfreq * 4))
        freqs, psd = signal.welch(data, fs=self.sfreq, nperseg=nperseg)
        return freqs, psd
    
    def compute_band_power(self, freqs: np.ndarray, psd: np.ndarray, 
                           band: Tuple[float, float]) -> float:
        """Potencia en banda"""
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not np.any(idx):
            return 0.0
        return np.trapz(psd[idx], freqs[idx])
    
    def compute_spectral_entropy(self, psd: np.ndarray) -> float:
        """Entropia espectral"""
        psd_sum = np.sum(psd)
        if psd_sum < 1e-10:
            return 0.0
        psd_norm = psd / psd_sum
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    
    def compute_sef(self, freqs: np.ndarray, psd: np.ndarray, percentile: float = 0.95) -> float:
        """Spectral Edge Frequency"""
        cumsum = np.cumsum(psd)
        if cumsum[-1] > 0:
            sef_idx = np.searchsorted(cumsum, percentile * cumsum[-1])
            return freqs[min(sef_idx, len(freqs)-1)]
        return 0.0
    
    def compute_hjorth(self, data: np.ndarray) -> Dict[str, float]:
        """Parametros de Hjorth"""
        activity = np.var(data)
        d1 = np.diff(data)
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
    
    def compute_temporal(self, data: np.ndarray) -> Dict[str, float]:
        """Features temporales"""
        features = {}
        features['skewness'] = float(stats.skew(data))
        features['kurtosis'] = float(stats.kurtosis(data))
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(data - np.mean(data)))) > 0)
        features['zero_crossing_rate'] = zero_crossings / len(data)
        
        features['line_length'] = np.sum(np.abs(np.diff(data)))
        features['line_length_norm'] = features['line_length'] / (len(data) - 1)
        features['mean_abs_value'] = np.mean(np.abs(data))
        features['rms'] = np.sqrt(np.mean(data**2))
        features['variance'] = np.var(data)
        features['iqr'] = float(stats.iqr(data))
        
        return features
    
    def compute_sample_entropy(self, data: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        """Sample entropy (simplificado)"""
        n = len(data)
        if n > 500:
            data = data[::max(1, n//500)]
            n = len(data)
        
        if n < 50:
            return 0.0
        
        r = r_factor * np.std(data)
        if r < 1e-10:
            return 0.0
        
        def count_matches(template_len):
            count = 0
            templates = np.array([data[i:i+template_len] for i in range(n - template_len)])
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
        except:
            return 0.0
    
    def compute_cross_channel(self, ch1: np.ndarray, ch2: np.ndarray) -> Dict[str, float]:
        """Features cross-channel"""
        features = {}
        
        if np.std(ch1) > 1e-10 and np.std(ch2) > 1e-10:
            features['cross_correlation'] = float(np.corrcoef(ch1, ch2)[0, 1])
        else:
            features['cross_correlation'] = 0.0
        
        try:
            freqs, coh = signal.coherence(ch1, ch2, fs=self.sfreq, 
                                          nperseg=min(len(ch1), int(self.sfreq * 2)))
            
            for band_name, (f_low, f_high) in self.config.spectral_bands.items():
                idx = np.logical_and(freqs >= f_low, freqs <= f_high)
                features[f'coherence_{band_name}'] = float(np.mean(coh[idx])) if np.any(idx) else 0.0
        except:
            for band_name in self.config.spectral_bands.keys():
                features[f'coherence_{band_name}'] = 0.0
        
        try:
            env1 = np.abs(signal.hilbert(ch1))
            env2 = np.abs(signal.hilbert(ch2))
            features['envelope_correlation'] = float(np.corrcoef(env1, env2)[0, 1])
        except:
            features['envelope_correlation'] = 0.0
        
        return features
    
    def extract_features(self, data: np.ndarray, channel_names: List[str]) -> Dict[str, float]:
        """
        Extrae todas las features de una epoca.
        data: shape (n_channels, n_samples)
        """
        n_channels = data.shape[0]
        features = {}
        
        for ch_idx in range(n_channels):
            ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}"
            ch_clean = ch_name.replace(' ', '_').replace('-', '_')
            ch_data = data[ch_idx]
            
            # PSD y band powers
            freqs, psd = self.compute_psd(ch_data)
            total_power = max(np.trapz(psd, freqs), 1e-20)
            
            band_powers = {}
            for band_name, band_range in self.config.spectral_bands.items():
                bp = self.compute_band_power(freqs, psd, band_range)
                band_powers[band_name] = bp
                features[f'{ch_clean}_{band_name}_abs'] = bp
                features[f'{ch_clean}_{band_name}_rel'] = bp / total_power
            
            features[f'{ch_clean}_total_power'] = total_power
            features[f'{ch_clean}_spectral_entropy'] = self.compute_spectral_entropy(psd)
            features[f'{ch_clean}_peak_freq'] = freqs[np.argmax(psd)] if len(psd) > 0 else 0
            features[f'{ch_clean}_sef95'] = self.compute_sef(freqs, psd, 0.95)
            features[f'{ch_clean}_sef50'] = self.compute_sef(freqs, psd, 0.50)
            
            # Ratios
            eps = 1e-10
            delta = max(band_powers.get('delta', eps), eps)
            theta = max(band_powers.get('theta', eps), eps)
            alpha = max(band_powers.get('alpha', eps), eps)
            sigma = max(band_powers.get('sigma', eps), eps)
            beta = max(band_powers.get('beta', eps), eps)
            
            features[f'{ch_clean}_delta_beta_ratio'] = delta / beta
            features[f'{ch_clean}_theta_alpha_ratio'] = theta / alpha
            features[f'{ch_clean}_slowing_ratio'] = (delta + theta) / (alpha + beta)
            features[f'{ch_clean}_spindle_ratio'] = sigma / (delta + theta + alpha + beta)
            features[f'{ch_clean}_delta_theta_ratio'] = delta / theta
            features[f'{ch_clean}_alpha_theta_ratio'] = alpha / theta
            features[f'{ch_clean}_slow_fast_ratio'] = (delta + theta + alpha) / (sigma + beta)
            
            # Hjorth
            for h_name, h_val in self.compute_hjorth(ch_data).items():
                features[f'{ch_clean}_{h_name}'] = h_val
            
            # Temporal
            for t_name, t_val in self.compute_temporal(ch_data).items():
                features[f'{ch_clean}_{t_name}'] = t_val
            
            # Sample entropy
            features[f'{ch_clean}_sample_entropy'] = self.compute_sample_entropy(ch_data)
        
        # Cross-channel
        if n_channels >= 2:
            for cf_name, cf_val in self.compute_cross_channel(data[0], data[1]).items():
                features[f'cross_{cf_name}'] = cf_val
        
        if self.feature_names is None:
            self.feature_names = list(features.keys())
        
        return features


# =============================================================================
# CALCULADOR PAC
# =============================================================================

class PACCalculator:
    """Calcula PAC theta-gamma"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fs = config.sampling_rate
        
        if TENSORPAC_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pac = Pac(
                    idpac=(1, 0, 0),
                    f_pha=list(config.theta_band),
                    f_amp=list(config.gamma_band),
                    dcomplex='wavelet',
                    width=7,
                    verbose=False
                )
        else:
            self.pac = None
    
    def compute(self, data: np.ndarray) -> float:
        """Calcula PAC para una ventana"""
        if self.pac is None:
            return self._fallback_pac(data)
        
        if len(data) < int(2 * self.fs):
            return np.nan
        
        # Filtrar
        sos = signal.butter(4, [self.config.bandpass_low, self.config.bandpass_high], 
                           btype='bandpass', fs=self.fs, output='sos')
        data_filt = signal.sosfiltfilt(sos, data)
        
        # Normalizar
        data_norm = (data_filt - np.mean(data_filt)) / (np.std(data_filt) + 1e-10)
        
        if np.ptp(data_norm) > 8.0:
            return np.nan
        
        data_reshaped = data_norm[np.newaxis, :]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pac_value = self.pac.filterfit(self.fs, data_reshaped, data_reshaped)
            return float(pac_value[0, 0, 0])
        except:
            return np.nan
    
    def _fallback_pac(self, data: np.ndarray) -> float:
        """PAC simplificado sin tensorpac"""
        try:
            # Filtrar theta
            sos_theta = signal.butter(4, list(self.config.theta_band), 
                                     btype='bandpass', fs=self.fs, output='sos')
            theta = signal.sosfiltfilt(sos_theta, data)
            
            # Filtrar gamma
            sos_gamma = signal.butter(4, list(self.config.gamma_band), 
                                     btype='bandpass', fs=self.fs, output='sos')
            gamma = signal.sosfiltfilt(sos_gamma, data)
            
            # Fase de theta
            theta_phase = np.angle(signal.hilbert(theta))
            
            # Amplitud de gamma
            gamma_amp = np.abs(signal.hilbert(gamma))
            
            # MVL simplificado
            z = gamma_amp * np.exp(1j * theta_phase)
            mvl = np.abs(np.mean(z))
            
            return mvl
        except:
            return np.nan


# =============================================================================
# CARGADOR DE DATOS
# =============================================================================

class DataLoader:
    """Carga datos de Sleep-EDF"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def get_subject_files(self) -> List[Dict[str, Path]]:
        """Encuentra pares PSG + Hypnogram"""
        psg_files = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno_files = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        
        hypno_map = {}
        for hypno_path in hypno_files:
            codigo = hypno_path.stem.replace("-Hypnogram", "")
            if len(codigo) >= 6:
                hypno_map[codigo[:-1]] = hypno_path
        
        subject_files = []
        for psg_path in psg_files:
            codigo = psg_path.stem.replace("-PSG", "")
            if len(codigo) >= 7 and codigo.endswith('0'):
                base = codigo[:-1]
                if base in hypno_map:
                    subject_files.append({
                        'psg': psg_path,
                        'hypno': hypno_map[base],
                        'subject_id': codigo
                    })
        
        return subject_files
    
    def load_raw(self, psg_path: Path) -> Optional[mne.io.Raw]:
        """Carga archivo EEG"""
        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
            
            # Seleccionar canales
            available = []
            for target in self.config.target_channels:
                if target in raw.ch_names:
                    available.append(target)
                else:
                    for ch in raw.ch_names:
                        if target.lower() in ch.lower():
                            available.append(ch)
                            break
            
            if not available:
                return None
            
            raw.pick_channels(available)
            
            if raw.info['sfreq'] != self.config.sampling_rate:
                raw.resample(self.config.sampling_rate)
            
            raw.filter(self.config.bandpass_low, self.config.bandpass_high, verbose=False)
            
            return raw
        except Exception as e:
            logger.error(f"Error cargando {psg_path}: {e}")
            return None
    
    def load_annotations(self, hypno_path: Path) -> mne.Annotations:
        """Carga hipnograma"""
        return mne.read_annotations(hypno_path)
    
    def extract_epochs_by_stage(self, raw: mne.io.Raw, annotations: mne.Annotations, 
                                 stages: List[str]) -> List[np.ndarray]:
        """Extrae epocas de etapas especificas"""
        epochs = []
        fs = raw.info['sfreq']
        epoch_samples = int(self.config.epoch_duration * fs)
        
        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            if desc in stages:
                start_sample = int(onset * fs)
                n_epochs_in_segment = int(duration / self.config.epoch_duration)
                
                for i in range(n_epochs_in_segment):
                    epoch_start = start_sample + i * epoch_samples
                    epoch_end = epoch_start + epoch_samples
                    
                    if epoch_end <= raw.n_times:
                        data, _ = raw[:, epoch_start:epoch_end]
                        
                        # Control de calidad
                        if np.max(np.abs(data)) < 500:  # uV threshold
                            epochs.append(data)
        
        return epochs


# =============================================================================
# MODELO H*
# =============================================================================

class HStarModel:
    """Modelo H* basado en RandomForest"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scaler = None
        self.classifier = None
        self.feature_names = None
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Entrena el modelo"""
        logger.info(f"Entrenando modelo H* con {X.shape[0]} muestras, {X.shape[1]} features")
        
        self.feature_names = feature_names
        
        # Split para validacion
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.config.sacred_seed, stratify=y
        )
        
        # Scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # RandomForest
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.sacred_seed,
            n_jobs=-1
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Validacion
        y_pred = self.classifier.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred)
        
        logger.info(f"F1 Score (validacion): {f1:.4f}")
        logger.info(classification_report(y_val, y_pred, target_names=['Wake', 'N3']))
        
        self.is_trained = True
    
    def predict_hstar(self, features: Dict[str, float]) -> float:
        """Predice H* (probabilidad de Wake)"""
        if not self.is_trained:
            raise RuntimeError("Modelo no entrenado")
        
        X = np.array([[features.get(fn, 0.0) for fn in self.feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        # H* = probabilidad de Wake (clase 0)
        proba = self.classifier.predict_proba(X_scaled)
        return float(proba[0, 0])
    
    def save(self, path: Path):
        """Guarda modelo"""
        model_data = {
            'scaler': self.scaler,
            'classifier': self.classifier,
            'feature_names': self.feature_names
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Modelo guardado: {path}")
    
    def load(self, path: Path):
        """Carga modelo"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        logger.info(f"Modelo cargado: {path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class Pipeline:
    """Pipeline completo"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_extractor = FeatureExtractor(config)
        self.pac_calculator = PACCalculator(config)
        self.hstar_model = HStarModel(config)
        
        config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_hstar_from_edf(self) -> bool:
        """Entrena modelo H* extrayendo features de los EDF"""
        
        logger.info("=" * 70)
        logger.info("FASE 1: EXTRACCION DE FEATURES Y ENTRENAMIENTO DE H*")
        logger.info("=" * 70)
        
        subject_files = self.data_loader.get_subject_files()
        
        if self.config.max_subjects:
            subject_files = subject_files[:self.config.max_subjects]
        
        logger.info(f"Procesando {len(subject_files)} sujetos...")
        
        all_features = []
        all_labels = []
        
        for subj_info in tqdm(subject_files, desc="Extrayendo features"):
            try:
                raw = self.data_loader.load_raw(subj_info['psg'])
                if raw is None:
                    continue
                
                annotations = self.data_loader.load_annotations(subj_info['hypno'])
                channel_names = raw.ch_names
                
                # Extraer epocas Wake
                wake_epochs = self.data_loader.extract_epochs_by_stage(
                    raw, annotations, ['Sleep stage W']
                )
                
                # Extraer epocas N3
                n3_epochs = self.data_loader.extract_epochs_by_stage(
                    raw, annotations, ['Sleep stage 3', 'Sleep stage 4']
                )
                
                # Extraer features de Wake (label = 0)
                for epoch in wake_epochs[:500]:  # Limitar por sujeto
                    features = self.feature_extractor.extract_features(epoch, channel_names)
                    all_features.append([features[fn] for fn in self.feature_extractor.feature_names])
                    all_labels.append(0)
                
                # Extraer features de N3 (label = 1)
                for epoch in n3_epochs[:500]:
                    features = self.feature_extractor.extract_features(epoch, channel_names)
                    all_features.append([features[fn] for fn in self.feature_extractor.feature_names])
                    all_labels.append(1)
                
            except Exception as e:
                logger.error(f"Error procesando {subj_info['subject_id']}: {e}")
                continue
        
        if len(all_features) < 100:
            logger.error("Insuficientes datos extraidos")
            return False
        
        X = np.nan_to_num(np.array(all_features), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.array(all_labels)
        
        logger.info(f"Datos extraidos: {X.shape}")
        logger.info(f"Wake: {np.sum(y == 0)}, N3: {np.sum(y == 1)}")
        
        # Entrenar modelo
        self.hstar_model.train(X, y, self.feature_extractor.feature_names)
        
        # Guardar modelo
        model_path = self.config.output_dir / 'hstar_model.pkl'
        self.hstar_model.save(model_path)
        
        # Guardar features para reusar
        features_path = self.config.output_dir / 'training_features.npz'
        np.savez(features_path, X=X, y=y, feature_names=self.feature_extractor.feature_names)
        logger.info(f"Features guardadas: {features_path}")
        
        return True
    
    def find_transitions(self, annotations: mne.Annotations) -> List[Dict]:
        """Encuentra transiciones Wake -> Sleep"""
        transitions = []
        wake_states = ['Sleep stage W']
        sleep_states = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']
        
        for i in range(len(annotations.description) - 1):
            current = annotations.description[i]
            next_state = annotations.description[i + 1]
            
            if current in wake_states and next_state in sleep_states:
                transitions.append({
                    'index': i,
                    'time': annotations.onset[i + 1],
                    'from_state': current,
                    'to_state': next_state
                })
        
        return transitions
    
    def analyze_transition(self, raw: mne.io.Raw, transition: Dict) -> Optional[Dict]:
        """Analiza una transicion individual"""
        fs = raw.info['sfreq']
        transition_time = transition['time']
        channel_names = raw.ch_names
        
        # Extraer ventana
        start_time = max(0, transition_time - self.config.window_before_transition)
        end_time = min(raw.times[-1], transition_time + self.config.window_after_transition)
        
        start_sample = int(start_time * fs)
        end_sample = int(end_time * fs)
        
        if end_sample - start_sample < int(60 * fs):
            return None
        
        data, _ = raw[:, start_sample:end_sample]
        transition_in_window = transition_time - start_time
        
        # Ventanas deslizantes
        window_samples = int(self.config.sliding_window_sec * fs)
        step_samples = int(self.config.sliding_step_sec * fs)
        
        n_samples = data.shape[1]
        
        hstar_values = []
        pac_values = []
        times = []
        
        for i in range((n_samples - window_samples) // step_samples + 1):
            start = i * step_samples
            end = start + window_samples
            
            if end > n_samples:
                break
            
            window_data = data[:, start:end]
            center_time = (start + window_samples // 2) / fs
            relative_time = center_time - transition_in_window
            
            # H*
            features = self.feature_extractor.extract_features(window_data, channel_names)
            hstar = self.hstar_model.predict_hstar(features)
            
            # PAC
            pac = self.pac_calculator.compute(window_data[0])
            
            times.append(relative_time)
            hstar_values.append(hstar)
            pac_values.append(pac)
        
        if len(times) < 10:
            return None
        
        # Detectar onsets
        def detect_onset(values, times):
            valid_idx = ~np.isnan(values)
            if np.sum(valid_idx) < 5:
                return None
            
            values = np.array(values)[valid_idx]
            times = np.array(times)[valid_idx]
            
            # Baseline: primeros valores (Wake estable)
            baseline_mask = times < -60
            if np.sum(baseline_mask) < 3:
                baseline_mask = np.arange(len(values)) < len(values) // 3
            
            baseline_mean = np.mean(values[baseline_mask])
            baseline_std = np.std(values[baseline_mask])
            
            if baseline_std < 1e-10:
                return None
            
            threshold = baseline_mean - self.config.onset_threshold_sd * baseline_std
            
            # Buscar primer cruce
            for t, v in zip(times, values):
                if t >= -60 and v < threshold:
                    return t
            
            return None
        
        onset_hstar = detect_onset(hstar_values, times)
        onset_pac = detect_onset(pac_values, times)
        
        delta_t = None
        if onset_hstar is not None and onset_pac is not None:
            delta_t = onset_hstar - onset_pac
        
        return {
            'transition_time': transition_time,
            'from_state': transition['from_state'],
            'to_state': transition['to_state'],
            'onset_hstar': onset_hstar,
            'onset_pac': onset_pac,
            'delta_t': delta_t,
            'hstar_first': delta_t > 0 if delta_t is not None else None
        }
    
    def run_precedence_analysis(self) -> Tuple[pd.DataFrame, Dict]:
        """Ejecuta analisis de precedencia"""
        
        logger.info("=" * 70)
        logger.info("FASE 2: ANALISIS DE PRECEDENCIA H* vs PAC")
        logger.info("=" * 70)
        
        # Verificar modelo
        model_path = self.config.output_dir / 'hstar_model.pkl'
        if model_path.exists():
            self.hstar_model.load(model_path)
            # Cargar feature names
            features_path = self.config.output_dir / 'training_features.npz'
            if features_path.exists():
                data = np.load(features_path, allow_pickle=True)
                self.feature_extractor.feature_names = list(data['feature_names'])
        elif not self.hstar_model.is_trained:
            logger.info("Modelo no encontrado, entrenando...")
            if not self.train_hstar_from_edf():
                raise RuntimeError("No se pudo entrenar modelo H*")
        
        subject_files = self.data_loader.get_subject_files()
        
        if self.config.max_subjects:
            subject_files = subject_files[:self.config.max_subjects]
        
        logger.info(f"Analizando {len(subject_files)} sujetos...")
        
        all_results = []
        
        for subj_info in tqdm(subject_files, desc="Analizando transiciones"):
            try:
                raw = self.data_loader.load_raw(subj_info['psg'])
                if raw is None:
                    continue
                
                annotations = self.data_loader.load_annotations(subj_info['hypno'])
                transitions = self.find_transitions(annotations)
                
                for trans in transitions:
                    result = self.analyze_transition(raw, trans)
                    if result is not None:
                        result['subject_id'] = subj_info['subject_id']
                        all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error en {subj_info['subject_id']}: {e}")
                continue
        
        logger.info(f"Transiciones analizadas: {len(all_results)}")
        
        if len(all_results) == 0:
            raise RuntimeError("No se encontraron transiciones validas")
        
        df = pd.DataFrame(all_results)
        df.to_csv(self.config.output_dir / 'precedence_results.csv', index=False)
        
        # Estadisticas
        stats = self.compute_statistics(df)
        
        return df, stats
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """Computa estadisticas"""
        
        logger.info("=" * 70)
        logger.info("ESTADISTICAS")
        logger.info("=" * 70)
        
        valid = df['delta_t'].dropna()
        
        if len(valid) < 5:
            return {'status': 'insufficient_data'}
        
        n_total = len(valid)
        n_hstar_first = (valid > 0).sum()
        n_pac_first = (valid < 0).sum()
        
        pct_hstar_first = 100 * n_hstar_first / n_total
        
        mean_delta = valid.mean()
        std_delta = valid.std()
        median_delta = valid.median()
        
        # Tests
        n_pos = int((valid > 0).sum())
        n_neg = int((valid < 0).sum())
        
        if n_pos + n_neg > 0:
            p_binomial = binomtest(n_pos, n_pos + n_neg, 0.5, alternative='greater').pvalue
            try:
                _, p_wilcox = wilcoxon(valid, alternative='greater')
            except:
                p_wilcox = np.nan
        else:
            p_binomial = np.nan
            p_wilcox = np.nan
        
        logger.info(f"Transiciones validas: {n_total}")
        logger.info(f"H* primero: {n_hstar_first} ({pct_hstar_first:.1f}%)")
        logger.info(f"PAC primero: {n_pac_first} ({100*n_pac_first/n_total:.1f}%)")
        logger.info(f"Delta_t medio: {mean_delta:.1f} +/- {std_delta:.1f} s")
        logger.info(f"Delta_t mediana: {median_delta:.1f} s")
        logger.info(f"p-value (binomial): {p_binomial:.6f}")
        
        return {
            'n_total': n_total,
            'n_hstar_first': n_hstar_first,
            'n_pac_first': n_pac_first,
            'pct_hstar_first': pct_hstar_first,
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'median_delta': median_delta,
            'p_binomial': p_binomial,
            'p_wilcoxon': p_wilcox,
            'status': 'success'
        }
    
    def evaluate_criteria(self, stats: Dict) -> Dict:
        """Evalua criterios pre-registrados"""
        
        logger.info("=" * 70)
        logger.info("EVALUACION DE CRITERIOS")
        logger.info("=" * 70)
        
        if stats.get('status') != 'success':
            return {'verdict': 'INCONCLUSIVE', 'reason': 'Insufficient data'}
        
        # Criterio 1
        crit1_pass = stats['mean_delta'] >= self.config.criterion_delta_t_threshold
        logger.info(f"Criterio 1 (Delta_t >= {self.config.criterion_delta_t_threshold}s): "
                   f"{'[PASS]' if crit1_pass else '[FAIL]'} ({stats['mean_delta']:.1f}s)")
        
        # Criterio 2
        crit2_pass = stats['pct_hstar_first'] >= self.config.criterion_consistency_threshold * 100
        logger.info(f"Criterio 2 (>= {self.config.criterion_consistency_threshold*100:.0f}%): "
                   f"{'[PASS]' if crit2_pass else '[FAIL]'} ({stats['pct_hstar_first']:.1f}%)")
        
        # Criterio 3
        crit3_pass = stats['p_binomial'] < self.config.criterion_p_value_threshold
        logger.info(f"Criterio 3 (p < {self.config.criterion_p_value_threshold}): "
                   f"{'[PASS]' if crit3_pass else '[FAIL]'} (p = {stats['p_binomial']:.6f})")
        
        # Veredicto
        all_passed = crit1_pass and crit2_pass and crit3_pass
        falsified = stats['pct_hstar_first'] < 50
        
        if all_passed:
            verdict = 'VALIDATED'
            description = 'Arquitectura H* -> PAC soportada'
        elif falsified:
            verdict = 'FALSIFIED'
            description = 'PAC precede a H* en mayoria - Arquitectura falsificada'
        else:
            verdict = 'INCONCLUSIVE'
            description = 'Tendencia pero no alcanza criterios'
        
        return {
            'verdict': verdict,
            'description': description,
            'criterion_1': crit1_pass,
            'criterion_2': crit2_pass,
            'criterion_3': crit3_pass
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("""
================================================================================
     NIVEL 3: TEST DE PRECEDENCIA TEMPORAL H* -> PAC
     Autopsychic Fold Hypothesis
================================================================================

HIPOTESIS:
    Durante transiciones Wake -> Sleep, H* declina ANTES que PAC

CRITERIOS PRE-REGISTRADOS:
    1. Delta_t medio >= 10 segundos
    2. >= 70% transiciones con H* primero
    3. p < 0.01
    
FALSIFICACION:
    Si PAC precede a H* en >= 50% de transiciones
================================================================================
    """)
    
    config = Config()
    pipeline = Pipeline(config)
    
    try:
        # Fase 1: Entrenar H* (si no existe)
        model_path = config.output_dir / 'hstar_model.pkl'
        if not model_path.exists():
            pipeline.train_hstar_from_edf()
        
        # Fase 2: Analisis de precedencia
        df, stats = pipeline.run_precedence_analysis()
        
        # Fase 3: Evaluacion
        criteria = pipeline.evaluate_criteria(stats)
        
        # Veredicto
        print("\n" + "=" * 80)
        print(f">>> NIVEL 3: {criteria['verdict']} <<<")
        print("=" * 80)
        print(f"\n{criteria['description']}")
        print("=" * 80)
        
        # Guardar resultados
        final = {
            'timestamp': datetime.now().isoformat(),
            'statistics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in stats.items()},
            'criteria': criteria
        }
        
        with open(config.output_dir / 'final_results.json', 'w') as f:
            json.dump(final, f, indent=2, default=str)
        
        print(f"\n[OUTPUT] Resultados en: {config.output_dir}")
        
        return df, stats, criteria
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    main()