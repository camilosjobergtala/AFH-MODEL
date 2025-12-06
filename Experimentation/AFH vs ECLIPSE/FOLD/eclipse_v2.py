#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
AFH TRANSITION ANALYSIS + ECLIPSE v2.0
Test de Precedencia H* → PAC (P-∇-3) con Validación Rigurosa
═══════════════════════════════════════════════════════════════════════════════

PREDICCIÓN AFH (P-∇-3):
    Durante transiciones Vigilia→Sueño, H* desciende ANTES que PAC.
    
    Especificación:
    - Δt(H* → PAC) = 10-30 segundos
    - Precedencia en ≥70% de transiciones
    
ECLIPSE v2.0 PROTOCOL:
    1. Split criptográfico (70% desarrollo, 30% validación)
    2. Criterios pre-registrados ANTES de análisis
    3. Desarrollo en training set (explorar, ajustar)
    4. Validación SINGLE-SHOT en holdout
    5. Resultado binario terminal

CORRECCIÓN v2.0:
    - H* ya NO incluye coherencia delta-gamma (evita redundancia con PAC)
    - H* mide condiciones organizacionales INDEPENDIENTES del Pliegue

CRITERIO DE FALSACIÓN (F-∇-3):
    Si Δt < 5s o precedencia < 50% en VALIDACIÓN → Arquitectura falsificada

Author: Camilo Sjöberg Tala, M.D.
Date: 2025-12-05
Version: 2.0.0 (ECLIPSE)
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal, stats
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
import warnings
import logging
import json
import time
from datetime import datetime
import hashlib

# PAC calculation
try:
    from tensorpac import Pac
    TENSORPAC_AVAILABLE = True
except ImportError:
    TENSORPAC_AVAILABLE = False
    print("⚠️ tensorpac not available - using manual PAC calculation")

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE v2.0 CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EclipseConfig:
    """
    ECLIPSE v2.0 Protocol Configuration
    
    Cryptographic split + pre-registered criteria + single-shot validation
    """
    
    # SACRED SEED - NO CAMBIAR JAMÁS
    sacred_seed: int = 42
    
    # Split ratios
    dev_ratio: float = 0.70      # 70% desarrollo
    val_ratio: float = 0.30      # 30% validación
    
    # PRE-REGISTERED CRITERIA (P-∇-3)
    # Estos valores están LOCKED antes de ver datos
    precedence_threshold: float = 0.70    # ≥70% de transiciones
    expected_lag_min: float = 10.0        # Lag mínimo esperado (s)
    expected_lag_max: float = 30.0        # Lag máximo esperado (s)
    min_lag_for_precedence: float = 5.0   # Mínimo para contar como precedencia
    significance_alpha: float = 0.05      # Alpha para t-test
    
    # FALSIFICATION THRESHOLDS (F-∇-3)
    falsification_precedence: float = 0.50  # Si <50% → falsificado
    falsification_lag: float = 5.0          # Si lag <5s → falsificado
    
    # Minimum requirements
    min_transitions_dev: int = 15
    min_transitions_val: int = 10
    min_subjects_dev: int = 10
    min_subjects_val: int = 5


@dataclass
class TransitionConfig:
    """Configuración para análisis de transiciones"""
    
    # Rutas
    data_dir: Path = None
    output_dir: Path = None
    
    # Parámetros de señal
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    notch_freq: float = 50.0
    
    # Canales objetivo
    target_channels: List[str] = field(default_factory=lambda: [
        'EEG Fpz-Cz',
        'EEG Pz-Oz'
    ])
    
    # Bandas para PAC (∇ - Pliegue)
    delta_band: Tuple[float, float] = (2.0, 4.0)
    gamma_band: Tuple[float, float] = (30.0, 50.0)
    
    # Bandas para H* (INDEPENDIENTES de PAC)
    theta_band: Tuple[float, float] = (4.0, 8.0)
    alpha_band: Tuple[float, float] = (8.0, 12.0)
    beta_band: Tuple[float, float] = (13.0, 30.0)
    
    # Parámetros de transición
    transition_window_before: float = 180.0  # 3 minutos antes
    transition_window_after: float = 180.0   # 3 minutos después
    sliding_window_size: float = 30.0        # ventana de 30s
    sliding_window_step: float = 10.0        # paso de 10s
    
    # Estados Sleep-EDF
    wake_state: str = 'Sleep stage W'
    n1_state: str = 'Sleep stage 1'
    n2_state: str = 'Sleep stage 2'
    n3_state: str = 'Sleep stage 3'
    
    # Transiciones a detectar
    include_wake_n2: bool = True
    include_wake_n3: bool = True
    
    # ECLIPSE
    eclipse: EclipseConfig = field(default_factory=EclipseConfig)
    
    # Límite de sujetos (None = todos)
    n_subjects: Optional[int] = None


class TransitionEvent(NamedTuple):
    """Representa una transición de estado"""
    onset: float
    from_state: str
    to_state: str
    subject_id: str
    duration_before: float
    duration_after: float


# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE v2.0 SPLIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class EclipseSplitManager:
    """
    Gestiona split criptográfico de sujetos
    
    INMUTABLE: Una vez creado, el split no puede cambiar
    """
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.split_hash = None
        self.dev_subjects = []
        self.val_subjects = []
        self._locked = False
    
    def create_split(self, subject_ids: List[str]) -> Dict:
        """
        Crea split criptográfico de sujetos
        
        El split es DETERMINÍSTICO dado el seed y la lista de sujetos
        """
        if self._locked:
            raise RuntimeError("Split ya está locked - no se puede modificar")
        
        # Ordenar para reproducibilidad
        sorted_ids = sorted(subject_ids)
        
        # Hash de entrada para verificación
        input_hash = hashlib.sha256(
            f"{self.config.sacred_seed}:{','.join(sorted_ids)}".encode()
        ).hexdigest()[:16]
        
        # Split determinístico
        np.random.seed(self.config.sacred_seed)
        indices = np.random.permutation(len(sorted_ids))
        
        n_dev = int(len(sorted_ids) * self.config.dev_ratio)
        
        dev_indices = indices[:n_dev]
        val_indices = indices[n_dev:]
        
        self.dev_subjects = [sorted_ids[i] for i in dev_indices]
        self.val_subjects = [sorted_ids[i] for i in val_indices]
        
        # Hash del split
        self.split_hash = hashlib.sha256(
            f"{input_hash}:{','.join(self.dev_subjects)}:{','.join(self.val_subjects)}".encode()
        ).hexdigest()[:16]
        
        self._locked = True
        
        return {
            'input_hash': input_hash,
            'split_hash': self.split_hash,
            'n_total': len(sorted_ids),
            'n_dev': len(self.dev_subjects),
            'n_val': len(self.val_subjects),
            'dev_subjects': self.dev_subjects,
            'val_subjects': self.val_subjects,
            'seed': self.config.sacred_seed
        }
    
    def verify_split(self, expected_hash: str) -> bool:
        """Verifica integridad del split"""
        return self.split_hash == expected_hash
    
    def get_split_certificate(self) -> Dict:
        """Genera certificado del split para documentación"""
        return {
            'timestamp': datetime.now().isoformat(),
            'sacred_seed': self.config.sacred_seed,
            'split_hash': self.split_hash,
            'n_dev': len(self.dev_subjects),
            'n_val': len(self.val_subjects),
            'dev_ratio': self.config.dev_ratio,
            'protocol': 'ECLIPSE v2.0'
        }


# ═══════════════════════════════════════════════════════════════════════════
# H* INDEX CALCULATOR (CORREGIDO - SIN REDUNDANCIA CON PAC)
# ═══════════════════════════════════════════════════════════════════════════

class HStarCalculator:
    """
    Calcula H* Index: Condición organizacional que habilita el Pliegue
    
    VERSIÓN 2.0 - CORREGIDA:
    - NO incluye coherencia delta-gamma (eso es casi PAC)
    - Mide condiciones INDEPENDIENTES del Pliegue
    
    Componentes:
    1. Coherencia theta-alpha (coordinación local, NO delta-gamma)
    2. Coherencia alpha-beta (integración cortical)
    3. Complejidad LZ normalizada (organización informacional)
    4. Estabilidad temporal (autocorrelación)
    5. Entropía espectral (distribución de potencia)
    """
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.fs = config.sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, 
                        band: Tuple[float, float]) -> np.ndarray:
        """Filtro pasabanda Butterworth"""
        nyq = self.fs / 2
        low = band[0] / nyq
        high = min(band[1] / nyq, 0.99)
        
        if low >= high or low <= 0:
            return data
            
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, data)
    
    def compute_envelope_correlation(self, data: np.ndarray,
                                     band1: Tuple[float, float],
                                     band2: Tuple[float, float]) -> float:
        """
        Calcula correlación de envolventes entre dos bandas
        """
        try:
            signal1 = self.bandpass_filter(data, band1)
            signal2 = self.bandpass_filter(data, band2)
            
            env1 = np.abs(hilbert(signal1))
            env2 = np.abs(hilbert(signal2))
            
            if np.std(env1) < 1e-10 or np.std(env2) < 1e-10:
                return 0.0
            
            corr = np.corrcoef(env1, env2)[0, 1]
            return float(np.abs(corr))
            
        except Exception:
            return 0.0
    
    def compute_lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """Complejidad de Lempel-Ziv normalizada"""
        try:
            binary = (data > np.median(data)).astype(int)
            s = ''.join(map(str, binary))
            n = len(s)
            
            if n == 0:
                return 0.0
            
            i, c, l, k, k_max = 0, 1, 1, 1, 1
            
            while True:
                if s[i + k - 1] != s[l + k - 1]:
                    if k > k_max:
                        k_max = k
                    i += 1
                    if i == l:
                        c += 1
                        l += k_max
                        if l + 1 > n:
                            break
                        i, k, k_max = 0, 1, 1
                    else:
                        k = 1
                else:
                    k += 1
                    if l + k > n:
                        c += 1
                        break
            
            b = n / np.log2(n) if n > 1 else 1
            return float(np.clip(c / b, 0, 1))
            
        except Exception:
            return 0.5
    
    def compute_autocorr_stability(self, data: np.ndarray, 
                                   max_lag: int = 50) -> float:
        """Estabilidad via decaimiento de autocorrelación"""
        try:
            n = len(data)
            if n < max_lag * 2:
                max_lag = n // 4
            if max_lag < 2:
                return 0.5
            
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
            autocorr = np.correlate(data_norm, data_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            
            threshold = 1 / np.e
            crossings = np.where(autocorr[:max_lag] < threshold)[0]
            tau = crossings[0] if len(crossings) > 0 else max_lag
            
            return float(np.clip(tau / max_lag, 0, 1))
            
        except Exception:
            return 0.5
    
    def compute_spectral_entropy(self, data: np.ndarray) -> float:
        """Entropía espectral normalizada"""
        try:
            freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(len(data), 256))
            psd_norm = psd / (np.sum(psd) + 1e-10)
            psd_norm = psd_norm[psd_norm > 0]
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            max_entropy = np.log2(len(psd_norm)) if len(psd_norm) > 0 else 1
            return float(np.clip(entropy / max_entropy, 0, 1))
        except Exception:
            return 0.5
    
    def compute_relative_band_power(self, data: np.ndarray,
                                    band: Tuple[float, float]) -> float:
        """Potencia relativa en una banda"""
        try:
            freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(len(data), 256))
            total_power = np.sum(psd)
            band_mask = (freqs >= band[0]) & (freqs <= band[1])
            band_power = np.sum(psd[band_mask])
            return float(band_power / (total_power + 1e-10))
        except Exception:
            return 0.0
    
    def compute_h_star_index(self, data: np.ndarray) -> Dict:
        """
        Calcula H* Index compuesto (VERSIÓN 2.0 - SIN REDUNDANCIA PAC)
        
        H* = w1*Coh_θα + w2*Coh_αβ + w3*LZ + w4*Stability + w5*Entropy
        
        NOTA: NO incluye coherencia delta-gamma porque eso es casi PAC
        """
        data = self._preprocess(data)
        
        if len(data) < self.fs * 2:
            return self._empty_result()
        
        # Componente 1: Coherencia theta-alpha (coordinación local)
        # Esto mide integración talamocortical sin tocar bandas PAC
        coh_theta_alpha = self.compute_envelope_correlation(
            data, 
            self.config.theta_band,  # 4-8 Hz
            self.config.alpha_band   # 8-12 Hz
        )
        
        # Componente 2: Coherencia alpha-beta (integración cortical)
        coh_alpha_beta = self.compute_envelope_correlation(
            data,
            self.config.alpha_band,  # 8-12 Hz
            self.config.beta_band    # 13-30 Hz
        )
        
        # Componente 3: Complejidad LZ (organización informacional)
        complexity = self.compute_lempel_ziv_complexity(data)
        
        # Componente 4: Estabilidad temporal
        stability = self.compute_autocorr_stability(data)
        
        # Componente 5: Entropía espectral
        spectral_entropy = self.compute_spectral_entropy(data)
        
        # Componente 6: Ratio alpha/theta (marcador de arousal)
        alpha_power = self.compute_relative_band_power(data, self.config.alpha_band)
        theta_power = self.compute_relative_band_power(data, self.config.theta_band)
        alpha_theta_ratio = alpha_power / (theta_power + 1e-10)
        alpha_theta_norm = np.clip(alpha_theta_ratio / 3.0, 0, 1)  # Normalizar ~[0,1]
        
        # PESOS CORREGIDOS (sin delta-gamma)
        w_coh_ta = 0.20       # Coherencia theta-alpha
        w_coh_ab = 0.15       # Coherencia alpha-beta
        w_complexity = 0.20   # Complejidad LZ
        w_stability = 0.20    # Estabilidad
        w_entropy = 0.15      # Entropía espectral
        w_arousal = 0.10      # Ratio alpha/theta
        
        h_star = (
            w_coh_ta * coh_theta_alpha +
            w_coh_ab * coh_alpha_beta +
            w_complexity * complexity +
            w_stability * stability +
            w_entropy * spectral_entropy +
            w_arousal * alpha_theta_norm
        )
        
        return {
            'h_star': float(h_star),
            'coherence_theta_alpha': float(coh_theta_alpha),
            'coherence_alpha_beta': float(coh_alpha_beta),
            'complexity_lz': float(complexity),
            'stability': float(stability),
            'spectral_entropy': float(spectral_entropy),
            'alpha_theta_ratio': float(alpha_theta_norm),
            'valid': True
        }
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa señal"""
        sos = signal.butter(4, [self.config.lowcut, self.config.highcut],
                           btype='bandpass', fs=self.fs, output='sos')
        filtered = signal.sosfiltfilt(sos, data)
        
        b_notch, a_notch = signal.iirnotch(self.config.notch_freq, Q=30, fs=self.fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        return (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
    
    def _empty_result(self) -> Dict:
        return {
            'h_star': np.nan, 'coherence_theta_alpha': np.nan,
            'coherence_alpha_beta': np.nan, 'complexity_lz': np.nan,
            'stability': np.nan, 'spectral_entropy': np.nan,
            'alpha_theta_ratio': np.nan, 'valid': False
        }


# ═══════════════════════════════════════════════════════════════════════════
# PAC CALCULATOR (SIN CAMBIOS - OPERACIONALIZA ∇)
# ═══════════════════════════════════════════════════════════════════════════

class PACCalculator:
    """
    Calcula Phase-Amplitude Coupling (PAC) delta→gamma
    
    Operacionaliza el Pliegue Autopsíquico (∇):
    - PAC alto = convergencia temporal activa
    - PAC bajo = sin convergencia
    """
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.fs = config.sampling_rate
        
        if TENSORPAC_AVAILABLE:
            self.pac_obj = Pac(
                idpac=(1, 0, 0),
                f_pha=list(config.delta_band),
                f_amp=list(config.gamma_band),
                dcomplex='wavelet',
                width=7
            )
        else:
            self.pac_obj = None
    
    def bandpass_filter(self, data: np.ndarray, 
                        band: Tuple[float, float]) -> np.ndarray:
        nyq = self.fs / 2
        low = band[0] / nyq
        high = min(band[1] / nyq, 0.99)
        if low >= high or low <= 0:
            return data
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, data)
    
    def compute_pac_manual(self, data: np.ndarray) -> float:
        """PAC via Mean Vector Length"""
        try:
            delta = self.bandpass_filter(data, self.config.delta_band)
            gamma = self.bandpass_filter(data, self.config.gamma_band)
            
            delta_phase = np.angle(hilbert(delta))
            gamma_amp = np.abs(hilbert(gamma))
            gamma_amp = (gamma_amp - np.mean(gamma_amp)) / (np.std(gamma_amp) + 1e-10)
            
            complex_signal = gamma_amp * np.exp(1j * delta_phase)
            return float(np.abs(np.mean(complex_signal)))
        except Exception:
            return np.nan
    
    def compute_pac(self, data: np.ndarray) -> Dict:
        """Calcula PAC delta→gamma"""
        data = self._preprocess(data)
        
        if len(data) < self.fs * 2:
            return {'pac': np.nan, 'valid': False}
        
        try:
            if self.pac_obj is not None:
                data_reshaped = data[np.newaxis, :]
                pac_value = self.pac_obj.filterfit(self.fs, data_reshaped, data_reshaped)
                pac_value = float(pac_value[0, 0, 0])
            else:
                pac_value = self.compute_pac_manual(data)
            
            return {'pac': pac_value, 'valid': not np.isnan(pac_value)}
        except Exception:
            return {'pac': np.nan, 'valid': False}
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        sos = signal.butter(4, [self.config.lowcut, self.config.highcut],
                           btype='bandpass', fs=self.fs, output='sos')
        filtered = signal.sosfiltfilt(sos, data)
        b_notch, a_notch = signal.iirnotch(self.config.notch_freq, Q=30, fs=self.fs)
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        return (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# TRANSITION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

class TransitionDetector:
    """Detecta transiciones Wake→N2 y Wake→N3"""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
    
    def find_transitions(self, annotations: mne.Annotations,
                        subject_id: str) -> List[TransitionEvent]:
        """Encuentra transiciones Wake→Sleep"""
        transitions = []
        
        events = sorted([
            {'onset': ann['onset'], 'duration': ann['duration'], 
             'description': ann['description']}
            for ann in annotations
        ], key=lambda x: x['onset'])
        
        target_states = []
        if self.config.include_wake_n2:
            target_states.append(self.config.n2_state)
        if self.config.include_wake_n3:
            target_states.append(self.config.n3_state)
        
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Wake → N2/N3 directa
            if (current['description'] == self.config.wake_state and
                next_event['description'] in target_states):
                
                transitions.append(TransitionEvent(
                    onset=next_event['onset'],
                    from_state='Wake',
                    to_state=next_event['description'].split()[-1],
                    subject_id=subject_id,
                    duration_before=current['duration'],
                    duration_after=next_event['duration']
                ))
            
            # Wake → N1 → N2/N3 (N1 corto)
            elif (current['description'] == self.config.wake_state and
                  next_event['description'] == self.config.n1_state and
                  i + 2 < len(events)):
                
                next_next = events[i + 2]
                if (next_next['description'] in target_states and
                    next_event['duration'] < 120):
                    
                    transitions.append(TransitionEvent(
                        onset=next_next['onset'],
                        from_state='Wake',
                        to_state=next_next['description'].split()[-1],
                        subject_id=subject_id,
                        duration_before=current['duration'] + next_event['duration'],
                        duration_after=next_next['duration']
                    ))
        
        # Filtrar con contexto suficiente
        return [t for t in transitions if t.duration_before >= 60 and t.duration_after >= 60]


# ═══════════════════════════════════════════════════════════════════════════
# PRECEDENCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class PrecedenceAnalyzer:
    """Analiza precedencia temporal H* → PAC"""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.h_star_calc = HStarCalculator(config)
        self.pac_calc = PACCalculator(config)
    
    def analyze_single_transition(self, raw: mne.io.Raw,
                                  transition: TransitionEvent,
                                  channel: str) -> Dict:
        """Analiza una transición individual"""
        fs = raw.info['sfreq']
        
        window_start = max(0, transition.onset - self.config.transition_window_before)
        window_end = min(raw.times[-1], transition.onset + self.config.transition_window_after)
        
        start_sample = int(window_start * fs)
        end_sample = int(window_end * fs)
        
        try:
            ch_idx = raw.ch_names.index(channel)
            data = raw.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)[0]
        except Exception as e:
            return self._empty_result(transition)
        
        if len(data) < fs * 60:
            return self._empty_result(transition)
        
        # Ventanas deslizantes
        window_samples = int(self.config.sliding_window_size * fs)
        step_samples = int(self.config.sliding_window_step * fs)
        
        timepoints, h_star_series, pac_series = [], [], []
        
        n_windows = (len(data) - window_samples) // step_samples + 1
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            if end > len(data):
                break
            
            epoch = data[start:end]
            t_relative = window_start + (start / fs) - transition.onset
            timepoints.append(t_relative)
            
            h_star_result = self.h_star_calc.compute_h_star_index(epoch)
            h_star_series.append(h_star_result['h_star'])
            
            pac_result = self.pac_calc.compute_pac(epoch)
            pac_series.append(pac_result['pac'])
        
        timepoints = np.array(timepoints)
        h_star_series = np.array(h_star_series)
        pac_series = np.array(pac_series)
        
        # Limpiar NaNs
        valid_mask = ~(np.isnan(h_star_series) | np.isnan(pac_series))
        if np.sum(valid_mask) < 5:
            return self._empty_result(transition)
        
        timepoints = timepoints[valid_mask]
        h_star_series = h_star_series[valid_mask]
        pac_series = pac_series[valid_mask]
        
        # Normalizar
        h_star_norm = (h_star_series - np.nanmean(h_star_series)) / (np.nanstd(h_star_series) + 1e-10)
        pac_norm = (pac_series - np.nanmean(pac_series)) / (np.nanstd(pac_series) + 1e-10)
        
        # Suavizar
        h_star_smooth = uniform_filter1d(h_star_norm, size=3)
        pac_smooth = uniform_filter1d(pac_norm, size=3)
        
        # Detectar puntos de caída
        h_star_drop_idx = self._find_drop_point(h_star_smooth, timepoints)
        pac_drop_idx = self._find_drop_point(pac_smooth, timepoints)
        
        # Calcular lag
        if h_star_drop_idx is not None and pac_drop_idx is not None:
            h_star_drop_time = timepoints[h_star_drop_idx]
            pac_drop_time = timepoints[pac_drop_idx]
            lag_seconds = pac_drop_time - h_star_drop_time
            h_star_precedes = lag_seconds > self.config.eclipse.min_lag_for_precedence
        else:
            h_star_drop_time, pac_drop_time, lag_seconds = np.nan, np.nan, np.nan
            h_star_precedes = False
        
        # Cross-correlation
        xcorr_lag = self._compute_xcorr_lag(h_star_smooth, pac_smooth)
        
        return {
            'subject_id': transition.subject_id,
            'channel': channel,
            'transition_onset': transition.onset,
            'from_state': transition.from_state,
            'to_state': transition.to_state,
            'timepoints': timepoints.tolist(),
            'h_star_series': h_star_series.tolist(),
            'pac_series': pac_series.tolist(),
            'h_star_drop_time': float(h_star_drop_time) if not np.isnan(h_star_drop_time) else None,
            'pac_drop_time': float(pac_drop_time) if not np.isnan(pac_drop_time) else None,
            'lag_seconds': float(lag_seconds) if not np.isnan(lag_seconds) else None,
            'h_star_precedes': h_star_precedes,
            'xcorr_lag_seconds': float(xcorr_lag) if xcorr_lag is not None else None,
            'h_star_pre': float(np.nanmean(h_star_series[timepoints < 0])),
            'h_star_post': float(np.nanmean(h_star_series[timepoints > 0])),
            'pac_pre': float(np.nanmean(pac_series[timepoints < 0])),
            'pac_post': float(np.nanmean(pac_series[timepoints > 0])),
            'valid': True
        }
    
    def _find_drop_point(self, series: np.ndarray, timepoints: np.ndarray) -> Optional[int]:
        """Encuentra punto de caída cerca de t=0"""
        search_mask = np.abs(timepoints) < 120
        if np.sum(search_mask) < 3:
            return None
        
        search_indices = np.where(search_mask)[0]
        search_series = series[search_mask]
        
        for i in range(len(search_series) - 1):
            if search_series[i] > 0 and search_series[i + 1] <= 0:
                return search_indices[i]
        
        return search_indices[np.argmin(search_series)]
    
    def _compute_xcorr_lag(self, series1: np.ndarray, series2: np.ndarray) -> Optional[float]:
        """Calcula lag via cross-correlation"""
        try:
            if len(series1) < 5:
                return None
            xcorr = np.correlate(series1, series2, mode='full')
            lags = np.arange(-len(series1) + 1, len(series1))
            max_idx = np.argmax(xcorr)
            return lags[max_idx] * self.config.sliding_window_step
        except Exception:
            return None
    
    def _empty_result(self, transition: TransitionEvent) -> Dict:
        return {'subject_id': transition.subject_id, 
                'transition_onset': transition.onset, 'valid': False}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSOR CON ECLIPSE v2.0
# ═══════════════════════════════════════════════════════════════════════════

class AFHTransitionProcessor:
    """Procesador principal con protocolo ECLIPSE v2.0"""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.detector = TransitionDetector(config)
        self.analyzer = PrecedenceAnalyzer(config)
        self.split_manager = EclipseSplitManager(config.eclipse)
    
    def discover_subjects(self) -> List[str]:
        """Descubre todos los sujetos disponibles"""
        psg_files = list(self.config.data_dir.glob('*PSG.edf'))
        return [f.stem.replace('-PSG', '') for f in psg_files]
    
    def load_subject(self, psg_file: Path, hypno_file: Path) -> Optional[Dict]:
        """Carga datos de un sujeto"""
        try:
            raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
            annotations = mne.read_annotations(str(hypno_file))
            raw.set_annotations(annotations)
            
            available_channels = []
            for target in self.config.target_channels:
                if target in raw.ch_names:
                    available_channels.append(target)
            
            if not available_channels:
                for ch in raw.ch_names:
                    if 'eeg' in ch.lower():
                        available_channels.append(ch)
                        if len(available_channels) >= 2:
                            break
            
            if not available_channels:
                return None
            
            if raw.info['sfreq'] != self.config.sampling_rate:
                raw.resample(self.config.sampling_rate)
            
            return {'raw': raw, 'channels': available_channels, 
                    'annotations': raw.annotations}
        except Exception as e:
            logger.error(f"Error cargando: {e}")
            return None
    
    def process_subject(self, subject_id: str) -> List[Dict]:
        """Procesa un sujeto"""
        psg_file = self.config.data_dir / f"{subject_id}-PSG.edf"
        hypno_id = subject_id[:-1] + 'C'
        hypno_file = self.config.data_dir / f"{hypno_id}-Hypnogram.edf"
        
        if not psg_file.exists() or not hypno_file.exists():
            return []
        
        loaded = self.load_subject(psg_file, hypno_file)
        if loaded is None:
            return []
        
        transitions = self.detector.find_transitions(loaded['annotations'], subject_id)
        if len(transitions) < 1:
            return []
        
        results = []
        for trans_idx, transition in enumerate(transitions):
            for channel in loaded['channels']:
                result = self.analyzer.analyze_single_transition(
                    loaded['raw'], transition, channel
                )
                result['transition_idx'] = trans_idx
                results.append(result)
        
        return [r for r in results if r.get('valid', False)]
    
    def run_eclipse_analysis(self) -> Dict:
        """
        Ejecuta análisis completo con protocolo ECLIPSE v2.0
        
        1. Descubrir sujetos
        2. Split criptográfico
        3. Fase DESARROLLO (explorar, ajustar)
        4. Fase VALIDACIÓN (single-shot, terminal)
        5. Veredicto final
        """
        
        print("\n" + "=" * 80)
        print("🔬 AFH TRANSITION ANALYSIS + ECLIPSE v2.0")
        print("   Test de Precedencia H* → PAC (P-∇-3)")
        print("=" * 80)
        
        start_time = time.time()
        
        # ═══════════════════════════════════════════════════════════════
        # FASE 0: SETUP
        # ═══════════════════════════════════════════════════════════════
        
        print("\n📋 FASE 0: SETUP ECLIPSE")
        print("-" * 40)
        
        # Descubrir sujetos
        all_subjects = self.discover_subjects()
        if self.config.n_subjects:
            all_subjects = all_subjects[:self.config.n_subjects]
        
        print(f"  Sujetos disponibles: {len(all_subjects)}")
        
        # Split criptográfico
        split_info = self.split_manager.create_split(all_subjects)
        
        print(f"  Split hash: {split_info['split_hash']}")
        print(f"  Desarrollo: {split_info['n_dev']} sujetos")
        print(f"  Validación: {split_info['n_val']} sujetos")
        
        # PRE-REGISTRAR CRITERIOS
        print("\n📝 CRITERIOS PRE-REGISTRADOS (LOCKED):")
        print(f"  • Precedencia threshold: ≥{self.config.eclipse.precedence_threshold*100:.0f}%")
        print(f"  • Lag esperado: [{self.config.eclipse.expected_lag_min}, {self.config.eclipse.expected_lag_max}] s")
        print(f"  • Alpha: {self.config.eclipse.significance_alpha}")
        print(f"  • Falsificación si: precedencia <{self.config.eclipse.falsification_precedence*100:.0f}% OR lag <{self.config.eclipse.falsification_lag}s")
        
        # ═══════════════════════════════════════════════════════════════
        # FASE 1: DESARROLLO
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "=" * 80)
        print("📊 FASE 1: DESARROLLO")
        print("=" * 80)
        
        dev_results = []
        for idx, subject_id in enumerate(self.split_manager.dev_subjects, 1):
            print(f"[DEV {idx}/{len(self.split_manager.dev_subjects)}] {subject_id}...", end=" ")
            results = self.process_subject(subject_id)
            if results:
                dev_results.extend(results)
                print(f"✓ {len(results)} análisis")
            else:
                print("✗")
        
        print(f"\n  Total análisis desarrollo: {len(dev_results)}")
        
        if len(dev_results) < self.config.eclipse.min_transitions_dev:
            print(f"\n❌ Insuficientes transiciones en desarrollo")
            return {'error': 'insufficient_dev_transitions', 'phase': 'development'}
        
        # Análisis desarrollo
        dev_analysis = self._analyze_phase(dev_results, "DESARROLLO")
        
        # ═══════════════════════════════════════════════════════════════
        # FASE 2: VALIDACIÓN (SINGLE-SHOT)
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "=" * 80)
        print("🎯 FASE 2: VALIDACIÓN (SINGLE-SHOT)")
        print("=" * 80)
        print("⚠️  ADVERTENCIA: Esta fase es TERMINAL")
        print("   Los resultados determinan el veredicto final")
        print("-" * 40)
        
        val_results = []
        for idx, subject_id in enumerate(self.split_manager.val_subjects, 1):
            print(f"[VAL {idx}/{len(self.split_manager.val_subjects)}] {subject_id}...", end=" ")
            results = self.process_subject(subject_id)
            if results:
                val_results.extend(results)
                print(f"✓ {len(results)} análisis")
            else:
                print("✗")
        
        print(f"\n  Total análisis validación: {len(val_results)}")
        
        if len(val_results) < self.config.eclipse.min_transitions_val:
            print(f"\n⚠️  Pocas transiciones en validación - resultados con alta incertidumbre")
        
        # Análisis validación
        val_analysis = self._analyze_phase(val_results, "VALIDACIÓN")
        
        # ═══════════════════════════════════════════════════════════════
        # FASE 3: VEREDICTO TERMINAL
        # ═══════════════════════════════════════════════════════════════
        
        print("\n" + "=" * 80)
        print("⚖️  VEREDICTO TERMINAL ECLIPSE")
        print("=" * 80)
        
        verdict = self._compute_verdict(val_analysis)
        
        elapsed_time = time.time() - start_time
        
        # Compilar resultados finales
        final_results = {
            'eclipse_version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': elapsed_time / 60,
            
            'split': {
                'hash': split_info['split_hash'],
                'n_dev': split_info['n_dev'],
                'n_val': split_info['n_val'],
                'seed': self.config.eclipse.sacred_seed
            },
            
            'pre_registered_criteria': {
                'precedence_threshold': self.config.eclipse.precedence_threshold,
                'expected_lag_range': [self.config.eclipse.expected_lag_min, 
                                       self.config.eclipse.expected_lag_max],
                'significance_alpha': self.config.eclipse.significance_alpha,
                'falsification_precedence': self.config.eclipse.falsification_precedence,
                'falsification_lag': self.config.eclipse.falsification_lag
            },
            
            'development': dev_analysis,
            'validation': val_analysis,
            'verdict': verdict,
            
            'raw_results': {
                'development': dev_results,
                'validation': val_results
            }
        }
        
        return final_results
    
    def _analyze_phase(self, results: List[Dict], phase_name: str) -> Dict:
        """Analiza resultados de una fase"""
        
        valid_results = [r for r in results if r.get('valid') and r.get('lag_seconds') is not None]
        
        if len(valid_results) == 0:
            return {'error': 'no_valid_results', 'n_transitions': 0}
        
        lags = [r['lag_seconds'] for r in valid_results]
        precedes = [r['h_star_precedes'] for r in valid_results]
        
        n_total = len(valid_results)
        n_precedes = sum(precedes)
        precedence_rate = n_precedes / n_total
        
        mean_lag = np.mean(lags)
        std_lag = np.std(lags)
        median_lag = np.median(lags)
        
        lags_when_precedes = [l for l, p in zip(lags, precedes) if p]
        mean_lag_when_precedes = np.mean(lags_when_precedes) if lags_when_precedes else np.nan
        
        # T-test
        if len(lags) > 2:
            t_stat, p_value = stats.ttest_1samp(lags, 0, alternative='greater')
        else:
            t_stat, p_value = np.nan, 1.0
        
        # Cambios pre/post
        h_star_pre = np.mean([r['h_star_pre'] for r in valid_results])
        h_star_post = np.mean([r['h_star_post'] for r in valid_results])
        pac_pre = np.mean([r['pac_pre'] for r in valid_results])
        pac_post = np.mean([r['pac_post'] for r in valid_results])
        
        print(f"\n  📈 Resultados {phase_name}:")
        print(f"     Transiciones: {n_total}")
        print(f"     H* precede PAC: {n_precedes}/{n_total} ({precedence_rate*100:.1f}%)")
        print(f"     Lag medio: {mean_lag:.1f} ± {std_lag:.1f} s")
        print(f"     Lag (cuando precede): {mean_lag_when_precedes:.1f} s")
        print(f"     T-test: t={t_stat:.2f}, p={p_value:.4f}")
        
        return {
            'n_transitions': n_total,
            'n_subjects': len(set(r['subject_id'] for r in valid_results)),
            'precedence_rate': float(precedence_rate),
            'n_precedes': n_precedes,
            'lag_mean': float(mean_lag),
            'lag_std': float(std_lag),
            'lag_median': float(median_lag),
            'lag_when_precedes': float(mean_lag_when_precedes) if not np.isnan(mean_lag_when_precedes) else None,
            't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
            'p_value': float(p_value),
            'h_star_pre': float(h_star_pre),
            'h_star_post': float(h_star_post),
            'pac_pre': float(pac_pre),
            'pac_post': float(pac_post)
        }
    
    def _compute_verdict(self, val_analysis: Dict) -> Dict:
        """Computa veredicto final basado en criterios pre-registrados"""
        
        if 'error' in val_analysis:
            return {
                'verdict': 'INCONCLUSIVE',
                'reason': 'Insufficient validation data',
                'criteria_evaluated': False
            }
        
        eclipse = self.config.eclipse
        
        # Criterio 1: Tasa de precedencia
        precedence_rate = val_analysis['precedence_rate']
        crit1_pass = precedence_rate >= eclipse.precedence_threshold
        crit1_fail_hard = precedence_rate < eclipse.falsification_precedence
        
        # Criterio 2: Lag en rango
        lag = val_analysis.get('lag_when_precedes')
        if lag is not None:
            crit2_pass = eclipse.expected_lag_min <= lag <= eclipse.expected_lag_max
            crit2_fail_hard = lag < eclipse.falsification_lag
        else:
            crit2_pass = False
            crit2_fail_hard = True
        
        # Criterio 3: Significancia
        p_value = val_analysis['p_value']
        crit3_pass = p_value < eclipse.significance_alpha
        
        # Contar criterios
        n_passed = sum([crit1_pass, crit2_pass, crit3_pass])
        n_failed_hard = sum([crit1_fail_hard, crit2_fail_hard])
        
        # Determinar veredicto
        if n_failed_hard >= 1:
            verdict = 'FALSIFIED'
            reason = 'At least one hard falsification criterion met'
        elif n_passed >= 2:
            verdict = 'SUPPORTED'
            reason = f'{n_passed}/3 criteria passed'
        elif n_passed == 1:
            verdict = 'MIXED'
            reason = 'Only 1/3 criteria passed'
        else:
            verdict = 'FALSIFIED'
            reason = 'No criteria passed'
        
        print(f"\n  Criterio 1 (precedencia ≥{eclipse.precedence_threshold*100:.0f}%): ", end="")
        print(f"{'✅ PASS' if crit1_pass else '❌ FAIL'} ({precedence_rate*100:.1f}%)")
        
        print(f"  Criterio 2 (lag {eclipse.expected_lag_min}-{eclipse.expected_lag_max}s): ", end="")
        print(f"{'✅ PASS' if crit2_pass else '❌ FAIL'} ({lag:.1f}s)" if lag else "❌ FAIL (N/A)")
        
        print(f"  Criterio 3 (p < {eclipse.significance_alpha}): ", end="")
        print(f"{'✅ PASS' if crit3_pass else '❌ FAIL'} (p={p_value:.4f})")
        
        print(f"\n  {'='*50}")
        
        if verdict == 'SUPPORTED':
            print(f"  ✅ VEREDICTO: HIPÓTESIS P-∇-3 SOPORTADA")
            print(f"     Arquitectura H* → ∇ VALIDADA")
        elif verdict == 'FALSIFIED':
            print(f"  ❌ VEREDICTO: HIPÓTESIS P-∇-3 FALSIFICADA")
            print(f"     Criterio F-∇-3 cumplido")
            print(f"     Arquitectura H* → ∇ NO SOPORTADA")
        else:
            print(f"  ⚠️  VEREDICTO: EVIDENCIA MIXTA")
            print(f"     Resultados no concluyentes")
        
        print(f"  {'='*50}")
        
        return {
            'verdict': verdict,
            'reason': reason,
            'criteria': {
                'precedence': {'passed': crit1_pass, 'value': precedence_rate, 
                              'threshold': eclipse.precedence_threshold},
                'lag_range': {'passed': crit2_pass, 'value': lag,
                             'range': [eclipse.expected_lag_min, eclipse.expected_lag_max]},
                'significance': {'passed': crit3_pass, 'value': p_value,
                                'threshold': eclipse.significance_alpha}
            },
            'n_passed': n_passed,
            'hard_falsification': n_failed_hard >= 1
        }
    
    def save_results(self, results: Dict, output_dir: Path):
        """Guarda resultados completos"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separar raw results
        raw_results = results.pop('raw_results', {})
        
        # Guardar JSON principal
        with open(output_dir / 'eclipse_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Guardar detalles
        for phase in ['development', 'validation']:
            phase_results = raw_results.get(phase, [])
            if phase_results:
                df = pd.DataFrame([{
                    'subject_id': r['subject_id'],
                    'channel': r.get('channel', ''),
                    'to_state': r.get('to_state', ''),
                    'lag_seconds': r.get('lag_seconds'),
                    'h_star_precedes': r.get('h_star_precedes'),
                    'h_star_pre': r.get('h_star_pre'),
                    'h_star_post': r.get('h_star_post'),
                    'pac_pre': r.get('pac_pre'),
                    'pac_post': r.get('pac_post')
                } for r in phase_results])
                df.to_csv(output_dir / f'{phase}_details.csv', index=False)
        
        # Guardar certificado ECLIPSE
        certificate = {
            'protocol': 'ECLIPSE v2.0',
            'timestamp': datetime.now().isoformat(),
            'split_hash': results['split']['hash'],
            'sacred_seed': results['split']['seed'],
            'verdict': results['verdict']['verdict'],
            'pre_registered_criteria': results['pre_registered_criteria']
        }
        
        with open(output_dir / 'eclipse_certificate.json', 'w') as f:
            json.dump(certificate, f, indent=2)
        
        print(f"\n📁 Resultados guardados en: {output_dir}")
    
    def generate_figures(self, results: Dict, output_dir: Path):
        """Genera figuras de resultados"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        raw_results = results.get('raw_results', {})
        
        for idx, (phase, ax_row) in enumerate(zip(['development', 'validation'], [0, 1])):
            phase_results = raw_results.get(phase, [])
            valid_results = [r for r in phase_results if r.get('valid') and r.get('lag_seconds')]
            
            if not valid_results:
                continue
            
            lags = [r['lag_seconds'] for r in valid_results]
            precedes = [r['h_star_precedes'] for r in valid_results]
            
            # Histograma de lags
            ax1 = axes[ax_row, 0]
            ax1.hist(lags, bins=20, edgecolor='black', alpha=0.7, 
                    color='steelblue' if phase == 'development' else 'darkorange')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='H* = PAC')
            ax1.axvline(x=np.mean(lags), color='green', linestyle='-', linewidth=2, 
                       label=f'Mean = {np.mean(lags):.1f}s')
            ax1.set_xlabel('Lag (segundos)', fontsize=11)
            ax1.set_ylabel('Frecuencia', fontsize=11)
            ax1.set_title(f'{phase.upper()}: Distribución de Lag H* → PAC', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Tasa de precedencia
            ax2 = axes[ax_row, 1]
            n_precedes = sum(precedes)
            n_total = len(precedes)
            colors = ['green', 'red']
            ax2.bar(['H* precede', 'PAC precede'], [n_precedes, n_total - n_precedes], 
                   color=colors, alpha=0.7, edgecolor='black')
            ax2.axhline(y=n_total * 0.7, color='blue', linestyle='--', linewidth=2,
                       label=f'Threshold 70% = {n_total * 0.7:.0f}')
            ax2.set_ylabel('Número de transiciones', fontsize=11)
            ax2.set_title(f'{phase.upper()}: Precedencia ({n_precedes}/{n_total} = {n_precedes/n_total*100:.1f}%)', 
                         fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'eclipse_analysis_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Figura guardada: {output_dir / 'eclipse_analysis_summary.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AFH Transition Analysis + ECLIPSE v2.0"
    )
    
    parser.add_argument(
        '--data-dir', type=str,
        default=r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette',
        help='Ruta al directorio sleep-cassette'
    )
    
    parser.add_argument(
        '--output-dir', type=str,
        default='./afh_eclipse_results',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--n-subjects', type=int, default=None,
        help='Número de sujetos (default: todos)'
    )
    
    args = parser.parse_args()
    
    # Configuración
    config = TransitionConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        n_subjects=args.n_subjects
    )
    
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: Directorio no encontrado: {config.data_dir}")
        return 1
    
    # Ejecutar
    processor = AFHTransitionProcessor(config)
    
    try:
        results = processor.run_eclipse_analysis()
        
        if 'error' not in results:
            processor.save_results(results, config.output_dir)
            processor.generate_figures(results, config.output_dir)
            
            # Resumen final
            verdict = results.get('verdict', {}).get('verdict', 'UNKNOWN')
            
            print("\n" + "=" * 80)
            print("🔬 RESUMEN FINAL - ECLIPSE v2.0")
            print("=" * 80)
            
            if verdict == 'SUPPORTED':
                print("""
  ✅ RESULTADO: ARQUITECTURA AFH SOPORTADA
  
  La condición organizacional (H*) desciende ANTES que el
  Pliegue Autopsíquico (PAC) durante transiciones Wake→Sleep.
  
  Esto valida la predicción P-∇-3: H* HABILITA ∇.
                """)
            elif verdict == 'FALSIFIED':
                print("""
  ❌ RESULTADO: ARQUITECTURA AFH FALSIFICADA
  
  H* NO precede consistentemente a PAC, o el lag es
  insuficiente para soportar la arquitectura H* → ∇.
  
  Criterio F-∇-3 cumplido.
  
  Siguiente paso: Explorar operacionalizaciones alternativas.
                """)
            else:
                print(f"""
  ⚠️  RESULTADO: EVIDENCIA MIXTA
  
  Algunos criterios pasados, otros fallidos.
  Revisar resultados detallados.
                """)
            
            print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Análisis interrumpido")
        return 1
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())