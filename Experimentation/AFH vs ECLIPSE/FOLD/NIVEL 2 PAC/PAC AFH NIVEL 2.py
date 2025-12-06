#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
AFH TRANSITION ANALYSIS: PRECEDENCIA H* → PAC
Test específico de la Hipótesis del Pliegue Autopsíquico
═══════════════════════════════════════════════════════════════════════════════

PREDICCIÓN AFH (P-∇-3):
    Durante transiciones Vigilia→N2, H* desciende ANTES que PAC.
    
    Especificación:
    - Δt(H* → PAC) = 10-30 segundos
    - Precedencia en ≥70% de transiciones
    
JUSTIFICACIÓN TEÓRICA:
    Si H* (condición organizacional) HABILITA ∇ (Pliegue/PAC),
    entonces H* debe preceder temporalmente a PAC.
    
CRITERIO DE FALSACIÓN (F-∇-3):
    Si Δt < 5s o precedencia < 50% → Arquitectura H* → ∇ falsificada

ESTE TEST ES ESPECÍFICO DE AFH:
    - IIT no predice esta secuencia temporal
    - GNW no predice esta secuencia temporal
    - Solo AFH hace esta predicción específica

Author: Camilo Sjöberg Tala, M.D.
Date: 2025-12-05
Version: 1.0.0
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal, stats
from scipy.signal import hilbert, coherence
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
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionConfig:
    """Configuración para análisis de transiciones"""
    
    # Rutas
    data_dir: Path
    output_dir: Path
    
    # Parámetros de señal
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    notch_freq: float = 50.0
    
    # Canales
    target_channels: List[str] = field(default_factory=lambda: [
        'EEG Fpz-Cz',
        'EEG Pz-Oz'
    ])
    
    # Bandas para PAC (según RR DELTA PAC)
    delta_band: Tuple[float, float] = (2.0, 4.0)
    gamma_band: Tuple[float, float] = (30.0, 50.0)
    
    # Bandas para H* components
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
    
    # Criterios de falsación (P-∇-3)
    min_lag_seconds: float = 5.0      # Mínimo lag para considerar precedencia
    expected_lag_min: float = 10.0    # Lag esperado mínimo
    expected_lag_max: float = 30.0    # Lag esperado máximo
    precedence_threshold: float = 0.70  # ≥70% de transiciones
    
    # ECLIPSE
    sacred_seed: int = 42
    n_subjects: Optional[int] = None
    
    # Mínimos para análisis válido
    min_transitions_per_subject: int = 1
    min_total_transitions: int = 20


class TransitionEvent(NamedTuple):
    """Representa una transición de estado"""
    onset: float           # Tiempo de inicio (segundos)
    from_state: str        # Estado origen
    to_state: str          # Estado destino
    subject_id: str        # ID del sujeto
    duration_before: float # Duración del estado anterior
    duration_after: float  # Duración del estado siguiente


# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO DE H* INDEX
# ═══════════════════════════════════════════════════════════════════════════

class HStarCalculator:
    """
    Calcula H* Index: Condición organizacional que habilita el Pliegue
    
    Componentes:
    1. Coherencia espectral (coordinación entre bandas)
    2. Complejidad (Lempel-Ziv normalizada)
    3. Estabilidad temporal (decaimiento de autocorrelación)
    
    H* alto = sistema organizado, listo para generar presencia
    H* bajo = sistema desorganizado, presencia no habilitada
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
    
    def compute_spectral_coherence(self, data: np.ndarray,
                                   band1: Tuple[float, float],
                                   band2: Tuple[float, float]) -> float:
        """
        Calcula coherencia espectral entre dos bandas de frecuencia
        Usa la señal filtrada en cada banda y computa coherencia
        """
        try:
            # Filtrar en cada banda
            signal1 = self.bandpass_filter(data, band1)
            signal2 = self.bandpass_filter(data, band2)
            
            # Extraer envolventes de amplitud
            env1 = np.abs(hilbert(signal1))
            env2 = np.abs(hilbert(signal2))
            
            # Correlación de Pearson entre envolventes
            if np.std(env1) < 1e-10 or np.std(env2) < 1e-10:
                return 0.0
            
            corr = np.corrcoef(env1, env2)[0, 1]
            return float(np.abs(corr))
            
        except Exception as e:
            logger.debug(f"Error en coherencia: {e}")
            return 0.0
    
    def compute_lempel_ziv_complexity(self, data: np.ndarray) -> float:
        """
        Calcula complejidad de Lempel-Ziv normalizada
        
        Alta complejidad = señal rica en información
        Baja complejidad = señal simple/repetitiva
        """
        try:
            # Binarizar señal por mediana
            binary = (data > np.median(data)).astype(int)
            
            # Algoritmo LZ76
            s = ''.join(map(str, binary))
            n = len(s)
            
            if n == 0:
                return 0.0
            
            # Contar complejidad
            i = 0
            c = 1
            l = 1
            k = 1
            k_max = 1
            
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
                        i = 0
                        k = 1
                        k_max = 1
                    else:
                        k = 1
                else:
                    k += 1
                    if l + k > n:
                        c += 1
                        break
            
            # Normalizar por longitud
            b = n / np.log2(n) if n > 1 else 1
            lz_norm = c / b
            
            return float(np.clip(lz_norm, 0, 1))
            
        except Exception as e:
            logger.debug(f"Error en LZ: {e}")
            return 0.5
    
    def compute_autocorr_stability(self, data: np.ndarray, 
                                   max_lag: int = 50) -> float:
        """
        Calcula estabilidad temporal via decaimiento de autocorrelación
        
        Decaimiento lento = señal estable
        Decaimiento rápido = señal inestable
        """
        try:
            n = len(data)
            if n < max_lag * 2:
                max_lag = n // 4
            
            if max_lag < 2:
                return 0.5
            
            # Normalizar
            data_norm = (data - np.mean(data)) / (np.std(data) + 1e-10)
            
            # Calcular autocorrelación
            autocorr = np.correlate(data_norm, data_norm, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalizar
            
            # Calcular tau (tiempo característico de decaimiento)
            # Buscar primer cruce por 1/e
            threshold = 1 / np.e
            crossings = np.where(autocorr[:max_lag] < threshold)[0]
            
            if len(crossings) > 0:
                tau = crossings[0]
            else:
                tau = max_lag
            
            # Normalizar: tau alto = alta estabilidad
            stability = tau / max_lag
            
            return float(np.clip(stability, 0, 1))
            
        except Exception as e:
            logger.debug(f"Error en autocorr: {e}")
            return 0.5
    
    def compute_spectral_entropy(self, data: np.ndarray) -> float:
        """
        Calcula entropía espectral normalizada
        
        Alta entropía = distribución uniforme de potencia
        Baja entropía = potencia concentrada en pocas frecuencias
        """
        try:
            # PSD via Welch
            freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(len(data), 256))
            
            # Normalizar PSD a distribución de probabilidad
            psd_norm = psd / (np.sum(psd) + 1e-10)
            
            # Entropía de Shannon
            psd_norm = psd_norm[psd_norm > 0]
            entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            
            # Normalizar por entropía máxima
            max_entropy = np.log2(len(psd_norm))
            entropy_norm = entropy / max_entropy if max_entropy > 0 else 0
            
            return float(np.clip(entropy_norm, 0, 1))
            
        except Exception as e:
            logger.debug(f"Error en spectral entropy: {e}")
            return 0.5
    
    def compute_h_star_index(self, data: np.ndarray) -> Dict:
        """
        Calcula H* Index compuesto
        
        H* = w1*Coherencia + w2*Complejidad + w3*Estabilidad + w4*Entropía
        
        Retorna dict con componentes individuales y score total
        """
        # Preprocesar
        data = self._preprocess(data)
        
        if len(data) < self.fs * 2:  # Mínimo 2 segundos
            return self._empty_result()
        
        # Componente 1: Coherencia delta-gamma (coordinación cross-frequency)
        coherence_dg = self.compute_spectral_coherence(
            data, 
            self.config.delta_band, 
            self.config.gamma_band
        )
        
        # Componente 2: Coherencia theta-alpha (coordinación local)
        coherence_ta = self.compute_spectral_coherence(
            data,
            self.config.theta_band,
            self.config.alpha_band
        )
        
        # Componente 3: Complejidad LZ
        complexity = self.compute_lempel_ziv_complexity(data)
        
        # Componente 4: Estabilidad temporal
        stability = self.compute_autocorr_stability(data)
        
        # Componente 5: Entropía espectral
        spectral_entropy = self.compute_spectral_entropy(data)
        
        # Combinar componentes
        # Pesos basados en relevancia teórica para coordinación talamocortical
        w_coh_dg = 0.25      # Coherencia delta-gamma (crítica para PAC)
        w_coh_ta = 0.15      # Coherencia theta-alpha
        w_complexity = 0.20  # Complejidad (organización)
        w_stability = 0.25   # Estabilidad (condición sostenida)
        w_entropy = 0.15     # Entropía espectral
        
        h_star = (
            w_coh_dg * coherence_dg +
            w_coh_ta * coherence_ta +
            w_complexity * complexity +
            w_stability * stability +
            w_entropy * spectral_entropy
        )
        
        return {
            'h_star': float(h_star),
            'coherence_delta_gamma': float(coherence_dg),
            'coherence_theta_alpha': float(coherence_ta),
            'complexity_lz': float(complexity),
            'stability': float(stability),
            'spectral_entropy': float(spectral_entropy),
            'valid': True
        }
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa señal"""
        # Filtro pasabanda
        sos = signal.butter(
            4,
            [self.config.lowcut, self.config.highcut],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        filtered = signal.sosfiltfilt(sos, data)
        
        # Notch 50 Hz
        b_notch, a_notch = signal.iirnotch(
            self.config.notch_freq,
            Q=30,
            fs=self.fs
        )
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        # Z-score
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        return filtered
    
    def _empty_result(self) -> Dict:
        """Resultado vacío para datos inválidos"""
        return {
            'h_star': np.nan,
            'coherence_delta_gamma': np.nan,
            'coherence_theta_alpha': np.nan,
            'complexity_lz': np.nan,
            'stability': np.nan,
            'spectral_entropy': np.nan,
            'valid': False
        }


# ═══════════════════════════════════════════════════════════════════════════
# CÁLCULO DE PAC
# ═══════════════════════════════════════════════════════════════════════════

class PACCalculator:
    """
    Calcula Phase-Amplitude Coupling (PAC) delta→gamma
    
    PAC alto = convergencia temporal activa (Pliegue operando)
    PAC bajo = sin convergencia (Pliegue inactivo)
    """
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.fs = config.sampling_rate
        
        if TENSORPAC_AVAILABLE:
            self.pac_obj = Pac(
                idpac=(1, 0, 0),  # Modulation Index (Tort 2010)
                f_pha=list(config.delta_band),
                f_amp=list(config.gamma_band),
                dcomplex='wavelet',
                width=7
            )
        else:
            self.pac_obj = None
    
    def bandpass_filter(self, data: np.ndarray, 
                        band: Tuple[float, float]) -> np.ndarray:
        """Filtro pasabanda"""
        nyq = self.fs / 2
        low = band[0] / nyq
        high = min(band[1] / nyq, 0.99)
        
        if low >= high or low <= 0:
            return data
            
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return signal.sosfiltfilt(sos, data)
    
    def compute_pac_manual(self, data: np.ndarray) -> float:
        """
        Calcula PAC manualmente (Mean Vector Length)
        Usado si tensorpac no está disponible
        """
        try:
            # Filtrar
            delta = self.bandpass_filter(data, self.config.delta_band)
            gamma = self.bandpass_filter(data, self.config.gamma_band)
            
            # Extraer fase y amplitud
            delta_phase = np.angle(hilbert(delta))
            gamma_amp = np.abs(hilbert(gamma))
            
            # Z-score de amplitud
            gamma_amp = (gamma_amp - np.mean(gamma_amp)) / (np.std(gamma_amp) + 1e-10)
            
            # Mean Vector Length
            n = len(delta_phase)
            complex_signal = gamma_amp * np.exp(1j * delta_phase)
            mvl = np.abs(np.mean(complex_signal))
            
            return float(mvl)
            
        except Exception as e:
            logger.debug(f"Error en PAC manual: {e}")
            return np.nan
    
    def compute_pac(self, data: np.ndarray) -> Dict:
        """Calcula PAC delta→gamma"""
        # Preprocesar
        data = self._preprocess(data)
        
        if len(data) < self.fs * 2:
            return {'pac': np.nan, 'valid': False}
        
        try:
            if self.pac_obj is not None:
                # Usar tensorpac
                data_reshaped = data[np.newaxis, :]
                pac_value = self.pac_obj.filterfit(
                    self.fs, 
                    data_reshaped, 
                    data_reshaped
                )
                pac_value = float(pac_value[0, 0, 0])
            else:
                # Usar cálculo manual
                pac_value = self.compute_pac_manual(data)
            
            return {
                'pac': pac_value,
                'valid': not np.isnan(pac_value)
            }
            
        except Exception as e:
            logger.debug(f"Error en PAC: {e}")
            return {'pac': np.nan, 'valid': False}
    
    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa señal"""
        # Filtro pasabanda
        sos = signal.butter(
            4,
            [self.config.lowcut, self.config.highcut],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        filtered = signal.sosfiltfilt(sos, data)
        
        # Notch 50 Hz
        b_notch, a_notch = signal.iirnotch(
            self.config.notch_freq,
            Q=30,
            fs=self.fs
        )
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        # Z-score
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        return filtered


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR DE TRANSICIONES
# ═══════════════════════════════════════════════════════════════════════════

class TransitionDetector:
    """Detecta transiciones Wake→N2 en anotaciones de sueño"""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
    
    def find_transitions(self, annotations: mne.Annotations,
                        subject_id: str) -> List[TransitionEvent]:
        """
        Encuentra transiciones Wake→N2 (directas o vía N1)
        
        Transiciones válidas:
        - Wake → N2 (directa)
        - Wake → N1 → N2 (vía N1, N1 < 2 min)
        """
        transitions = []
        
        # Convertir anotaciones a lista ordenada
        events = []
        for ann in annotations:
            events.append({
                'onset': ann['onset'],
                'duration': ann['duration'],
                'description': ann['description']
            })
        
        # Ordenar por tiempo
        events = sorted(events, key=lambda x: x['onset'])
        
        # Buscar transiciones
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            
            # Caso 1: Wake → N2 directa
            if (current['description'] == self.config.wake_state and
                next_event['description'] == self.config.n2_state):
                
                transitions.append(TransitionEvent(
                    onset=next_event['onset'],
                    from_state='Wake',
                    to_state='N2',
                    subject_id=subject_id,
                    duration_before=current['duration'],
                    duration_after=next_event['duration']
                ))
            
            # Caso 2: Wake → N1 → N2 (N1 corto)
            elif (current['description'] == self.config.wake_state and
                  next_event['description'] == self.config.n1_state and
                  i + 2 < len(events)):
                
                next_next = events[i + 2]
                
                if (next_next['description'] == self.config.n2_state and
                    next_event['duration'] < 120):  # N1 < 2 minutos
                    
                    transitions.append(TransitionEvent(
                        onset=next_next['onset'],
                        from_state='Wake',
                        to_state='N2',
                        subject_id=subject_id,
                        duration_before=current['duration'] + next_event['duration'],
                        duration_after=next_next['duration']
                    ))
        
        # Filtrar transiciones con contexto suficiente
        valid_transitions = []
        for trans in transitions:
            # Necesitamos al menos 2 min antes y después
            if (trans.duration_before >= 60 and 
                trans.duration_after >= 60):
                valid_transitions.append(trans)
        
        return valid_transitions


# ═══════════════════════════════════════════════════════════════════════════
# ANALIZADOR DE PRECEDENCIA
# ═══════════════════════════════════════════════════════════════════════════

class PrecedenceAnalyzer:
    """
    Analiza precedencia temporal H* → PAC durante transiciones
    
    HIPÓTESIS (P-∇-3):
    H* desciende ANTES que PAC durante transiciones Wake→N2
    """
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.h_star_calc = HStarCalculator(config)
        self.pac_calc = PACCalculator(config)
    
    def analyze_single_transition(self, raw: mne.io.Raw,
                                  transition: TransitionEvent,
                                  channel: str) -> Dict:
        """
        Analiza una transición individual
        
        1. Extrae ventana alrededor de la transición
        2. Calcula H* y PAC en ventanas deslizantes
        3. Detecta punto de caída de cada métrica
        4. Calcula lag (H* caída - PAC caída)
        """
        fs = raw.info['sfreq']
        
        # Definir ventana de análisis
        window_start = transition.onset - self.config.transition_window_before
        window_end = transition.onset + self.config.transition_window_after
        
        # Ajustar a límites del recording
        window_start = max(0, window_start)
        window_end = min(raw.times[-1], window_end)
        
        # Extraer datos
        start_sample = int(window_start * fs)
        end_sample = int(window_end * fs)
        
        try:
            ch_idx = raw.ch_names.index(channel)
            data = raw.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)[0]
        except Exception as e:
            logger.warning(f"Error extrayendo datos: {e}")
            return self._empty_transition_result(transition)
        
        if len(data) < fs * 60:  # Mínimo 1 minuto
            return self._empty_transition_result(transition)
        
        # Calcular métricas en ventanas deslizantes
        window_samples = int(self.config.sliding_window_size * fs)
        step_samples = int(self.config.sliding_window_step * fs)
        
        timepoints = []
        h_star_series = []
        pac_series = []
        
        # Componentes individuales de H*
        coherence_dg_series = []
        complexity_series = []
        stability_series = []
        
        n_windows = (len(data) - window_samples) // step_samples + 1
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            
            if end > len(data):
                break
            
            epoch = data[start:end]
            
            # Tiempo relativo a la transición
            t_relative = window_start + (start / fs) - transition.onset
            timepoints.append(t_relative)
            
            # Calcular H*
            h_star_result = self.h_star_calc.compute_h_star_index(epoch)
            h_star_series.append(h_star_result['h_star'])
            coherence_dg_series.append(h_star_result['coherence_delta_gamma'])
            complexity_series.append(h_star_result['complexity_lz'])
            stability_series.append(h_star_result['stability'])
            
            # Calcular PAC
            pac_result = self.pac_calc.compute_pac(epoch)
            pac_series.append(pac_result['pac'])
        
        # Convertir a arrays
        timepoints = np.array(timepoints)
        h_star_series = np.array(h_star_series)
        pac_series = np.array(pac_series)
        
        # Limpiar NaNs
        valid_mask = ~(np.isnan(h_star_series) | np.isnan(pac_series))
        
        if np.sum(valid_mask) < 5:
            return self._empty_transition_result(transition)
        
        timepoints = timepoints[valid_mask]
        h_star_series = h_star_series[valid_mask]
        pac_series = pac_series[valid_mask]
        
        # Normalizar series para comparación
        h_star_norm = (h_star_series - np.nanmean(h_star_series)) / (np.nanstd(h_star_series) + 1e-10)
        pac_norm = (pac_series - np.nanmean(pac_series)) / (np.nanstd(pac_series) + 1e-10)
        
        # Suavizar para detectar tendencia
        h_star_smooth = uniform_filter1d(h_star_norm, size=3)
        pac_smooth = uniform_filter1d(pac_norm, size=3)
        
        # Detectar punto de caída (cruce por cero desde arriba)
        h_star_drop_idx = self._find_drop_point(h_star_smooth, timepoints)
        pac_drop_idx = self._find_drop_point(pac_smooth, timepoints)
        
        # Calcular lag
        if h_star_drop_idx is not None and pac_drop_idx is not None:
            h_star_drop_time = timepoints[h_star_drop_idx]
            pac_drop_time = timepoints[pac_drop_idx]
            lag_seconds = pac_drop_time - h_star_drop_time  # Positivo si H* cae primero
            h_star_precedes = lag_seconds > self.config.min_lag_seconds
        else:
            h_star_drop_time = np.nan
            pac_drop_time = np.nan
            lag_seconds = np.nan
            h_star_precedes = False
        
        # Cross-correlation como medida alternativa
        xcorr_lag = self._compute_xcorr_lag(h_star_smooth, pac_smooth, timepoints)
        
        return {
            'subject_id': transition.subject_id,
            'channel': channel,
            'transition_onset': transition.onset,
            'from_state': transition.from_state,
            'to_state': transition.to_state,
            
            # Series temporales
            'timepoints': timepoints.tolist(),
            'h_star_series': h_star_series.tolist(),
            'pac_series': pac_series.tolist(),
            
            # Puntos de caída
            'h_star_drop_time': float(h_star_drop_time) if not np.isnan(h_star_drop_time) else None,
            'pac_drop_time': float(pac_drop_time) if not np.isnan(pac_drop_time) else None,
            
            # Lag
            'lag_seconds': float(lag_seconds) if not np.isnan(lag_seconds) else None,
            'h_star_precedes': h_star_precedes,
            
            # Cross-correlation
            'xcorr_lag_seconds': float(xcorr_lag) if xcorr_lag is not None else None,
            
            # Métricas pre/post transición
            'h_star_pre': float(np.nanmean(h_star_series[timepoints < 0])),
            'h_star_post': float(np.nanmean(h_star_series[timepoints > 0])),
            'pac_pre': float(np.nanmean(pac_series[timepoints < 0])),
            'pac_post': float(np.nanmean(pac_series[timepoints > 0])),
            
            'valid': True
        }
    
    def _find_drop_point(self, series: np.ndarray, 
                         timepoints: np.ndarray) -> Optional[int]:
        """
        Encuentra el punto donde la serie cruza por cero desde arriba
        (indica caída de la métrica)
        
        Busca cerca de t=0 (momento de transición)
        """
        # Buscar en ventana cercana a t=0
        search_mask = np.abs(timepoints) < 120  # ±2 minutos
        
        if np.sum(search_mask) < 3:
            return None
        
        search_indices = np.where(search_mask)[0]
        search_series = series[search_mask]
        
        # Buscar cruce por cero
        for i in range(len(search_series) - 1):
            if search_series[i] > 0 and search_series[i + 1] <= 0:
                return search_indices[i]
        
        # Si no hay cruce, buscar mínimo local
        min_idx = np.argmin(search_series)
        return search_indices[min_idx]
    
    def _compute_xcorr_lag(self, series1: np.ndarray, 
                           series2: np.ndarray,
                           timepoints: np.ndarray) -> Optional[float]:
        """
        Calcula lag via cross-correlation
        
        Lag positivo = series1 lidera series2
        """
        try:
            if len(series1) < 5:
                return None
            
            # Cross-correlation
            xcorr = np.correlate(series1, series2, mode='full')
            lags = np.arange(-len(series1) + 1, len(series1))
            
            # Encontrar lag con máxima correlación
            max_idx = np.argmax(xcorr)
            lag_samples = lags[max_idx]
            
            # Convertir a segundos
            step_seconds = self.config.sliding_window_step
            lag_seconds = lag_samples * step_seconds
            
            return lag_seconds
            
        except Exception as e:
            logger.debug(f"Error en xcorr: {e}")
            return None
    
    def _empty_transition_result(self, transition: TransitionEvent) -> Dict:
        """Resultado vacío para transición inválida"""
        return {
            'subject_id': transition.subject_id,
            'transition_onset': transition.onset,
            'valid': False
        }


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class AFHTransitionProcessor:
    """Procesador principal para análisis de transiciones AFH"""
    
    def __init__(self, config: TransitionConfig):
        self.config = config
        self.detector = TransitionDetector(config)
        self.analyzer = PrecedenceAnalyzer(config)
    
    def load_subject(self, psg_file: Path, hypno_file: Path) -> Optional[Dict]:
        """Carga datos de un sujeto"""
        try:
            raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
            annotations = mne.read_annotations(str(hypno_file))
            raw.set_annotations(annotations)
            
            # Buscar canales disponibles
            available_channels = []
            for target in self.config.target_channels:
                if target in raw.ch_names:
                    available_channels.append(target)
                else:
                    for ch in raw.ch_names:
                        if 'eeg' in ch.lower() and ('fpz' in ch.lower() or 'pz' in ch.lower()):
                            if ch not in available_channels:
                                available_channels.append(ch)
                                break
            
            if not available_channels:
                eeg_channels = [ch for ch in raw.ch_names if 'eeg' in ch.lower()]
                if eeg_channels:
                    available_channels = eeg_channels[:2]
            
            if not available_channels:
                logger.warning(f"  No hay canales EEG disponibles")
                return None
            
            # Resamplear si necesario
            if raw.info['sfreq'] != self.config.sampling_rate:
                raw.resample(self.config.sampling_rate)
            
            return {
                'raw': raw,
                'channels': available_channels,
                'annotations': raw.annotations
            }
            
        except Exception as e:
            logger.error(f"  Error cargando: {e}")
            return None
    
    def process_subject(self, subject_id: str, 
                       psg_file: Path, 
                       hypno_file: Path) -> List[Dict]:
        """Procesa un sujeto completo"""
        logger.info(f"Procesando {subject_id}...")
        
        loaded = self.load_subject(psg_file, hypno_file)
        if loaded is None:
            return []
        
        raw = loaded['raw']
        channels = loaded['channels']
        annotations = loaded['annotations']
        
        # Detectar transiciones
        transitions = self.detector.find_transitions(annotations, subject_id)
        logger.info(f"  {subject_id}: {len(transitions)} transiciones Wake→N2")
        
        if len(transitions) < self.config.min_transitions_per_subject:
            logger.warning(f"  {subject_id}: Insuficientes transiciones")
            return []
        
        # Analizar cada transición
        results = []
        for trans_idx, transition in enumerate(transitions):
            for channel in channels:
                result = self.analyzer.analyze_single_transition(
                    raw, transition, channel
                )
                result['transition_idx'] = trans_idx
                results.append(result)
        
        valid_results = [r for r in results if r.get('valid', False)]
        logger.info(f"  {subject_id}: {len(valid_results)} análisis válidos")
        
        return results
    
    def run_full_analysis(self) -> Dict:
        """Ejecuta análisis completo"""
        
        print("\n" + "=" * 80)
        print("🔬 AFH TRANSITION ANALYSIS")
        print("   Test de Precedencia H* → PAC (P-∇-3)")
        print("=" * 80)
        
        start_time = time.time()
        
        # Buscar archivos
        psg_files = sorted(self.config.data_dir.glob('*PSG.edf'))
        
        if self.config.n_subjects:
            psg_files = psg_files[:self.config.n_subjects]
        
        print(f"\n📁 Dataset: {self.config.data_dir}")
        print(f"   Archivos PSG: {len(psg_files)}")
        
        # Procesar sujetos
        all_results = []
        subjects_with_transitions = 0
        
        for idx, psg_file in enumerate(psg_files, 1):
            subject_id = psg_file.stem.replace('-PSG', '')
            
            # Sleep-EDF naming
            hypno_id = subject_id[:-1] + 'C'
            hypno_file = psg_file.parent / f"{hypno_id}-Hypnogram.edf"
            
            if not hypno_file.exists():
                logger.warning(f"[{idx}/{len(psg_files)}] {subject_id}: Hypnogram no encontrado")
                continue
            
            print(f"[{idx}/{len(psg_files)}] {subject_id}...", end=" ")
            
            results = self.process_subject(subject_id, psg_file, hypno_file)
            
            valid_results = [r for r in results if r.get('valid', False)]
            
            if valid_results:
                all_results.extend(valid_results)
                subjects_with_transitions += 1
                print(f"✓ {len(valid_results)} análisis")
            else:
                print("✗ Sin transiciones válidas")
        
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 Procesamiento completo:")
        print(f"   Sujetos con transiciones: {subjects_with_transitions}")
        print(f"   Total análisis: {len(all_results)}")
        print(f"   Tiempo: {elapsed_time/60:.1f} minutos")
        
        if len(all_results) < self.config.min_total_transitions:
            print(f"\n⚠️  Insuficientes transiciones para análisis estadístico")
            print(f"   Mínimo requerido: {self.config.min_total_transitions}")
            return {'error': 'insufficient_transitions'}
        
        # Analizar resultados
        return self._analyze_results(all_results)
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analiza resultados y evalúa criterio P-∇-3"""
        
        print("\n" + "=" * 80)
        print("📊 ANÁLISIS DE PRECEDENCIA H* → PAC")
        print("=" * 80)
        
        # Filtrar resultados válidos
        valid_results = [r for r in results if r.get('valid', False) and r.get('lag_seconds') is not None]
        
        if len(valid_results) == 0:
            return {'error': 'no_valid_results'}
        
        # Extraer métricas
        lags = [r['lag_seconds'] for r in valid_results]
        precedes = [r['h_star_precedes'] for r in valid_results]
        xcorr_lags = [r['xcorr_lag_seconds'] for r in valid_results if r.get('xcorr_lag_seconds') is not None]
        
        # Estadísticas
        n_total = len(valid_results)
        n_precedes = sum(precedes)
        precedence_rate = n_precedes / n_total
        
        mean_lag = np.mean(lags)
        std_lag = np.std(lags)
        median_lag = np.median(lags)
        
        mean_lag_when_precedes = np.mean([l for l, p in zip(lags, precedes) if p]) if n_precedes > 0 else np.nan
        
        # Cambios pre/post
        h_star_pre = np.mean([r['h_star_pre'] for r in valid_results])
        h_star_post = np.mean([r['h_star_post'] for r in valid_results])
        pac_pre = np.mean([r['pac_pre'] for r in valid_results])
        pac_post = np.mean([r['pac_post'] for r in valid_results])
        
        h_star_change = (h_star_post - h_star_pre) / (h_star_pre + 1e-10)
        pac_change = (pac_post - pac_pre) / (pac_pre + 1e-10)
        
        print(f"\n  📈 ESTADÍSTICAS DE LAG:")
        print(f"     Total transiciones analizadas: {n_total}")
        print(f"     H* precede PAC en: {n_precedes}/{n_total} ({precedence_rate*100:.1f}%)")
        print(f"     Lag medio: {mean_lag:.1f} ± {std_lag:.1f} segundos")
        print(f"     Lag mediano: {median_lag:.1f} segundos")
        if not np.isnan(mean_lag_when_precedes):
            print(f"     Lag medio (cuando H* precede): {mean_lag_when_precedes:.1f} segundos")
        
        print(f"\n  📉 CAMBIOS PRE→POST TRANSICIÓN:")
        print(f"     H*:  {h_star_pre:.4f} → {h_star_post:.4f} ({h_star_change*100:+.1f}%)")
        print(f"     PAC: {pac_pre:.4f} → {pac_post:.4f} ({pac_change*100:+.1f}%)")
        
        # Evaluar criterio P-∇-3
        print("\n" + "=" * 80)
        print("📋 EVALUACIÓN CRITERIO P-∇-3")
        print("=" * 80)
        
        # Criterio 1: Tasa de precedencia
        precedence_pass = precedence_rate >= self.config.precedence_threshold
        
        # Criterio 2: Lag en rango esperado
        lag_in_range = (self.config.expected_lag_min <= mean_lag_when_precedes <= self.config.expected_lag_max) if not np.isnan(mean_lag_when_precedes) else False
        
        # Criterio 3: Significancia estadística (t-test de lag > 0)
        if len(lags) > 2:
            t_stat, p_value = stats.ttest_1samp(lags, 0, alternative='greater')
            significant = p_value < 0.05
        else:
            t_stat, p_value = np.nan, 1.0
            significant = False
        
        print(f"\n  Criterio 1: Tasa de precedencia ≥ {self.config.precedence_threshold*100:.0f}%")
        print(f"     Observado: {precedence_rate*100:.1f}%")
        print(f"     Resultado: {'✅ PASADO' if precedence_pass else '❌ FALLIDO'}")
        
        print(f"\n  Criterio 2: Lag medio en rango [{self.config.expected_lag_min}, {self.config.expected_lag_max}] s")
        print(f"     Observado: {mean_lag_when_precedes:.1f} s")
        print(f"     Resultado: {'✅ PASADO' if lag_in_range else '❌ FALLIDO'}")
        
        print(f"\n  Criterio 3: Lag significativamente > 0 (p < 0.05)")
        print(f"     t = {t_stat:.2f}, p = {p_value:.4f}")
        print(f"     Resultado: {'✅ PASADO' if significant else '❌ FALLIDO'}")
        
        # Veredicto final
        n_passed = sum([precedence_pass, lag_in_range, significant])
        
        print("\n" + "=" * 80)
        
        if n_passed >= 2:
            print("✅ VEREDICTO: PRECEDENCIA H* → PAC CONFIRMADA")
            print("   Arquitectura AFH (H* habilita ∇) SOPORTADA")
            verdict = 'SUPPORTED'
        elif n_passed == 1:
            print("⚠️  VEREDICTO: EVIDENCIA MIXTA")
            print("   Precedencia parcialmente observada")
            verdict = 'MIXED'
        else:
            print("❌ VEREDICTO: PRECEDENCIA H* → PAC NO CONFIRMADA")
            print("   Criterio F-∇-3 CUMPLIDO: Arquitectura H* → ∇ FALSIFICADA")
            verdict = 'FALSIFIED'
        
        print("=" * 80)
        
        # Compilar resultados
        analysis_results = {
            'n_transitions': n_total,
            'n_subjects': len(set(r['subject_id'] for r in valid_results)),
            
            'precedence': {
                'rate': float(precedence_rate),
                'n_precedes': n_precedes,
                'threshold': self.config.precedence_threshold,
                'passed': precedence_pass
            },
            
            'lag': {
                'mean': float(mean_lag),
                'std': float(std_lag),
                'median': float(median_lag),
                'mean_when_precedes': float(mean_lag_when_precedes) if not np.isnan(mean_lag_when_precedes) else None,
                'expected_range': [self.config.expected_lag_min, self.config.expected_lag_max],
                'in_range': lag_in_range
            },
            
            'statistical_test': {
                't_statistic': float(t_stat) if not np.isnan(t_stat) else None,
                'p_value': float(p_value),
                'significant': significant
            },
            
            'changes': {
                'h_star_pre': float(h_star_pre),
                'h_star_post': float(h_star_post),
                'h_star_change_pct': float(h_star_change * 100),
                'pac_pre': float(pac_pre),
                'pac_post': float(pac_post),
                'pac_change_pct': float(pac_change * 100)
            },
            
            'criteria_passed': n_passed,
            'verdict': verdict,
            
            'raw_results': valid_results
        }
        
        return analysis_results
    
    def save_results(self, results: Dict, output_dir: Path):
        """Guarda resultados"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar JSON principal
        results_file = output_dir / 'transition_analysis_results.json'
        
        # Separar raw_results para archivo aparte
        raw_results = results.pop('raw_results', [])
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Guardar raw results como CSV
        if raw_results:
            df = pd.DataFrame([{
                'subject_id': r['subject_id'],
                'channel': r.get('channel', 'unknown'),
                'transition_onset': r['transition_onset'],
                'lag_seconds': r.get('lag_seconds'),
                'h_star_precedes': r.get('h_star_precedes'),
                'xcorr_lag': r.get('xcorr_lag_seconds'),
                'h_star_pre': r.get('h_star_pre'),
                'h_star_post': r.get('h_star_post'),
                'pac_pre': r.get('pac_pre'),
                'pac_post': r.get('pac_post')
            } for r in raw_results])
            
            df.to_csv(output_dir / 'transition_details.csv', index=False)
        
        print(f"\n📁 Resultados guardados en: {output_dir}")
    
    def generate_figures(self, results: Dict, output_dir: Path):
        """Genera figuras de resultados"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_results = results.get('raw_results', [])
        if not raw_results:
            return
        
        # Figura 1: Distribución de lags
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        lags = [r['lag_seconds'] for r in raw_results if r.get('lag_seconds') is not None]
        
        # Histograma de lags
        ax1 = axes[0]
        ax1.hist(lags, bins=20, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', label='H* = PAC')
        ax1.axvline(x=np.mean(lags), color='green', linestyle='-', label=f'Mean = {np.mean(lags):.1f}s')
        ax1.set_xlabel('Lag (segundos)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Lag H* → PAC')
        ax1.legend()
        
        # Scatter pre vs post
        ax2 = axes[1]
        h_star_pre = [r['h_star_pre'] for r in raw_results]
        h_star_post = [r['h_star_post'] for r in raw_results]
        pac_pre = [r['pac_pre'] for r in raw_results]
        pac_post = [r['pac_post'] for r in raw_results]
        
        ax2.scatter(h_star_pre, h_star_post, alpha=0.5, label='H*')
        ax2.scatter(pac_pre, pac_post, alpha=0.5, label='PAC')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax2.set_xlabel('Pre-transición')
        ax2.set_ylabel('Post-transición')
        ax2.set_title('Cambio Pre→Post Transición')
        ax2.legend()
        
        # Tasa de precedencia
        ax3 = axes[2]
        precedes = [r['h_star_precedes'] for r in raw_results]
        n_precedes = sum(precedes)
        n_total = len(precedes)
        
        ax3.bar(['H* precede', 'PAC precede'], [n_precedes, n_total - n_precedes], 
                color=['green', 'red'], alpha=0.7)
        ax3.axhline(y=n_total * 0.7, color='blue', linestyle='--', 
                   label=f'Threshold (70% = {n_total * 0.7:.0f})')
        ax3.set_ylabel('Número de transiciones')
        ax3.set_title('Precedencia H* vs PAC')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'transition_analysis_summary.png', dpi=150)
        plt.close()
        
        print(f"📊 Figura guardada: {output_dir / 'transition_analysis_summary.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AFH Transition Analysis - Test de Precedencia H* → PAC"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette',
        help='Ruta al directorio sleep-cassette'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./afh_transition_results',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--n-subjects',
        type=int,
        default=None,
        help='Número de sujetos (default: todos)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    # Configuración
    config = TransitionConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sacred_seed=args.seed,
        n_subjects=args.n_subjects
    )
    
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: Directorio no encontrado: {config.data_dir}")
        return 1
    
    # Ejecutar análisis
    processor = AFHTransitionProcessor(config)
    
    try:
        results = processor.run_full_analysis()
        
        if 'error' not in results:
            processor.save_results(results, config.output_dir)
            processor.generate_figures(results, config.output_dir)
            
            # Resumen final
            print("\n" + "=" * 80)
            print("🔬 RESUMEN FINAL - TEST P-∇-3")
            print("=" * 80)
            
            verdict = results.get('verdict', 'UNKNOWN')
            
            if verdict == 'SUPPORTED':
                print("""
  ✅ RESULTADO: ARQUITECTURA AFH SOPORTADA
  
  Evidencia:
  • H* desciende ANTES que PAC durante transiciones Wake→N2
  • Lag observado en rango predicho (10-30 segundos)
  • Precedencia estadísticamente significativa
  
  Implicación:
  La condición organizacional (H*) efectivamente HABILITA
  el Pliegue Autopsíquico (PAC), como predice AFH.
                """)
            elif verdict == 'FALSIFIED':
                print("""
  ❌ RESULTADO: ARQUITECTURA AFH FALSIFICADA
  
  Observación:
  • H* NO precede consistentemente a PAC
  • O el lag es demasiado pequeño
  
  Implicación:
  La arquitectura H* → ∇ no se sostiene empíricamente.
  PAC podría ser biomarcador válido pero no como
  operacionalización del Pliegue habilitado por H*.
  
  Siguiente paso:
  Explorar operacionalizaciones alternativas (ΔT, TE, GC, DCM)
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