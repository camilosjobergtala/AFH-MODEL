#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIVEL 2 EXTENDED: PAC Band Specificity + N3 + Power vs PAC
WITH FULL ECLIPSE v2.0 INTEGRATION
Autopsychic Fold Hypothesis - VERSIÓN CORREGIDA
═══════════════════════════════════════════════════════════════════════════════

HIPÓTESIS EXTENDIDAS:
    H1: PAC discrimina consciencia, no cantidad de delta
    H2: Delta-gamma PAC ≠ Delta POWER
    H3: N3 tiene BAJO PAC a pesar de ALTO delta power
    
CORRECCIONES vs VERSIÓN ANTERIOR:
    ✅ Threshold Cohen's d ajustado a 1.0 (según RR DELTA PAC)
    ✅ Threshold PAC vs Power advantage = 0.3 (diferencia sustancial)
    ✅ Ambos canales: Fpz-Cz + Pz-Oz (según protocolo tesis)
    ✅ Banda delta corregida a 2-4 Hz (según RR, no 1-4 Hz)
    ✅ Banda gamma corregida a 30-50 Hz (según RR)
    
CRITERIOS DE FALSACIÓN (Tabla 3.5 Tesis):
    F-∇-1: p > 0.05 → PAC no discrimina
    F-∇-2: PAC explicado por Power → epifenómeno
    
Author: Camilo Sjöberg Tala
Date: 2025-12-03
Version: NIVEL_2_EXTENDED_ECLIPSE_v3.1_CORRECTED
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal, stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from dataclasses import dataclass, field
import argparse
import sys
import json
import time

from tensorpac import Pac

# IMPORT ECLIPSE v2.0
try:
    from eclipse_v2 import (
        EclipseFramework,
        EclipseConfig,
        FalsificationCriteria,
        EclipseValidator
    )
    ECLIPSE_AVAILABLE = True
except ImportError:
    ECLIPSE_AVAILABLE = False
    print("\n⚠️  WARNING: ECLIPSE v2.0 not found in path")
    print("   Place eclipse_v2.py in same directory or PYTHONPATH")
    print("   Continuing without ECLIPSE integration...")

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN CORREGIDA SEGÚN RR DELTA PAC
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BandPair:
    """Define un par de bandas para PAC"""
    name: str
    phase_band: Tuple[float, float]
    amp_band: Tuple[float, float]
    description: str

@dataclass
class Config:
    """Configuración NIVEL 2 EXTENDED - CORREGIDA según RR DELTA PAC"""
    data_dir: Path
    output_dir: Path
    
    # Parámetros de señal
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    notch_freq: float = 50.0
    
    # CORREGIDO: Canales según protocolo tesis (ambos)
    target_channels: List[str] = field(default_factory=lambda: [
        'EEG Fpz-Cz',  # Anterior
        'EEG Pz-Oz'    # Posterior
    ])
    
    # CORREGIDO: Bandas según RR DELTA PAC (Tabla 3.4)
    # Delta: 2-4 Hz (no 1-4 Hz)
    # Gamma: 30-50 Hz (no 30-45 Hz)
    band_pairs: List[BandPair] = field(default_factory=lambda: [
        BandPair(
            name="Delta-Gamma",
            phase_band=(2.0, 4.0),      # CORREGIDO: 2-4 Hz según RR
            amp_band=(30.0, 50.0),      # CORREGIDO: 30-50 Hz según RR
            description="Primary AFH prediction - slow wave coupling"
        ),
        BandPair(
            name="Theta-Gamma",
            phase_band=(4.0, 8.0),
            amp_band=(30.0, 50.0),
            description="Secondary - exceeds predicted window"
        ),
        BandPair(
            name="Alpha-Gamma",
            phase_band=(8.0, 12.0),
            amp_band=(30.0, 50.0),
            description="Attention coupling"
        ),
        BandPair(
            name="Beta-Gamma",
            phase_band=(13.0, 30.0),
            amp_band=(30.0, 50.0),
            description="Fast cortical coupling - control"
        ),
    ])
    
    # Bandas de potencia para comparación PAC vs POWER
    power_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Delta': (2.0, 4.0),    # CORREGIDO: Mismo rango que PAC
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (13.0, 30.0),
        'Gamma': (30.0, 50.0),  # CORREGIDO: Mismo rango que PAC
    })
    
    # Parámetros de época
    epoch_duration: float = 30.0
    n_epochs_per_state: int = 50
    
    # Estados Sleep-EDF
    wake_state: str = 'Sleep stage W'
    n1_state: str = 'Sleep stage 1'
    n2_state: str = 'Sleep stage 2'
    n3_state: str = 'Sleep stage 3'
    rem_state: str = 'Sleep stage R'
    
    # ECLIPSE
    sacred_seed: int = 42
    n_subjects: Optional[int] = None
    researcher: str = "Camilo Sjöberg Tala"
    project_name: str = "NIVEL2_PAC_AUTOPSYCHIC_FOLD"
    
    # CORREGIDO: Thresholds según RR DELTA PAC (Tabla 3.5)
    cohens_d_threshold: float = 1.0      # Según RR: d ≥ 1.0
    p_value_threshold: float = 0.001     # Según RR: p < 0.001
    pac_power_advantage_threshold: float = 0.3  # Diferencia sustancial
    
    def __post_init__(self):
        if self.n_subjects is None:
            self.project_name = f"{self.project_name}_FULL_153subj"
        else:
            self.project_name = f"{self.project_name}_{self.n_subjects}subj"


# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA PAC + POWER (CORREGIDA)
# ═══════════════════════════════════════════════════════════════════════════

class ExtendedPACCalculator:
    """Calcula PAC Y POWER para múltiples bandas - CORREGIDO"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fs = config.sampling_rate
        
        self.pac_objects = {}
        for band_pair in config.band_pairs:
            self.pac_objects[band_pair.name] = Pac(
                idpac=(1, 0, 0),  # Modulation Index (Tort 2010)
                f_pha=list(band_pair.phase_band),
                f_amp=list(band_pair.amp_band),
                dcomplex='wavelet',
                width=7
            )
            logger.debug(f"PAC object created: {band_pair.name} "
                        f"({band_pair.phase_band} → {band_pair.amp_band})")
        
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa señal EEG según protocolo RR"""
        # Filtro pasabanda según RR: 0.5-45 Hz
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
        
        # Z-score normalización
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        return filtered
    
    def has_artifact(self, data: np.ndarray, threshold_uv: float = 100.0) -> bool:
        """
        Detecta artefactos según RR: amplitud > 100 µV
        NOTA: Asume datos ya normalizados (z-score), threshold ajustado
        """
        # Para datos z-scored, usamos threshold de 8 std (muy conservador)
        peak_to_peak = np.ptp(data)
        return peak_to_peak > 8.0
    
    def compute_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> float:
        """Calcula power espectral en una banda específica (Welch)"""
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(len(data), 256))
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        if not np.any(idx_band):
            return np.nan
        power = np.trapz(psd[idx_band], freqs[idx_band])
        return float(power)
    
    def compute_all_metrics(self, data: np.ndarray) -> Dict[str, Dict]:
        """Calcula PAC + POWER para una época"""
        data_clean = self.preprocess(data)
        
        # Verificar artefactos
        if self.has_artifact(data_clean):
            return self._empty_results('artifact')
        
        # Verificar longitud mínima
        min_samples = int(2.0 * self.fs)
        if len(data_clean) < min_samples:
            return self._empty_results('too_short')
        
        # Compute PAC para cada par de bandas
        pac_results = {}
        data_reshaped = data_clean[np.newaxis, :]
        
        for band_name, pac_obj in self.pac_objects.items():
            try:
                pac_value = pac_obj.filterfit(self.fs, data_reshaped, data_reshaped)
                pac_value = float(pac_value[0, 0, 0])
                
                pac_results[band_name] = {
                    'value': pac_value,
                    'valid': True,
                    'reject_reason': None
                }
            except Exception as e:
                logger.warning(f"Error en PAC {band_name}: {e}")
                pac_results[band_name] = {
                    'value': np.nan,
                    'valid': False,
                    'reject_reason': f'computation_error: {e}'
                }
        
        # Compute POWER para cada banda
        power_results = {}
        for band_name, band_range in self.config.power_bands.items():
            try:
                power_val = self.compute_band_power(data_clean, band_range)
                power_results[band_name] = {
                    'value': power_val,
                    'valid': not np.isnan(power_val)
                }
            except Exception as e:
                logger.warning(f"Error en Power {band_name}: {e}")
                power_results[band_name] = {
                    'value': np.nan,
                    'valid': False
                }
        
        return {
            'pac': pac_results,
            'power': power_results
        }
    
    def _empty_results(self, reason: str) -> Dict:
        """Genera resultados vacíos para época rechazada"""
        return {
            'pac': {
                band_name: {'value': np.nan, 'valid': False, 'reject_reason': reason}
                for band_name in self.pac_objects.keys()
            },
            'power': {
                band_name: {'value': np.nan, 'valid': False}
                for band_name in self.config.power_bands.keys()
            }
        }


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR SLEEP-EDF (CORREGIDO: AMBOS CANALES)
# ═══════════════════════════════════════════════════════════════════════════

class ExtendedSleepEDFProcessor:
    """Procesa Sleep-EDF con PAC + POWER - CORREGIDO para ambos canales"""
    
    def __init__(self, config: Config):
        self.config = config
        self.calculator = ExtendedPACCalculator(config)
    
    def load_subject(self, psg_file: Path, hypno_file: Path) -> Optional[Dict]:
        """Carga datos de un sujeto con ambos canales"""
        try:
            raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
            annotations = mne.read_annotations(str(hypno_file))
            raw.set_annotations(annotations)
            
            available_states = set(raw.annotations.description)
            
            # CORREGIDO: Buscar ambos canales target
            available_channels = []
            for target in self.config.target_channels:
                if target in raw.ch_names:
                    available_channels.append(target)
                else:
                    # Buscar variantes
                    for ch in raw.ch_names:
                        if target.lower().replace(' ', '') in ch.lower().replace(' ', ''):
                            available_channels.append(ch)
                            break
            
            if not available_channels:
                # Fallback: cualquier canal EEG
                eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper()]
                if eeg_channels:
                    available_channels = eeg_channels[:2]
                else:
                    logger.warning(f"  No hay canales EEG disponibles")
                    return None
            
            logger.info(f"  Canales: {available_channels}")
            raw.pick_channels(available_channels)
            
            if raw.info['sfreq'] != self.config.sampling_rate:
                raw.resample(self.config.sampling_rate)
            
            return {
                'raw': raw,
                'channels': available_channels,
                'available_states': available_states
            }
        except Exception as e:
            logger.error(f"  Error cargando: {e}")
            return None
    
    def extract_epochs_for_state(self, raw, state_label: str, n_epochs: int) -> List[Dict]:
        """Extrae epochs de un estado específico para todos los canales"""
        state_annotations = [ann for ann in raw.annotations if ann['description'] == state_label]
        
        if not state_annotations:
            return []
        
        epochs_data = []
        n_channels = len(raw.ch_names)
        
        for ann in state_annotations:
            start_sample = int(ann['onset'] * raw.info['sfreq'])
            duration_samples = int(ann['duration'] * raw.info['sfreq'])
            
            if duration_samples < self.config.epoch_duration * raw.info['sfreq']:
                continue
            
            data_segment = raw.get_data(start=start_sample, stop=start_sample + duration_samples)
            
            epoch_samples = int(self.config.epoch_duration * raw.info['sfreq'])
            n_possible = data_segment.shape[1] // epoch_samples
            
            for i in range(min(n_possible, n_epochs - len(epochs_data))):
                start = i * epoch_samples
                end = start + epoch_samples
                
                epoch_dict = {
                    'data': {},
                    'channels': raw.ch_names
                }
                for ch_idx, ch_name in enumerate(raw.ch_names):
                    epoch_dict['data'][ch_name] = data_segment[ch_idx, start:end]
                
                epochs_data.append(epoch_dict)
                
                if len(epochs_data) >= n_epochs:
                    break
            
            if len(epochs_data) >= n_epochs:
                break
        
        return epochs_data
    
    def process_subject(self, subject_id: str, psg_file: Path, hypno_file: Path) -> pd.DataFrame:
        """Procesa un sujeto completo con ambos canales"""
        logger.info(f"Procesando {subject_id}...")
        
        loaded = self.load_subject(psg_file, hypno_file)
        if loaded is None:
            return pd.DataFrame()
        
        raw = loaded['raw']
        channels = loaded['channels']
        available_states = loaded['available_states']
        
        # Estados a extraer
        states_to_extract = {
            'Wake': self.config.wake_state,
            'N2': self.config.n2_state,
            'N3': self.config.n3_state,
        }
        
        if self.config.rem_state in available_states:
            states_to_extract['REM'] = self.config.rem_state
        
        all_results = []
        
        for state_name, state_label in states_to_extract.items():
            if state_label not in available_states:
                logger.debug(f"  {state_name} no disponible")
                continue
            
            epochs = self.extract_epochs_for_state(raw, state_label, self.config.n_epochs_per_state)
            logger.info(f"  {subject_id}: {len(epochs)} {state_name} epochs")
            
            if len(epochs) == 0:
                continue
            
            for epoch_idx, epoch_dict in enumerate(epochs):
                # CORREGIDO: Procesar cada canal
                for ch_name in channels:
                    if ch_name not in epoch_dict['data']:
                        continue
                    
                    epoch_data = epoch_dict['data'][ch_name]
                    metrics = self.calculator.compute_all_metrics(epoch_data)
                    
                    # Guardar PAC results
                    for band_name, pac_result in metrics['pac'].items():
                        result = {
                            'subject_id': subject_id,
                            'channel': ch_name,
                            'state': state_name,
                            'epoch_idx': epoch_idx,
                            'band_pair': band_name,
                            'value': pac_result['value'],
                            'valid': pac_result['valid'],
                            'metric_type': 'PAC'
                        }
                        all_results.append(result)
                    
                    # Guardar POWER results
                    for power_band_name, power_result in metrics['power'].items():
                        result = {
                            'subject_id': subject_id,
                            'channel': ch_name,
                            'state': state_name,
                            'epoch_idx': epoch_idx,
                            'band_pair': power_band_name,
                            'value': power_result['value'],
                            'valid': power_result['valid'],
                            'metric_type': 'POWER'
                        }
                        all_results.append(result)
        
        logger.info(f"  {subject_id}: {len(all_results)} measurements total")
        return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ANÁLISIS ESTADÍSTICO
# ═══════════════════════════════════════════════════════════════════════════

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calcula Cohen's d para dos grupos"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_holdout_metrics(holdout_pac: pd.DataFrame, 
                            holdout_power: pd.DataFrame,
                            config: Config) -> Dict:
    """
    Computa todas las métricas en holdout set
    CORREGIDO: Thresholds según RR DELTA PAC
    """
    
    results = {
        'tests': {},
        'summary': {},
        'falsification_criteria': {}
    }
    
    valid_pac = holdout_pac[holdout_pac['valid']].copy()
    valid_power = holdout_power[holdout_power['valid']].copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: PAC Wake vs N2 (Delta-Gamma) - CRITERIO F-∇-1
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🔬 TEST 1: Delta-Gamma PAC (Wake vs N2) - Criterio F-∇-1")
    print("=" * 80)
    
    delta_pac = valid_pac[valid_pac['band_pair'] == 'Delta-Gamma']
    
    # Promediar por sujeto primero (evitar pseudoreplicación)
    pac_by_subject = delta_pac.groupby(['subject_id', 'state'])['value'].mean().reset_index()
    
    pac_wake = pac_by_subject[pac_by_subject['state'] == 'Wake']['value'].values
    pac_n2 = pac_by_subject[pac_by_subject['state'] == 'N2']['value'].values
    
    pac_wake = pac_wake[~np.isnan(pac_wake)]
    pac_n2 = pac_n2[~np.isnan(pac_n2)]
    
    if len(pac_wake) > 1 and len(pac_n2) > 1:
        pac_d_wake_n2 = compute_cohens_d(pac_wake, pac_n2)
        _, pac_p_wake_n2 = stats.ttest_ind(pac_wake, pac_n2, alternative='greater')
        
        print(f"  N sujetos Wake: {len(pac_wake)}")
        print(f"  N sujetos N2: {len(pac_n2)}")
        print(f"  PAC Wake: {np.mean(pac_wake):.4f} ± {np.std(pac_wake, ddof=1):.4f}")
        print(f"  PAC N2:   {np.mean(pac_n2):.4f} ± {np.std(pac_n2, ddof=1):.4f}")
        print(f"  Cohen's d: {pac_d_wake_n2:.3f} (threshold: {config.cohens_d_threshold})")
        print(f"  p-value:   {pac_p_wake_n2:.6f} (threshold: {config.p_value_threshold})")
        
        # Evaluación según RR
        d_pass = pac_d_wake_n2 >= config.cohens_d_threshold
        p_pass = pac_p_wake_n2 < config.p_value_threshold
        
        if d_pass and p_pass:
            print(f"\n  ✅ CRITERIO F-∇-1: PASADO (d={pac_d_wake_n2:.2f} ≥ {config.cohens_d_threshold}, p={pac_p_wake_n2:.4f} < {config.p_value_threshold})")
        else:
            print(f"\n  ❌ CRITERIO F-∇-1: FALLIDO")
            if not d_pass:
                print(f"     Cohen's d = {pac_d_wake_n2:.2f} < {config.cohens_d_threshold}")
            if not p_pass:
                print(f"     p-value = {pac_p_wake_n2:.4f} ≥ {config.p_value_threshold}")
        
        results['tests']['pac_wake_vs_n2'] = {
            'cohens_d': float(pac_d_wake_n2),
            'p_value': float(pac_p_wake_n2),
            'wake_mean': float(np.mean(pac_wake)),
            'wake_std': float(np.std(pac_wake, ddof=1)),
            'n2_mean': float(np.mean(pac_n2)),
            'n2_std': float(np.std(pac_n2, ddof=1)),
            'n_wake': len(pac_wake),
            'n_n2': len(pac_n2),
            'd_threshold_passed': d_pass,
            'p_threshold_passed': p_pass
        }
        
        results['falsification_criteria']['F_nabla_1_cohens_d'] = float(pac_d_wake_n2)
        results['falsification_criteria']['F_nabla_1_pvalue'] = float(pac_p_wake_n2)
    else:
        print("  ⚠️  Datos insuficientes para Test 1")
        results['tests']['pac_wake_vs_n2'] = {'error': 'insufficient_data'}
        results['falsification_criteria']['F_nabla_1_cohens_d'] = 0.0
        results['falsification_criteria']['F_nabla_1_pvalue'] = 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: PAC vs POWER - CRITERIO F-∇-2
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🔥 TEST 2: PAC vs POWER Discrimination - Criterio F-∇-2")
    print("=" * 80)
    
    delta_power = valid_power[valid_power['band_pair'] == 'Delta']
    
    # Promediar por sujeto
    power_by_subject = delta_power.groupby(['subject_id', 'state'])['value'].mean().reset_index()
    
    power_wake = power_by_subject[power_by_subject['state'] == 'Wake']['value'].values
    power_n2 = power_by_subject[power_by_subject['state'] == 'N2']['value'].values
    
    power_wake = power_wake[~np.isnan(power_wake)]
    power_n2 = power_n2[~np.isnan(power_n2)]
    
    if len(power_wake) > 1 and len(power_n2) > 1:
        power_d_wake_n2 = compute_cohens_d(power_wake, power_n2)
        
        # Calcular ventaja PAC vs Power
        pac_d = results['falsification_criteria'].get('F_nabla_1_cohens_d', 0.0)
        pac_advantage = pac_d - power_d_wake_n2
        
        print(f"  Delta POWER Cohen's d (Wake vs N2): {power_d_wake_n2:.3f}")
        print(f"  Delta-Gamma PAC Cohen's d:          {pac_d:.3f}")
        print(f"  PAC advantage:                      {pac_advantage:.3f} (threshold: {config.pac_power_advantage_threshold})")
        
        advantage_pass = pac_advantage > config.pac_power_advantage_threshold
        
        if advantage_pass:
            print(f"\n  ✅ CRITERIO F-∇-2: PASADO - PAC NO es epifenómeno de Power")
        else:
            print(f"\n  ❌ CRITERIO F-∇-2: FALLIDO - PAC podría ser epifenómeno de Power")
        
        results['tests']['pac_vs_power'] = {
            'power_cohens_d': float(power_d_wake_n2),
            'pac_cohens_d': float(pac_d),
            'pac_advantage': float(pac_advantage),
            'threshold': config.pac_power_advantage_threshold,
            'passed': advantage_pass
        }
        
        results['falsification_criteria']['F_nabla_2_pac_advantage'] = float(pac_advantage)
    else:
        print("  ⚠️  Datos insuficientes para Test 2")
        results['tests']['pac_vs_power'] = {'error': 'insufficient_data'}
        results['falsification_criteria']['F_nabla_2_pac_advantage'] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: N3 PARADOX
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🚨 TEST 3: N3 PARADOX (Alto Power, Bajo PAC)")
    print("=" * 80)
    
    if 'N3' in valid_pac['state'].unique():
        # PAC en N3
        pac_n3_data = pac_by_subject[pac_by_subject['state'] == 'N3']['value'].values
        pac_n3_data = pac_n3_data[~np.isnan(pac_n3_data)]
        
        # Power en N3
        power_n3_data = power_by_subject[power_by_subject['state'] == 'N3']['value'].values
        power_n3_data = power_n3_data[~np.isnan(power_n3_data)]
        
        if len(pac_n3_data) > 0 and len(pac_wake) > 0:
            print("\n  📊 PAC por estado (Delta-Gamma):")
            print(f"     Wake: {np.mean(pac_wake):.4f} ± {np.std(pac_wake, ddof=1):.4f}")
            print(f"     N2:   {np.mean(pac_n2):.4f} ± {np.std(pac_n2, ddof=1):.4f}")
            print(f"     N3:   {np.mean(pac_n3_data):.4f} ± {np.std(pac_n3_data, ddof=1):.4f}")
            
            # Verificar gradiente PAC: Wake > N2 > N3
            pac_gradient_correct = (np.mean(pac_wake) > np.mean(pac_n2) > np.mean(pac_n3_data))
            
            results['tests']['n3_paradox'] = {
                'pac_wake': float(np.mean(pac_wake)),
                'pac_n2': float(np.mean(pac_n2)),
                'pac_n3': float(np.mean(pac_n3_data)),
                'pac_gradient_correct': pac_gradient_correct
            }
            
            if len(power_n3_data) > 0 and len(power_wake) > 0:
                print("\n  📊 POWER por estado (Delta):")
                print(f"     Wake: {np.mean(power_wake):.6f}")
                print(f"     N2:   {np.mean(power_n2):.6f}")
                print(f"     N3:   {np.mean(power_n3_data):.6f}")
                
                # Verificar gradiente Power: N3 > N2 > Wake (opuesto a PAC)
                power_gradient_opposite = (np.mean(power_n3_data) > np.mean(power_n2) > np.mean(power_wake))
                
                results['tests']['n3_paradox']['power_wake'] = float(np.mean(power_wake))
                results['tests']['n3_paradox']['power_n2'] = float(np.mean(power_n2))
                results['tests']['n3_paradox']['power_n3'] = float(np.mean(power_n3_data))
                results['tests']['n3_paradox']['power_gradient_opposite'] = power_gradient_opposite
                
                # N3 Paradox confirmado si:
                # PAC: Wake > N2 > N3 (consciencia decrece)
                # Power: N3 > N2 > Wake (power aumenta)
                paradox_confirmed = pac_gradient_correct and power_gradient_opposite
                
                results['tests']['n3_paradox']['paradox_confirmed'] = paradox_confirmed
                
                if paradox_confirmed:
                    print("\n  ✅ N3 PARADOX CONFIRMADO:")
                    print("     PAC decrece (Wake > N2 > N3) = pérdida de convergencia temporal")
                    print("     Power aumenta (N3 > N2 > Wake) = más delta, menos consciencia")
                    print("     → PAC captura consciencia, no cantidad de delta")
                elif pac_gradient_correct:
                    print("\n  ⚠️  Gradiente PAC correcto pero Power no es opuesto")
                else:
                    print("\n  ❌ Gradiente PAC incorrecto - predicción AFH no soportada")
            
            results['falsification_criteria']['n3_paradox_confirmed'] = results['tests']['n3_paradox'].get('paradox_confirmed', False)
        else:
            print("  ⚠️  Datos N3 insuficientes")
            results['tests']['n3_paradox'] = {'error': 'insufficient_n3_data'}
    else:
        print("  ⚠️  Estado N3 no disponible en holdout")
        results['tests']['n3_paradox'] = {'error': 'n3_not_available'}
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: GRADIENTE DE FRECUENCIA (Nivel 2)
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📈 TEST 4: Gradiente de Frecuencia (Nivel 2 Validación)")
    print("=" * 80)
    
    band_results = {}
    for band_pair in config.band_pairs:
        band_name = band_pair.name
        band_pac = valid_pac[valid_pac['band_pair'] == band_name]
        
        if len(band_pac) == 0:
            continue
        
        band_by_subject = band_pac.groupby(['subject_id', 'state'])['value'].mean().reset_index()
        
        b_wake = band_by_subject[band_by_subject['state'] == 'Wake']['value'].values
        b_n2 = band_by_subject[band_by_subject['state'] == 'N2']['value'].values
        
        b_wake = b_wake[~np.isnan(b_wake)]
        b_n2 = b_n2[~np.isnan(b_n2)]
        
        if len(b_wake) > 1 and len(b_n2) > 1:
            d = compute_cohens_d(b_wake, b_n2)
            band_results[band_name] = d
            print(f"  {band_name}: Cohen's d = {d:.3f}")
    
    if band_results:
        # Ordenar por Cohen's d
        sorted_bands = sorted(band_results.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n  Ranking (mayor discriminación primero):")
        for i, (band, d) in enumerate(sorted_bands, 1):
            marker = "✓" if band == "Delta-Gamma" else " "
            print(f"    {i}. {band}: d = {d:.3f} {marker}")
        
        # Verificar si Delta-Gamma está en top 2
        top_bands = [b[0] for b in sorted_bands[:2]]
        delta_in_top = "Delta-Gamma" in top_bands
        
        results['tests']['frequency_gradient'] = {
            'band_cohens_d': band_results,
            'ranking': [b[0] for b in sorted_bands],
            'delta_gamma_in_top2': delta_in_top
        }
        
        if delta_in_top:
            print(f"\n  ✅ Delta-Gamma en top 2 - soporta ventana temporal predicha (≤4 Hz)")
        else:
            print(f"\n  ⚠️  Delta-Gamma no en top 2 - ventana temporal podría ser más amplia")
    
    # ═══════════════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📋 RESUMEN DE CRITERIOS DE FALSACIÓN")
    print("=" * 80)
    
    f1_d = results['falsification_criteria'].get('F_nabla_1_cohens_d', 0)
    f1_p = results['falsification_criteria'].get('F_nabla_1_pvalue', 1)
    f2_adv = results['falsification_criteria'].get('F_nabla_2_pac_advantage', 0)
    
    f1_passed = (f1_d >= config.cohens_d_threshold) and (f1_p < config.p_value_threshold)
    f2_passed = f2_adv > config.pac_power_advantage_threshold
    
    print(f"\n  F-∇-1 (PAC discrimina Wake vs N2):")
    print(f"    Cohen's d = {f1_d:.3f} {'≥' if f1_d >= config.cohens_d_threshold else '<'} {config.cohens_d_threshold} → {'✅' if f1_d >= config.cohens_d_threshold else '❌'}")
    print(f"    p-value = {f1_p:.4f} {'<' if f1_p < config.p_value_threshold else '≥'} {config.p_value_threshold} → {'✅' if f1_p < config.p_value_threshold else '❌'}")
    print(f"    RESULTADO: {'✅ PASADO' if f1_passed else '❌ FALLIDO'}")
    
    print(f"\n  F-∇-2 (PAC ≠ epifenómeno de Power):")
    print(f"    PAC advantage = {f2_adv:.3f} {'>' if f2_adv > config.pac_power_advantage_threshold else '≤'} {config.pac_power_advantage_threshold} → {'✅' if f2_passed else '❌'}")
    print(f"    RESULTADO: {'✅ PASADO' if f2_passed else '❌ FALLIDO'}")
    
    # Veredicto final
    n_passed = sum([f1_passed, f2_passed])
    n_failed = 2 - n_passed
    
    results['summary'] = {
        'F_nabla_1_passed': f1_passed,
        'F_nabla_2_passed': f2_passed,
        'criteria_passed': n_passed,
        'criteria_failed': n_failed
    }
    
    print("\n" + "=" * 80)
    if n_failed >= 2:
        print("❌ VEREDICTO: PLIEGUE AUTOPSÍQUICO FALSIFICADO")
        print("   (≥2 criterios de falsación cumplidos)")
        results['summary']['verdict'] = 'FALSIFIED'
    elif n_passed == 2:
        print("✅ VEREDICTO: PLIEGUE AUTOPSÍQUICO SOPORTADO (Niveles 1-2)")
        print("   (Todos los criterios pasados)")
        results['summary']['verdict'] = 'SUPPORTED'
    else:
        print("⚠️  VEREDICTO: EVIDENCIA MIXTA")
        print(f"   ({n_passed}/2 criterios pasados)")
        results['summary']['verdict'] = 'MIXED'
    print("=" * 80)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS PRINCIPAL CON ECLIPSE v2.0
# ═══════════════════════════════════════════════════════════════════════════

def run_extended_analysis_with_eclipse(config: Config):
    """Ejecuta análisis completo con ECLIPSE v2.0 integration"""
    
    print("\n" + "=" * 80)
    print("🔬 NIVEL 2 EXTENDED + ECLIPSE v2.0 ANALYSIS")
    print("   Autopsychic Fold Hypothesis - Validación PAC δ→γ")
    print("=" * 80)
    print(f"\n📋 Parámetros según RR DELTA PAC:")
    print(f"   Banda delta (fase): {config.band_pairs[0].phase_band} Hz")
    print(f"   Banda gamma (amplitud): {config.band_pairs[0].amp_band} Hz")
    print(f"   Threshold Cohen's d: ≥ {config.cohens_d_threshold}")
    print(f"   Threshold p-value: < {config.p_value_threshold}")
    print(f"   Canales: {config.target_channels}")
    
    if not ECLIPSE_AVAILABLE:
        print("\n❌ ERROR: ECLIPSE v2.0 not available")
        return
    
    processor = ExtendedSleepEDFProcessor(config)
    
    # Cargar sujetos
    psg_files = sorted(config.data_dir.glob('*PSG.edf'))
    if config.n_subjects:
        psg_files = psg_files[:config.n_subjects]
    
    print(f"\n📁 Dataset: {config.data_dir}")
    print(f"   PSG files: {len(list(config.data_dir.glob('*PSG.edf')))}")
    print(f"   Procesando: {len(psg_files)} sujetos")
    
    if len(psg_files) > 50:
        estimated_time_min = len(psg_files) * 2.0  # ~2 min por sujeto con 2 canales
        print(f"   ⏱️  Tiempo estimado: ~{estimated_time_min/60:.1f} horas")
    
    start_time = time.time()
    
    all_data = []
    subject_ids = []
    
    # Checkpoint
    checkpoint_frequency = 10
    checkpoint_file = config.output_dir / 'checkpoint_data.pkl'
    
    for idx, psg_file in enumerate(psg_files, 1):
        subject_id = psg_file.stem.replace('-PSG', '')
        subject_ids.append(subject_id)
        
        # Sleep-EDF naming: SC4001E0 -> SC4001EC
        hypno_id = subject_id[:-1] + 'C'
        hypno_file = psg_file.parent / f"{hypno_id}-Hypnogram.edf"
        
        if not hypno_file.exists():
            logger.warning(f"  [{idx}/{len(psg_files)}] {subject_id}: Hypnogram no encontrado")
            continue
        
        progress_pct = (idx / len(psg_files)) * 100
        print(f"[{idx}/{len(psg_files)}] ({progress_pct:.1f}%) Processing {subject_id}...")
        
        df = processor.process_subject(subject_id, psg_file, hypno_file)
        
        if not df.empty:
            all_data.append(df)
        
        # Checkpoint
        if idx % checkpoint_frequency == 0 and all_data:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_df = pd.concat(all_data, ignore_index=True)
            checkpoint_df.to_pickle(checkpoint_file)
            logger.info(f"  Checkpoint: {len(all_data)} subjects")
    
    if not all_data:
        print(f"\n❌ No se procesaron datos válidos")
        return
    
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    
    full_df = pd.concat(all_data, ignore_index=True)
    n_subjects_processed = len(set(full_df['subject_id']))
    
    print(f"\n📊 Procesamiento completo:")
    print(f"   Sujetos: {n_subjects_processed}")
    print(f"   Measurements: {len(full_df)}")
    print(f"   Tiempo: {elapsed_min:.1f} min ({elapsed_min/60:.2f} hrs)")
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE STAGE 1: SPLIT
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🔒 ECLIPSE STAGE 1: IRREVERSIBLE SUBJECT SPLIT")
    print("=" * 80)
    
    eclipse_config = EclipseConfig(
        project_name=config.project_name,
        researcher=config.researcher,
        sacred_seed=config.sacred_seed,
        development_ratio=0.7,
        holdout_ratio=0.3,
        output_dir=str(config.output_dir / 'eclipse_v2')
    )
    
    eclipse = EclipseFramework(eclipse_config)
    
    processed_subject_ids = sorted(list(set(full_df['subject_id'])))
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(processed_subject_ids)
    
    print(f"   Development: {len(dev_subjects)} subjects (70%)")
    print(f"   Holdout: {len(holdout_subjects)} subjects (30%)")
    
    dev_df = full_df[full_df['subject_id'].isin(dev_subjects)].copy()
    holdout_df = full_df[full_df['subject_id'].isin(holdout_subjects)].copy()
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE STAGE 2: CRITERIOS PRE-REGISTRADOS
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📋 ECLIPSE STAGE 2: PRE-REGISTERED CRITERIA (según RR DELTA PAC)")
    print("=" * 80)
    
    criteria = [
        FalsificationCriteria(
            name="delta_gamma_pac_cohens_d",
            threshold=config.cohens_d_threshold,  # 1.0 según RR
            comparison=">=",
            description=f"Delta-Gamma PAC Cohen's d ≥ {config.cohens_d_threshold}",
            is_required=True
        ),
        FalsificationCriteria(
            name="delta_gamma_pac_pvalue",
            threshold=config.p_value_threshold,  # 0.001 según RR
            comparison="<",
            description=f"Delta-Gamma PAC p-value < {config.p_value_threshold}",
            is_required=True
        ),
        FalsificationCriteria(
            name="pac_vs_power_advantage",
            threshold=config.pac_power_advantage_threshold,  # 0.3
            comparison=">",
            description=f"PAC advantage > {config.pac_power_advantage_threshold} (no epifenómeno)",
            is_required=True
        ),
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE STAGE 4: SINGLE-SHOT HOLDOUT VALIDATION
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🎯 ECLIPSE STAGE 4: SINGLE-SHOT HOLDOUT VALIDATION")
    print("=" * 80)
    print("⚠️  CRÍTICO: Esta es la ÚNICA oportunidad de testear hipótesis")
    
    holdout_pac = holdout_df[holdout_df['metric_type'] == 'PAC']
    holdout_power = holdout_df[holdout_df['metric_type'] == 'POWER']
    
    validation_results = compute_holdout_metrics(holdout_pac, holdout_power, config)
    
    # Wrap para ECLIPSE
    eclipse_validation = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_df,
        final_model={},
        validation_function=lambda model, data: validation_results
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE STAGE 5: FINAL ASSESSMENT
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📊 ECLIPSE STAGE 5: FINAL ASSESSMENT")
    print("=" * 80)
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results={},
        validation_results=eclipse_validation,
        generate_reports=True,
        compute_integrity=True
    )
    
    # Guardar resultados
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    pac_df = full_df[full_df['metric_type'] == 'PAC']
    power_df = full_df[full_df['metric_type'] == 'POWER']
    
    pac_df.to_csv(config.output_dir / 'nivel2_pac_results.csv', index=False)
    power_df.to_csv(config.output_dir / 'nivel2_power_results.csv', index=False)
    
    with open(config.output_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados guardados en: {config.output_dir}")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("🔬 RESUMEN FINAL - VALIDACIÓN PLIEGUE AUTOPSÍQUICO")
    print("=" * 80)
    
    verdict = validation_results.get('summary', {}).get('verdict', 'UNKNOWN')
    
    if verdict == 'SUPPORTED':
        print("""
  ✅ RESULTADO: PLIEGUE AUTOPSÍQUICO SOPORTADO (Niveles 1-2)
  
  Evidencia:
  • PAC δ→γ discrimina Wake vs N2 con efecto grande (d ≥ 1.0)
  • PAC no es epifenómeno de delta power
  • Gradiente de consciencia capturado: Wake > N2 > N3
  
  Siguiente paso: RR DELTA PAC Stage 2 en CAP Sleep Database (N=108)
        """)
    elif verdict == 'FALSIFIED':
        print("""
  ❌ RESULTADO: PLIEGUE AUTOPSÍQUICO FALSIFICADO
  
  Criterios fallidos:
  • F-∇-1: PAC no discrimina significativamente (d < 1.0 o p ≥ 0.001)
  • F-∇-2: PAC podría ser epifenómeno de delta power
  
  Implicación: Buscar operacionalizaciones alternativas (ΔT, TE, GC, DCM)
        """)
    else:
        print(f"""
  ⚠️  RESULTADO: EVIDENCIA MIXTA
  
  Algunos criterios pasados, otros fallidos.
  Revisar validation_results.json para detalles.
        """)
    
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NIVEL 2 EXTENDED - Autopsychic Fold PAC Validation (CORRECTED)"
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
        default='./nivel2_pac_validation_results',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--n-subjects',
        type=int,
        help='Número de sujetos (default: TODOS)',
        default=None
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sacred_seed=args.seed,
        n_subjects=args.n_subjects
    )
    
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: Directorio no encontrado: {config.data_dir}")
        sys.exit(1)
    
    try:
        run_extended_analysis_with_eclipse(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Análisis interrumpido")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()