#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIVEL 2 EXTENDED: PAC Band Specificity + N3 + Power vs PAC
WITH FULL ECLIPSE v2.0 INTEGRATION
Autopsychic Fold Hypothesis
═══════════════════════════════════════════════════════════════════════════════

HIPÓTESIS EXTENDIDAS:
    H1: PAC discrimina consciencia, no cantidad de delta
    H2: Delta-gamma PAC ≠ Delta POWER
    H3: N3 tiene BAJO PAC a pesar de ALTO delta power
    
ECLIPSE v2.0 INTEGRATION:
    ✅ Stage 1: Irreversible 70/30 subject split
    ✅ Stage 2: Pre-registered falsification criteria
    ✅ Stage 3: Development phase (optional CV if needed)
    ✅ Stage 4: Single-shot holdout validation
    ✅ Stage 5: Final assessment with EIS, STDS, integrity verification
    
TESTS CRÍTICOS:
    Test 1: PAC Wake > N2 > N3 (a pesar de power N3 > N2 > Wake)
    Test 2: Delta-gamma PAC discrimina MEJOR que Delta POWER
    Test 3: REM PAC ≈ Wake PAC (si consciencia ON en ambos)

Author: Camilo Sjöberg Tala
Date: 2025-10-22
Version: NIVEL_2_EXTENDED_ECLIPSE_v3.0

EJECUCIÓN:
    python NIVEL2_EXTENDED_ECLIPSE_FULL.py          # Todos los sujetos (default)
    python NIVEL2_EXTENDED_ECLIPSE_FULL.py --n-subjects 10  # Solo 10 sujetos
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
# CONFIGURACIÓN EXTENDIDA
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
    """Configuración NIVEL 2 EXTENDED con ECLIPSE"""
    data_dir: Path
    output_dir: Path
    
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    notch_freq: float = 50.0
    
    band_pairs: List[BandPair] = field(default_factory=lambda: [
        BandPair(
            name="Delta-Gamma",
            phase_band=(1.0, 4.0),
            amp_band=(30.0, 45.0),
            description="Slow wave coupling"
        ),
        BandPair(
            name="Theta-Gamma",
            phase_band=(4.0, 8.0),
            amp_band=(30.0, 45.0),
            description="AFH prediction"
        ),
        BandPair(
            name="Alpha-Gamma",
            phase_band=(8.0, 12.0),
            amp_band=(30.0, 45.0),
            description="Attention coupling"
        ),
        BandPair(
            name="Beta-Gamma",
            phase_band=(13.0, 30.0),
            amp_band=(30.0, 45.0),
            description="Fast cortical coupling"
        ),
    ])
    
    power_bands: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 12.0),
        'Beta': (13.0, 30.0),
        'Gamma': (30.0, 45.0),
    })
    
    epoch_duration: float = 30.0
    n_epochs_per_state: int = 50  # Reducido de 100 a 50 para procesar 153 sujetos más rápido
    
    wake_state: str = 'Sleep stage W'
    n1_state: str = 'Sleep stage 1'
    n2_state: str = 'Sleep stage 2'
    n3_state: str = 'Sleep stage 3'
    rem_state: str = 'Sleep stage R'
    
    sacred_seed: int = 42
    n_subjects: Optional[int] = None
    
    # ECLIPSE configuration
    researcher: str = "Camilo Sjöberg Tala"
    project_name: str = "NIVEL2_EXTENDED_PAC_N3_POWER"
    
    def __post_init__(self):
        """Ajustar nombre del proyecto según cantidad de sujetos"""
        if self.n_subjects is None:
            self.project_name = f"{self.project_name}_FULL_DATASET"
        else:
            self.project_name = f"{self.project_name}_{self.n_subjects}subj"


# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA PAC + POWER
# ═══════════════════════════════════════════════════════════════════════════

class ExtendedPACCalculator:
    """Calcula PAC Y POWER para múltiples bandas"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fs = config.sampling_rate
        
        self.pac_objects = {}
        for band_pair in config.band_pairs:
            self.pac_objects[band_pair.name] = Pac(
                idpac=(1, 0, 0),
                f_pha=list(band_pair.phase_band),
                f_amp=list(band_pair.amp_band),
                dcomplex='wavelet',
                width=7
            )
            logger.info(f"PAC object created: {band_pair.name}")
        
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa señal EEG"""
        sos = signal.butter(
            4,
            [self.config.lowcut, self.config.highcut],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        filtered = signal.sosfiltfilt(sos, data)
        
        b_notch, a_notch = signal.iirnotch(
            self.config.notch_freq,
            Q=30,
            fs=self.fs
        )
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
        
        return filtered
    
    def has_artifact(self, data: np.ndarray) -> bool:
        """Detecta artefactos"""
        peak_to_peak = np.ptp(data)
        return peak_to_peak > 8.0
    
    def compute_band_power(self, data: np.ndarray, band: Tuple[float, float]) -> float:
        """Calcula power espectral en una banda específica"""
        freqs, psd = signal.welch(data, fs=self.fs, nperseg=min(len(data), 256))
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        power = np.trapz(psd[idx_band], freqs[idx_band])
        return float(power)
    
    def compute_all_metrics(self, data: np.ndarray) -> Dict[str, Dict]:
        """Calcula PAC + POWER"""
        data_clean = self.preprocess(data)
        
        if self.has_artifact(data_clean):
            return {
                'pac': {
                    band_name: {
                        'pac': np.nan,
                        'valid': False,
                        'reject_reason': 'artifact'
                    }
                    for band_name in self.pac_objects.keys()
                },
                'power': {
                    band_name: np.nan
                    for band_name in self.config.power_bands.keys()
                }
            }
        
        min_samples = int(2.0 * self.fs)
        if len(data_clean) < min_samples:
            return {
                'pac': {
                    band_name: {
                        'pac': np.nan,
                        'valid': False,
                        'reject_reason': 'too_short'
                    }
                    for band_name in self.pac_objects.keys()
                },
                'power': {
                    band_name: np.nan
                    for band_name in self.config.power_bands.keys()
                }
            }
        
        # Compute PAC
        pac_results = {}
        data_reshaped = data_clean[np.newaxis, :]
        
        for band_name, pac_obj in self.pac_objects.items():
            try:
                pac_value = pac_obj.filterfit(self.fs, data_reshaped, data_reshaped)
                pac_value = float(pac_value[0, 0, 0])
                
                pac_results[band_name] = {
                    'pac': pac_value,
                    'valid': True,
                    'reject_reason': None
                }
            except Exception as e:
                logger.warning(f"Error en PAC {band_name}: {e}")
                pac_results[band_name] = {
                    'pac': np.nan,
                    'valid': False,
                    'reject_reason': f'computation_error: {e}'
                }
        
        # Compute POWER
        power_results = {}
        for band_name, band_range in self.config.power_bands.items():
            try:
                power_val = self.compute_band_power(data_clean, band_range)
                power_results[band_name] = power_val
            except Exception as e:
                logger.warning(f"Error en Power {band_name}: {e}")
                power_results[band_name] = np.nan
        
        return {
            'pac': pac_results,
            'power': power_results
        }


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR SLEEP-EDF
# ═══════════════════════════════════════════════════════════════════════════

class ExtendedSleepEDFProcessor:
    """Procesa Sleep-EDF con PAC + POWER para múltiples estados"""
    
    def __init__(self, config: Config):
        self.config = config
        self.calculator = ExtendedPACCalculator(config)
    
    def load_subject(self, psg_file: Path, hypno_file: Path) -> Optional[Dict]:
        """Carga datos de un sujeto"""
        try:
            raw = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
            annotations = mne.read_annotations(str(hypno_file))
            raw.set_annotations(annotations)
            
            available_states = set(raw.annotations.description)
            logger.info(f"  Estados disponibles: {available_states}")
            
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper()]
            if not eeg_channels:
                logger.warning(f"  No hay canales EEG")
                return None
            
            channel = 'EEG Pz-Oz' if 'EEG Pz-Oz' in eeg_channels else eeg_channels[0]
            logger.info(f"  Usando canal: {channel}")
            raw.pick_channels([channel])
            
            if raw.info['sfreq'] != self.config.sampling_rate:
                raw.resample(self.config.sampling_rate)
            
            return {
                'raw': raw,
                'channel': channel,
                'available_states': available_states
            }
        except Exception as e:
            logger.error(f"  Error cargando: {e}")
            return None
    
    def extract_epochs_for_state(self, raw, state_label: str, n_epochs: int) -> List[np.ndarray]:
        """Extrae epochs de un estado específico"""
        state_annotations = [ann for ann in raw.annotations if ann['description'] == state_label]
        
        if not state_annotations:
            return []
        
        epochs_data = []
        for ann in state_annotations:
            start_sample = int(ann['onset'] * raw.info['sfreq'])
            duration_samples = int(ann['duration'] * raw.info['sfreq'])
            
            if duration_samples < self.config.epoch_duration * raw.info['sfreq']:
                continue
            
            data_segment = raw.get_data(start=start_sample, stop=start_sample + duration_samples)[0]
            
            epoch_samples = int(self.config.epoch_duration * raw.info['sfreq'])
            n_possible = len(data_segment) // epoch_samples
            
            for i in range(min(n_possible, n_epochs - len(epochs_data))):
                start = i * epoch_samples
                end = start + epoch_samples
                epochs_data.append(data_segment[start:end])
                
                if len(epochs_data) >= n_epochs:
                    break
            
            if len(epochs_data) >= n_epochs:
                break
        
        return epochs_data
    
    def process_subject(self, subject_id: str, psg_file: Path, hypno_file: Path) -> pd.DataFrame:
        """Procesa un sujeto completo"""
        logger.info(f"Procesando {subject_id}...")
        
        loaded = self.load_subject(psg_file, hypno_file)
        if loaded is None:
            return pd.DataFrame()
        
        raw = loaded['raw']
        available_states = loaded['available_states']
        
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
                logger.info(f"  {state_name} no disponible")
                continue
            
            epochs = self.extract_epochs_for_state(raw, state_label, self.config.n_epochs_per_state)
            logger.info(f"  {subject_id}: {len(epochs)} {state_name} epochs")
            
            if len(epochs) == 0:
                continue
            
            for epoch_idx, epoch_data in enumerate(epochs):
                metrics = self.calculator.compute_all_metrics(epoch_data)
                
                # Guardar PAC results
                for band_name, pac_result in metrics['pac'].items():
                    result = {
                        'subject_id': subject_id,
                        'state': state_name,
                        'epoch_idx': epoch_idx,
                        'band_pair': band_name,
                        'pac': pac_result['pac'],
                        'valid': pac_result['valid'],
                        'metric_type': 'PAC'
                    }
                    all_results.append(result)
                
                # Guardar POWER results (una vez por epoch, no por cada band_pair)
                for power_band_name, power_val in metrics['power'].items():
                    result = {
                        'subject_id': subject_id,
                        'state': state_name,
                        'epoch_idx': epoch_idx,
                        'band_pair': power_band_name,
                        'pac': power_val,  # usando 'pac' column para consistencia
                        'valid': not np.isnan(power_val),
                        'metric_type': 'POWER'
                    }
                    all_results.append(result)
        
        logger.info(f"  {subject_id}: {len(all_results)} measurements")
        return pd.DataFrame(all_results)


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS CON ECLIPSE v2.0 INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

def run_extended_analysis_with_eclipse(config: Config):
    """Ejecuta análisis completo con ECLIPSE v2.0 integration"""
    
    print("\n" + "=" * 80)
    print("🔬 NIVEL 2 EXTENDED + ECLIPSE v2.0 ANALYSIS")
    print("=" * 80)
    
    if not ECLIPSE_AVAILABLE:
        print("\n❌ ERROR: ECLIPSE v2.0 not available")
        print("   Cannot proceed without ECLIPSE integration")
        print("   Please ensure eclipse_v2.py is in path")
        return
    
    processor = ExtendedSleepEDFProcessor(config)
    
    # Cargar sujetos
    psg_files = sorted(config.data_dir.glob('*PSG.edf'))
    if config.n_subjects:
        psg_files = psg_files[:config.n_subjects]
    
    print(f"\n📁 Diagnóstico de archivos:")
    print(f"   Directorio: {config.data_dir}")
    print(f"   PSG files encontrados: {len(list(config.data_dir.glob('*PSG.edf')))}")
    print(f"   Hypnogram files encontrados: {len(list(config.data_dir.glob('*Hypnogram.edf')))}")
    
    # Mostrar primeros 3 pares de archivos
    print(f"\n   Primeros archivos encontrados:")
    for psg in list(config.data_dir.glob('*PSG.edf'))[:3]:
        print(f"     PSG: {psg.name}")
    for hypno in list(config.data_dir.glob('*Hypnogram.edf'))[:3]:
        print(f"     Hypnogram: {hypno.name}")
    
    print(f"\n🔧 Procesando {len(psg_files)} sujetos...")
    
    if len(psg_files) > 50:
        estimated_time_min = len(psg_files) * 1.5  # ~1.5 min por sujeto
        estimated_time_hrs = estimated_time_min / 60
        print(f"⏱️  Tiempo estimado: ~{estimated_time_hrs:.1f} horas")
        print(f"   (checkpoints guardados cada 10 sujetos)")
    
    import time
    start_time = time.time()
    
    all_data = []
    subject_ids = []
    
    # Checkpoint para guardar progreso cada N sujetos
    checkpoint_frequency = 10
    checkpoint_file = config.output_dir / 'checkpoint_data.pkl'
    
    for idx, psg_file in enumerate(psg_files, 1):
        subject_id = psg_file.stem.replace('-PSG', '')
        subject_ids.append(subject_id)
        
        # Sleep-EDF usa formato: SC4001E0-PSG.edf y SC4001EC-Hypnogram.edf
        # Reemplazar último dígito por 'C' para hypnogram
        hypno_id = subject_id[:-1] + 'C'  # SC4001E0 -> SC4001EC
        hypno_file = psg_file.parent / f"{hypno_id}-Hypnogram.edf"
        
        if not hypno_file.exists():
            logger.warning(f"  [{idx}/{len(psg_files)}] {subject_id}: Hypnogram no encontrado (buscando {hypno_file.name})")
            continue
        
        # Progress indicator
        progress_pct = (idx / len(psg_files)) * 100
        print(f"[{idx}/{len(psg_files)}] ({progress_pct:.1f}%) Processing {subject_id}...")
        
        logger.info(f"[{idx}/{len(psg_files)}] {subject_id}")
        df = processor.process_subject(subject_id, psg_file, hypno_file)
        
        if not df.empty:
            all_data.append(df)
        
        # Guardar checkpoint cada N sujetos
        if idx % checkpoint_frequency == 0 and all_data:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_df = pd.concat(all_data, ignore_index=True)
            checkpoint_df.to_pickle(checkpoint_file)
            logger.info(f"  Checkpoint saved: {len(all_data)} subjects processed")
    
    if not all_data:
        print(f"\n❌ No se procesaron datos válidos")
        print(f"   Total PSG files encontrados: {len(psg_files)}")
        print(f"   Sujetos con hypnogram: 0")
        print(f"\n💡 SOLUCIÓN:")
        print(f"   1. Verifica que los archivos hypnogram existan en: {config.data_dir}")
        print(f"   2. El formato esperado es: SC4001EC-Hypnogram.edf (nota la 'C' al final)")
        print(f"   3. Si tienes otro formato, ajusta el código en la línea de hypno_id")
        return
    
    # Calcular tiempo total
    elapsed_time = time.time() - start_time
    elapsed_min = elapsed_time / 60
    elapsed_hrs = elapsed_min / 60
    
    full_df = pd.concat(all_data, ignore_index=True)
    
    n_subjects_processed = len(set(full_df['subject_id']))
    
    print(f"\n📊 Datos procesados exitosamente:")
    print(f"  Total PSG files encontrados: {len(psg_files)}")
    print(f"  Sujetos procesados con éxito: {n_subjects_processed}")
    print(f"  Total measurements: {len(full_df)}")
    print(f"  Estados únicos: {sorted(full_df['state'].unique())}")
    print(f"  Bandas PAC: {sorted(full_df[full_df['metric_type']=='PAC']['band_pair'].unique())}")
    print(f"  Bandas POWER: {sorted(full_df[full_df['metric_type']=='POWER']['band_pair'].unique())}")
    print(f"\n⏱️  Tiempo de procesamiento: {elapsed_min:.1f} minutos ({elapsed_hrs:.2f} horas)")
    print(f"  Promedio: {elapsed_min/n_subjects_processed:.1f} min/sujeto")
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE v2.0 STAGE 1: IRREVERSIBLE SPLIT
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
    
    # Use only successfully processed subjects
    processed_subject_ids = sorted(list(set(full_df['subject_id'])))
    print(f"  Sujetos válidos para split: {len(processed_subject_ids)}")
    
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(processed_subject_ids)
    
    print(f"✅ Split completed:")
    print(f"   Development: {len(dev_subjects)} subjects")
    print(f"   Holdout: {len(holdout_subjects)} subjects")
    print(f"   Split locked with seed: {config.sacred_seed}")
    
    # Split dataframe
    dev_df = full_df[full_df['subject_id'].isin(dev_subjects)].copy()
    holdout_df = full_df[full_df['subject_id'].isin(holdout_subjects)].copy()
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE v2.0 STAGE 2: REGISTER FALSIFICATION CRITERIA
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📋 ECLIPSE STAGE 2: PRE-REGISTERED CRITERIA")
    print("=" * 80)
    
    criteria = [
        # Test 1: PAC Wake > N2 (Cohen's d)
        FalsificationCriteria(
            name="delta_gamma_pac_cohens_d_wake_vs_n2",
            threshold=0.5,
            comparison=">=",
            description="Delta-Gamma PAC Cohen's d (Wake vs N2) >= 0.5",
            is_required=True
        ),
        
        # Test 2: PAC Wake > N2 (p-value)
        FalsificationCriteria(
            name="delta_gamma_pac_pvalue_wake_vs_n2",
            threshold=0.05,
            comparison="<",
            description="Delta-Gamma PAC p-value (Wake vs N2) < 0.05",
            is_required=True
        ),
        
        # Test 3: PAC discriminates better than POWER
        FalsificationCriteria(
            name="pac_vs_power_advantage",
            threshold=0.0,
            comparison=">",
            description="PAC Cohen's d > POWER Cohen's d",
            is_required=True
        ),
        
        # Test 4: N3 Paradox - PAC Wake > N3
        FalsificationCriteria(
            name="n3_paradox_wake_vs_n3",
            threshold=0.0,
            comparison=">",
            description="PAC Wake > PAC N3 (paradox test)",
            is_required=True
        ),
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    print("✅ Criteria registered:")
    for crit in criteria:
        print(f"   • {crit}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE v2.0 STAGE 3: DEVELOPMENT (EXPLORATORY)
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🔬 ECLIPSE STAGE 3: DEVELOPMENT PHASE (Exploratory)")
    print("=" * 80)
    print("⚠️  Note: Development phase used for parameter exploration only")
    print("   Final validation uses holdout set (Stage 4)")
    
    # Separate PAC and POWER
    pac_df = full_df[full_df['metric_type'] == 'PAC'].copy()
    power_df = full_df[full_df['metric_type'] == 'POWER'].copy()
    
    dev_pac = pac_df[pac_df['subject_id'].isin(dev_subjects)]
    dev_power = power_df[power_df['subject_id'].isin(dev_subjects)]
    
    # Development exploration (no ECLIPSE tracking needed here)
    print(f"\n📊 Development set exploration:")
    print(f"   PAC measurements: {len(dev_pac)}")
    print(f"   POWER measurements: {len(dev_power)}")
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE v2.0 STAGE 4: SINGLE-SHOT HOLDOUT VALIDATION
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("🎯 ECLIPSE STAGE 4: SINGLE-SHOT HOLDOUT VALIDATION")
    print("=" * 80)
    print("⚠️  CRITICAL: This is the ONLY chance to test hypotheses")
    print("   No multiple attempts allowed")
    
    holdout_pac = pac_df[pac_df['subject_id'].isin(holdout_subjects)]
    holdout_power = power_df[power_df['subject_id'].isin(holdout_subjects)]
    
    # Compute validation metrics
    validation_results = compute_holdout_metrics(
        holdout_pac, 
        holdout_power, 
        config
    )
    
    # Wrap in ECLIPSE validation
    eclipse_validation = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_df,
        final_model={},  # No model in this analysis
        validation_function=lambda model, data: validation_results
    )
    
    # ═════════════════════════════════════════════════════════════════════════
    # ECLIPSE v2.0 STAGE 5: FINAL ASSESSMENT WITH EIS, STDS
    # ═════════════════════════════════════════════════════════════════════════
    
    print("\n" + "=" * 80)
    print("📊 ECLIPSE STAGE 5: FINAL ASSESSMENT + INTEGRITY METRICS")
    print("=" * 80)
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results={},  # No CV in this analysis
        validation_results=eclipse_validation,
        generate_reports=True,
        compute_integrity=True
    )
    
    # Display final summary
    print("\n" + "=" * 80)
    print("🎯 ECLIPSE v2.0 FINAL SUMMARY")
    print("=" * 80)
    print(eclipse.generate_summary())
    
    # Verify integrity
    print("\n" + "=" * 80)
    print("🔐 ECLIPSE INTEGRITY VERIFICATION")
    print("=" * 80)
    eclipse.verify_integrity()
    
    # Save all results
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pac_df.to_csv(config.output_dir / 'nivel2_extended_pac.csv', index=False)
    power_df.to_csv(config.output_dir / 'nivel2_extended_power.csv', index=False)
    
    # Save validation results as JSON
    with open(config.output_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"\n📁 Results saved to: {config.output_dir}")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS NIVEL 2 EXTENDED + ECLIPSE v2.0 FINALIZADO")
    print("=" * 80)


def compute_holdout_metrics(holdout_pac: pd.DataFrame, 
                            holdout_power: pd.DataFrame,
                            config: Config) -> Dict:
    """Computa todas las métricas en holdout set"""
    
    results = {}
    
    valid_pac = holdout_pac[holdout_pac['valid']].copy()
    valid_power = holdout_power[holdout_power['valid']].copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: PAC Wake vs N2 (Delta-Gamma)
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n🔬 TEST 1: Delta-Gamma PAC (Wake vs N2)")
    print("─" * 80)
    
    delta_pac = valid_pac[valid_pac['band_pair'] == 'Delta-Gamma']
    
    pac_wake = delta_pac[delta_pac['state'] == 'Wake']['pac'].values
    pac_n2 = delta_pac[delta_pac['state'] == 'N2']['pac'].values
    
    pac_wake = pac_wake[~np.isnan(pac_wake)]
    pac_n2 = pac_n2[~np.isnan(pac_n2)]
    
    pac_d_wake_n2 = 0.0  # Default
    
    if len(pac_wake) > 0 and len(pac_n2) > 0:
        pooled_std = np.sqrt(
            ((len(pac_wake)-1)*np.var(pac_wake,ddof=1) + 
             (len(pac_n2)-1)*np.var(pac_n2,ddof=1)) /
            (len(pac_wake) + len(pac_n2) - 2)
        )
        pac_d_wake_n2 = (np.mean(pac_wake) - np.mean(pac_n2)) / (pooled_std + 1e-10)
        _, pac_p_wake_n2 = stats.ttest_ind(pac_wake, pac_n2, alternative='greater')
        
        print(f"  Cohen's d: {pac_d_wake_n2:.3f}")
        print(f"  p-value:   {pac_p_wake_n2:.6f}")
        print(f"  PAC Wake:  {np.mean(pac_wake):.4f} ± {np.std(pac_wake, ddof=1):.4f}")
        print(f"  PAC N2:    {np.mean(pac_n2):.4f} ± {np.std(pac_n2, ddof=1):.4f}")
        
        results['delta_gamma_pac_cohens_d_wake_vs_n2'] = float(pac_d_wake_n2)
        results['delta_gamma_pac_pvalue_wake_vs_n2'] = float(pac_p_wake_n2)
    else:
        print("  ⚠️  Insufficient data for PAC Wake vs N2")
        results['delta_gamma_pac_cohens_d_wake_vs_n2'] = 0.0
        results['delta_gamma_pac_pvalue_wake_vs_n2'] = 1.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: POWER vs PAC
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n🔥 TEST 2: POWER vs PAC Discrimination")
    print("─" * 80)
    
    delta_power = valid_power[valid_power['band_pair'] == 'Delta']
    
    power_wake = delta_power[delta_power['state'] == 'Wake']['pac'].values
    power_n2 = delta_power[delta_power['state'] == 'N2']['pac'].values
    
    power_wake = power_wake[~np.isnan(power_wake)]
    power_n2 = power_n2[~np.isnan(power_n2)]
    
    power_d_wake_n2 = 0.0  # Default
    
    if len(power_wake) > 0 and len(power_n2) > 0:
        pooled_std = np.sqrt(
            ((len(power_wake)-1)*np.var(power_wake,ddof=1) + 
             (len(power_n2)-1)*np.var(power_n2,ddof=1)) /
            (len(power_wake) + len(power_n2) - 2)
        )
        power_d_wake_n2 = (np.mean(power_wake) - np.mean(power_n2)) / (pooled_std + 1e-10)
        
        print(f"  Delta POWER Cohen's d: {power_d_wake_n2:.3f}")
        print(f"  Delta PAC Cohen's d:   {pac_d_wake_n2:.3f}")
        print(f"  Difference:            {pac_d_wake_n2 - power_d_wake_n2:.3f}")
        
        results['power_cohens_d_wake_vs_n2'] = float(power_d_wake_n2)
        results['pac_vs_power_advantage'] = float(pac_d_wake_n2 - power_d_wake_n2)
        
        if pac_d_wake_n2 > power_d_wake_n2 + 0.5:
            print("\n  ✅ PAC DISCRIMINA SIGNIFICATIVAMENTE MEJOR")
        elif pac_d_wake_n2 > power_d_wake_n2:
            print("\n  ⚠️  PAC discrimina mejor (modesto)")
        else:
            print("\n  ❌ POWER discrimina igual o mejor")
    else:
        print("  ⚠️  Insufficient data for POWER analysis")
        results['power_cohens_d_wake_vs_n2'] = 0.0
        results['pac_vs_power_advantage'] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: N3 Paradox
    # ═══════════════════════════════════════════════════════════════════════
    
    print("\n🚨 TEST 3: N3 PARADOX")
    print("─" * 80)
    
    if 'N3' in valid_pac['state'].unique():
        pac_n3 = delta_pac[delta_pac['state'] == 'N3']['pac'].values
        pac_n3 = pac_n3[~np.isnan(pac_n3)]
        
        if len(pac_n3) > 0 and len(pac_wake) > 0:
            print(f"  PAC Wake: {np.mean(pac_wake):.4f}")
            print(f"  PAC N2:   {np.mean(pac_n2):.4f}")
            print(f"  PAC N3:   {np.mean(pac_n3):.4f}")
            
            results['pac_wake_mean'] = float(np.mean(pac_wake))
            results['pac_n2_mean'] = float(np.mean(pac_n2))
            results['pac_n3_mean'] = float(np.mean(pac_n3))
            results['n3_paradox_wake_vs_n3'] = float(np.mean(pac_wake) - np.mean(pac_n3))
            
            if np.mean(pac_wake) > np.mean(pac_n2) > np.mean(pac_n3):
                print("\n  ✅ PATRÓN CORRECTO: Wake > N2 > N3")
            else:
                print("\n  ❌ PATRÓN INCORRECTO")
        else:
            print("  ⚠️  Insufficient N3 data")
            results['n3_paradox_wake_vs_n3'] = 0.0
        
        # Power in N3
        power_n3 = delta_power[delta_power['state'] == 'N3']['pac'].values
        power_n3 = power_n3[~np.isnan(power_n3)]
        
        if len(power_n3) > 0 and len(power_wake) > 0:
            print(f"\n  Power Wake: {np.mean(power_wake):.4f}")
            print(f"  Power N2:   {np.mean(power_n2):.4f}")
            print(f"  Power N3:   {np.mean(power_n3):.4f}")
            
            results['power_wake_mean'] = float(np.mean(power_wake))
            results['power_n2_mean'] = float(np.mean(power_n2))
            results['power_n3_mean'] = float(np.mean(power_n3))
    else:
        print("  ⚠️  N3 not available in holdout")
        results['n3_paradox_wake_vs_n3'] = 0.0
    
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NIVEL 2 EXTENDED with FULL ECLIPSE v2.0 Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default='./nivel2_extended_eclipse_results',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--n-subjects',
        type=int,
        help='Número específico de sujetos (default: TODOS los disponibles)',
        default=None
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    if args.n_subjects:
        n_subjects = args.n_subjects
        print(f"🔬 MODO CUSTOM: {n_subjects} sujetos")
    else:
        n_subjects = None  # SIEMPRE PROCESAR TODOS
        print("🔬 MODO COMPLETO: TODOS los sujetos disponibles (153 sujetos)")
    
    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sacred_seed=args.seed,
        n_subjects=n_subjects
    )
    
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: No se encontró el directorio de datos:")
        print(f"   {config.data_dir}")
        sys.exit(1)
    
    try:
        run_extended_analysis_with_eclipse(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Análisis interrumpido")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR durante el análisis:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()