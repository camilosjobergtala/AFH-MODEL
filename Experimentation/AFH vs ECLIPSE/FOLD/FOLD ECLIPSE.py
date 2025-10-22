#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
ΔtFold Analysis on Sleep-EDF Database using ECLIPSE v2.0
═══════════════════════════════════════════════════════════════════════════════

ANÁLISIS COMPLETO CON DATOS REALES

Hypothesis H1: Δt_Fold_wake > Δt_Fold_sleep

Author: Based on ECLIPSE v2.0 by Camilo Sjöberg Tala
Date: 2025-10-21

EJECUCIÓN:
    # Modo piloto (10 sujetos, ~5 min)
    python run_sleep_edf_analysis.py --pilot
    
    # Análisis completo (todos los sujetos, ~1-2 horas)
    python run_sleep_edf_analysis.py --full
    
    # Personalizado
    python run_sleep_edf_analysis.py --n-subjects 20 --data-dir /ruta/datos
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
from dataclasses import dataclass
import argparse
import sys

# Import ECLIPSE v2.0 framework
try:
    from eclipse_v2 import (
        EclipseFramework, EclipseConfig, FalsificationCriteria
    )
except ImportError:
    print("ERROR: No se encontró eclipse_v2.py")
    print("Asegúrate de que eclipse_v2.py esté en el mismo directorio")
    sys.exit(1)

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Configuración completa del análisis"""
    # Rutas
    data_dir: Path
    output_dir: Path
    
    # Procesamiento de señal
    sampling_rate: float = 100.0
    lowcut: float = 1.0
    highcut: float = 20.0
    notch_freq: float = 50.0
    
    # Banda ΔtFold
    freq_band_low: float = 2.0
    freq_band_high: float = 9.0
    
    # Ventanas de transición
    pre_start: float = -60.0   # segundos antes de transición
    pre_end: float = -30.0     # termina 30s antes de transición
    post_start: float = 30.0   # empieza 30s después de transición
    post_end: float = 60.0     # termina 60s después
    
    # Control de calidad
    artifact_threshold_uv: float = 200.0
    min_coherence: float = 0.1
    
    # Estados
    wake_states: List[str] = None
    sleep_states: List[str] = None
    
    # ECLIPSE
    sacred_seed: int = 42
    n_subjects: Optional[int] = None  # None = todos
    
    def __post_init__(self):
        if self.wake_states is None:
            self.wake_states = ['Sleep stage W']
        if self.sleep_states is None:
            self.sleep_states = ['Sleep stage 1', 'Sleep stage 2']


# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA DE ΔtFold
# ═══════════════════════════════════════════════════════════════════════════

class DeltaTFoldCalculator:
    """Calcula Δt_Fold: group delay en coherencia 2-9 Hz"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fs = config.sampling_rate
        
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocesa señal EEG
        
        Args:
            data: (n_channels, n_samples)
            
        Returns:
            Datos preprocesados
        """
        # Bandpass
        sos = signal.butter(
            4,
            [self.config.lowcut, self.config.highcut],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        filtered = signal.sosfiltfilt(sos, data, axis=1)
        
        # Notch
        b_notch, a_notch = signal.iirnotch(
            self.config.notch_freq,
            Q=30,
            fs=self.fs
        )
        filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=1)
        
        # Z-score por canal
        filtered = (filtered - filtered.mean(axis=1, keepdims=True)) / \
                   (filtered.std(axis=1, keepdims=True) + 1e-10)
        
        return filtered
    
    def has_artifact(self, data: np.ndarray) -> bool:
        """Detecta artefactos por amplitud"""
        peak_to_peak = np.ptp(data, axis=1)
        return np.any(peak_to_peak > self.config.artifact_threshold_uv)
    
    def compute_metrics(
        self,
        ch1: np.ndarray,
        ch2: np.ndarray
    ) -> Dict[str, float]:
        """
        Calcula coherencia y group delay
        
        Args:
            ch1, ch2: Series temporales (1D)
            
        Returns:
            {'tau': group delay (s), 'wPLI': robustez, 'coherence': fuerza, 'valid': bool}
        """
        # Coherencia
        f, Cxy = signal.coherence(
            ch1, ch2,
            fs=self.fs,
            nperseg=int(4 * self.fs),
            noverlap=int(2 * self.fs)
        )
        
        # Extraer banda 2-9 Hz
        mask = (f >= self.config.freq_band_low) & (f <= self.config.freq_band_high)
        f_band = f[mask]
        Cxy_band = Cxy[mask]
        
        mean_coh = np.mean(Cxy_band)
        
        if mean_coh < self.config.min_coherence:
            return {
                'tau': np.nan,
                'wPLI': np.nan,
                'coherence': mean_coh,
                'valid': False
            }
        
        # Cross-spectrum para fase
        f_csd, Pxy = signal.csd(
            ch1, ch2,
            fs=self.fs,
            nperseg=int(4 * self.fs),
            noverlap=int(2 * self.fs)
        )
        
        mask_csd = (f_csd >= self.config.freq_band_low) & \
                   (f_csd <= self.config.freq_band_high)
        f_band_csd = f_csd[mask_csd]
        Pxy_band = Pxy[mask_csd]
        
        # Fase
        phase = np.angle(Pxy_band)
        phase_unwrap = np.unwrap(phase)
        
        # Group delay: τ = -(1/2π) * dφ/df
        if len(f_band_csd) > 1:
            dphi_df = np.gradient(phase_unwrap, f_band_csd)
            tau = -np.median(dphi_df) / (2 * np.pi)
        else:
            tau = np.nan
        
        # wPLI (robustez contra volumen conducción)
        imag = np.imag(Pxy_band)
        wPLI = np.abs(np.mean(np.abs(imag) * np.sign(imag))) / \
               (np.mean(np.abs(imag)) + 1e-10)
        
        return {
            'tau': np.abs(tau) if not np.isnan(tau) else np.nan,
            'wPLI': wPLI,
            'coherence': mean_coh,
            'valid': True
        }
    
    def compute_delta_t_fold(self, data: np.ndarray) -> Dict[str, any]:
        """
        Calcula Δt_Fold completo
        
        Args:
            data: (2, n_samples) [Fpz-Cz, Pz-Oz]
            
        Returns:
            {'delta_t_fold': |τ|, 'wPLI', 'coherence', 'valid', 'reject_reason'}
        """
        # Preprocesar
        data_clean = self.preprocess(data)
        
        # Chequear artefactos
        if self.has_artifact(data_clean):
            return {
                'delta_t_fold': np.nan,
                'wPLI': np.nan,
                'coherence': np.nan,
                'valid': False,
                'reject_reason': 'artifact'
            }
        
        # Calcular métricas
        metrics = self.compute_metrics(data_clean[0], data_clean[1])
        
        return {
            'delta_t_fold': metrics['tau'],
            'wPLI': metrics['wPLI'],
            'coherence': metrics['coherence'],
            'valid': metrics['valid'],
            'reject_reason': None if metrics['valid'] else 'low_coherence'
        }


# ═══════════════════════════════════════════════════════════════════════════
# CARGADOR DE SLEEP-EDF
# ═══════════════════════════════════════════════════════════════════════════

class SleepEDFLoader:
    """Carga y procesa archivos Sleep-EDF"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def get_subjects(self, subset: str = 'SC') -> List[str]:
        """Obtiene lista de sujetos disponibles"""
        pattern = f"{subset}*PSG.edf"
        files = sorted(self.config.data_dir.glob(pattern))
        
        subjects = []
        for f in files:
            # SC4001E0-PSG.edf -> SC4001
            name = f.stem.split('-')[0]
            subject_id = name[:6]
            if subject_id not in subjects:
                subjects.append(subject_id)
        
        logger.info(f"Encontrados {len(subjects)} sujetos en subset {subset}")
        return subjects
    
    def load_recording(
        self,
        subject_id: str,
        night: int = 0
    ) -> Tuple[mne.io.Raw, pd.DataFrame]:
        """
        Carga PSG y hypnograma
        
        Returns:
            (raw, hypnogram_df)
        """
        import glob
        
        night_code = 'E0' if night == 0 else 'E1'
        
        # PSG
        psg_file = self.config.data_dir / f"{subject_id}{night_code}-PSG.edf"
        if not psg_file.exists():
            raise FileNotFoundError(f"No existe: {psg_file}")
        
        raw = mne.io.read_raw_edf(psg_file, preload=True, verbose=False)
        
        # Hypnograma - en Sleep-EDF los hypnogramas usan sufijo EC (no E0/E1)
        hypno_file = self.config.data_dir / f"{subject_id}EC-Hypnogram.edf"
        
        if not hypno_file.exists():
            raise FileNotFoundError(f"No se encontró hypnogram para {subject_id}")
        
        annotations = mne.read_annotations(str(hypno_file))
        
        hypno_df = pd.DataFrame({
            'onset': annotations.onset,
            'duration': annotations.duration,
            'description': annotations.description
        })
        
        return raw, hypno_df


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR DE TRANSICIONES
# ═══════════════════════════════════════════════════════════════════════════

class TransitionDetector:
    """Encuentra transiciones vigilia ↔ sueño"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def find_transitions(
        self,
        hypnogram: pd.DataFrame,
        from_states: List[str],
        to_states: List[str]
    ) -> List[Dict]:
        """Encuentra transiciones entre estados"""
        transitions = []
        
        for i in range(len(hypnogram) - 1):
            current = hypnogram.iloc[i]
            next_stage = hypnogram.iloc[i + 1]
            
            if (current['description'] in from_states and
                next_stage['description'] in to_states):
                
                transitions.append({
                    'transition_idx': i,
                    'from_state': current['description'],
                    'to_state': next_stage['description'],
                    'transition_time': next_stage['onset']
                })
        
        return transitions
    
    def extract_windows(
        self,
        raw: mne.io.Raw,
        transition_time: float,
        channels: List[str] = ['EEG Fpz-Cz', 'EEG Pz-Oz']
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extrae ventanas pre/post alrededor de transición
        
        Returns:
            (pre_data, post_data) o (None, None) si falla
        """
        fs = raw.info['sfreq']
        
        # Definir ventanas
        pre_start = transition_time + self.config.pre_start
        pre_end = transition_time + self.config.pre_end
        post_start = transition_time + self.config.post_start
        post_end = transition_time + self.config.post_end
        
        # Convertir a muestras
        pre_start_samp = int(pre_start * fs)
        pre_end_samp = int(pre_end * fs)
        post_start_samp = int(post_start * fs)
        post_end_samp = int(post_end * fs)
        
        try:
            pre_data, _ = raw[channels, pre_start_samp:pre_end_samp]
            post_data, _ = raw[channels, post_start_samp:post_end_samp]
            return pre_data, post_data
        except:
            return None, None


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

class SleepFoldAnalysis:
    """Pipeline completo de análisis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = SleepEDFLoader(config)
        self.calculator = DeltaTFoldCalculator(config)
        self.detector = TransitionDetector(config)
        
    def process_subject(
        self,
        subject_id: str,
        night: int = 0
    ) -> pd.DataFrame:
        """Procesa un sujeto completo"""
        try:
            raw, hypnogram = self.loader.load_recording(subject_id, night)
            
            # Encontrar transiciones W→S y S→W
            trans_ws = self.detector.find_transitions(
                hypnogram,
                self.config.wake_states,
                self.config.sleep_states
            )
            
            trans_sw = self.detector.find_transitions(
                hypnogram,
                self.config.sleep_states,
                self.config.wake_states
            )
            
            all_trans = trans_ws + trans_sw
            
            if not all_trans:
                return pd.DataFrame()
            
            # Procesar cada transición
            results = []
            
            for trans in all_trans:
                pre_data, post_data = self.detector.extract_windows(
                    raw,
                    trans['transition_time']
                )
                
                if pre_data is None:
                    continue
                
                # Calcular ΔtFold
                pre_metrics = self.calculator.compute_delta_t_fold(pre_data)
                post_metrics = self.calculator.compute_delta_t_fold(post_data)
                
                results.append({
                    'subject_id': subject_id,
                    'night': night,
                    'transition_idx': trans['transition_idx'],
                    'from_state': trans['from_state'],
                    'to_state': trans['to_state'],
                    'transition_time': trans['transition_time'],
                    
                    'delta_t_fold_pre': pre_metrics['delta_t_fold'],
                    'wPLI_pre': pre_metrics['wPLI'],
                    'coherence_pre': pre_metrics['coherence'],
                    'valid_pre': pre_metrics['valid'],
                    
                    'delta_t_fold_post': post_metrics['delta_t_fold'],
                    'wPLI_post': post_metrics['wPLI'],
                    'coherence_post': post_metrics['coherence'],
                    'valid_post': post_metrics['valid'],
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"Error procesando {subject_id}: {e}")
            return pd.DataFrame()
    
    def prepare_dataset(
        self,
        subject_ids: List[str]
    ) -> pd.DataFrame:
        """Procesa múltiples sujetos"""
        all_data = []
        
        for i, subject_id in enumerate(subject_ids, 1):
            logger.info(f"Procesando {subject_id} ({i}/{len(subject_ids)})...")
            
            # Solo procesar noche 0 (E0)
            try:
                df = self.process_subject(subject_id, night=0)
                if not df.empty:
                    all_data.append(df)
            except FileNotFoundError as e:
                logger.debug(f"Archivo no encontrado para {subject_id}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error procesando {subject_id}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No se pudo extraer ningún dato válido!")
        
        dataset = pd.concat(all_data, ignore_index=True)
        
        # Filtrar válidos
        dataset_valid = dataset[
            dataset['valid_pre'] & dataset['valid_post']
        ].copy()
        
        logger.info(
            f"\nDataset compilado: {len(dataset)} total, "
            f"{len(dataset_valid)} válidos ({len(dataset_valid)/len(dataset)*100:.1f}%)"
        )
        
        return dataset_valid


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS CON ECLIPSE
# ═══════════════════════════════════════════════════════════════════════════

def run_eclipse_analysis(config: Config):
    """Ejecuta análisis completo con ECLIPSE v2.0"""
    
    print("\n" + "=" * 80)
    print("🧠 ΔtFold SLEEP-EDF ANALYSIS WITH ECLIPSE v2.0")
    print("=" * 80)
    print(f"Hypothesis H1: Δt_Fold_wake > Δt_Fold_sleep")
    print(f"Frequency band: {config.freq_band_low}-{config.freq_band_high} Hz")
    print(f"Wake states: {config.wake_states}")
    print(f"Sleep states: {config.sleep_states}")
    print("=" * 80)
    
    # Inicializar
    analysis = SleepFoldAnalysis(config)
    
    # Obtener sujetos
    all_subjects = analysis.loader.get_subjects('SC')
    
    if config.n_subjects:
        subjects = all_subjects[:config.n_subjects]
        logger.info(f"\nUsando {len(subjects)} sujetos (limitado)")
    else:
        subjects = all_subjects
        logger.info(f"\nUsando TODOS los {len(subjects)} sujetos")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PREPARAR DATOS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("STEP 1: PREPARACIÓN DE DATOS")
    print("-" * 80)
    
    dataset = analysis.prepare_dataset(subjects)
    
    # Asignar labels (basado en estado PRE-transición)
    dataset['is_wake_pre'] = dataset['from_state'].isin(config.wake_states).astype(int)
    dataset['consciousness'] = dataset['is_wake_pre']
    dataset['delta_t_fold'] = dataset['delta_t_fold_pre']
    
    print(f"\nDataset final:")
    print(f"  • Total transiciones: {len(dataset)}")
    print(f"  • Muestras vigilia: {dataset['consciousness'].sum()}")
    print(f"  • Muestras sueño: {(~dataset['consciousness'].astype(bool)).sum()}")
    print(f"  • Sujetos únicos: {dataset['subject_id'].nunique()}")
    
    # Stats exploratorias
    wake_vals = dataset[dataset['consciousness']==1]['delta_t_fold']
    sleep_vals = dataset[dataset['consciousness']==0]['delta_t_fold']
    
    print(f"\n📊 Estadísticas exploratorias:")
    print(f"  • Vigilia:  mean={wake_vals.mean():.4f}s, std={wake_vals.std():.4f}s, n={len(wake_vals)}")
    print(f"  • Sueño:    mean={sleep_vals.mean():.4f}s, std={sleep_vals.std():.4f}s, n={len(sleep_vals)}")
    print(f"  • Diferencia: {(wake_vals.mean()-sleep_vals.mean())*1000:.2f} ms")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 1: SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("ECLIPSE STAGE 1: SPLIT IRREVERSIBLE (NIVEL SUJETO)")
    print("-" * 80)
    
    eclipse_config = EclipseConfig(
        project_name="DeltaTFold_SleepEDF_RealData",
        researcher="Sleep Analysis",
        sacred_seed=config.sacred_seed,
        development_ratio=0.7,
        holdout_ratio=0.3,
        output_dir=str(config.output_dir / "eclipse_v2")
    )
    
    eclipse = EclipseFramework(eclipse_config)
    
    # Split a nivel sujeto
    unique_subjects = dataset['subject_id'].unique().tolist()
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(unique_subjects)
    
    dev_data = dataset[dataset['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = dataset[dataset['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
    print(f"\n✅ Split completado:")
    print(f"  • Development: {len(dev_subjects)} sujetos, {len(dev_data)} transiciones")
    print(f"  • Holdout: {len(holdout_subjects)} sujetos, {len(holdout_data)} transiciones")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 2: CRITERIOS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("ECLIPSE STAGE 2: CRITERIOS PRE-REGISTRADOS")
    print("-" * 80)
    
    criteria = [
        FalsificationCriteria(
            name="Cohen's d (Wake vs Sleep)",
            threshold=0.3,
            comparison=">=",
            description="Tamaño del efecto >= 0.3 (efecto mediano)",
            is_required=True
        ),
        FalsificationCriteria(
            name="p-value (one-tailed t-test)",
            threshold=0.05,
            comparison="<=",
            description="p < 0.05 para H1: vigilia > sueño",
            is_required=True
        ),
        FalsificationCriteria(
            name="Mean difference (ms)",
            threshold=10.0,
            comparison=">=",
            description="Diferencia media >= 10 ms",
            is_required=False
        )
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    print("\n✅ Criterios registrados:")
    for crit in criteria:
        print(f"  • {crit}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 3: DESARROLLO
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("ECLIPSE STAGE 3: DESARROLLO CON NESTED CV")
    print("-" * 80)
    
    def train_fn(train_indices, **kwargs):
        return {'metadata': 'no model needed'}
    
    def val_fn(model, val_indices, **kwargs):
        val_df = dev_data.iloc[val_indices]
        
        wake = val_df[val_df['consciousness']==1]['delta_t_fold'].values
        sleep = val_df[val_df['consciousness']==0]['delta_t_fold'].values
        
        wake = wake[~np.isnan(wake)]
        sleep = sleep[~np.isnan(sleep)]
        
        if len(wake) == 0 or len(sleep) == 0:
            return {
                "Cohen's d (Wake vs Sleep)": 0.0,
                "p-value (one-tailed t-test)": 1.0,
                "Mean difference (ms)": 0.0
            }
        
        pooled_std = np.sqrt(
            ((len(wake)-1)*np.var(wake,ddof=1) + (len(sleep)-1)*np.var(sleep,ddof=1)) /
            (len(wake) + len(sleep) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(sleep)) / (pooled_std + 1e-10)
        
        _, p_val = stats.ttest_ind(wake, sleep, alternative='greater')
        
        mean_diff_ms = (np.mean(wake) - np.mean(sleep)) * 1000
        
        return {
            "Cohen's d (Wake vs Sleep)": cohens_d,
            "p-value (one-tailed t-test)": p_val,
            "Mean difference (ms)": mean_diff_ms
        }
    
    dev_results = eclipse.stage3_development(
        development_data=list(range(len(dev_data))),
        training_function=train_fn,
        validation_function=val_fn
    )
    
    print("\n✅ Desarrollo completado:")
    # En ECLIPSE v2.0 los resultados están en 'metrics' no en 'cv_results'
    if 'metrics' in dev_results:
        metrics_list = dev_results['metrics']
        avg = pd.DataFrame(metrics_list).mean()
        cohens_d_key = "Cohen's d (Wake vs Sleep)"
        pval_key = "p-value (one-tailed t-test)"
        diff_key = "Mean difference (ms)"
        print(f"  • Cohen's d (promedio): {avg[cohens_d_key]:.3f}")
        print(f"  • p-value (promedio): {avg[pval_key]:.4f}")
        print(f"  • Diferencia (ms): {avg[diff_key]:.2f}")
    else:
        print("  • Resultados de CV completados (ver arriba)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 4: VALIDACIÓN HOLDOUT
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("ECLIPSE STAGE 4: VALIDACIÓN SINGLE-SHOT EN HOLDOUT")
    print("-" * 80)
    print("⚠️  ÚNICA EVALUACIÓN EN DATOS DE HOLDOUT ⚠️")
    print("-" * 80)
    
    final_model = train_fn(list(range(len(dev_data))))
    
    def final_val_fn(model, holdout_df, **kwargs):
        wake = holdout_df[holdout_df['consciousness']==1]['delta_t_fold'].values
        sleep = holdout_df[holdout_df['consciousness']==0]['delta_t_fold'].values
        
        wake = wake[~np.isnan(wake)]
        sleep = sleep[~np.isnan(sleep)]
        
        if len(wake) == 0 or len(sleep) == 0:
            return {
                "Cohen's d (Wake vs Sleep)": 0.0,
                "p-value (one-tailed t-test)": 1.0,
                "Mean difference (ms)": 0.0
            }
        
        pooled_std = np.sqrt(
            ((len(wake)-1)*np.var(wake,ddof=1) + (len(sleep)-1)*np.var(sleep,ddof=1)) /
            (len(wake) + len(sleep) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(sleep)) / (pooled_std + 1e-10)
        
        _, p_val = stats.ttest_ind(wake, sleep, alternative='greater')
        
        return {
            "Cohen's d (Wake vs Sleep)": cohens_d,
            "p-value (one-tailed t-test)": p_val,
            "Mean difference (ms)": (np.mean(wake) - np.mean(sleep)) * 1000,
            'n_wake': len(wake),
            'n_sleep': len(sleep),
            'mean_wake_s': np.mean(wake),
            'mean_sleep_s': np.mean(sleep),
            'std_wake_s': np.std(wake, ddof=1),
            'std_sleep_s': np.std(sleep, ddof=1)
        }
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=final_val_fn
    )
    
    if val_results is None:
        print("\n❌ Validación cancelada")
        return
    
    print("\n🎯 RESULTADOS HOLDOUT:")
    for k, v in val_results.items():
        if isinstance(v, float):
            print(f"  • {k}: {v:.4f}")
        else:
            print(f"  • {k}: {v}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 5: ASSESSMENT FINAL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "-" * 80)
    print("ECLIPSE STAGE 5: ASSESSMENT FINAL CON EIS/STDS")
    print("-" * 80)
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True
    )
    
    # Resumen
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS COMPLETO - RESUMEN FINAL")
    print("=" * 80)
    print(eclipse.generate_summary())
    
    print(f"\n📁 Resultados guardados en: {config.output_dir / 'eclipse_v2'}")
    print("\nArchivos generados:")
    print("  • *_REPORT.html - Reporte completo")
    print("  • *_EIS_REPORT.txt - Eclipse Integrity Score")
    print("  • *_STDS_REPORT.txt - Data Snooping Test")
    print("  • lockfile.json - Registro de integridad")
    
    # Verificar integridad
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS FINALIZADO CON ÉXITO")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ΔtFold Analysis on Sleep-EDF with ECLIPSE v2.0",
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
        default='./sleep_fold_results',
        help='Directorio de salida'
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--pilot',
        action='store_true',
        help='Modo piloto (10 sujetos, ~5-10 min)'
    )
    
    group.add_argument(
        '--full',
        action='store_true',
        help='Análisis completo (todos los sujetos, ~1-2 horas)'
    )
    
    group.add_argument(
        '--n-subjects',
        type=int,
        help='Número específico de sujetos'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad'
    )
    
    args = parser.parse_args()
    
    # Determinar n_subjects
    if args.pilot:
        n_subjects = 10
    elif args.full:
        n_subjects = None
    elif args.n_subjects:
        n_subjects = args.n_subjects
    else:
        # Default: full
        logger.warning("No se especificó modo. Usando --full por defecto.")
        n_subjects = None
    
    # Configurar
    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sacred_seed=args.seed,
        n_subjects=n_subjects
    )
    
    # Verificar que existe el directorio de datos
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: No se encontró el directorio de datos:")
        print(f"   {config.data_dir}")
        print(f"\nAsegúrate de:")
        print(f"  1. Haber descargado Sleep-EDF Database")
        print(f"  2. Descomprimido en la ubicación correcta")
        print(f"  3. Especificar la ruta con --data-dir si está en otro lugar")
        sys.exit(1)
    
    # Crear directorio de salida
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ejecutar
    try:
        run_eclipse_analysis(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Análisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR durante el análisis:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()