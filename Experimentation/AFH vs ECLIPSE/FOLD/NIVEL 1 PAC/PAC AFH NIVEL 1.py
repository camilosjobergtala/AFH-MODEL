#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIVEL 1: Phase-Amplitude Coupling (PAC) Theta-Gamma Test
Autopsychic Fold Hypothesis - ECLIPSE v2.0
═══════════════════════════════════════════════════════════════════════════════

HIPÓTESIS:
    H1: PAC_theta-gamma_wake > PAC_theta-gamma_N2
    
OPERACIONALIZACIÓN DEL FOLD:
    Fold_intensity = PAC(theta_phase, gamma_amplitude)
    - Theta (4-8 Hz) = procesamiento cortical lento
    - Gamma (30-50 Hz) = procesamiento sensoriomotor rápido
    - PAC = acoplamiento fase-amplitud (Modulation Index)

CRITERIO DE FALSIFICACIÓN:
    Si Cohen's d < 0.5 O p ≥ 0.05 → FOLD FALSIFICADO → ABANDONAR AFH

Author: Camilo Sjöberg Tala
Date: 2025-01-XX
Version: NIVEL_1_v1.0

EJECUCIÓN:
    # Modo piloto (10 sujetos, ~10 min)
    python nivel1_pac_test.py --pilot
    
    # Análisis completo (50 sujetos, ~1-2 horas)
    python nivel1_pac_test.py --full --n-subjects 50
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

# Tensorpac para PAC
# Tensorpac para PAC
from tensorpac import Pac

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
    """Configuración NIVEL 1: PAC Test"""
    # Rutas
    data_dir: Path
    output_dir: Path
    
    # Procesamiento de señal
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0  # Cambió de 80 a 45 Hz (bajo Nyquist de 50 Hz)
    notch_freq: float = 50.0
    
    # PAC theta-gamma
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 45.0)  # Cambió de 50 a 45 Hz para estar bajo Nyquist
    
    # Epochs
    epoch_duration: float = 30.0
    n_epochs_per_state: int = 100
    
    # Control de calidad
    artifact_threshold_uv: float = 200.0
    min_pac_value: float = 0.0
    
    # Estados
    wake_state: str = 'Sleep stage W'
    n2_state: str = 'Sleep stage 2'
    
    # ECLIPSE
    sacred_seed: int = 42
    n_subjects: Optional[int] = None
    
    # Canal a usar
    channel_name: str = 'Pz-Oz'


# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA PAC
# ═══════════════════════════════════════════════════════════════════════════

class PACCalculator:
    """Calcula Phase-Amplitude Coupling theta-gamma"""
    
    def __init__(self, config: Config):
        self.config = config
        self.fs = config.sampling_rate
        
        # Crear objeto PAC de tensorpac
        self.pac = Pac(
            idpac=(1, 0, 0),
            f_pha=list(config.theta_band),
            f_amp=list(config.gamma_band),
            dcomplex='wavelet',
            width=7
        )
        
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocesa señal EEG
        
        Args:
            data: (n_samples,) señal 1D
            
        Returns:
            Señal preprocesada
        """
        # Bandpass 0.5-80 Hz
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
    
    def has_artifact(self, data: np.ndarray) -> bool:
        """Detecta artefactos por amplitud"""
        peak_to_peak = np.ptp(data)
        return peak_to_peak > 8.0
    
    def compute_pac(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calcula PAC en una epoch
        
        Args:
            data: (n_samples,) señal 1D de 30s
            
        Returns:
            {'pac': Modulation Index, 'valid': bool, 'reject_reason': str}
        """
        # Preprocesar
        data_clean = self.preprocess(data)
        
        # Chequear artefactos
        if self.has_artifact(data_clean):
            return {
                'pac': np.nan,
                'valid': False,
                'reject_reason': 'artifact'
            }
        
        # Chequear longitud mínima
        min_samples = int(2.0 * self.fs)
        if len(data_clean) < min_samples:
            return {
                'pac': np.nan,
                'valid': False,
                'reject_reason': 'too_short'
            }
        
        # Calcular PAC
        data_reshaped = data_clean[np.newaxis, :]
        
        try:
            pac_value = self.pac.filterfit(self.fs, data_reshaped, data_reshaped)
            pac_value = float(pac_value[0, 0, 0])
            
            return {
                'pac': pac_value,
                'valid': True,
                'reject_reason': None
            }
            
        except Exception as e:
            logger.warning(f"Error en PAC: {e}")
            return {
                'pac': np.nan,
                'valid': False,
                'reject_reason': f'computation_error: {e}'
            }


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR DE DATOS SLEEP-EDF
# ═══════════════════════════════════════════════════════════════════════════

class SleepEDFProcessor:
    """Carga y procesa datos Sleep-EDF"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pac_calc = PACCalculator(config)
        
    def find_subject_files(self) -> List[Dict[str, Path]]:
        """Encuentra pares PSG + Hypnogram usando lógica Sleep-EDF correcta"""
        
        # DEBUG
        print(f"\n🔍 DEBUG INFO:")
        print(f"  Buscando en: {self.config.data_dir}")
        print(f"  Directorio existe? {self.config.data_dir.exists()}")
        print(f"  Es directorio? {self.config.data_dir.is_dir()}")
        
        if self.config.data_dir.exists():
            all_files = list(self.config.data_dir.glob("*"))
            print(f"  Total archivos en carpeta: {len(all_files)}")
            if len(all_files) > 0:
                print(f"  Primeros 3 archivos:")
                for f in all_files[:3]:
                    print(f"    - {f.name}")
        
        # Buscar archivos PSG y Hypnogram
        psg_files = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno_files = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        
        print(f"  Archivos PSG encontrados: {len(psg_files)}")
        print(f"  Archivos Hypnogram encontrados: {len(hypno_files)}")
        if len(psg_files) > 0:
            print(f"  Primer PSG: {psg_files[0].name}")
        if len(hypno_files) > 0:
            print(f"  Primer Hypnogram: {hypno_files[0].name}")
        print()
        # FIN DEBUG
        
        # Crear mapa de hypnograms por código base
        # Ejemplo: SC4512EW-Hypnogram.edf -> base: SC4512E
        hypno_map = {}
        for hypno_path in hypno_files:
            codigo_hypno = hypno_path.stem.replace("-Hypnogram", "")
            if len(codigo_hypno) >= 6:
                # Quitar último carácter (W, 0, etc)
                base_letra = codigo_hypno[:-1]
                hypno_map[base_letra] = hypno_path
        
        # Emparejar PSG con Hypnogram
        # Ejemplo: SC4212E0-PSG.edf -> base: SC4212E
        subject_files = []
        for psg_path in psg_files:
            codigo_psg = psg_path.stem.replace("-PSG", "")
            if len(codigo_psg) >= 7 and codigo_psg.endswith('0'):
                # Quitar el '0' final
                base_letra = codigo_psg[:-1]
                
                if base_letra in hypno_map:
                    subject_files.append({
                        'psg': psg_path,
                        'hypno': hypno_map[base_letra],
                        'subject_id': codigo_psg
                    })
        
        logger.info(f"Encontrados {len(subject_files)} sujetos con PSG + Hypnogram emparejados")
        
        if self.config.n_subjects is not None:
            subject_files = subject_files[:self.config.n_subjects]
            logger.info(f"Limitado a {len(subject_files)} sujetos")
        
        return subject_files
    
    def load_subject(self, psg_path: Path, hypno_path: Path) -> Tuple[mne.io.Raw, mne.Annotations]:
        """Carga PSG y anotaciones"""
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annotations = mne.read_annotations(hypno_path)
        
        # DEBUG: Mostrar estados disponibles (solo para el primer sujeto)
        unique_descriptions = set(annotations.description)
        if len(unique_descriptions) > 0:
            logger.info(f"  Estados disponibles: {sorted(unique_descriptions)}")
        
        return raw, annotations
    
    def extract_epochs_for_state(
        self,
        raw: mne.io.Raw,
        annotations: mne.Annotations,
        state: str
    ) -> List[np.ndarray]:
        """
        Extrae epochs de 30s para un estado específico
        
        Returns:
            List de arrays (n_samples,) para el canal EEG disponible
        """
        # Buscar canal disponible
        available_channel = None
        
        # Lista de canales preferidos en orden
        preferred_channels = ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']
        
        for ch in preferred_channels:
            if ch in raw.ch_names:
                available_channel = ch
                break
        
        # Si no encuentra ninguno preferido, usar el primer canal EEG
        if available_channel is None:
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper() or ch.startswith('EEG')]
            if len(eeg_channels) > 0:
                available_channel = eeg_channels[0]
            else:
                logger.warning(f"No se encontró ningún canal EEG. Canales disponibles: {raw.ch_names[:5]}")
                return []
        
        logger.info(f"  Usando canal: {available_channel}")
        
        raw_ch = raw.copy().pick_channels([available_channel])
        
        # Buscar epochs del estado deseado directamente en las anotaciones
        # En Sleep-EDF los estados son: "Sleep stage W", "Sleep stage 1", etc.
        fs = raw.info['sfreq']
        epoch_samples = int(self.config.epoch_duration * fs)
        
        epochs_list = []
        for i, desc in enumerate(annotations.description):
            if desc == state:
                onset_time = annotations.onset[i]
                onset_sample = int(onset_time * fs)
                offset_sample = onset_sample + epoch_samples
                
                if offset_sample <= raw_ch.n_times:
                    data, _ = raw_ch[:, onset_sample:offset_sample]
                    epoch_data = data[0, :]
                    epochs_list.append(epoch_data)
                    
                    if len(epochs_list) >= self.config.n_epochs_per_state:
                        break
        
        return epochs_list
    
    def process_subject(self, subject_info: Dict) -> pd.DataFrame:
        """
        Procesa un sujeto completo
        
        Returns:
            DataFrame con PAC por epoch
        """
        subject_id = subject_info['subject_id']
        logger.info(f"Procesando {subject_id}...")
        
        try:
            raw, annotations = self.load_subject(
                subject_info['psg'],
                subject_info['hypno']
            )
        except Exception as e:
            logger.error(f"Error cargando {subject_id}: {e}")
            return pd.DataFrame()
        
        wake_epochs = self.extract_epochs_for_state(
            raw, annotations, self.config.wake_state
        )
        
        n2_epochs = self.extract_epochs_for_state(
            raw, annotations, self.config.n2_state
        )
        
        logger.info(f"  {subject_id}: {len(wake_epochs)} Wake, {len(n2_epochs)} N2 epochs")
        
        if len(wake_epochs) == 0 or len(n2_epochs) == 0:
            logger.warning(f"  {subject_id}: Insuficientes epochs, saltando")
            return pd.DataFrame()
        
        results = []
        
        for i, epoch_data in enumerate(wake_epochs):
            pac_result = self.pac_calc.compute_pac(epoch_data)
            results.append({
                'subject_id': subject_id,
                'state': 'wake',
                'consciousness': 1,
                'epoch_idx': i,
                'pac': pac_result['pac'],
                'valid': pac_result['valid'],
                'reject_reason': pac_result['reject_reason']
            })
        
        for i, epoch_data in enumerate(n2_epochs):
            pac_result = self.pac_calc.compute_pac(epoch_data)
            results.append({
                'subject_id': subject_id,
                'state': 'n2',
                'consciousness': 0,
                'epoch_idx': i,
                'pac': pac_result['pac'],
                'valid': pac_result['valid'],
                'reject_reason': pac_result['reject_reason']
            })
        
        df = pd.DataFrame(results)
        
        n_rejected = (~df['valid']).sum()
        if n_rejected > 0:
            logger.info(f"  {subject_id}: {n_rejected} epochs rechazadas")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS PRINCIPAL CON ECLIPSE v2.0
# ═══════════════════════════════════════════════════════════════════════════

def run_eclipse_analysis(config: Config):
    """Ejecuta análisis completo NIVEL 1 con ECLIPSE v2.0"""
    
    logger.info("=" * 80)
    logger.info("NIVEL 1: PAC THETA-GAMMA TEST - ECLIPSE v2.0")
    logger.info("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROCESAR DATOS
    # ─────────────────────────────────────────────────────────────────────────
    
    processor = SleepEDFProcessor(config)
    subject_files = processor.find_subject_files()
    
    if len(subject_files) == 0:
        logger.error("No se encontraron archivos Sleep-EDF")
        return
    
    logger.info(f"Procesando {len(subject_files)} sujetos...")
    
    all_data = []
    for i, subj_info in enumerate(subject_files, 1):
        logger.info(f"[{i}/{len(subject_files)}] {subj_info['subject_id']}")
        df = processor.process_subject(subj_info)
        if not df.empty:
            all_data.append(df)
    
    if len(all_data) == 0:
        logger.error("No se procesaron datos válidos")
        return
    
    full_df = pd.concat(all_data, ignore_index=True)
    
    valid_df = full_df[full_df['valid']].copy()
    
    logger.info(f"\nDatos procesados:")
    logger.info(f"  Total epochs: {len(full_df)}")
    logger.info(f"  Epochs válidas: {len(valid_df)}")
    logger.info(f"  Rechazadas: {len(full_df) - len(valid_df)}")
    logger.info(f"  Wake válidas: {(valid_df['consciousness']==1).sum()}")
    logger.info(f"  N2 válidas: {(valid_df['consciousness']==0).sum()}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v2.0 STAGE 1: IRREVERSIBLE SPLIT
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v2.0 STAGE 1: IRREVERSIBLE DATA SPLIT (70/30)")
    print("=" * 80)
    
    eclipse_config = EclipseConfig(
        project_name="NIVEL_1_PAC_THETA_GAMMA",
        researcher="Camilo Sjöberg Tala",
        sacred_seed=config.sacred_seed,
        development_ratio=0.7,
        holdout_ratio=0.3,
        n_folds_cv=5,
        output_dir=str(config.output_dir / 'eclipse_v2')
    )
    
    eclipse = EclipseFramework(eclipse_config)
    
    unique_subjects = valid_df['subject_id'].unique().tolist()
    
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(
        data_identifiers=unique_subjects
    )
    
    dev_data = valid_df[valid_df['subject_id'].isin(dev_subjects)].copy()
    holdout_data = valid_df[valid_df['subject_id'].isin(holdout_subjects)].copy()
    
    logger.info(f"\nSplit completado:")
    logger.info(f"  Development: {len(dev_subjects)} sujetos, {len(dev_data)} epochs")
    logger.info(f"  Holdout: {len(holdout_subjects)} sujetos, {len(holdout_data)} epochs")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v2.0 STAGE 2: CRITERIA REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v2.0 STAGE 2: PRE-REGISTRATION OF FALSIFICATION CRITERIA")
    print("=" * 80)
    
    criteria = [
        FalsificationCriteria(
            name="Cohen's d (Wake vs Sleep)",
            threshold=0.5,
            comparison=">=",
            description="Medium-to-large effect size (Cohen's d >= 0.5)",
            is_required=True
        ),
        FalsificationCriteria(
            name="p-value (one-tailed t-test)",
            threshold=0.05,
            comparison="<",
            description="Statistical significance (p < 0.05)",
            is_required=True
        )
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    print("\n📋 PRE-REGISTERED ANALYSIS PLAN:")
    print(f"""
NIVEL 1: AUTOPSYCHIC FOLD - PAC THETA-GAMMA TEST

HIPÓTESIS:
    H1: PAC_theta-gamma_wake > PAC_theta-gamma_N2
    H0: PAC_wake = PAC_N2

OPERACIONALIZACIÓN:
    Fold_intensity = PAC(theta_phase, gamma_amplitude)
    - Theta: {config.theta_band[0]}-{config.theta_band[1]} Hz (fase)
    - Gamma: {config.gamma_band[0]}-{config.gamma_band[1]} Hz (amplitud)
    - PAC: Modulation Index (Tort et al. 2010)
    - Método: Tensorpac con wavelet

DATASET:
    Sleep-EDF Database
    - {len(dev_subjects)} sujetos development
    - {len(holdout_subjects)} sujetos holdout
    - States: Wake vs N2 sleep
    - Epochs: {config.epoch_duration}s

ANÁLISIS ESTADÍSTICO:
    Independent samples t-test (wake vs N2)
    Cohen's d con pooled standard deviation
    One-tailed test (direccional: wake > N2)

CRITERIOS DE FALSIFICACIÓN VINCULANTES:
""")
    for crit in criteria:
        print(f"    • {crit}")
    
    print("""
COMPROMISOS:
    - No modificar análisis post-hoc
    - Publicar resultado (positivo o negativo)
    - Código público en GitHub
    - Una sola evaluación en holdout
""")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v2.0 STAGE 3: DEVELOPMENT & CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v2.0 STAGE 3: DEVELOPMENT (5-Fold Cross-Validation)")
    print("=" * 80)
    
    # ─────────────────────────────────────────────────────────────────────────
    def train_fn(dev_indices, **kwargs):
        """Función de 'entrenamiento' (no hay modelo, solo data)"""
        return dev_data
    
    def val_fn(model, val_indices, **kwargs):
        """Función de validación (calcula stats en CV fold)"""
        val_df = model.iloc[val_indices]
        
        wake = val_df[val_df['consciousness']==1]['pac'].values
        n2 = val_df[val_df['consciousness']==0]['pac'].values
        
        wake = wake[~np.isnan(wake)]
        n2 = n2[~np.isnan(n2)]
        
        if len(wake) == 0 or len(n2) == 0:
            return {
                "Cohen's d (Wake vs Sleep)": 0.0,
                "p-value (one-tailed t-test)": 1.0,
                "Mean PAC Wake": 0.0,
                "Mean PAC N2": 0.0
            }
        
        pooled_std = np.sqrt(
            ((len(wake)-1)*np.var(wake,ddof=1) + (len(n2)-1)*np.var(n2,ddof=1)) /
            (len(wake) + len(n2) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(n2)) / (pooled_std + 1e-10)
        
        _, p_val = stats.ttest_ind(wake, n2, alternative='greater')
        
        return {
            "Cohen's d (Wake vs Sleep)": cohens_d,
            "p-value (one-tailed t-test)": p_val,
            "Mean PAC Wake": np.mean(wake),
            "Mean PAC N2": np.mean(n2)
        }
    
    dev_results = eclipse.stage3_development(
        development_data=list(range(len(dev_data))),
        training_function=train_fn,
        validation_function=val_fn
    )
    
    print("\n✅ Desarrollo completado")
    if 'metrics' in dev_results:
        metrics_list = dev_results['metrics']
        avg = pd.DataFrame(metrics_list).mean()
        
        cohens_d_key = "Cohen's d (Wake vs Sleep)"
        pval_key = "p-value (one-tailed t-test)"
        mean_wake_key = "Mean PAC Wake"
        mean_n2_key = "Mean PAC N2"
        
        print(f"  • Cohen's d (promedio CV): {avg[cohens_d_key]:.3f}")
        print(f"  • p-value (promedio CV): {avg[pval_key]:.4f}")
        print(f"  • PAC Wake (promedio): {avg[mean_wake_key]:.4f}")
        print(f"  • PAC N2 (promedio): {avg[mean_n2_key]:.4f}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 4: VALIDACIÓN HOLDOUT (SINGLE-SHOT)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE STAGE 4: VALIDACIÓN SINGLE-SHOT EN HOLDOUT")
    print("=" * 80)
    print("⚠️  ÚNICA EVALUACIÓN EN DATOS DE HOLDOUT ⚠️")
    print("=" * 80)
    
    final_model = dev_data
    
    def final_val_fn(model, holdout_df, **kwargs):
        """Validación final en holdout"""
        wake = holdout_df[holdout_df['consciousness']==1]['pac'].values
        n2 = holdout_df[holdout_df['consciousness']==0]['pac'].values
        
        wake = wake[~np.isnan(wake)]
        n2 = n2[~np.isnan(n2)]
        
        if len(wake) == 0 or len(n2) == 0:
            return {
                "Cohen's d (Wake vs Sleep)": 0.0,
                "p-value (one-tailed t-test)": 1.0,
            }
        
        pooled_std = np.sqrt(
            ((len(wake)-1)*np.var(wake,ddof=1) + (len(n2)-1)*np.var(n2,ddof=1)) /
            (len(wake) + len(n2) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(n2)) / (pooled_std + 1e-10)
        
        _, p_val = stats.ttest_ind(wake, n2, alternative='greater')
        
        return {
            "Cohen's d (Wake vs Sleep)": cohens_d,
            "p-value (one-tailed t-test)": p_val,
            "Mean PAC Wake": np.mean(wake),
            "Mean PAC N2": np.mean(n2),
            "Std PAC Wake": np.std(wake, ddof=1),
            "Std PAC N2": np.std(n2, ddof=1),
            "n_wake": len(wake),
            "n_n2": len(n2)
        }
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=final_val_fn
    )
    
    if val_results is None:
        print("\n❌ Validación cancelada por el usuario")
        return
    
    print("\n🎯 RESULTADOS HOLDOUT:")
    for k, v in val_results.items():
        if isinstance(v, float):
            print(f"  • {k}: {v:.4f}")
        else:
            print(f"  • {k}: {v}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v2.0 STAGE 5: FINAL ASSESSMENT WITH EIS & STDS
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v2.0 STAGE 5: FINAL ASSESSMENT")
    print("=" * 80)
    print("Computing Eclipse Integrity Score (EIS) & Statistical Test for Data Snooping (STDS)...")
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True  # ✨ NEW IN v2.0: EIS + STDS
    )
    
    # Mostrar métricas v2.0
    if 'integrity_metrics' in final_assessment:
        integrity = final_assessment['integrity_metrics']
        
        print("\n📊 ECLIPSE v2.0 INTEGRITY METRICS:")
        
        # Eclipse Integrity Score (EIS)
        if 'eis' in integrity:
            eis_data = integrity['eis']
            print(f"\n🔒 Eclipse Integrity Score (EIS): {eis_data.get('eis', 0):.4f}")
            print("  Components:")
            components = eis_data.get('components', {})
            if components:
                print(f"    • Pre-registration: {components.get('preregistration', 0):.3f}")
                print(f"    • Split strength: {components.get('split_strength', 0):.3f}")
                print(f"    • Protocol adherence: {components.get('protocol_adherence', 0):.3f}")
                print(f"    • Leakage risk: {components.get('leakage_risk', 0):.3f}")
                print(f"    • Transparency: {components.get('transparency', 0):.3f}")
        
        # Statistical Test for Data Snooping (STDS)
        if 'stds' in integrity:
            stds_data = integrity['stds']
            if stds_data.get('status') == 'success':
                p_value = stds_data.get('p_value', 1.0)
                print(f"\n📈 Statistical Test for Data Snooping (STDS):")
                print(f"    • p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("    • ⚠️  WARNING: Possible data snooping detected (p < 0.05)")
                else:
                    print("    • ✅ No evidence of data snooping")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VEREDICTO FINAL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("🔥 VEREDICTO FINAL - NIVEL 1 PAC THETA-GAMMA")
    print("=" * 80)
    
    # Extraer métricas desde el diccionario anidado
    metrics = val_results.get('metrics', val_results)
    cohens_d = metrics["Cohen's d (Wake vs Sleep)"]
    p_value = metrics["p-value (one-tailed t-test)"]
    
    print(f"\n📊 Resultados Holdout:")
    print(f"  • Cohen's d = {cohens_d:.3f}")
    print(f"  • p-value = {p_value:.4f}")
    print(f"  • PAC Wake = {metrics.get('Mean PAC Wake', 0):.4f}")
    print(f"  • PAC N2 = {metrics.get('Mean PAC N2', 0):.4f}")
    
    print(f"\n✅ Criterios Pre-Registrados:")
    print(f"  • Cohen's d ≥ 0.5: {'✅ SÍ' if cohens_d >= 0.5 else '❌ NO'} (d={cohens_d:.3f})")
    print(f"  • p < 0.05: {'✅ SÍ' if p_value < 0.05 else '❌ NO'} (p={p_value:.4f})")
    
    # Evaluación de criterios
    criteria_met = cohens_d >= 0.5 and p_value < 0.05
    
    if criteria_met:
        print("\n" + "=" * 80)
        print("✅ NIVEL 1: FOLD VALIDADO")
        print("=" * 80)
        print("\n🎯 CONCLUSIÓN:")
        print("  • Existe convergencia temporal theta-gamma que discrimina consciencia")
        print("  • El Autopsychic Fold tiene evidencia preliminar robusta")
        print("  • Effect size es medium-to-large (Cohen's d >= 0.5)")
        print("  • Significancia estadística alcanzada (p < 0.05)")
        print("\n🚀 PRÓXIMO PASO:")
        print("  → Proceder a NIVEL 2: Especificidad de bandas frecuenciales")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ NIVEL 1: FOLD FALSIFICADO")
        print("=" * 80)
        print("\n💔 CONCLUSIÓN:")
        print("  • NO existe convergencia temporal medible en theta-gamma")
        print("  • El Autopsychic Fold NO tiene evidencia empírica")
        print("  • Criterios de falsificación vinculantes NO cumplidos")
        print("\n⚖️  DECISIÓN HONESTA:")
        print("  → ABANDONAR AFH en su forma actual")
        print("  → Publicar resultado negativo con transparencia total")
        print("  → Re-evaluar premisas teóricas fundamentales")
        print("=" * 80)
    
    # Resumen ECLIPSE v2.0
    print("\n" + "=" * 80)
    print("📋 ECLIPSE v2.0 SUMMARY")
    print("=" * 80)
    print(eclipse.generate_summary())
    
    print(f"\n📁 Archivos generados:")
    print(f"  • Carpeta: {config.output_dir / 'eclipse_v2'}")
    print(f"  • Reporte principal: {eclipse_config.project_name}_REPORT.html")
    print(f"  • EIS Report: {eclipse_config.project_name}_EIS_REPORT.txt")
    print(f"  • STDS Report: {eclipse_config.project_name}_STDS_REPORT.txt")
    print(f"  • Lockfile: lockfile.json (integridad criptográfica)")
    
    # Verificar integridad
    print("\n🔐 Verificando integridad del pipeline...")
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS NIVEL 1 FINALIZADO")
    print("=" * 80)
    print("\n🏆 ECLIPSE v2.0 Features Utilizados:")
    print("  ✅ Eclipse Integrity Score (EIS) - Rigor cuantificado")
    print("  ✅ Statistical Test for Data Snooping (STDS) - P-hacking detection")
    print("  ✅ Automated Code Auditor - Protocol compliance")
    print("  ✅ Irreversible split - No data leakage")
    print("  ✅ Single-shot validation - No second chances")
    print("  ✅ Cryptographic integrity - Full reproducibility")
    print("\n" + "=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NIVEL 1: PAC Theta-Gamma Test with ECLIPSE v2.0",
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
        default='./nivel1_pac_results',
        help='Directorio de salida'
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--pilot',
        action='store_true',
        help='Modo piloto (10 sujetos, ~10 min)'
    )
    
    group.add_argument(
        '--full',
        action='store_true',
        help='Análisis completo (todos los sujetos)'
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
    
    if args.pilot:
        n_subjects = 10
    elif args.full:
        n_subjects = None
    elif args.n_subjects:
        n_subjects = args.n_subjects
    else:
        logger.warning("No se especificó modo. Usando --pilot por defecto.")
        n_subjects = 10
    
    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        sacred_seed=args.seed,
        n_subjects=n_subjects
    )
    
    if not config.data_dir.exists():
        print(f"\n❌ ERROR: No se encontró el directorio de datos:")
        print(f"   {config.data_dir}")
        print(f"\nDescarga Sleep-EDF y ajusta --data-dir")
        sys.exit(1)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
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