#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIVEL 1: Phase-Amplitude Coupling (PAC) Theta-Gamma Test
Autopsychic Fold Hypothesis - ECLIPSE v3.0
═══════════════════════════════════════════════════════════════════════════════

VERSIÓN: ANÁLISIS COMPLETO (153 sujetos)

HIPÓTESIS:
    H1: PAC_theta-gamma_wake > PAC_theta-gamma_N2
    
OPERACIONALIZACIÓN DEL FOLD:
    Fold_intensity = PAC(theta_phase, gamma_amplitude)
    - Theta (4-8 Hz) = procesamiento cortical lento
    - Gamma (30-45 Hz) = procesamiento sensoriomotor rápido
    - PAC = acoplamiento fase-amplitud (Modulation Index)

CRITERIO DE FALSIFICACIÓN:
    Si Cohen's d < 0.5 O p ≥ 0.05 → FOLD FALSIFICADO → ABANDONAR AFH

Author: Camilo Sjöberg Tala
Date: 2025-12-09
Version: NIVEL_1_v2.2_FULL (ECLIPSE v3.0) - Fixed f-string syntax

EJECUCIÓN:
    python PAC_AFH_NIVEL1_FULL_v2.py
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from dataclasses import dataclass
import sys

# Suprimir warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
mne.set_log_level('ERROR')

# Configurar logging - menos verboso
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suprimir logging de tensorpac
logging.getLogger('tensorpac').setLevel(logging.ERROR)

# Tensorpac para PAC
from tensorpac import Pac

# Import ECLIPSE v3.0 framework
try:
    from eclipse_v3 import (
        EclipseFramework, EclipseConfig, FalsificationCriteria
    )
except ImportError:
    print("ERROR: No se encontró eclipse_v3.py")
    print("Asegúrate de que eclipse_v3.py esté en el mismo directorio")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

# Keys for metrics (avoid backslash issues in f-strings)
KEY_COHENS_D = "Cohen's d (Wake vs Sleep)"
KEY_PVALUE = "p-value (one-tailed t-test)"
KEY_PAC_WAKE = "Mean PAC Wake"
KEY_PAC_N2 = "Mean PAC N2"
KEY_STD_WAKE = "Std PAC Wake"
KEY_STD_N2 = "Std PAC N2"


@dataclass
class Config:
    """Configuración NIVEL 1: PAC Test - FULL ANALYSIS"""
    # Rutas
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./nivel1_pac_results_v3_FULL')
    
    # Procesamiento de señal
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    notch_freq: float = 50.0
    
    # PAC theta-gamma
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 45.0)
    
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
    n_subjects: Optional[int] = None  # None = TODOS los sujetos
    
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
        
        # Crear objeto PAC de tensorpac (suprimir warning)
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
        """Detecta artefactos por amplitud"""
        peak_to_peak = np.ptp(data)
        return peak_to_peak > 8.0
    
    def compute_pac(self, data: np.ndarray) -> Dict[str, float]:
        """Calcula PAC en una epoch"""
        data_clean = self.preprocess(data)
        
        if self.has_artifact(data_clean):
            return {'pac': np.nan, 'valid': False, 'reject_reason': 'artifact'}
        
        min_samples = int(2.0 * self.fs)
        if len(data_clean) < min_samples:
            return {'pac': np.nan, 'valid': False, 'reject_reason': 'too_short'}
        
        data_reshaped = data_clean[np.newaxis, :]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pac_value = self.pac.filterfit(self.fs, data_reshaped, data_reshaped)
            pac_value = float(pac_value[0, 0, 0])
            return {'pac': pac_value, 'valid': True, 'reject_reason': None}
        except Exception as e:
            return {'pac': np.nan, 'valid': False, 'reject_reason': 'computation_error: {}'.format(e)}


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR DE DATOS SLEEP-EDF
# ═══════════════════════════════════════════════════════════════════════════

class SleepEDFProcessor:
    """Carga y procesa datos Sleep-EDF"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pac_calc = PACCalculator(config)
        
    def find_subject_files(self) -> List[Dict[str, Path]]:
        """Encuentra pares PSG + Hypnogram"""
        print("\n[SEARCH] Buscando archivos en: {}".format(self.config.data_dir))
        
        psg_files = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno_files = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        
        print("  Archivos PSG: {}".format(len(psg_files)))
        print("  Archivos Hypnogram: {}".format(len(hypno_files)))
        
        hypno_map = {}
        for hypno_path in hypno_files:
            codigo_hypno = hypno_path.stem.replace("-Hypnogram", "")
            if len(codigo_hypno) >= 6:
                base_letra = codigo_hypno[:-1]
                hypno_map[base_letra] = hypno_path
        
        subject_files = []
        for psg_path in psg_files:
            codigo_psg = psg_path.stem.replace("-PSG", "")
            if len(codigo_psg) >= 7 and codigo_psg.endswith('0'):
                base_letra = codigo_psg[:-1]
                if base_letra in hypno_map:
                    subject_files.append({
                        'psg': psg_path,
                        'hypno': hypno_map[base_letra],
                        'subject_id': codigo_psg
                    })
        
        print("  Sujetos emparejados: {}".format(len(subject_files)))
        
        if self.config.n_subjects is not None:
            subject_files = subject_files[:self.config.n_subjects]
            print("  Limitado a: {} sujetos".format(len(subject_files)))
        
        return subject_files
    
    def load_subject(self, psg_path: Path, hypno_path: Path) -> Tuple[mne.io.Raw, mne.Annotations]:
        """Carga PSG y anotaciones"""
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annotations = mne.read_annotations(hypno_path)
        return raw, annotations
    
    def extract_epochs_for_state(self, raw: mne.io.Raw, annotations: mne.Annotations, state: str) -> List[np.ndarray]:
        """Extrae epochs de 30s para un estado específico"""
        available_channel = None
        preferred_channels = ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']
        
        for ch in preferred_channels:
            if ch in raw.ch_names:
                available_channel = ch
                break
        
        if available_channel is None:
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper()]
            if len(eeg_channels) > 0:
                available_channel = eeg_channels[0]
            else:
                return []
        
        raw_ch = raw.copy().pick_channels([available_channel])
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
                    epochs_list.append(data[0, :])
                    
                    if len(epochs_list) >= self.config.n_epochs_per_state:
                        break
        
        return epochs_list
    
    def process_subject(self, subject_info: Dict) -> pd.DataFrame:
        """Procesa un sujeto completo"""
        subject_id = subject_info['subject_id']
        
        try:
            raw, annotations = self.load_subject(subject_info['psg'], subject_info['hypno'])
        except Exception as e:
            logger.error("Error cargando {}: {}".format(subject_id, e))
            return pd.DataFrame()
        
        wake_epochs = self.extract_epochs_for_state(raw, annotations, self.config.wake_state)
        n2_epochs = self.extract_epochs_for_state(raw, annotations, self.config.n2_state)
        
        if len(wake_epochs) == 0 or len(n2_epochs) == 0:
            return pd.DataFrame()
        
        results = []
        
        for i, epoch_data in enumerate(wake_epochs):
            pac_result = self.pac_calc.compute_pac(epoch_data)
            results.append({
                'subject_id': subject_id, 'state': 'wake', 'consciousness': 1,
                'epoch_idx': i, 'pac': pac_result['pac'],
                'valid': pac_result['valid'], 'reject_reason': pac_result['reject_reason']
            })
        
        for i, epoch_data in enumerate(n2_epochs):
            pac_result = self.pac_calc.compute_pac(epoch_data)
            results.append({
                'subject_id': subject_id, 'state': 'n2', 'consciousness': 0,
                'epoch_idx': i, 'pac': pac_result['pac'],
                'valid': pac_result['valid'], 'reject_reason': pac_result['reject_reason']
            })
        
        return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS PRINCIPAL CON ECLIPSE v3.0
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis():
    """Ejecuta análisis COMPLETO NIVEL 1 con ECLIPSE v3.0"""
    
    config = Config()
    
    print("=" * 80)
    print("NIVEL 1: PAC THETA-GAMMA TEST - ANALISIS COMPLETO")
    print("   ECLIPSE v3.0 - Autopsychic Fold Hypothesis")
    print("=" * 80)
    print("\n[OUTPUT] Directorio de salida: {}".format(config.output_dir))
    print("[SEED] Sacred seed: {}".format(config.sacred_seed))
    print("[MODE] Modo: ANALISIS COMPLETO (todos los sujetos)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROCESAR DATOS
    # ─────────────────────────────────────────────────────────────────────────
    
    processor = SleepEDFProcessor(config)
    subject_files = processor.find_subject_files()
    
    if len(subject_files) == 0:
        print("\n[ERROR] No se encontraron archivos Sleep-EDF")
        return
    
    print("\n[PROCESSING] Procesando {} sujetos...".format(len(subject_files)))
    print("   (Esto tomara aproximadamente 1-2 horas)")
    print()
    
    all_data = []
    for i, subj_info in enumerate(subject_files, 1):
        # Progreso cada 10 sujetos
        if i % 10 == 0 or i == 1:
            print("   [{}/{}] Procesando {}...".format(i, len(subject_files), subj_info['subject_id']))
        
        df = processor.process_subject(subj_info)
        if not df.empty:
            all_data.append(df)
    
    if len(all_data) == 0:
        print("\n[ERROR] No se procesaron datos validos")
        return
    
    full_df = pd.concat(all_data, ignore_index=True)
    valid_df = full_df[full_df['valid']].copy()
    
    n_total = len(full_df)
    n_valid = len(valid_df)
    pct_valid = 100 * n_valid / n_total
    n_wake = (valid_df['consciousness'] == 1).sum()
    n_n2 = (valid_df['consciousness'] == 0).sum()
    
    print("\n[DATA] Datos procesados:")
    print("  - Sujetos procesados: {}".format(len(all_data)))
    print("  - Total epochs: {}".format(n_total))
    print("  - Epochs validas: {} ({:.1f}%)".format(n_valid, pct_valid))
    print("  - Wake validas: {}".format(n_wake))
    print("  - N2 validas: {}".format(n_n2))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v3.0 STAGE 1: IRREVERSIBLE SPLIT (FORCE NEW)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v3.0 STAGE 1: IRREVERSIBLE DATA SPLIT (70/30)")
    print("=" * 80)
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    eclipse_config = EclipseConfig(
        project_name="NIVEL_1_PAC_THETA_GAMMA_FULL",
        researcher="Camilo Sjoberg Tala",
        sacred_seed=config.sacred_seed,
        development_ratio=0.7,
        holdout_ratio=0.3,
        n_folds_cv=5,
        output_dir=str(config.output_dir / 'eclipse_v3'),
        non_interactive=False,
        stds_alpha=0.05,
        audit_pass_threshold=70.0
    )
    
    eclipse = EclipseFramework(eclipse_config)
    
    unique_subjects = valid_df['subject_id'].unique().tolist()
    print("\n[SPLIT] Sujetos unicos para split: {}".format(len(unique_subjects)))
    
    # FORCE=TRUE para crear nuevo split (no usar el del piloto)
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(
        data_identifiers=unique_subjects,
        force=True  # IMPORTANTE: Forzar nuevo split
    )
    
    dev_data = valid_df[valid_df['subject_id'].isin(dev_subjects)].copy()
    holdout_data = valid_df[valid_df['subject_id'].isin(holdout_subjects)].copy()
    
    print("\n[OK] Split completado:")
    print("  - Development: {} sujetos, {} epochs".format(len(dev_subjects), len(dev_data)))
    print("  - Holdout: {} sujetos, {} epochs".format(len(holdout_subjects), len(holdout_data)))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v3.0 STAGE 2: CRITERIA REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v3.0 STAGE 2: PRE-REGISTRATION OF FALSIFICATION CRITERIA")
    print("=" * 80)
    
    criteria = [
        FalsificationCriteria(
            name=KEY_COHENS_D,
            threshold=0.5,
            comparison=">=",
            description="Medium-to-large effect size (Cohen's d >= 0.5)",
            is_required=True
        ),
        FalsificationCriteria(
            name=KEY_PVALUE,
            threshold=0.05,
            comparison="<",
            description="Statistical significance (p < 0.05)",
            is_required=True
        )
    ]
    
    eclipse.stage2_register_criteria(criteria, force=True)
    
    print("""
PRE-REGISTERED ANALYSIS PLAN:

NIVEL 1: AUTOPSYCHIC FOLD - PAC THETA-GAMMA TEST (FULL)

HIPOTESIS:
    H1: PAC_theta-gamma_wake > PAC_theta-gamma_N2
    H0: PAC_wake = PAC_N2

OPERACIONALIZACION:
    Fold_intensity = PAC(theta_phase, gamma_amplitude)
    - Theta: {}-{} Hz (fase)
    - Gamma: {}-{} Hz (amplitud)
    - PAC: Modulation Index (Tort et al. 2010)

DATASET:
    Sleep-EDF Database (COMPLETO)
    - {} sujetos development
    - {} sujetos holdout
    - {} epochs development
    - {} epochs holdout

CRITERIOS DE FALSIFICACION VINCULANTES:
    - Cohen's d >= 0.5 [REQUIRED]
    - p-value < 0.05 [REQUIRED]
""".format(
        config.theta_band[0], config.theta_band[1],
        config.gamma_band[0], config.gamma_band[1],
        len(dev_subjects), len(holdout_subjects),
        len(dev_data), len(holdout_data)
    ))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v3.0 STAGE 3: DEVELOPMENT & CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v3.0 STAGE 3: DEVELOPMENT (5-Fold Cross-Validation)")
    print("=" * 80)
    
    def train_fn(dev_indices, **kwargs):
        return dev_data
    
    def val_fn(model, val_indices, **kwargs):
        val_df = model.iloc[val_indices]
        
        wake = val_df[val_df['consciousness'] == 1]['pac'].values
        n2 = val_df[val_df['consciousness'] == 0]['pac'].values
        
        wake = wake[~np.isnan(wake)]
        n2 = n2[~np.isnan(n2)]
        
        if len(wake) == 0 or len(n2) == 0:
            return {
                KEY_COHENS_D: 0.0,
                KEY_PVALUE: 1.0,
                KEY_PAC_WAKE: 0.0,
                KEY_PAC_N2: 0.0
            }
        
        pooled_std = np.sqrt(
            ((len(wake) - 1) * np.var(wake, ddof=1) + (len(n2) - 1) * np.var(n2, ddof=1)) /
            (len(wake) + len(n2) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(n2)) / (pooled_std + 1e-10)
        _, p_val = stats.ttest_ind(wake, n2, alternative='greater')
        
        return {
            KEY_COHENS_D: cohens_d,
            KEY_PVALUE: p_val,
            KEY_PAC_WAKE: np.mean(wake),
            KEY_PAC_N2: np.mean(n2)
        }
    
    dev_results = eclipse.stage3_development(
        development_data=list(range(len(dev_data))),
        training_function=train_fn,
        validation_function=val_fn
    )
    
    print("\n[OK] Desarrollo completado")
    if 'aggregated_metrics' in dev_results:
        agg = dev_results['aggregated_metrics']
        d_mean = agg[KEY_COHENS_D]['mean']
        d_std = agg[KEY_COHENS_D]['std']
        p_mean = agg[KEY_PVALUE]['mean']
        p_std = agg[KEY_PVALUE]['std']
        pac_wake_mean = agg[KEY_PAC_WAKE]['mean']
        pac_n2_mean = agg[KEY_PAC_N2]['mean']
        
        print("  - Cohen's d (CV): {:.3f} +/- {:.3f}".format(d_mean, d_std))
        print("  - p-value (CV): {:.4f} +/- {:.4f}".format(p_mean, p_std))
        print("  - PAC Wake: {:.4f}".format(pac_wake_mean))
        print("  - PAC N2: {:.4f}".format(pac_n2_mean))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE STAGE 4: VALIDACIÓN HOLDOUT (SINGLE-SHOT)
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v3.0 STAGE 4: VALIDACION SINGLE-SHOT EN HOLDOUT")
    print("=" * 80)
    print(">>> UNICA EVALUACION EN DATOS DE HOLDOUT <<<")
    print("=" * 80)
    
    def final_val_fn(model, holdout_df, **kwargs):
        wake = holdout_df[holdout_df['consciousness'] == 1]['pac'].values
        n2 = holdout_df[holdout_df['consciousness'] == 0]['pac'].values
        
        wake = wake[~np.isnan(wake)]
        n2 = n2[~np.isnan(n2)]
        
        if len(wake) == 0 or len(n2) == 0:
            return {KEY_COHENS_D: 0.0, KEY_PVALUE: 1.0}
        
        pooled_std = np.sqrt(
            ((len(wake) - 1) * np.var(wake, ddof=1) + (len(n2) - 1) * np.var(n2, ddof=1)) /
            (len(wake) + len(n2) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(n2)) / (pooled_std + 1e-10)
        _, p_val = stats.ttest_ind(wake, n2, alternative='greater')
        
        return {
            KEY_COHENS_D: cohens_d,
            KEY_PVALUE: p_val,
            KEY_PAC_WAKE: np.mean(wake),
            KEY_PAC_N2: np.mean(n2),
            KEY_STD_WAKE: np.std(wake, ddof=1),
            KEY_STD_N2: np.std(n2, ddof=1),
            "n_wake": len(wake),
            "n_n2": len(n2)
        }
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=dev_data,
        validation_function=final_val_fn,
        force=True  # Forzar nueva validacion
    )
    
    if val_results is None:
        print("\n[CANCELLED] Validacion cancelada")
        return
    
    print("\n[HOLDOUT RESULTS]:")
    metrics = val_results.get('metrics', val_results)
    for k, v in metrics.items():
        if isinstance(v, float):
            print("  - {}: {:.4f}".format(k, v))
        else:
            print("  - {}: {}".format(k, v))
    
    # ─────────────────────────────────────────────────────────────────────────
    # ECLIPSE v3.0 STAGE 5: FINAL ASSESSMENT
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("ECLIPSE v3.0 STAGE 5: FINAL ASSESSMENT")
    print("=" * 80)
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True
    )
    
    # Mostrar métricas v3.0
    if 'integrity_metrics' in final_assessment:
        integrity = final_assessment['integrity_metrics']
        
        print("\n[ECLIPSE v3.0 INTEGRITY METRICS]:")
        
        if 'eis' in integrity:
            eis_data = integrity['eis']
            eis_value = eis_data.get('eis', 0)
            eis_interp = eis_data.get('interpretation', 'N/A')
            print("\n[EIS] Eclipse Integrity Score: {:.4f}".format(eis_value))
            print("   {}".format(eis_interp))
            components = eis_data.get('components', {})
            if components:
                print("  Components:")
                print("    - Pre-registration: {:.3f}".format(components.get('preregistration_score', 0)))
                print("    - Split strength: {:.3f}".format(components.get('split_strength', 0)))
                print("    - Protocol adherence: {:.3f}".format(components.get('protocol_adherence', 0)))
                print("    - Leakage score: {:.3f}".format(components.get('leakage_score', 0)))
                print("    - Transparency: {:.3f}".format(components.get('transparency_score', 0)))
        
        if 'stds' in integrity:
            stds_data = integrity['stds']
            if stds_data.get('status') == 'success':
                max_z = stds_data.get('max_z_score', 0)
                mean_z = stds_data.get('mean_z_score', 0)
                risk_level = stds_data.get('risk_level', 'N/A')
                
                print("\n[STDS] Statistical Test for Data Snooping:")
                print("    - Max z-score: {:+.4f}".format(max_z))
                print("    - Mean z-score: {:+.4f}".format(mean_z))
                print("    - Risk Level: {}".format(risk_level))
                
                if max_z > 3:
                    print("    - [WARNING] Very unusual (|z| > 3)")
                elif max_z > 2:
                    print("    - [NOTABLE] |z| > 2")
                else:
                    print("    - [OK] Normal range")
    
    # ─────────────────────────────────────────────────────────────────────────
    # VEREDICTO FINAL
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print(">>> VEREDICTO FINAL - NIVEL 1 PAC THETA-GAMMA (FULL) <<<")
    print("=" * 80)
    
    cohens_d = metrics[KEY_COHENS_D]
    p_value = metrics[KEY_PVALUE]
    
    print("\n[HOLDOUT] Resultados (n={} sujetos):".format(len(holdout_subjects)))
    print("  - Cohen's d = {:.3f}".format(cohens_d))
    print("  - p-value = {:.6f}".format(p_value))
    print("  - PAC Wake = {:.4f} +/- {:.4f}".format(
        metrics.get(KEY_PAC_WAKE, 0), metrics.get(KEY_STD_WAKE, 0)))
    print("  - PAC N2 = {:.4f} +/- {:.4f}".format(
        metrics.get(KEY_PAC_N2, 0), metrics.get(KEY_STD_N2, 0)))
    print("  - n_wake = {}, n_n2 = {}".format(
        int(metrics.get('n_wake', 0)), int(metrics.get('n_n2', 0))))
    
    print("\n[CRITERIA] Pre-Registrados:")
    d_pass = cohens_d >= 0.5
    p_pass = p_value < 0.05
    d_status = "[PASS]" if d_pass else "[FAIL]"
    p_status = "[PASS]" if p_pass else "[FAIL]"
    print("  - Cohen's d >= 0.5: {} (d={:.3f})".format(d_status, cohens_d))
    print("  - p < 0.05: {} (p={:.6f})".format(p_status, p_value))
    
    criteria_met = d_pass and p_pass
    
    if criteria_met:
        print("\n" + "=" * 80)
        print(">>> NIVEL 1: FOLD VALIDADO <<<")
        print("=" * 80)
        print("\n[CONCLUSION]:")
        print("  - Existe convergencia temporal theta-gamma que discrimina consciencia")
        print("  - El Autopsychic Fold tiene evidencia empirica robusta")
        print("  - Effect size medium-to-large confirmado")
        print("  - Significancia estadistica alcanzada")
        print("\n[NEXT STEP]:")
        print("  -> Proceder a NIVEL 2: Especificidad de bandas frecuenciales")
    else:
        print("\n" + "=" * 80)
        print(">>> NIVEL 1: FOLD FALSIFICADO <<<")
        print("=" * 80)
        print("\n[CONCLUSION]:")
        print("  - NO existe convergencia temporal medible en theta-gamma")
        print("  - El Autopsychic Fold NO tiene evidencia empirica suficiente")
        print("  - Criterios de falsificacion vinculantes NO cumplidos")
        print("\n[DECISION]:")
        print("  -> ABANDONAR AFH en su forma actual")
        print("  -> Publicar resultado negativo con transparencia total")
    
    print("\n" + "=" * 80)
    print("[FILES] Archivos generados en: {}".format(config.output_dir / 'eclipse_v3'))
    print("=" * 80)
    
    # Verificar integridad
    print("\n[VERIFY] Verificando integridad...")
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("[DONE] ANALISIS NIVEL 1 COMPLETO FINALIZADO")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        run_full_analysis()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Analisis interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print("\n\n[ERROR] Durante el analisis:")
        print("   {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)