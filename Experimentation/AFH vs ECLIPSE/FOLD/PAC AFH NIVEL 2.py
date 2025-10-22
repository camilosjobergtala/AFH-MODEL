#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
NIVEL 2: PAC Band Specificity Test
Autopsychic Fold Hypothesis - ECLIPSE v2.0
═══════════════════════════════════════════════════════════════════════════════

HIPÓTESIS:
    H1: PAC_theta-gamma tiene MAYOR discriminación que otras bandas
    H2: Si TODAS las bandas discriminan igual → NO hay especificidad
    
BANDAS TESTEADAS:
    1. Delta-Gamma (1-4 Hz, 30-45 Hz)
    2. Theta-Gamma (4-8 Hz, 30-45 Hz)   ← NIVEL 1 validado (d=2.437)
    3. Alpha-Gamma (8-12 Hz, 30-45 Hz)
    4. Beta-Gamma (13-30 Hz, 30-45 Hz)

PREDICCIÓN AFH:
    Cohen's d_theta-gamma > d_otros
    
CRITERIO DE FALSIFICACIÓN NIVEL 2:
    Si d_theta-gamma ≤ promedio(d_otros) → NO hay especificidad → Replantear AFH

Author: Camilo Sjöberg Tala
Date: 2025-10-22
Version: NIVEL_2_v2.0_FIXED

EJECUCIÓN:
    python NIVEL2_PAC_BAND_SPECIFICITY_FIXED.py --pilot
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

from tensorpac import Pac

warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN MULTI-BANDA
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
    """Configuración NIVEL 2: Multi-Band PAC Test"""
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
            description="AFH prediction - MAIN"
        ),
        BandPair(
            name="Alpha-Gamma",
            phase_band=(8.0, 12.0),
            amp_band=(30.0, 45.0),
            description="Attention-related coupling"
        ),
        BandPair(
            name="Beta-Gamma",
            phase_band=(13.0, 30.0),
            amp_band=(30.0, 45.0),
            description="Fast cortical coupling"
        ),
    ])
    
    epoch_duration: float = 30.0
    n_epochs_per_state: int = 100
    
    wake_state: str = 'Sleep stage W'
    n2_state: str = 'Sleep stage 2'
    
    sacred_seed: int = 42
    n_subjects: Optional[int] = None


# ═══════════════════════════════════════════════════════════════════════════
# CALCULADORA PAC MULTI-BANDA
# ═══════════════════════════════════════════════════════════════════════════

class MultiPACCalculator:
    """Calcula PAC para múltiples pares de bandas"""
    
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
            logger.info(f"PAC object created: {band_pair.name} - "
                       f"phase {band_pair.phase_band} Hz, amp {band_pair.amp_band} Hz")
        
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
    
    def compute_all_pac(self, data: np.ndarray) -> Dict[str, Dict]:
        """
        Calcula PAC para TODAS las bandas en una epoch
        
        Returns:
            {
                'Delta-Gamma': {'pac': X, 'valid': True, ...},
                'Theta-Gamma': {'pac': Y, 'valid': True, ...},
                ...
            }
        """
        data_clean = self.preprocess(data)
        
        if self.has_artifact(data_clean):
            return {
                band_name: {
                    'pac': np.nan,
                    'valid': False,
                    'reject_reason': 'artifact'
                }
                for band_name in self.pac_objects.keys()
            }
        
        min_samples = int(2.0 * self.fs)
        if len(data_clean) < min_samples:
            return {
                band_name: {
                    'pac': np.nan,
                    'valid': False,
                    'reject_reason': 'too_short'
                }
                for band_name in self.pac_objects.keys()
            }
        
        results = {}
        data_reshaped = data_clean[np.newaxis, :]
        
        for band_name, pac_obj in self.pac_objects.items():
            try:
                pac_value = pac_obj.filterfit(self.fs, data_reshaped, data_reshaped)
                pac_value = float(pac_value[0, 0, 0])
                
                results[band_name] = {
                    'pac': pac_value,
                    'valid': True,
                    'reject_reason': None
                }
            except Exception as e:
                logger.warning(f"Error en PAC {band_name}: {e}")
                results[band_name] = {
                    'pac': np.nan,
                    'valid': False,
                    'reject_reason': f'computation_error: {e}'
                }
        
        return results


# ═══════════════════════════════════════════════════════════════════════════
# PROCESADOR SLEEP-EDF CON MULTI-PAC (USANDO LÓGICA NIVEL 1)
# ═══════════════════════════════════════════════════════════════════════════

class SleepEDFProcessor:
    """Carga y procesa datos Sleep-EDF con multi-PAC"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pac_calc = MultiPACCalculator(config)
        
    def find_subject_files(self) -> List[Dict[str, Path]]:
        """Encuentra pares PSG + Hypnogram"""
        
        psg_files = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno_files = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        
        logger.info(f"PSG files: {len(psg_files)}, Hypnogram files: {len(hypno_files)}")
        
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
        
        logger.info(f"Encontrados {len(subject_files)} sujetos emparejados")
        
        if self.config.n_subjects is not None:
            subject_files = subject_files[:self.config.n_subjects]
            logger.info(f"Limitado a {len(subject_files)} sujetos")
        
        return subject_files
    
    def load_subject(self, psg_path: Path, hypno_path: Path) -> Tuple[mne.io.Raw, mne.Annotations]:
        """Carga PSG y anotaciones - IGUAL QUE NIVEL 1"""
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annotations = mne.read_annotations(hypno_path)
        
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
        Extrae epochs de 30s para un estado - IGUAL QUE NIVEL 1
        """
        available_channel = None
        
        preferred_channels = ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']
        
        for ch in preferred_channels:
            if ch in raw.ch_names:
                available_channel = ch
                break
        
        if available_channel is None:
            eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch.upper() or ch.startswith('EEG')]
            if len(eeg_channels) > 0:
                available_channel = eeg_channels[0]
            else:
                logger.warning(f"No se encontró ningún canal EEG")
                return []
        
        logger.info(f"  Usando canal: {available_channel}")
        
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
                    epoch_data = data[0, :]
                    epochs_list.append(epoch_data)
                    
                    if len(epochs_list) >= self.config.n_epochs_per_state:
                        break
        
        return epochs_list
    
    def process_subject(self, subject_info: Dict) -> pd.DataFrame:
        """Procesa un sujeto con TODAS las bandas PAC"""
        
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
        
        wake_epochs = self.extract_epochs_for_state(raw, annotations, self.config.wake_state)
        n2_epochs = self.extract_epochs_for_state(raw, annotations, self.config.n2_state)
        
        logger.info(f"  {subject_id}: {len(wake_epochs)} Wake, {len(n2_epochs)} N2 epochs")
        
        if len(wake_epochs) == 0 or len(n2_epochs) == 0:
            logger.warning(f"  {subject_id}: Insuficientes epochs")
            return pd.DataFrame()
        
        # Calcular PAC para TODAS las bandas, TODAS las epochs
        results = []
        
        # Wake epochs
        for i, epoch_data in enumerate(wake_epochs):
            pac_results = self.pac_calc.compute_all_pac(epoch_data)
            
            for band_name, pac_info in pac_results.items():
                results.append({
                    'subject_id': subject_id,
                    'state': 'wake',
                    'consciousness': 1,
                    'epoch_idx': i,
                    'band_pair': band_name,
                    'pac': pac_info['pac'],
                    'valid': pac_info['valid'],
                    'reject_reason': pac_info['reject_reason']
                })
        
        # N2 epochs
        for i, epoch_data in enumerate(n2_epochs):
            pac_results = self.pac_calc.compute_all_pac(epoch_data)
            
            for band_name, pac_info in pac_results.items():
                results.append({
                    'subject_id': subject_id,
                    'state': 'n2',
                    'consciousness': 0,
                    'epoch_idx': i,
                    'band_pair': band_name,
                    'pac': pac_info['pac'],
                    'valid': pac_info['valid'],
                    'reject_reason': pac_info['reject_reason']
                })
        
        df = pd.DataFrame(results)
        
        n_rejected = (~df['valid']).sum()
        if n_rejected > 0:
            logger.info(f"  {subject_id}: {n_rejected} measurements rechazadas")
        
        return df


# ═══════════════════════════════════════════════════════════════════════════
# ANÁLISIS MULTI-BANDA
# ═══════════════════════════════════════════════════════════════════════════

def run_multiband_analysis(config: Config):
    """Ejecuta análisis NIVEL 2 con múltiples bandas"""
    
    logger.info("=" * 80)
    logger.info("NIVEL 2: PAC BAND SPECIFICITY TEST - ECLIPSE v2.0")
    logger.info("=" * 80)
    
    processor = SleepEDFProcessor(config)
    subject_files = processor.find_subject_files()
    
    if len(subject_files) == 0:
        logger.error("No se encontraron archivos")
        return
    
    logger.info(f"Procesando {len(subject_files)} sujetos con {len(config.band_pairs)} pares de bandas...")
    
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
    logger.info(f"  Total measurements: {len(full_df)}")
    logger.info(f"  Valid measurements: {len(valid_df)}")
    
    for band_name in valid_df['band_pair'].unique():
        band_data = valid_df[valid_df['band_pair'] == band_name]
        n_wake = (band_data['consciousness'] == 1).sum()
        n_n2 = (band_data['consciousness'] == 0).sum()
        logger.info(f"  {band_name}: {n_wake} Wake, {n_n2} N2")
    
    # ANÁLISIS POR BANDA
    print("\n" + "=" * 80)
    print("ANÁLISIS INDEPENDIENTE POR BANDA")
    print("=" * 80)
    
    band_results = {}
    
    for band_pair in config.band_pairs:
        band_name = band_pair.name
        print(f"\n{'='*80}")
        print(f"BANDA: {band_name}")
        print(f"  Phase: {band_pair.phase_band} Hz")
        print(f"  Amplitude: {band_pair.amp_band} Hz")
        print(f"{'='*80}")
        
        band_data = valid_df[valid_df['band_pair'] == band_name].copy()
        
        if len(band_data) == 0:
            print(f"⚠️  Sin datos válidos para {band_name}")
            continue
        
        unique_subjects = band_data['subject_id'].unique()
        
        np.random.seed(config.sacred_seed)
        n_dev = int(0.7 * len(unique_subjects))
        shuffled_subjects = np.random.permutation(unique_subjects)
        dev_subjects = shuffled_subjects[:n_dev]
        holdout_subjects = shuffled_subjects[n_dev:]
        
        dev_data = band_data[band_data['subject_id'].isin(dev_subjects)].copy()
        holdout_data = band_data[band_data['subject_id'].isin(holdout_subjects)].copy()
        
        print(f"  Development: {len(dev_subjects)} sujetos, {len(dev_data)} epochs")
        print(f"  Holdout: {len(holdout_subjects)} sujetos, {len(holdout_data)} epochs")
        
        wake = holdout_data[holdout_data['consciousness']==1]['pac'].values
        n2 = holdout_data[holdout_data['consciousness']==0]['pac'].values
        
        wake = wake[~np.isnan(wake)]
        n2 = n2[~np.isnan(n2)]
        
        if len(wake) == 0 or len(n2) == 0:
            print(f"⚠️  Insuficientes datos en holdout para {band_name}")
            continue
        
        pooled_std = np.sqrt(
            ((len(wake)-1)*np.var(wake,ddof=1) + (len(n2)-1)*np.var(n2,ddof=1)) /
            (len(wake) + len(n2) - 2)
        )
        cohens_d = (np.mean(wake) - np.mean(n2)) / (pooled_std + 1e-10)
        
        _, p_val = stats.ttest_ind(wake, n2, alternative='greater')
        
        print(f"\n📊 RESULTADOS HOLDOUT {band_name}:")
        print(f"  PAC Wake:  {np.mean(wake):.4f} ± {np.std(wake, ddof=1):.4f}")
        print(f"  PAC N2:    {np.mean(n2):.4f} ± {np.std(n2, ddof=1):.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}")
        print(f"  p-value:   {p_val:.6f}")
        print(f"  Ratio:     {np.mean(wake)/np.mean(n2):.2f}:1")
        
        band_results[band_name] = {
            'cohens_d': cohens_d,
            'p_value': p_val,
            'mean_wake': np.mean(wake),
            'mean_n2': np.mean(n2),
            'std_wake': np.std(wake, ddof=1),
            'std_n2': np.std(n2, ddof=1),
            'n_wake': len(wake),
            'n_n2': len(n2),
            'ratio': np.mean(wake)/np.mean(n2)
        }
    
    # COMPARACIÓN FINAL
    print("\n" + "=" * 80)
    print("🔥 COMPARACIÓN ENTRE BANDAS - NIVEL 2")
    print("=" * 80)
    
    print("\n📊 TABLA COMPARATIVA:")
    header = f"{'Banda':<20} {'Cohen d':>12} {'p-value':>12} {'PAC Wake':>12} {'PAC N2':>12} {'Ratio':>8}"
    print(header)
    print("-" * 90)
    
    for band_name in ['Delta-Gamma', 'Theta-Gamma', 'Alpha-Gamma', 'Beta-Gamma']:
        if band_name in band_results:
            res = band_results[band_name]
            print(f"{band_name:<20} {res['cohens_d']:>12.3f} {res['p_value']:>12.6f} "
                  f"{res['mean_wake']:>12.4f} {res['mean_n2']:>12.4f} {res['ratio']:>8.2f}")
    
    if len(band_results) >= 2:
        sorted_bands = sorted(band_results.items(), key=lambda x: x[1]['cohens_d'], reverse=True)
        winner_name, winner_stats = sorted_bands[0]
        
        print(f"\n🏆 MAYOR DISCRIMINACIÓN: {winner_name}")
        print(f"   Cohen's d = {winner_stats['cohens_d']:.3f}")
        
        if 'Theta-Gamma' in band_results:
            theta_d = band_results['Theta-Gamma']['cohens_d']
            other_ds = [v['cohens_d'] for k, v in band_results.items() if k != 'Theta-Gamma']
            
            if len(other_ds) > 0:
                mean_other_d = np.mean(other_ds)
                max_other_d = np.max(other_ds)
                
                print(f"\n🎯 EVALUACIÓN AFH:")
                print(f"   Theta-Gamma d: {theta_d:.3f}")
                print(f"   Promedio otros: {mean_other_d:.3f}")
                print(f"   Máximo otros: {max_other_d:.3f}")
                print(f"   Diferencia vs promedio: {theta_d - mean_other_d:.3f}")
                print(f"   Diferencia vs máximo: {theta_d - max_other_d:.3f}")
                
                if theta_d > mean_other_d and theta_d > max_other_d:
                    print(f"\n✅ NIVEL 2: ESPECIFICIDAD FUERTE")
                    print(f"   Theta-Gamma supera tanto promedio como máximo")
                    print(f"   AFH predicción VALIDADA")
                elif theta_d > mean_other_d:
                    print(f"\n⚠️  NIVEL 2: ESPECIFICIDAD DÉBIL")
                    print(f"   Theta-Gamma supera promedio pero NO máximo")
                    print(f"   Especificidad parcial")
                else:
                    print(f"\n❌ NIVEL 2: ESPECIFICIDAD NO CONFIRMADA")
                    print(f"   Theta-Gamma NO tiene ventaja sobre otras bandas")
                    print(f"   Replantear especificidad de AFH")
    
    # Guardar resultados
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(band_results).T
    output_file = config.output_dir / 'nivel2_band_comparison.csv'
    results_df.to_csv(output_file)
    logger.info(f"\n📁 Resultados guardados: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS NIVEL 2 FINALIZADO")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NIVEL 2: PAC Band Specificity Test",
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
        default='./nivel2_band_specificity_results',
        help='Directorio de salida'
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--pilot',
        action='store_true',
        help='Modo piloto (10 sujetos)'
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
        sys.exit(1)
    
    try:
        run_multiband_analysis(config)
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