#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AFH NIVEL 3b - ECLIPSE v4.4 - VALIDACIÓN H* v2
================================================================================

CONTEXTO:
  - H* v1 (Hjorth + DFA) fue INCONCLUSIVE en validación previa
  - Análisis exploratorio identificó theta_beta_ratio como mejor candidato (65.2%)
  - Este script valida H* v2 en el HOLDOUT NO TOCADO

H* v2 DEFINICIÓN:
  - Componente principal: theta_beta_ratio
  - Alternativa: combinación theta_beta + delta_beta (hstar_v5_ratios)

IMPORTANTE:
  - Usamos el MISMO split que v4.3 (sacred_seed=2025)
  - Solo analizamos HOLDOUT (los 46 sujetos no tocados)
  - Development ya fue "contaminado" por exploración

================================================================================
"""

import os
import sys
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.stats import binomtest, wilcoxon
from scipy.signal import butter, filtfilt, hilbert, welch

warnings.filterwarnings('ignore')

__version__ = "4.4.0"

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DEFAULT_PATHS = [
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\nivel3b_validation_v2"

# ============================================================================
# H* v2 DEFINICIÓN (PREREGISTRADA)
# ============================================================================

HSTAR_V2_DEFINITION = {
    'version': '2.0',
    'primary_metric': 'theta_beta_ratio',
    'components': ['theta_power', 'beta_power'],
    'formula': 'theta_power / beta_power',
    'bands': {
        'theta': [4, 8],
        'beta': [13, 30]
    },
    'normalization': 'baseline_only (t < -30s)',
    'justification': 'Selected from exploratory analysis: 65.2% precedence vs PAC (N=23)',
    'exploratory_reference': 'nivel3b_exploratory/hstar_candidates_ranking.csv',
    'date_selected': datetime.now(timezone.utc).isoformat()
}

HSTAR_V2_ALTERNATIVE = {
    'name': 'hstar_v2_combined',
    'components': ['theta_beta_ratio', 'delta_beta_ratio'],
    'weights': [0.5, 0.5],
    'justification': 'Combined top-2 candidates from exploratory (65.2% + 63.2%)'
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'min_wake_epochs_after': 2,
    'exclude_stages': ['?', 'M'],
}

ONSET_DETECTION = {
    'threshold_sd': 1.5,
    'consecutive_windows': 2,
    'window_size_s': 30,
    'window_step_s': 10,
}

SUBJECT_INCLUSION = {
    'min_transitions': 1,  # Bajado de 3 a 1 debido a datos limitados en holdout
}

STATISTICAL_TESTS = {
    'primary': 'binomial_subject_level',
    'sensitivity': 'wilcoxon_signed_rank',
}

# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.999))
    high = max(low + 0.001, min(highcut / nyq, 0.999))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def compute_band_power(signal, fs, band, nperseg=256):
    """Potencia en banda de frecuencia específica."""
    try:
        freqs, psd = welch(signal, fs, nperseg=min(nperseg, len(signal)//2))
        idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
        return np.trapz(psd[idx], freqs[idx])
    except:
        return np.nan


def compute_theta_beta_ratio(signal, fs):
    """H* v2 primary metric: theta/beta power ratio."""
    theta = compute_band_power(signal, fs, (4, 8))
    beta = compute_band_power(signal, fs, (13, 30))
    if beta <= 0 or np.isnan(beta):
        return np.nan
    return theta / beta


def compute_delta_beta_ratio(signal, fs):
    """H* v2 secondary metric: delta/beta power ratio."""
    delta = compute_band_power(signal, fs, (0.5, 4))
    beta = compute_band_power(signal, fs, (13, 30))
    if beta <= 0 or np.isnan(beta):
        return np.nan
    return delta / beta


def compute_pac_mvl(signal, fs, phase_freq=(4, 8), amp_freq=(30, 45)):
    """Phase-Amplitude Coupling via Mean Vector Length."""
    try:
        theta = bandpass_filter(signal, phase_freq[0], phase_freq[1], fs)
        theta_phase = np.angle(hilbert(theta))
        gamma = bandpass_filter(signal, amp_freq[0], amp_freq[1], fs)
        gamma_amp = np.abs(hilbert(gamma))
        n = len(theta_phase)
        mvl = np.abs(np.sum(gamma_amp * np.exp(1j * theta_phase))) / n
        return mvl / np.mean(gamma_amp) if np.mean(gamma_amp) > 0 else 0
    except:
        return np.nan


# ============================================================================
# SLEEP-EDF PROCESSOR
# ============================================================================

class SleepEDFProcessor:
    STAGE_MAPPING = {
        'Sleep stage W': 'W', 'Sleep stage 1': 'N1', 'Sleep stage 2': 'N2',
        'Sleep stage 3': 'N3', 'Sleep stage 4': 'N4', 'Sleep stage R': 'R',
        'Sleep stage ?': '?', 'Movement time': 'M'
    }
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.window_size = ONSET_DETECTION['window_size_s']
        self.window_step = ONSET_DETECTION['window_step_s']
    
    def find_psg_files(self):
        files = []
        # Buscar recursivamente
        psg_files = list(self.data_path.rglob("*-PSG.edf"))
        if not psg_files:
            psg_files = list(self.data_path.rglob("*PSG.edf"))
        
        print(f"   Encontrados {len(psg_files)} archivos PSG")
        
        for psg_file in sorted(psg_files):
            name = psg_file.name
            subject_id = name.replace('-PSG.edf', '').replace('PSG.edf', '')
            hyp_file = None
            
            # Buscar en mismo directorio
            parent = psg_file.parent
            hyp_candidates = list(parent.glob(f"{subject_id}*Hypnogram.edf"))
            if hyp_candidates:
                hyp_file = hyp_candidates[0]
            
            if hyp_file is None and len(subject_id) >= 2:
                hyp_prefix = subject_id[:-1] + 'C'
                hyp_candidates = list(parent.glob(f"{hyp_prefix}*Hypnogram.edf"))
                if hyp_candidates:
                    hyp_file = hyp_candidates[0]
            
            # Buscar globalmente si no está en el mismo dir
            if hyp_file is None:
                hyp_candidates = list(self.data_path.rglob(f"{subject_id}*Hypnogram.edf"))
                if hyp_candidates:
                    hyp_file = hyp_candidates[0]
                elif len(subject_id) >= 2:
                    hyp_prefix = subject_id[:-1] + 'C'
                    hyp_candidates = list(self.data_path.rglob(f"{hyp_prefix}*Hypnogram.edf"))
                    if hyp_candidates:
                        hyp_file = hyp_candidates[0]
            
            if hyp_file:
                files.append((subject_id, psg_file, hyp_file))
        
        print(f"   Pareados con Hypnogram: {len(files)}")
        return files
    
    def load_subject(self, psg_path, hyp_path):
        try:
            import mne
            raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
            
            eeg_channel = None
            for ch in ['EEG Pz-Oz', 'EEG Fpz-Cz']:
                if ch in raw.ch_names:
                    eeg_channel = ch
                    break
            if eeg_channel is None:
                for ch in raw.ch_names:
                    if 'EEG' in ch.upper():
                        eeg_channel = ch
                        break
            if eeg_channel is None:
                return None
            
            raw.pick_channels([eeg_channel])
            signal = raw.get_data()[0]
            fs = raw.info['sfreq']
            
            annotations = mne.read_annotations(str(hyp_path))
            total_duration = len(signal) / fs
            n_epochs = int(total_duration // 30)
            stages = ['?'] * n_epochs
            
            for ann in annotations:
                stage = self.STAGE_MAPPING.get(ann['description'], '?')
                start_epoch = int(ann['onset'] // 30)
                end_epoch = int((ann['onset'] + ann['duration']) // 30)
                for e in range(start_epoch, min(end_epoch, n_epochs)):
                    stages[e] = stage
            
            return {'signal': signal, 'fs': fs, 'stages': stages}
        except:
            return None
    
    def find_stable_transitions(self, stages):
        transitions = []
        sleep_stages = ['N1', 'N2', 'N3', 'N4']
        exclude = set(TRANSITION_CRITERIA['exclude_stages'])
        min_sleep = TRANSITION_CRITERIA['min_sleep_epochs_before']
        min_wake = TRANSITION_CRITERIA['min_wake_epochs_after']
        
        for i in range(min_sleep, len(stages) - min_wake):
            if stages[i] != 'W' or stages[i-1] not in sleep_stages:
                continue
            pre_stages = stages[max(0, i-min_sleep):i]
            if any(s in exclude or s == 'W' for s in pre_stages):
                continue
            if not all(s in sleep_stages for s in pre_stages):
                continue
            post_stages = stages[i:min(len(stages), i+min_wake)]
            if not all(s == 'W' for s in post_stages):
                continue
            transitions.append({'epoch_idx': i, 'from_stage': stages[i-1], 'time_seconds': i * 30})
        return transitions
    
    def extract_transition_window(self, signal, fs, transition_time, pre_seconds=120, post_seconds=120):
        start_sample = int((transition_time - pre_seconds) * fs)
        end_sample = int((transition_time + post_seconds) * fs)
        if start_sample < 0 or end_sample > len(signal):
            return None
        return signal[start_sample:end_sample]
    
    def compute_metrics_timeseries(self, window, fs):
        """Calcula H* v2 y PAC en ventanas deslizantes."""
        window_samples = int(self.window_size * fs)
        step_samples = int(self.window_step * fs)
        n_windows = (len(window) - window_samples) // step_samples + 1
        
        times = []
        theta_beta = []
        delta_beta = []
        pac = []
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            segment = window[start:end]
            center_time = (start + window_samples/2) / fs - 120
            
            times.append(center_time)
            theta_beta.append(compute_theta_beta_ratio(segment, fs))
            delta_beta.append(compute_delta_beta_ratio(segment, fs))
            pac.append(compute_pac_mvl(segment, fs))
        
        times = np.array(times)
        theta_beta = np.array(theta_beta)
        delta_beta = np.array(delta_beta)
        pac = np.array(pac)
        
        # Normalizar con baseline
        baseline_mask = times < -30
        
        def normalize(values, mask):
            baseline_vals = values[mask]
            baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
            if len(baseline_vals) < 2:
                return values
            mu, sigma = np.mean(baseline_vals), np.std(baseline_vals)
            if sigma < 1e-10:
                return values - mu
            return (values - mu) / sigma
        
        theta_beta_z = normalize(theta_beta, baseline_mask)
        delta_beta_z = normalize(delta_beta, baseline_mask)
        pac_z = normalize(pac, baseline_mask)
        
        # H* v2 primary = theta_beta_ratio
        hstar_v2_primary = theta_beta_z
        
        # H* v2 combined = (theta_beta + delta_beta) / 2
        hstar_v2_combined = (theta_beta_z + delta_beta_z) / 2
        
        return {
            'times': times,
            'hstar_v2_primary': hstar_v2_primary,
            'hstar_v2_combined': hstar_v2_combined,
            'pac': pac_z
        }
    
    def detect_onset(self, values, times, baseline_end=-30):
        threshold = ONSET_DETECTION['threshold_sd']
        k_consecutive = ONSET_DETECTION['consecutive_windows']
        
        mask = times >= baseline_end
        v = values[mask]
        t = times[mask]
        
        above = (~np.isnan(v)) & (v > threshold)
        
        run = 0
        for i, ok in enumerate(above):
            if ok:
                run += 1
                if run >= k_consecutive:
                    return t[i - k_consecutive + 1]
            else:
                run = 0
        return None


# ============================================================================
# VALIDACIÓN
# ============================================================================

@dataclass
class FalsificationCriteria:
    name: str
    threshold: float
    comparison: str
    description: str
    is_required: bool = True
    
    def evaluate(self, value: float) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        ops = {">=": lambda x, y: x >= y, "<=": lambda x, y: x <= y,
               ">": lambda x, y: x > y, "<": lambda x, y: x < y}
        return ops.get(self.comparison, lambda x, y: False)(value, self.threshold)
    
    def to_dict(self):
        return asdict(self)


class HstarV2Validation:
    def __init__(self, data_path: str, output_dir: str):
        self.processor = SleepEDFProcessor(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sacred_seed = 2025  # MISMO seed que v4.3
        
        # Criterios de falsificación
        self.criteria = [
            FalsificationCriteria(
                "pct_subjects_hstar_dominant", 55.0, ">=",
                "≥55% sujetos con H* v2 dominante", True
            ),
            FalsificationCriteria(
                "mean_subject_pct_hstar_first", 50.0, ">",
                "Media de %H* first por sujeto > 50%", True
            ),
            FalsificationCriteria(
                "p_value_primary", 0.05, "<",
                "p < 0.05 en test primario (binomial)", True
            ),
        ]
        
        print(f"\n{'='*70}")
        print(f"🔬 ECLIPSE v{__version__} - VALIDACIÓN H* v2")
        print(f"{'='*70}")
        print(f"\n📋 H* v2 DEFINICIÓN:")
        print(f"   Primary: {HSTAR_V2_DEFINITION['primary_metric']}")
        print(f"   Formula: {HSTAR_V2_DEFINITION['formula']}")
        print(f"   Bands: theta={HSTAR_V2_DEFINITION['bands']['theta']}, beta={HSTAR_V2_DEFINITION['bands']['beta']}")
        print(f"\n📋 VALIDACIÓN:")
        print(f"   Solo HOLDOUT (no tocado en exploración)")
        print(f"   Sacred seed: {self.sacred_seed}")
        print(f"{'='*70}\n")
    
    def get_holdout_ids(self, all_subject_ids):
        """Reproduce el split de v4.3 y retorna solo holdout."""
        np.random.seed(self.sacred_seed)
        shuffled = np.array(all_subject_ids).copy()
        np.random.shuffle(shuffled)
        
        n_dev = int(len(all_subject_ids) * 0.7)
        holdout_ids = set(shuffled[n_dev:].tolist())
        
        return holdout_ids
    
    def run(self):
        print("📂 Buscando archivos...")
        files = self.processor.find_psg_files()
        print(f"   Total: {len(files)}")
        
        if not files:
            print("❌ No se encontraron archivos")
            return None
        
        # Obtener holdout IDs
        all_ids = [f[0] for f in files]
        holdout_ids = self.get_holdout_ids(all_ids)
        
        # Filtrar solo holdout
        holdout_files = [(sid, psg, hyp) for sid, psg, hyp in files if sid in holdout_ids]
        print(f"   Holdout: {len(holdout_files)} sujetos")
        
        # Procesar
        print(f"\n{'='*70}")
        print("PROCESANDO HOLDOUT")
        print(f"{'='*70}")
        
        all_transitions = []
        subject_summaries = []
        
        for i, (subject_id, psg_path, hyp_path) in enumerate(holdout_files):
            print(f"\r   [{i+1}/{len(holdout_files)}] {subject_id}...", end="", flush=True)
            
            data = self.processor.load_subject(psg_path, hyp_path)
            if data is None:
                continue
            
            transitions = self.processor.find_stable_transitions(data['stages'])
            subject_transitions = []
            
            for trans in transitions:
                window = self.processor.extract_transition_window(
                    data['signal'], data['fs'], trans['time_seconds']
                )
                if window is None:
                    continue
                
                metrics = self.processor.compute_metrics_timeseries(window, data['fs'])
                
                # H* v2 primary (theta_beta_ratio)
                hstar_onset = self.processor.detect_onset(metrics['hstar_v2_primary'], metrics['times'])
                pac_onset = self.processor.detect_onset(metrics['pac'], metrics['times'])
                
                if hstar_onset is not None and pac_onset is not None:
                    delta = hstar_onset - pac_onset
                    trans_data = {
                        'subject_id': subject_id,
                        'from_stage': trans['from_stage'],
                        'hstar_v2_onset': hstar_onset,
                        'pac_onset': pac_onset,
                        'delta': delta,
                        'hstar_first': delta < 0,
                    }
                    all_transitions.append(trans_data)
                    subject_transitions.append(trans_data)
            
            if subject_transitions:
                n_trans = len(subject_transitions)
                n_hstar = sum(1 for t in subject_transitions if t['hstar_first'])
                pct_hstar = 100 * n_hstar / n_trans
                subject_summaries.append({
                    'subject_id': subject_id,
                    'n_transitions': n_trans,
                    'n_hstar_first': n_hstar,
                    'pct_hstar_first': pct_hstar,
                    'hstar_dominant': pct_hstar > 50,
                    'mean_delta': np.mean([t['delta'] for t in subject_transitions]),
                })
        
        print(f"\n   Total transiciones: {len(all_transitions)}")
        print(f"   Sujetos con transiciones: {len(subject_summaries)}")
        
        # Analizar
        results = self._analyze(subject_summaries, all_transitions)
        
        # Evaluación de criterios
        print(f"\n{'='*70}")
        print("EVALUACIÓN DE CRITERIOS")
        print(f"{'='*70}")
        
        metrics = results['subject_level_metrics']
        evaluations = []
        
        for c in self.criteria:
            value = metrics.get(c.name)
            passed = c.evaluate(value) if value is not None else False
            evaluations.append({'criterion': c.to_dict(), 'observed_value': value, 'passed': passed})
            status = "✅ PASS" if passed else "❌ FAIL"
            val_str = f"{value:.4f}" if value is not None else "N/A"
            print(f"   {c.name}: {val_str} {c.comparison} {c.threshold} → {status}")
        
        required = [e for e in evaluations if e['criterion']['is_required']]
        n_passed = sum(1 for e in required if e['passed'])
        n_total = len(required)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required) else "FALSIFIED" if n_passed == 0 else "INCONCLUSIVE"
        
        print(f"\n{'='*70}")
        if verdict == "VALIDATED":
            print(f"✅ VERDICT: {verdict}")
        elif verdict == "FALSIFIED":
            print(f"❌ VERDICT: {verdict}")
        else:
            print(f"⚠️  VERDICT: {verdict}")
        print(f"   Required criteria passed: {n_passed}/{n_total}")
        print(f"{'='*70}")
        
        # Guardar resultados
        final_results = {
            'project_name': 'AFH_Nivel3b_HstarV2_Validation',
            'eclipse_version': __version__,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'hstar_v2_definition': HSTAR_V2_DEFINITION,
            'hstar_v2_alternative': HSTAR_V2_ALTERNATIVE,
            'validation_summary': results,
            'criteria_evaluation': evaluations,
            'verdict': verdict,
            'required_criteria_passed': f"{n_passed}/{n_total}",
            'methodological_notes': [
                "H* v2 = theta_beta_ratio (selected from exploratory analysis)",
                "Validation on HOLDOUT only (not used in exploration)",
                "Same split as v4.3 (sacred_seed=2025)",
                "Primary test: binomial on subject dominance with 95% CI (Wilson)",
            ]
        }
        
        with open(self.output_dir / "hstar_v2_validation_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        pd.DataFrame(all_transitions).to_csv(self.output_dir / "hstar_v2_transitions.csv", index=False)
        pd.DataFrame(subject_summaries).to_csv(self.output_dir / "hstar_v2_subject_summaries.csv", index=False)
        
        print(f"\n📁 Resultados en: {self.output_dir}")
        
        return final_results
    
    def _analyze(self, subjects, transitions):
        if not subjects:
            return {'n_subjects': 0, 'subject_level_metrics': {}}
        
        min_trans = SUBJECT_INCLUSION['min_transitions']
        eligible = [s for s in subjects if s['n_transitions'] >= min_trans]
        excluded = [s for s in subjects if s['n_transitions'] < min_trans]
        
        n_total = len(subjects)
        n_eligible = len(eligible)
        n_excluded = len(excluded)
        
        print(f"\n   📊 FILTRO DE INCLUSIÓN:")
        print(f"      Sujetos totales: {n_total}")
        print(f"      Elegibles (≥{min_trans} trans): {n_eligible}")
        print(f"      Excluidos (<{min_trans} trans): {n_excluded}")
        
        if not eligible:
            return {'n_subjects_total': n_total, 'n_subjects_eligible': 0, 'subject_level_metrics': {}}
        
        pcts = np.array([s['pct_hstar_first'] for s in eligible])
        n_hstar_dominant = sum(1 for s in eligible if s['hstar_dominant'])
        pct_subjects_dominant = 100 * n_hstar_dominant / n_eligible
        mean_pct = np.mean(pcts)
        std_pct = np.std(pcts)
        
        # Test primario: Binomial + IC 95% Wilson
        binom_result = binomtest(n_hstar_dominant, n_eligible, 0.5, alternative='greater')
        p_primary = binom_result.pvalue
        ci = binom_result.proportion_ci(confidence_level=0.95, method="wilson")
        
        # Test sensibilidad: Wilcoxon
        diffs = pcts - 50
        nonzero = diffs[diffs != 0]
        n_nonzero = len(nonzero)
        
        try:
            p_sensitivity = wilcoxon(nonzero, alternative='greater').pvalue if n_nonzero >= 10 else np.nan
        except:
            p_sensitivity = np.nan
        
        # Nivel transición
        n_trans = len(transitions)
        n_hstar_first = sum(1 for t in transitions if t['hstar_first'])
        pct_trans = 100 * n_hstar_first / n_trans if n_trans > 0 else 0
        mean_delta = np.mean([t['delta'] for t in transitions]) if transitions else 0
        
        print(f"\n   📊 NIVEL SUJETO (PRIMARIO, n={n_eligible}):")
        print(f"      H* v2 dominante: {n_hstar_dominant} ({pct_subjects_dominant:.1f}%)")
        print(f"      IC 95% Wilson: [{ci.low*100:.1f}%, {ci.high*100:.1f}%]")
        print(f"      Media %H* first: {mean_pct:.1f}% ± {std_pct:.1f}%")
        print(f"      p-value PRIMARIO (binomial): {p_primary:.4f}")
        if not np.isnan(p_sensitivity):
            print(f"      p-value sensibilidad (Wilcoxon): {p_sensitivity:.4f}")
        else:
            print(f"      p-value sensibilidad (Wilcoxon): N/A (n_nonzero={n_nonzero})")
        
        print(f"\n   📊 NIVEL TRANSICIÓN (secundario):")
        print(f"      N transiciones: {n_trans}")
        print(f"      H* v2 primero: {pct_trans:.1f}%")
        print(f"      Delta medio: {mean_delta:.1f}s")
        
        return {
            'n_subjects_total': n_total,
            'n_subjects_eligible': n_eligible,
            'n_subjects_excluded': n_excluded,
            'n_transitions': n_trans,
            'subject_level_metrics': {
                'pct_subjects_hstar_dominant': pct_subjects_dominant,
                'ci95_low': ci.low * 100,
                'ci95_high': ci.high * 100,
                'mean_subject_pct_hstar_first': mean_pct,
                'std_subject_pct_hstar_first': std_pct,
                'p_value_primary': p_primary,
                'p_value_sensitivity': p_sensitivity,
                'wilcoxon_n_nonzero': n_nonzero,
            },
            'transition_level_metrics': {
                'pct_hstar_first': pct_trans,
                'mean_delta': mean_delta,
                'n_hstar_first': n_hstar_first,
            }
        }


# ============================================================================
# MAIN
# ============================================================================

def find_data_path():
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and p.is_dir():
            psg_files = list(p.glob("*PSG.edf")) + list(p.glob("*-PSG.edf"))
            if psg_files:
                return p
    return None


def main():
    print("🔍 Buscando datos...")
    data_path = find_data_path()
    
    if data_path is None:
        print("❌ No se encontraron datos")
        sys.exit(1)
    
    print(f"✅ Datos: {data_path}")
    print(f"✅ Salida: {DEFAULT_OUTPUT}")
    
    validation = HstarV2Validation(str(data_path), DEFAULT_OUTPUT)
    result = validation.run()
    
    if result:
        print("\n" + "="*70)
        print("✅ VALIDACIÓN COMPLETADA")
        print("="*70)


if __name__ == "__main__":
    main()