#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AFH NIVEL 3b - ECLIPSE v4.3 FINAL
Metodología 100% defendible
================================================================================

CORRECCIONES v4.3 (sobre v4.2):
  1. IC 95% Wilson para proporción sujetos H*-dominant
  2. Filtro: solo sujetos con ≥3 transiciones entran al test primario
  3. Trazabilidad Wilcoxon: n_subjects_nonzero, n_subjects_total
  4. Eliminado self.fs redundante en SleepEDFProcessor

CORRECCIONES v4.2:
  1. Test primario: Binomial a nivel sujeto (preregistrado)
  2. Test sensibilidad: Wilcoxon signed-rank (no paramétrico)
  3. Onset: 2 ventanas consecutivas (reduce falsos positivos)
  4. Trazabilidad: H* y criterios guardados en JSONs
  5. NO hay min(p1, p2) - cada test es independiente

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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from scipy import stats
from scipy.stats import binomtest, wilcoxon
from scipy.signal import butter, filtfilt, hilbert

warnings.filterwarnings('ignore')

__version__ = "4.3.0"

# ============================================================================
# DEFINICIONES PREREGISTRADAS
# ============================================================================

H_STAR_DEFINITION = {
    'components': ['hjorth_complexity', 'dfa'],
    'weights': [0.5, 0.5],
    'normalization': 'baseline_only (t < -30s)',
    'justification': 'Selected in prior exploratory analysis (HCTSA-style ranking)',
    'reference': 'Nivel 3b exploratory analysis, Sleep-EDF'
}

TRANSITION_CRITERIA = {
    'min_sleep_epochs_before': 2,
    'min_wake_epochs_after': 2,
    'exclude_stages': ['?', 'M'],
    'description': 'Stable transitions only: ≥2 sleep epochs before, ≥2 wake epochs after'
}

ONSET_DETECTION = {
    'threshold_sd': 1.5,
    'consecutive_windows': 2,
    'window_size_s': 30,
    'window_step_s': 10,
    'description': 'First point where metric exceeds 1.5 SD for 2 consecutive windows'
}

STATISTICAL_TESTS = {
    'primary': 'binomial_subject_level',
    'sensitivity': 'wilcoxon_signed_rank',
    'description': 'Primary: binomial test on proportion of subjects with H*-dominant. Sensitivity: Wilcoxon on %H*-first vs 50%'
}

SUBJECT_INCLUSION = {
    'min_transitions': 3,
    'description': 'Subjects must have ≥3 valid transitions to enter primary analysis'
}

# ============================================================================
# ECLIPSE CORE
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

@dataclass 
class EclipseConfig:
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    output_dir: str = "./eclipse_results"


class EclipseFramework:
    VERSION = __version__
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA.json"
        self.results_file = self.output_dir / f"{config.project_name}_RESULTS.json"
        
        print(f"\n{'='*70}")
        print(f"🔬 ECLIPSE v{self.VERSION} - AFH Nivel 3b FINAL")
        print(f"{'='*70}")
        print(f"Project: {config.project_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Sacred Seed: {config.sacred_seed}")
        print(f"Split: {config.development_ratio*100:.0f}% dev / {config.holdout_ratio*100:.0f}% holdout")
        print(f"\n📋 H* PREDEFINIDO:")
        print(f"   {H_STAR_DEFINITION['components']} (weights: {H_STAR_DEFINITION['weights']})")
        print(f"\n📋 INCLUSIÓN: min {SUBJECT_INCLUSION['min_transitions']} transiciones/sujeto")
        print(f"📋 TEST PRIMARIO: {STATISTICAL_TESTS['primary']}")
        print(f"{'='*70}\n")
    
    def stage1_split(self, subject_ids: List[str], force: bool = False):
        print(f"\n{'='*70}")
        print("STAGE 1: SUBJECT-LEVEL SPLIT")
        print(f"{'='*70}")
        
        if self.split_file.exists() and not force:
            print("⚠️  Split existente, cargando...")
            with open(self.split_file) as f:
                data = json.load(f)
            return data['development_ids'], data['holdout_ids']
        
        np.random.seed(self.config.sacred_seed)
        shuffled = np.array(subject_ids).copy()
        np.random.shuffle(shuffled)
        
        n_dev = int(len(subject_ids) * self.config.development_ratio)
        dev_ids = shuffled[:n_dev].tolist()
        holdout_ids = shuffled[n_dev:].tolist()
        
        integrity_hash = hashlib.sha256(
            f"{self.config.sacred_seed}_{sorted(subject_ids)}".encode()
        ).hexdigest()
        
        split_data = {
            'project_name': self.config.project_name,
            'eclipse_version': self.VERSION,
            'split_timestamp': datetime.now(timezone.utc).isoformat(),
            'sacred_seed': self.config.sacred_seed,
            'total_subjects': len(subject_ids),
            'n_development': len(dev_ids),
            'n_holdout': len(holdout_ids),
            'development_ids': dev_ids,
            'holdout_ids': holdout_ids,
            'integrity_hash': integrity_hash,
            'h_star_definition': H_STAR_DEFINITION,
            'transition_criteria': TRANSITION_CRITERIA,
            'onset_detection': ONSET_DETECTION,
            'subject_inclusion': SUBJECT_INCLUSION,
            'binding_declaration': "Split by SUBJECT before any analysis. H* predefined."
        }
        
        with open(self.split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✅ Split completado:")
        print(f"   Total: {len(subject_ids)} sujetos")
        print(f"   Desarrollo: {len(dev_ids)} sujetos")
        print(f"   Holdout: {len(holdout_ids)} sujetos")
        print(f"   Hash: {integrity_hash[:16]}...")
        
        return dev_ids, holdout_ids
    
    def stage2_register_criteria(self, criteria: List[FalsificationCriteria], force: bool = False):
        print(f"\n{'='*70}")
        print("STAGE 2: REGISTER FALSIFICATION CRITERIA")
        print(f"{'='*70}")
        
        if self.criteria_file.exists() and not force:
            print("⚠️  Criterios existentes, cargando...")
            with open(self.criteria_file) as f:
                return json.load(f)
        
        criteria_list = [c.to_dict() for c in criteria]
        criteria_hash = hashlib.sha256(
            json.dumps(criteria_list, sort_keys=True).encode()
        ).hexdigest()
        
        registration = {
            'project_name': self.config.project_name,
            'eclipse_version': self.VERSION,
            'registration_timestamp': datetime.now(timezone.utc).isoformat(),
            'criteria': criteria_list,
            'criteria_hash': criteria_hash,
            'h_star_definition': H_STAR_DEFINITION,
            'transition_criteria': TRANSITION_CRITERIA,
            'onset_detection': ONSET_DETECTION,
            'subject_inclusion': SUBJECT_INCLUSION,
            'statistical_tests': STATISTICAL_TESTS,
            'analysis_level': 'SUBJECT (primary); transitions (secondary)',
            'binding_declaration': "CRITERIA DEFINED BEFORE DATA ANALYSIS"
        }
        
        with open(self.criteria_file, 'w') as f:
            json.dump(registration, f, indent=2)
        
        print(f"✅ {len(criteria)} criterios registrados:")
        for c in criteria:
            print(f"   • {c.name} {c.comparison} {c.threshold}")
        print(f"   Hash: {criteria_hash[:16]}...")
        print(f"   Primary test: {STATISTICAL_TESTS['primary']}")
        print(f"   Subject inclusion: min {SUBJECT_INCLUSION['min_transitions']} transitions")
        
        return registration
    
    def stage5_assessment(self, dev_results: Dict, val_results: Dict):
        print(f"\n{'='*70}")
        print("STAGE 5: FINAL ASSESSMENT")
        print(f"{'='*70}")
        
        with open(self.criteria_file) as f:
            criteria_data = json.load(f)
        
        criteria = [FalsificationCriteria(**c) for c in criteria_data['criteria']]
        metrics = val_results.get('subject_level_metrics', {})
        
        print("\n📊 EVALUACIÓN DE CRITERIOS (nivel sujeto, test primario):")
        
        evaluations = []
        for c in criteria:
            value = metrics.get(c.name)
            passed = c.evaluate(value) if value is not None else False
            evaluations.append({'criterion': c.to_dict(), 'observed_value': value, 'passed': passed})
            status = "✅ PASS" if passed else "❌ FAIL"
            val_str = f"{value:.4f}" if value is not None else "N/A"
            print(f"   {c.name}: {val_str} {c.comparison} {c.threshold} → {status}")
        
        required = [e for e in evaluations if e['criterion']['is_required']]
        n_passed = sum(1 for e in required if e['passed'])
        n_total = len(required)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required) else "INCONCLUSIVE"
        
        print(f"\n{'='*70}")
        print(f"{'✅' if verdict == 'VALIDATED' else '⚠️ '} VERDICT: {verdict}")
        print(f"   Required criteria passed: {n_passed}/{n_total}")
        print(f"{'='*70}")
        
        assessment = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'eclipse_version': self.VERSION,
            'assessment_timestamp': datetime.now(timezone.utc).isoformat(),
            'h_star_definition': H_STAR_DEFINITION,
            'subject_inclusion': SUBJECT_INCLUSION,
            'statistical_tests': STATISTICAL_TESTS,
            'development_summary': dev_results,
            'validation_summary': val_results,
            'criteria_evaluation': evaluations,
            'verdict': verdict,
            'required_criteria_passed': f"{n_passed}/{n_total}",
            'methodological_notes': [
                "H* predefined (Hjorth + DFA, equal weights)",
                "Baseline-only normalization (t < -30s)",
                "Subject-level primary analysis",
                f"Subject inclusion: min {SUBJECT_INCLUSION['min_transitions']} transitions",
                "Onset detection: 2 consecutive windows > 1.5 SD",
                "Primary test: binomial on subject dominance with 95% CI (Wilson)",
                "Sensitivity test: Wilcoxon signed-rank with traceability"
            ]
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        
        return assessment


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(0.001, min(lowcut / nyq, 0.999))
    high = max(low + 0.001, min(highcut / nyq, 0.999))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def compute_hjorth_complexity(signal):
    try:
        diff1 = np.diff(signal)
        diff2 = np.diff(diff1)
        var0, var1, var2 = np.var(signal), np.var(diff1), np.var(diff2)
        if var0 <= 0 or var1 <= 0:
            return np.nan
        return np.sqrt(var2 / var1) / np.sqrt(var1 / var0)
    except:
        return np.nan


def compute_dfa(signal, min_box=4, max_box=None):
    try:
        n = len(signal)
        if max_box is None:
            max_box = n // 4
        y = np.cumsum(signal - np.mean(signal))
        scales = np.unique(np.logspace(np.log10(min_box), np.log10(max_box), 20).astype(int))
        scales = scales[(scales >= min_box) & (scales <= max_box)]
        if len(scales) < 4:
            return np.nan
        fluctuations = []
        for scale in scales:
            n_segments = n // scale
            if n_segments < 1:
                fluctuations.append(np.nan)
                continue
            rms_list = []
            for i in range(n_segments):
                segment = y[i*scale:(i+1)*scale]
                if len(segment) < scale:
                    continue
                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                rms_list.append(np.sqrt(np.mean((segment - trend)**2)))
            fluctuations.append(np.mean(rms_list) if rms_list else np.nan)
        valid = ~np.isnan(fluctuations)
        if np.sum(valid) < 4:
            return np.nan
        alpha, _ = np.polyfit(np.log10(scales[valid]), np.log10(np.array(fluctuations)[valid]), 1)
        return alpha
    except:
        return np.nan


def compute_pac_mvl(signal, fs, phase_freq=(4, 8), amp_freq=(30, 45)):
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
        """
        Encuentra pares PSG-Hypnogram.
        Formato Sleep-EDF: SC4ssNE0-PSG.edf y SC4ssNEC-Hypnogram.edf
        donde ss=subject, N=night, E0=PSG, EC=Hypnogram
        """
        files = []
        psg_files = list(self.data_path.glob("*-PSG.edf"))
        if not psg_files:
            psg_files = list(self.data_path.glob("*PSG.edf"))
        
        print(f"   Encontrados {len(psg_files)} archivos PSG")
        
        for psg_file in sorted(psg_files):
            name = psg_file.name
            # SC4001E0-PSG.edf -> subject_id = SC4001E0
            subject_id = name.replace('-PSG.edf', '').replace('PSG.edf', '')
            
            # Buscar hypnogram de varias formas
            hyp_file = None
            
            # Forma 1: Mismo prefijo exacto (SC4001E0 -> SC4001E0-Hypnogram.edf)
            hyp_candidates = list(self.data_path.glob(f"{subject_id}*Hypnogram.edf"))
            if hyp_candidates:
                hyp_file = hyp_candidates[0]
            
            # Forma 2: Sleep-EDF format (SC4001E0 -> SC4001EC)
            # Cambiar último caracter antes del sufijo: E0 -> EC
            if hyp_file is None and len(subject_id) >= 2:
                # SC4001E0 -> SC4001E + C
                hyp_prefix = subject_id[:-1] + 'C'
                hyp_candidates = list(self.data_path.glob(f"{hyp_prefix}*Hypnogram.edf"))
                if hyp_candidates:
                    hyp_file = hyp_candidates[0]
            
            # Forma 3: Buscar por subject/night (SC4001 -> cualquier hypnogram SC4001*)
            if hyp_file is None and len(subject_id) >= 6:
                base_id = subject_id[:6]  # SC4001
                hyp_candidates = list(self.data_path.glob(f"{base_id}*Hypnogram.edf"))
                if hyp_candidates:
                    # Tomar el que corresponda a la misma noche
                    night = subject_id[6] if len(subject_id) > 6 else ''
                    for hc in hyp_candidates:
                        if night in hc.name:
                            hyp_file = hc
                            break
                    if hyp_file is None:
                        hyp_file = hyp_candidates[0]
            
            if hyp_file:
                files.append((subject_id, psg_file, hyp_file))
        
        print(f"   Pareados con Hypnogram: {len(files)}")
        
        # Debug: mostrar primeros pares si hay
        if files and len(files) <= 5:
            print("   Pares encontrados:")
            for sid, psg, hyp in files[:5]:
                print(f"      {psg.name} <-> {hyp.name}")
        elif files:
            print(f"   Ejemplo: {files[0][1].name} <-> {files[0][2].name}")
        
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
        window_samples = int(self.window_size * fs)
        step_samples = int(self.window_step * fs)
        n_windows = (len(window) - window_samples) // step_samples + 1
        
        times, hjorth_raw, dfa_raw, pac_raw = [], [], [], []
        
        for i in range(n_windows):
            start = i * step_samples
            end = start + window_samples
            segment = window[start:end]
            center_time = (start + window_samples/2) / fs - 120
            times.append(center_time)
            hjorth_raw.append(compute_hjorth_complexity(segment))
            dfa_raw.append(compute_dfa(segment))
            pac_raw.append(compute_pac_mvl(segment, fs))
        
        times = np.array(times)
        hjorth_raw = np.array(hjorth_raw)
        dfa_raw = np.array(dfa_raw)
        pac_raw = np.array(pac_raw)
        
        # Normalizar SOLO con baseline (t < -30s)
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
        
        hjorth_z = normalize(hjorth_raw, baseline_mask)
        dfa_z = normalize(dfa_raw, baseline_mask)
        pac_z = normalize(pac_raw, baseline_mask)
        hstar = (hjorth_z + dfa_z) / 2
        
        return {'times': times, 'hstar': hstar, 'pac': pac_z}
    
    def detect_onset(self, values, times, baseline_end=-30):
        """
        Detecta onset con 2 VENTANAS CONSECUTIVAS > threshold
        (Reduce falsos positivos por ruido)
        """
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
                    # Retornar el tiempo de la primera ventana de la secuencia
                    return t[i - k_consecutive + 1]
            else:
                run = 0
        
        return None


# ============================================================================
# ANALYSIS
# ============================================================================

class Nivel3bAnalysis:
    def __init__(self, sleep_edf_path: str, output_dir: str = "./nivel3b_eclipse_results"):
        self.processor = SleepEDFProcessor(sleep_edf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = EclipseConfig(
            project_name="AFH_Nivel3b_v43",
            researcher="Camilo Sjöberg Tala",
            sacred_seed=2025,
            development_ratio=0.7,
            holdout_ratio=0.3,
            output_dir=str(self.output_dir)
        )
        
        self.eclipse = EclipseFramework(self.config)
        
        # Criterios con TEST PRIMARIO (binomial)
        self.criteria = [
            FalsificationCriteria(
                "pct_subjects_hstar_dominant", 55.0, ">=",
                "≥55% sujetos con H* dominante (>50% de sus transiciones)", True
            ),
            FalsificationCriteria(
                "mean_subject_pct_hstar_first", 50.0, ">",
                "Media de %H* first por sujeto > 50%", True
            ),
            FalsificationCriteria(
                "p_value_primary", 0.05, "<",
                "p < 0.05 en test primario (binomial a nivel sujeto)", True
            ),
        ]
    
    def run_complete_analysis(self):
        print("\n" + "="*70)
        print("AFH NIVEL 3b - ECLIPSE v4.3 FINAL")
        print("="*70)
        
        # Buscar archivos
        print("\n📂 Buscando archivos...")
        files = self.processor.find_psg_files()
        
        if not files:
            print("❌ No se encontraron archivos PSG")
            return None
        
        subject_ids = [f[0] for f in files]
        print(f"   Total sujetos: {len(subject_ids)}")
        
        # Split
        dev_ids, holdout_ids = self.eclipse.stage1_split(subject_ids)
        
        # Criterios
        self.eclipse.stage2_register_criteria(self.criteria)
        
        # Procesar
        print(f"\n{'='*70}")
        print("PROCESANDO SUJETOS")
        print(f"{'='*70}")
        
        all_transitions = []
        subject_summaries = []
        files_dict = {f[0]: (f[1], f[2]) for f in files}
        
        for i, subject_id in enumerate(subject_ids):
            print(f"\r   [{i+1}/{len(subject_ids)}] {subject_id}...", end="", flush=True)
            
            psg_path, hyp_path = files_dict[subject_id]
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
                hstar_onset = self.processor.detect_onset(metrics['hstar'], metrics['times'])
                pac_onset = self.processor.detect_onset(metrics['pac'], metrics['times'])
                
                if hstar_onset is not None and pac_onset is not None:
                    delta = hstar_onset - pac_onset
                    trans_data = {
                        'subject_id': subject_id,
                        'from_stage': trans['from_stage'],
                        'hstar_onset': hstar_onset,
                        'pac_onset': pac_onset,
                        'delta': delta,
                        'hstar_first': delta < 0,
                        'in_development': subject_id in dev_ids
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
                    'in_development': subject_id in dev_ids
                })
        
        print(f"\n   Total transiciones: {len(all_transitions)}")
        print(f"   Sujetos con transiciones: {len(subject_summaries)}")
        
        # Separar
        dev_subjects = [s for s in subject_summaries if s['in_development']]
        holdout_subjects = [s for s in subject_summaries if not s['in_development']]
        dev_trans = [t for t in all_transitions if t['in_development']]
        holdout_trans = [t for t in all_transitions if not t['in_development']]
        
        print(f"\n   Desarrollo: {len(dev_subjects)} sujetos, {len(dev_trans)} transiciones")
        print(f"   Holdout: {len(holdout_subjects)} sujetos, {len(holdout_trans)} transiciones")
        
        # Analizar
        print(f"\n{'='*70}")
        print("STAGE 3: DEVELOPMENT")
        print(f"{'='*70}")
        dev_results = self._analyze(dev_subjects, dev_trans, "Development")
        
        print(f"\n{'='*70}")
        print("STAGE 4: HOLDOUT VALIDATION")
        print(f"{'='*70}")
        val_results = self._analyze(holdout_subjects, holdout_trans, "Holdout")
        val_results['status'] = 'success'
        
        # Assessment
        assessment = self.eclipse.stage5_assessment(dev_results, val_results)
        
        # Guardar
        pd.DataFrame(all_transitions).to_csv(self.output_dir / "all_transitions.csv", index=False)
        pd.DataFrame(subject_summaries).to_csv(self.output_dir / "subject_summaries.csv", index=False)
        
        print(f"\n📁 Resultados en: {self.output_dir}")
        
        return assessment
    
    def _analyze(self, subjects, transitions, label):
        if not subjects:
            return {'n_subjects': 0, 'subject_level_metrics': {}}
        
        min_trans = SUBJECT_INCLUSION['min_transitions']
        
        # Filtrar sujetos con suficientes transiciones
        eligible_subjects = [s for s in subjects if s['n_transitions'] >= min_trans]
        excluded_subjects = [s for s in subjects if s['n_transitions'] < min_trans]
        
        n_total = len(subjects)
        n_eligible = len(eligible_subjects)
        n_excluded = len(excluded_subjects)
        
        if not eligible_subjects:
            print(f"\n   ⚠️  {label}: No hay sujetos con ≥{min_trans} transiciones")
            return {
                'n_subjects_total': n_total,
                'n_subjects_eligible': 0,
                'n_subjects_excluded': n_excluded,
                'subject_level_metrics': {}
            }
        
        pcts = np.array([s['pct_hstar_first'] for s in eligible_subjects])
        
        # Sujetos con H* dominante (entre los elegibles)
        n_hstar_dominant = sum(1 for s in eligible_subjects if s['hstar_dominant'])
        pct_subjects_dominant = 100 * n_hstar_dominant / n_eligible
        
        # Media y SD
        mean_pct = np.mean(pcts)
        std_pct = np.std(pcts)
        
        # ============================================================
        # TEST PRIMARIO: Binomial a nivel sujeto + IC 95% Wilson
        # H0: proporción de sujetos H*-dominant = 50%
        # ============================================================
        binom_result = binomtest(n_hstar_dominant, n_eligible, 0.5, alternative='greater')
        p_primary = binom_result.pvalue
        ci = binom_result.proportion_ci(confidence_level=0.95, method="wilson")
        ci_low, ci_high = ci.low, ci.high
        
        # ============================================================
        # TEST SENSIBILIDAD: Wilcoxon signed-rank con trazabilidad
        # H0: mediana de (pct - 50) = 0
        # ============================================================
        diffs = pcts - 50
        nonzero_diffs = diffs[diffs != 0]
        n_subjects_total_wilcoxon = len(pcts)
        n_subjects_nonzero_wilcoxon = len(nonzero_diffs)
        
        try:
            if n_subjects_nonzero_wilcoxon >= 10:
                p_sensitivity = wilcoxon(nonzero_diffs, alternative='greater').pvalue
            else:
                p_sensitivity = np.nan
        except:
            p_sensitivity = np.nan
        
        # Transiciones (secundario)
        n_trans = len(transitions)
        n_hstar_first = sum(1 for t in transitions if t['hstar_first'])
        pct_trans = 100 * n_hstar_first / n_trans if n_trans > 0 else 0
        deltas = [t['delta'] for t in transitions]
        mean_delta = np.mean(deltas) if deltas else 0
        
        # Print
        print(f"\n   {label}:")
        print(f"   {'─'*50}")
        print(f"\n   📊 FILTRO DE INCLUSIÓN:")
        print(f"      Sujetos totales: {n_total}")
        print(f"      Elegibles (≥{min_trans} trans): {n_eligible}")
        print(f"      Excluidos (<{min_trans} trans): {n_excluded}")
        
        print(f"\n   📊 NIVEL SUJETO (PRIMARIO, n={n_eligible}):")
        print(f"      H* dominante: {n_hstar_dominant} ({pct_subjects_dominant:.1f}%)")
        print(f"      IC 95% Wilson: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
        print(f"      Media %H* first: {mean_pct:.1f}% ± {std_pct:.1f}%")
        print(f"      p-value PRIMARIO (binomial): {p_primary:.4f}")
        
        if not np.isnan(p_sensitivity):
            print(f"      p-value sensibilidad (Wilcoxon): {p_sensitivity:.4f}")
            print(f"         → n_total={n_subjects_total_wilcoxon}, n_nonzero={n_subjects_nonzero_wilcoxon}")
        else:
            print(f"      p-value sensibilidad (Wilcoxon): N/A")
            print(f"         → n_total={n_subjects_total_wilcoxon}, n_nonzero={n_subjects_nonzero_wilcoxon} (<10, insuficiente)")
        
        print(f"\n   📊 NIVEL TRANSICIÓN (secundario):")
        print(f"      N transiciones: {n_trans}")
        print(f"      H* primero: {pct_trans:.1f}%")
        print(f"      Delta medio: {mean_delta:.1f}s")
        
        return {
            'n_subjects_total': n_total,
            'n_subjects_eligible': n_eligible,
            'n_subjects_excluded': n_excluded,
            'n_transitions': n_trans,
            'subject_level_metrics': {
                'pct_subjects_hstar_dominant': pct_subjects_dominant,
                'ci95_subjects_hstar_dominant_low': ci_low * 100,
                'ci95_subjects_hstar_dominant_high': ci_high * 100,
                'mean_subject_pct_hstar_first': mean_pct,
                'std_subject_pct_hstar_first': std_pct,
                'p_value_primary': p_primary,
                'p_value_sensitivity_wilcoxon': p_sensitivity,
                'wilcoxon_n_subjects_total': n_subjects_total_wilcoxon,
                'wilcoxon_n_subjects_nonzero': n_subjects_nonzero_wilcoxon,
            },
            'transition_level_metrics': {
                'pct_hstar_first': pct_trans,
                'mean_delta': mean_delta,
                'n_hstar_first': n_hstar_first,
            },
            'subject_details': {
                'eligible': [s['subject_id'] for s in eligible_subjects],
                'excluded': [s['subject_id'] for s in excluded_subjects]
            }
        }


# ============================================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================================

DEFAULT_PATHS = [
    # Windows - Google Drive
    r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
    # Alternativas comunes
    r"C:\Users\Camilo\Google Drive\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette",
    r"D:\sleep-cassette",
    # Linux/Mac
    "/mnt/g/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette",
    os.path.expanduser("~/Google Drive/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette"),
    # Directorio actual
    "./sleep-cassette",
]

DEFAULT_OUTPUT = r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\nivel3b_eclipse_results"


def find_data_path():
    """Busca automáticamente el directorio de datos."""
    for path in DEFAULT_PATHS:
        p = Path(path)
        if p.exists() and p.is_dir():
            # Verificar que tenga archivos PSG
            psg_files = list(p.glob("*PSG.edf")) + list(p.glob("*-PSG.edf"))
            if psg_files:
                return p
    return None


def find_output_path():
    """Determina el directorio de salida."""
    # Intentar usar el default
    default = Path(DEFAULT_OUTPUT)
    try:
        default.mkdir(parents=True, exist_ok=True)
        return default
    except:
        pass
    
    # Fallback a directorio actual
    fallback = Path("./nivel3b_eclipse_results")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AFH Nivel 3b - ECLIPSE v4.3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python nivel3b_eclipse_v4.3.py                    # Automático
  python nivel3b_eclipse_v4.3.py --sleep-edf-path "G:/Mi unidad/..."
        """
    )
    parser.add_argument('--sleep-edf-path', type=str, default=None,
                        help='Ruta a sleep-cassette (auto-detecta si no se especifica)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directorio de salida (auto-detecta si no se especifica)')
    args = parser.parse_args()
    
    # Auto-detectar ruta de datos
    if args.sleep_edf_path:
        data_path = Path(args.sleep_edf_path)
    else:
        print("🔍 Buscando directorio de datos automáticamente...")
        data_path = find_data_path()
    
    if data_path is None or not data_path.exists():
        print("❌ No se encontró el directorio de datos.")
        print("   Rutas buscadas:")
        for p in DEFAULT_PATHS[:3]:
            print(f"      • {p}")
        print("\n   Usa: python nivel3b_eclipse_v4.3.py --sleep-edf-path \"TU_RUTA\"")
        sys.exit(1)
    
    print(f"✅ Datos encontrados: {data_path}")
    
    # Auto-detectar salida
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = find_output_path()
    
    print(f"✅ Salida: {output_path}")
    
    # Ejecutar análisis
    analysis = Nivel3bAnalysis(str(data_path), str(output_path))
    result = analysis.run_complete_analysis()
    
    if result:
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)
        print(f"Veredicto: {result['verdict']}")
        print(f"Criterios: {result['required_criteria_passed']}")
        print(f"\n📁 Resultados guardados en:")
        print(f"   {output_path}")
        
        # Mostrar archivos generados
        print("\n📄 Archivos generados:")
        for f in output_path.glob("*"):
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()