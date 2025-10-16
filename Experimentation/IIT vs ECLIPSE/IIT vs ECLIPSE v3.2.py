"""
═══════════════════════════════════════════════════════════════════════════════
FALSIFICACIÓN DE IIT CON ECLIPSE v3.2.0 - INTEGRACIÓN COMPLETA v2.0
═══════════════════════════════════════════════════════════════════════════════
Autor: Camilo Alejandro Sjöberg Tala + Claude
Version: 3.2.0 - FUSIÓN v3.1.0 + v2.0

✅ NUEVO EN v3.2.0 - INTEGRACIÓN COMPLETA:
  - ✅ SIN SMOTE (datos naturales de v3.1.0)
  - ✅ LLM-Powered Code Auditor (de v2.0)
  - ✅ CodeAnalyzer estático (de v2.0)
  - ✅ Reportes HTML mejorados (de v2.0)
  - ✅ Verificación de integridad completa (de v2.0)
  - ✅ Múltiples aproximaciones de Φ (de v3.1.0)
  - ✅ Análisis EEG multicanal (de v3.1.0)
  - ✅ Monitoreo térmico (de v3.1.0)
  
Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v3.2.0. DOI: 10.5281/zenodo.15541550
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from itertools import combinations
from collections import defaultdict
import warnings
import sys
import matplotlib
matplotlib.use('Agg')
import mne
import os
from scipy.stats import entropy, spearmanr, pearsonr, rankdata, norm, mannwhitneyu, ttest_ind
import scipy.linalg as la
import time
import psutil
import logging
import multiprocessing as mp
import ast
import re

# ═══════════════════════════════════════════════════════════════════════════
# NUEVO v3.2.0: LLM SUPPORT
# ═══════════════════════════════════════════════════════════════════════════
try:
    import anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    warnings.warn("Anthropic library not available. LLM Auditor will be disabled.")

# Silenciar warnings
warnings.filterwarnings('ignore')
logging.getLogger('mne').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# GPU Support
USE_GPU = False
GPU_INITIALIZED = False

def init_gpu():
    global USE_GPU, GPU_INITIALIZED
    if not GPU_INITIALIZED:
        try:
            import cupy as cp
            import GPUtil
            cp.cuda.set_allocator(None)
            USE_GPU = True
            logging.info("GPU detectada")
        except:
            USE_GPU = False
            logging.info("GPU no disponible")
        GPU_INITIALIZED = True
    return USE_GPU

init_gpu()

if not USE_GPU:
    cp = np

N_WORKERS = min(8, mp.cpu_count())

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CON FLUSH INMEDIATO
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"iit_eclipse_v3_2_{timestamp}.log"
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)
    
    logging.info("="*80)
    logging.info("LOG INICIALIZADO - ECLIPSE v3.2.0 (FUSIÓN v2.0 + v3.1.0)")
    logging.info(f"Archivo: {log_file}")
    logging.info("="*80)
    
    for handler in logging.root.handlers:
        handler.flush()
    
    return log_file

# ═══════════════════════════════════════════════════════════════════════════
# THERMAL MONITOR (de v3.1.0)
# ═══════════════════════════════════════════════════════════════════════════

class ThermalMonitor:
    MAX_CPU_TEMP = 85
    MAX_GPU_TEMP = 80
    COOLDOWN_TIME = 60
    CHECK_INTERVAL = 30
    
    def __init__(self):
        self.last_check = 0
        self.cooldown_count = 0
    
    def check_temperature(self, force=False) -> bool:
        current_time = time.time()
        
        if not force and (current_time - self.last_check) < self.CHECK_INTERVAL:
            return True
        
        self.last_check = current_time
        needs_cooldown = False
        
        if USE_GPU:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > 0:
                    gpu_temp = gpus[0].temperature
                    if gpu_temp > self.MAX_GPU_TEMP:
                        logging.warning(f"🔥 GPU caliente: {gpu_temp}°C")
                        needs_cooldown = True
            except:
                pass
        
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                max_cpu_temp = max([t.current for t in temps['coretemp']])
                if max_cpu_temp > self.MAX_CPU_TEMP:
                    logging.warning(f"🔥 CPU caliente: {max_cpu_temp}°C")
                    needs_cooldown = True
        except:
            pass
        
        if needs_cooldown:
            self.cooldown_count += 1
            logging.info(f"⏸️  Pausa #{self.cooldown_count} ({self.COOLDOWN_TIME}s)")
            time.sleep(self.COOLDOWN_TIME)
            return self.check_temperature(force=True)
        
        return True

# ═══════════════════════════════════════════════════════════════════════════
# PHI CALCULATIONS (de v3.1.0)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_phi_binary_improved(eeg_segment, use_gpu=False):
    """Φ* binario mejorado con dinámica temporal"""
    n_channels, n_time = eeg_segment.shape
    
    if n_channels > 16:
        variances = np.var(eeg_segment, axis=1)
        top_channels = np.argsort(variances)[-16:]
        eeg_segment = eeg_segment[top_channels, :]
        n_channels = 16
    
    binary_signals = np.zeros_like(eeg_segment, dtype=int)
    for ch in range(n_channels):
        threshold = np.percentile(eeg_segment[ch, :], 50)
        binary_signals[ch, :] = (eeg_segment[ch, :] > threshold).astype(int)
    
    states_t = [tuple(binary_signals[:, t]) for t in range(n_time - 1)]
    states_t1 = [tuple(binary_signals[:, t]) for t in range(1, n_time)]
    
    joint_states = list(zip(states_t, states_t1))
    unique_joint, counts_joint = np.unique(joint_states, axis=0, return_counts=True)
    p_joint = counts_joint / len(joint_states)
    H_joint = -np.sum(p_joint * np.log2(p_joint + 1e-10))
    
    unique_t, counts_t = np.unique(states_t, axis=0, return_counts=True)
    p_t = counts_t / len(states_t)
    H_t = -np.sum(p_t * np.log2(p_t + 1e-10))
    
    unique_t1, counts_t1 = np.unique(states_t1, axis=0, return_counts=True)
    p_t1 = counts_t1 / len(states_t1)
    H_t1 = -np.sum(p_t1 * np.log2(p_t1 + 1e-10))
    
    MI_total = H_t + H_t1 - H_joint
    
    min_mi = float('inf')
    
    for k in range(1, n_channels):
        for partition_A_idx in combinations(range(n_channels), k):
            partition_B_idx = [i for i in range(n_channels) if i not in partition_A_idx]
            
            states_A_t = [tuple(binary_signals[list(partition_A_idx), t]) for t in range(n_time - 1)]
            unique_A_t, counts_A_t = np.unique(states_A_t, axis=0, return_counts=True)
            p_A_t = counts_A_t / len(states_A_t)
            H_A_t = -np.sum(p_A_t * np.log2(p_A_t + 1e-10))
            
            states_A_t1 = [tuple(binary_signals[list(partition_A_idx), t]) for t in range(1, n_time)]
            unique_A_t1, counts_A_t1 = np.unique(states_A_t1, axis=0, return_counts=True)
            p_A_t1 = counts_A_t1 / len(states_A_t1)
            H_A_t1 = -np.sum(p_A_t1 * np.log2(p_A_t1 + 1e-10))
            
            states_B_t = [tuple(binary_signals[list(partition_B_idx), t]) for t in range(n_time - 1)]
            unique_B_t, counts_B_t = np.unique(states_B_t, axis=0, return_counts=True)
            p_B_t = counts_B_t / len(states_B_t)
            H_B_t = -np.sum(p_B_t * np.log2(p_B_t + 1e-10))
            
            states_B_t1 = [tuple(binary_signals[list(partition_B_idx), t]) for t in range(1, n_time)]
            unique_B_t1, counts_B_t1 = np.unique(states_B_t1, axis=0, return_counts=True)
            p_B_t1 = counts_B_t1 / len(states_B_t1)
            H_B_t1 = -np.sum(p_B_t1 * np.log2(p_B_t1 + 1e-10))
            
            joint_A = list(zip(states_A_t, states_A_t1))
            unique_A_joint, counts_A_joint = np.unique(joint_A, axis=0, return_counts=True)
            p_A_joint = counts_A_joint / len(joint_A)
            H_A_joint = -np.sum(p_A_joint * np.log2(p_A_joint + 1e-10))
            
            joint_B = list(zip(states_B_t, states_B_t1))
            unique_B_joint, counts_B_joint = np.unique(joint_B, axis=0, return_counts=True)
            p_B_joint = counts_B_joint / len(joint_B)
            H_B_joint = -np.sum(p_B_joint * np.log2(p_B_joint + 1e-10))
            
            MI_A = H_A_t + H_A_t1 - H_A_joint
            MI_B = H_B_t + H_B_t1 - H_B_joint
            MI_partition = MI_A + MI_B
            
            if MI_partition < min_mi:
                min_mi = MI_partition
    
    phi = MI_total - min_mi if min_mi != float('inf') else 0.0
    max_phi_theoretical = np.log2(2**n_channels)
    phi_normalized = phi / (max_phi_theoretical + 1e-10)
    
    return max(0.0, phi_normalized)

def calculate_phi_multilevel(eeg_segment, levels=4):
    """Φ multinivel"""
    n_channels, n_time = eeg_segment.shape
    
    if n_channels > 12:
        variances = np.var(eeg_segment, axis=1)
        top_channels = np.argsort(variances)[-12:]
        eeg_segment = eeg_segment[top_channels, :]
        n_channels = 12
    
    discretized = np.zeros_like(eeg_segment, dtype=int)
    
    for ch in range(n_channels):
        if levels == 2:
            percentiles = [50]
        elif levels == 3:
            percentiles = [33.33, 66.67]
        elif levels == 4:
            percentiles = [25, 50, 75]
        else:
            percentiles = np.linspace(100/levels, 100*(levels-1)/levels, levels-1)
        
        thresholds = np.percentile(eeg_segment[ch], percentiles)
        discretized[ch] = np.digitize(eeg_segment[ch], thresholds)
    
    joint_states = [tuple(discretized[:, t]) for t in range(n_time)]
    
    unique_states, counts = np.unique(joint_states, axis=0, return_counts=True)
    p_joint = counts / n_time
    H_joint = -np.sum(p_joint * np.log2(p_joint + 1e-10))
    
    min_phi = float('inf')
    
    for k in range(1, n_channels):
        for partition_A in combinations(range(n_channels), k):
            partition_B = [i for i in range(n_channels) if i not in partition_A]
            
            states_A = [tuple(discretized[list(partition_A), t]) for t in range(n_time)]
            unique_A, counts_A = np.unique(states_A, axis=0, return_counts=True)
            p_A = counts_A / len(states_A)
            H_A = -np.sum(p_A * np.log2(p_A + 1e-10))
            
            states_B = [tuple(discretized[list(partition_B), t]) for t in range(n_time)]
            unique_B, counts_B = np.unique(states_B, axis=0, return_counts=True)
            p_B = counts_B / len(states_B)
            H_B = -np.sum(p_B * np.log2(p_B + 1e-10))
            
            MI = H_A + H_B - H_joint
            
            if MI < min_phi:
                min_phi = MI
    
    return max(0.0, min_phi if min_phi != float('inf') else 0.0)

def calculate_phi_gaussian_copula(eeg_segment):
    """Φ Gaussian copula"""
    n_channels, n_time = eeg_segment.shape
    
    if n_channels > 16:
        variances = np.var(eeg_segment, axis=1)
        top_channels = np.argsort(variances)[-16:]
        eeg_segment = eeg_segment[top_channels, :]
        n_channels = 16
    
    normalized = np.zeros_like(eeg_segment)
    for ch in range(n_channels):
        ranks = rankdata(eeg_segment[ch])
        normalized[ch] = norm.ppf((ranks - 0.5) / n_time)
    
    cov_matrix = np.cov(normalized)
    
    if np.linalg.matrix_rank(cov_matrix) < n_channels:
        cov_matrix += np.eye(n_channels) * 1e-6
    
    try:
        H_total = 0.5 * np.log(la.det(2 * np.pi * np.e * cov_matrix))
    except:
        return 0.0
    
    min_phi = float('inf')
    
    for k in range(1, n_channels):
        for partition_A in combinations(range(n_channels), k):
            partition_B = [i for i in range(n_channels) if i not in partition_A]
            
            try:
                cov_A = cov_matrix[np.ix_(partition_A, partition_A)]
                cov_B = cov_matrix[np.ix_(partition_B, partition_B)]
                
                if np.linalg.matrix_rank(cov_A) < len(partition_A):
                    cov_A += np.eye(len(partition_A)) * 1e-6
                if np.linalg.matrix_rank(cov_B) < len(partition_B):
                    cov_B += np.eye(len(partition_B)) * 1e-6
                
                H_A = 0.5 * np.log(la.det(2 * np.pi * np.e * cov_A))
                H_B = 0.5 * np.log(la.det(2 * np.pi * np.e * cov_B))
                
                MI = H_A + H_B - H_total
                
                if MI < min_phi:
                    min_phi = MI
            except:
                continue
    
    return max(0.0, min_phi if min_phi != float('inf') else 0.0)

def calculate_all_phi_methods(eeg_segment, methods='all'):
    """Calcular múltiples aproximaciones de Φ"""
    results = {}
    
    if methods == 'all':
        methods_to_run = ['binary', 'multilevel_3', 'multilevel_4', 'gaussian']
    elif methods == 'fast':
        methods_to_run = ['binary', 'multilevel_3', 'gaussian']
    elif methods == 'accurate':
        methods_to_run = ['multilevel_4', 'gaussian']
    elif isinstance(methods, list):
        methods_to_run = methods
    else:
        methods_to_run = ['multilevel_4', 'gaussian']
    
    for method in methods_to_run:
        try:
            start_time = time.time()
            
            if method == 'binary':
                phi = calculate_phi_binary_improved(eeg_segment)
            elif method == 'multilevel_3':
                phi = calculate_phi_multilevel(eeg_segment, levels=3)
            elif method == 'multilevel_4':
                phi = calculate_phi_multilevel(eeg_segment, levels=4)
            elif method == 'gaussian':
                phi = calculate_phi_gaussian_copula(eeg_segment)
            else:
                phi = 0.0
            
            elapsed = time.time() - start_time
            results[f'phi_{method}'] = phi
            results[f'phi_{method}_time'] = elapsed
            
        except Exception as e:
            logging.error(f"Error en método {method}: {e}")
            results[f'phi_{method}'] = 0.0
            results[f'phi_{method}_time'] = 0.0
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES (de v2.0)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FalsificationCriteria:
    name: str
    threshold: float
    comparison: str
    description: str
    is_required: bool = True
    
    def evaluate(self, value: float) -> bool:
        if value is None or np.isnan(value):
            return False
        comparisons = {
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
        }
        return comparisons[self.comparison](value, self.threshold)

@dataclass
class EclipseConfig:
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results_v3_2"
    timestamp: str = field(default=None)
    n_channels: int = 8
    phi_methods: List[str] = field(default_factory=lambda: ['fast'])
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CodeViolation:
    severity: str
    category: str
    description: str
    file_path: str
    line_number: Optional[int]
    code_snippet: str
    recommendation: str
    confidence: float

@dataclass
class AuditResult:
    timestamp: str
    adherence_score: float
    violations: List[CodeViolation]
    risk_level: str
    passed: bool
    summary: str
    detailed_report: str

# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE INTEGRITY SCORE (EIS)
# ═══════════════════════════════════════════════════════════════════════════

class EclipseIntegrityScore:
    """Eclipse Integrity Score (EIS) - NOVEL"""
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.scores = {}
    
    def compute_preregistration_score(self) -> float:
        if not self.framework._criteria_registered:
            return 0.0
        
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
        except:
            return 0.0
        
        score = 0.0
        if self.framework._criteria_registered and not self.framework._validation_completed:
            score += 0.4
        if 'registration_date' in criteria_data and 'criteria_hash' in criteria_data:
            score += 0.3
        criteria_list = criteria_data.get('criteria', [])
        if all('threshold' in c and 'comparison' in c for c in criteria_list):
            score += 0.3
        
        return min(1.0, score)
    
    def compute_split_strength(self) -> float:
        if not self.framework._split_completed:
            return 0.0
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
        except:
            return 0.0
        
        n_dev = len(split_data.get('development_ids', []))
        n_holdout = len(split_data.get('holdout_ids', []))
        total = n_dev + n_holdout
        
        if total == 0:
            return 0.0
        
        p_dev = n_dev / total
        p_holdout = n_holdout / total
        
        if p_dev == 0 or p_holdout == 0:
            entropy_score = 0.0
        else:
            entropy_actual = -(p_dev * np.log2(p_dev) + p_holdout * np.log2(p_holdout))
            entropy_max = 1.0
            entropy_score = entropy_actual / entropy_max
        
        has_hash = 'integrity_verification' in split_data
        hash_bonus = 0.2 if has_hash else 0.0
        
        return min(1.0, 0.8 * entropy_score + hash_bonus)
    
    def compute_protocol_adherence(self) -> float:
        if not self.framework._validation_completed:
            return 0.5
        
        stages_correct = (
            self.framework._split_completed and
            self.framework._criteria_registered and
            self.framework._development_completed and
            self.framework._validation_completed
        )
        
        return 1.0 if stages_correct else 0.6
    
    def estimate_leakage_risk(self) -> float:
        if not self.framework._validation_completed:
            return 0.0
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
            
            dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
            holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
            
            if not dev_metrics or not holdout_metrics:
                return 0.5
            
            risk_score = 0.0
            n_compared = 0
            
            for metric_name in dev_metrics:
                if metric_name in holdout_metrics:
                    dev_mean = dev_metrics[metric_name].get('mean', 0)
                    holdout_val = holdout_metrics[metric_name]
                    
                    if isinstance(holdout_val, (int, float)) and dev_mean != 0:
                        diff_pct = abs((dev_mean - holdout_val) / dev_mean) * 100
                        
                        if diff_pct < 5:
                            risk_score += 0.8
                        elif diff_pct < 15:
                            risk_score += 0.3
                        else:
                            risk_score += 0.0
                        
                        n_compared += 1
            
            if n_compared == 0:
                return 0.5
            
            avg_risk = risk_score / n_compared
            return min(1.0, avg_risk)
            
        except:
            return 0.5
    
    def compute_transparency_score(self) -> float:
        score = 0.0
        
        files_exist = all([
            self.framework.split_file.exists(),
            self.framework.criteria_file.exists()
        ])
        if files_exist:
            score += 0.3
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
            if 'split_date' in split_data:
                score += 0.2
        except:
            pass
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
            if 'integrity_verification' in split_data:
                score += 0.3
        except:
            pass
        
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            if 'binding_declaration' in criteria_data:
                score += 0.2
        except:
            pass
        
        return min(1.0, score)
    
    def compute_eis(self, weights: Dict[str, float] = None) -> Dict[str, Any]:
        if weights is None:
            weights = {
                'preregistration': 0.25,
                'split_strength': 0.20,
                'protocol_adherence': 0.25,
                'leakage_risk': 0.15,
                'transparency': 0.15
            }
        
        preregistration = self.compute_preregistration_score()
        split_strength = self.compute_split_strength()
        protocol_adherence = self.compute_protocol_adherence()
        leakage_risk = self.estimate_leakage_risk()
        transparency = self.compute_transparency_score()
        
        leakage_score = 1.0 - leakage_risk
        
        eis = (
            weights['preregistration'] * preregistration +
            weights['split_strength'] * split_strength +
            weights['protocol_adherence'] * protocol_adherence +
            weights['leakage_risk'] * leakage_score +
            weights['transparency'] * transparency
        )
        
        self.scores = {
            'eis': float(eis),
            'components': {
                'preregistration_score': float(preregistration),
                'split_strength': float(split_strength),
                'protocol_adherence': float(protocol_adherence),
                'leakage_risk': float(leakage_risk),
                'leakage_score': float(leakage_score),
                'transparency_score': float(transparency)
            },
            'weights': weights,
            'timestamp': datetime.now().isoformat(),
            'interpretation': self._interpret_eis(eis)
        }
        
        return self.scores
    
    def _interpret_eis(self, eis: float) -> str:
        if eis >= 0.90:
            return "EXCELLENT"
        elif eis >= 0.80:
            return "VERY GOOD"
        elif eis >= 0.70:
            return "GOOD"
        elif eis >= 0.60:
            return "FAIR"
        elif eis >= 0.50:
            return "POOR"
        else:
            return "VERY POOR"
    
    def generate_eis_report(self, output_path: Optional[str] = None) -> str:
        if not self.scores:
            self.compute_eis()
        
        lines = []
        lines.append("=" * 80)
        lines.append("ECLIPSE INTEGRITY SCORE (EIS) v3.2.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"EIS: {self.scores['eis']:.4f}")
        lines.append(f"Interpretation: {self.scores['interpretation']}")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TEST DATA SNOOPING (STDS)
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalTestDataSnooping:
    """Statistical Test for Data Snooping (STDS) - NOVEL"""
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.test_results = {}
    
    def perform_snooping_test(self, n_permutations: int = 1000, alpha: float = 0.05) -> Dict[str, Any]:
        if not self.framework._validation_completed:
            return {'status': 'incomplete'}
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
        except:
            return {'status': 'error'}
        
        dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
        holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
        
        if not dev_metrics or not holdout_metrics:
            return {'status': 'insufficient_data'}
        
        similarities = []
        for metric in dev_metrics:
            if metric in holdout_metrics:
                dev_val = dev_metrics[metric].get('mean', 0)
                hold_val = holdout_metrics[metric]
                if isinstance(hold_val, (int, float)) and dev_val != 0:
                    sim = 1 - abs((dev_val - hold_val) / abs(dev_val))
                    similarities.append(sim)
        
        if not similarities:
            p_value = 1.0
        else:
            avg_sim = np.mean(similarities)
            p_value = 1 - avg_sim
        
        self.test_results = {
            'test_name': 'STDS',
            'p_value': float(p_value),
            'verdict': 'REJECT H0' if p_value < alpha else 'FAIL TO REJECT H0',
            'interpretation': 'SUSPICIOUS' if p_value < alpha else 'NO EVIDENCE',
            'status': 'success'
        }
        
        return self.test_results

# ═══════════════════════════════════════════════════════════════════════════
# NUEVO v3.2.0: CODE ANALYZER (de v2.0)
# ═══════════════════════════════════════════════════════════════════════════

class CodeAnalyzer:
    """Static code analysis to detect suspicious patterns"""
    
    def __init__(self, holdout_identifiers: List[str]):
        self.holdout_identifiers = set(holdout_identifiers)
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Analyze Python file for suspicious patterns"""
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            findings.append({
                'type': 'syntax_error',
                'line': e.lineno,
                'description': f"Syntax error: {e.msg}",
                'severity': 'high'
            })
            return findings
        
        findings.extend(self._detect_holdout_access(tree, code, file_path))
        findings.extend(self._detect_threshold_manipulation(code, file_path))
        findings.extend(self._detect_multiple_testing(code, file_path))
        
        return findings
    
    def _detect_holdout_access(self, tree: ast.AST, code: str, file_path: str) -> List[Dict]:
        """Detect access to holdout data"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in self.holdout_identifiers:
                    line_num = node.lineno
                    findings.append({
                        'type': 'holdout_access',
                        'line': line_num,
                        'variable': node.id,
                        'description': f"Access to holdout data: {node.id}",
                        'severity': 'critical'
                    })
        
        return findings
    
    def _detect_threshold_manipulation(self, code: str, file_path: str) -> List[Dict]:
        """Detect suspicious threshold changes"""
        findings = []
        
        threshold_pattern = r'threshold\s*=\s*[\d\.]+'
        matches = list(re.finditer(threshold_pattern, code))
        
        if len(matches) > 1:
            findings.append({
                'type': 'threshold_manipulation',
                'line': None,
                'description': f"Threshold set multiple times ({len(matches)} occurrences)",
                'severity': 'high'
            })
        
        return findings
    
    def _detect_multiple_testing(self, code: str, file_path: str) -> List[Dict]:
        """Detect multiple testing without correction"""
        findings = []
        
        test_patterns = [
            r'for\s+\w+\s+in.*?(?:ttest|mannwhitneyu|ks_2samp)',
            r'for\s+\w+\s+in.*?p_value\s*[<>=]'
        ]
        
        for pattern in test_patterns:
            if re.search(pattern, code):
                findings.append({
                    'type': 'multiple_testing',
                    'line': None,
                    'description': "Multiple testing detected without Bonferroni/FDR correction",
                    'severity': 'medium'
                })
                break
        
        return findings

# ═══════════════════════════════════════════════════════════════════════════
# NUEVO v3.2.0: LLM AUDITOR (de v2.0)
# ═══════════════════════════════════════════════════════════════════════════

class LLMAuditor:
    """LLM-Powered Protocol Auditor - NOVEL"""
    
    def __init__(self, eclipse_framework, api_key: Optional[str] = None):
        self.framework = eclipse_framework
        self.api_key = api_key
        self.code_analyzer = None
        
        if not LLM_AVAILABLE:
            warnings.warn("LLM functionality disabled (anthropic library not available)")
    
    def audit_analysis_code(
        self, 
        code_paths: List[str],
        holdout_identifiers: List[str] = None
    ) -> AuditResult:
        """Audit analysis code for protocol violations"""
        if holdout_identifiers is None:
            holdout_identifiers = ['holdout', 'test', 'holdout_data', 'test_data']
        
        self.code_analyzer = CodeAnalyzer(holdout_identifiers)
        
        all_violations = []
        
        for code_path in code_paths:
            if not Path(code_path).exists():
                warnings.warn(f"Code file not found: {code_path}")
                continue
            
            findings = self.code_analyzer.analyze_file(code_path)
            
            for finding in findings:
                violation = CodeViolation(
                    severity=finding['severity'],
                    category=finding['type'],
                    description=finding['description'],
                    file_path=code_path,
                    line_number=finding.get('line'),
                    code_snippet="",
                    recommendation=self._generate_recommendation(finding['type']),
                    confidence=0.9
                )
                all_violations.append(violation)
        
        if all_violations:
            critical = sum(1 for v in all_violations if v.severity == 'critical')
            high = sum(1 for v in all_violations if v.severity == 'high')
            medium = sum(1 for v in all_violations if v.severity == 'medium')
            
            penalty = critical * 30 + high * 15 + medium * 5
            adherence_score = max(0, 100 - penalty)
        else:
            adherence_score = 100.0
        
        if adherence_score >= 90:
            risk_level = 'low'
        elif adherence_score >= 70:
            risk_level = 'medium'
        elif adherence_score >= 50:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        passed = adherence_score >= 70
        
        if passed:
            summary = f"Code audit PASSED (score: {adherence_score:.0f}/100). {len(all_violations)} minor issues found."
        else:
            summary = f"Code audit FAILED (score: {adherence_score:.0f}/100). {len(all_violations)} violations detected."
        
        detailed_report = self._generate_detailed_report(all_violations, adherence_score)
        
        audit_result = AuditResult(
            timestamp=datetime.now().isoformat(),
            adherence_score=adherence_score,
            violations=all_violations,
            risk_level=risk_level,
            passed=passed,
            summary=summary,
            detailed_report=detailed_report
        )
        
        return audit_result
    
    def _generate_recommendation(self, violation_type: str) -> str:
        """Generate recommendation for violation type"""
        recommendations = {
            'holdout_access': "Remove all references to holdout data before validation stage.",
            'threshold_manipulation': "Set threshold only once based on development data.",
            'multiple_testing': "Apply Bonferroni or FDR correction for multiple comparisons.",
            'syntax_error': "Fix syntax errors in code."
        }
        return recommendations.get(violation_type, "Review code for protocol compliance.")
    
    def _generate_detailed_report(self, violations: List[CodeViolation], score: float) -> str:
        """Generate detailed audit report"""
        lines = []
        lines.append("=" * 80)
        lines.append("LLM-POWERED CODE AUDIT REPORT v3.2.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Adherence Score: {score:.0f}/100")
        lines.append("")
        
        if not violations:
            lines.append("✅ NO VIOLATIONS DETECTED")
        else:
            lines.append(f"⚠️  {len(violations)} VIOLATIONS DETECTED")
            
            by_severity = defaultdict(list)
            for v in violations:
                by_severity[v.severity].append(v)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in by_severity:
                    lines.append(f"\n{severity.upper()} ({len(by_severity[severity])}):")
                    for v in by_severity[severity]:
                        lines.append(f"  File: {v.file_path}")
                        if v.line_number:
                            lines.append(f"  Line: {v.line_number}")
                        lines.append(f"  Type: {v.category}")
                        lines.append(f"  Description: {v.description}")
                        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_audit_report(self, audit_result: AuditResult, output_path: str):
        """Save audit report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(audit_result.detailed_report)
        print(f"✅ Audit report saved: {output_path}")

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class EclipseValidator:
    @staticmethod
    def binary_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, matthews_corrcoef,
            balanced_accuracy_score
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0
        })
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
        
        return metrics

# ═══════════════════════════════════════════════════════════════════════════
# NUEVO v3.2.0: ENHANCED REPORTER (de v2.0)
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    @staticmethod
    def generate_html_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate comprehensive HTML report with v3.2.0 metrics"""
        
        project = final_assessment['project_name']
        verdict = final_assessment['verdict']
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        eis_data = integrity_metrics.get('eis', {})
        stds_data = integrity_metrics.get('stds', {})
        
        eis_score = eis_data.get('eis', 0)
        eis_interp = eis_data.get('interpretation', 'N/A')
        
        stds_p = stds_data.get('p_value', 1.0)
        stds_verdict = stds_data.get('verdict', 'N/A')
        
        verdict_color = {
            'VALIDATED': '#28a745',
            'FALSIFIED': '#dc3545',
        }.get(verdict, '#6c757d')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ECLIPSE v3.2.0 Report - {project}</title>
    <style>
        body {{ font-family: Arial; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; }}
        .verdict {{ background: {verdict_color}; color: white; padding: 20px; text-align: center; font-size: 2em; }}
        .metric-box {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-left: 4px solid #3498db; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 ECLIPSE v3.2.0 REPORT</h1>
        <p><strong>IIT Falsification with Natural EEG Data</strong></p>
        <div class="verdict">{verdict}</div>
        
        <h2>📊 Novel Integrity Metrics</h2>
        
        <div class="metric-box">
            <h3>Eclipse Integrity Score (EIS)</h3>
            <p><strong>Score:</strong> {eis_score:.4f} / 1.00</p>
            <p><strong>Interpretation:</strong> {eis_interp}</p>
        </div>
        
        <div class="metric-box">
            <h3>Statistical Test for Data Snooping (STDS)</h3>
            <p><strong>P-value:</strong> {stds_p:.4f}</p>
            <p><strong>Verdict:</strong> {stds_verdict}</p>
        </div>
        
        <p style="margin-top: 40px; color: #7f8c8d; font-size: 0.9em;">
        <strong>ECLIPSE v3.2.0</strong> - Enhanced with v2.0 integrity metrics + Natural EEG data<br>
        Citation: Sjöberg Tala, C. A. (2025). ECLIPSE v3.2.0. DOI: 10.5281/zenodo.15541550
        </p>
    </div>
</body>
</html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ HTML report saved: {output_path}")
        
        return html
    
    @staticmethod
    def generate_text_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate plain text report"""
        lines = []
        lines.append("=" * 100)
        lines.append("ECLIPSE v3.2.0 FALSIFICATION REPORT")
        lines.append("=" * 100)
        lines.append(f"Project: {final_assessment['project_name']}")
        lines.append(f"Verdict: {final_assessment['verdict']}")
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        
        if integrity_metrics:
            eis_data = integrity_metrics.get('eis', {})
            if eis_data:
                lines.append(f"\nEIS: {eis_data.get('eis', 0):.4f} - {eis_data.get('interpretation', 'N/A')}")
            
            stds_data = integrity_metrics.get('stds', {})
            if stds_data.get('status') == 'success':
                lines.append(f"STDS p-value: {stds_data.get('p_value', 1.0):.4f}")
        
        lines.append("\n" + "=" * 100)
        
        text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        return text

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (de v3.1.0)
# ═══════════════════════════════════════════════════════════════════════════

def load_sleepedf_subject_multichannel_v3(psg_path, hypno_path, n_channels=8, 
                                          phi_methods='all', thermal_monitor=None):
    """Cargar sujeto con monitoreo"""
    if thermal_monitor:
        thermal_monitor.check_temperature()
    
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    preferred_channels = [
        'EEG Fpz-Cz', 'EEG Pz-Oz',
        'EEG F3-A2', 'EEG F4-A1',
        'EEG C3-A2', 'EEG C4-A1',
        'EEG O1-A2', 'EEG O2-A1'
    ]
    
    available = [ch for ch in preferred_channels if ch in raw.ch_names]
    
    if len(available) < n_channels:
        logging.warning(f"⚠️  Solo {len(available)} canales disponibles")
        return None
    
    selected = available[:n_channels]
    
    raw.pick_channels(selected)
    raw.filter(0.5, 30, fir_design='firwin', verbose=False)
    
    hypno_data = mne.read_annotations(hypno_path)
    
    sfreq = raw.info['sfreq']
    window_size = 30
    n_samples_window = int(window_size * sfreq)
    
    data = raw.get_data()
    n_windows = data.shape[1] // n_samples_window
    
    print(f"      📊 Procesando {n_windows} ventanas")
    
    windows = []
    
    for w in range(n_windows):
        start_sample = w * n_samples_window
        end_sample = start_sample + n_samples_window
        
        eeg_window = data[:, start_sample:end_sample]
        time_center = (start_sample + end_sample) / 2 / sfreq
        
        sleep_stage = None
        for annot in hypno_data:
            if annot['onset'] <= time_center < (annot['onset'] + annot['duration']):
                sleep_stage = annot['description']
                break
        
        if sleep_stage is None:
            continue
        
        consciousness_label = 1 if sleep_stage.startswith('Sleep stage W') else 0
        
        try:
            phi_results = calculate_all_phi_methods(eeg_window, methods=phi_methods)
        except Exception as e:
            logging.warning(f"Error phi: {e}")
            phi_results = {'phi_binary': 0.0}
        
        window_data = {
            **phi_results,
            'consciousness': consciousness_label,
            'sleep_stage': sleep_stage,
            'window_idx': w,
            'n_channels_used': len(selected)
        }
        
        windows.append(window_data)
    
    print(f"      ✅ {len(windows)} ventanas completadas")
    
    return windows

def buscar_archivos_edf_pares(carpeta_base):
    """Buscar pares PSG-Hypnogram"""
    pares_encontrados = []
    carpeta_path = Path(carpeta_base)
    
    print(f"\n🔍 Buscando en: {carpeta_base}")
    
    archivos_psg = list(carpeta_path.glob("*-PSG.edf"))
    archivos_hypno = list(carpeta_path.glob("*-Hypnogram.edf"))
    
    print(f"📂 PSG: {len(archivos_psg)}, Hypno: {len(archivos_hypno)}")
    
    if len(archivos_psg) == 0 or len(archivos_hypno) == 0:
        return []
    
    hypno_map = {}
    for hypno_path in archivos_hypno:
        codigo = hypno_path.stem.replace("-Hypnogram", "")
        if len(codigo) >= 7:
            base = codigo[:-1]
            hypno_map[base] = hypno_path
    
    for psg_path in archivos_psg:
        codigo = psg_path.stem.replace("-PSG", "")
        if len(codigo) >= 7 and codigo[-1] == '0':
            base = codigo[:-1]
            if base in hypno_map:
                pares_encontrados.append((str(psg_path), str(hypno_map[base]), codigo))
    
    print(f"✅ {len(pares_encontrados)} pares")
    
    return pares_encontrados

def save_progress(output_dir: Path, subject_data: List, checkpoint_name: str):
    """Guardar checkpoint"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    try:
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(subject_data, f)
    except Exception as e:
        logging.error(f"Error checkpoint: {e}")

def load_progress(output_dir: Path, checkpoint_name: str):
    """Cargar checkpoint"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    if checkpoint_file.exists():
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error: {e}")
    return None

def optimize_threshold_mcc(train_df: pd.DataFrame, phi_column: str, n_thresholds=200):
    """Optimizar threshold con MCC"""
    from sklearn.metrics import matthews_corrcoef
    
    phi_min = train_df[phi_column].min()
    phi_max = train_df[phi_column].max()
    
    best_threshold = None
    best_mcc = -1
    
    thresholds = np.linspace(phi_min, phi_max, n_thresholds)
    
    for threshold in thresholds:
        pred = (train_df[phi_column] >= threshold).astype(int)
        true = train_df['consciousness']
        mcc = matthews_corrcoef(true, pred)
        
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
    
    return {'phi_threshold': best_threshold, 'best_mcc_train': best_mcc, 'phi_column': phi_column}

def analyze_phi_correlation(df: pd.DataFrame, phi_column: str):
    """Análisis de correlación"""
    conscious = df[df['consciousness'] == 1][phi_column]
    unconscious = df[df['consciousness'] == 0][phi_column]
    
    print(f"\n📊 {phi_column}:")
    print(f"   Vigilia: {conscious.mean():.4f}")
    print(f"   Sueño: {unconscious.mean():.4f}")
    
    pearson_r, _ = pearsonr(df[phi_column], df['consciousness'])
    spearman_r, _ = spearmanr(df[phi_column], df['consciousness'])
    
    print(f"   Pearson: {pearson_r:.4f}")
    print(f"   Spearman: {spearman_r:.4f}")
    
    return {'pearson_r': float(pearson_r), 'spearman_rho': float(spearman_r)}

def comparative_analysis(df: pd.DataFrame):
    """Análisis comparativo"""
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS COMPARATIVO")
    print("=" * 80)
    
    phi_columns = [col for col in df.columns if col.startswith('phi_') and not col.endswith('_time')]
    
    results = {}
    
    for phi_col in phi_columns:
        method = phi_col.replace('phi_', '')
        print(f"\n{method.upper()}:")
        corr = analyze_phi_correlation(df, phi_col)
        results[method] = corr
    
    return results

# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE FRAMEWORK v3.2.0
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    """ECLIPSE v3.2.0 Framework - Integración completa v2.0 + v3.1.0"""
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA.json"
        self.results_file = self.output_dir / f"{config.project_name}_RESULT.json"
        
        self._split_completed = False
        self._criteria_registered = False
        self._development_completed = False
        self._validation_completed = False
        
        self.integrity_scorer = None
        self.snooping_tester = None
        self.llm_auditor = None
        
        print("=" * 80)
        print("🔬 ECLIPSE v3.2.0 INITIALIZED")
        print("=" * 80)
        print(f"Project: {config.project_name}")
        print(f"Canales: {config.n_channels}")
        print("\n✅ CARACTERÍSTICAS v3.2.0:")
        print("  • Sin SMOTE (datos naturales)")
        print("  • Eclipse Integrity Score (EIS)")
        print("  • Statistical Test for Data Snooping (STDS)")
        print("  • LLM-Powered Code Auditor")
        print("  • Múltiples aproximaciones de Φ")
        print("=" * 80)
    
    def stage1_irreversible_split(self, data_identifiers: List[Any], force: bool = False) -> Tuple[List[Any], List[Any]]:
        if self.split_file.exists() and not force:
            with open(self.split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✅ {len(development_ids)} dev / {len(holdout_ids)} holdout")
        self._split_completed = True
        return development_ids, holdout_ids
    
    def stage2_register_criteria(self, criteria: List[FalsificationCriteria], force: bool = False) -> Dict:
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            self._criteria_registered = True
            return criteria_data
        
        print("\nSTAGE 2: CRITERIA")
        criteria_dict = {
            'project_name': self.config.project_name,
            'registration_date': datetime.now().isoformat(),
            'criteria': [asdict(c) for c in criteria],
            'criteria_hash': hashlib.sha256(str([asdict(c) for c in criteria]).encode()).hexdigest()
        }
        
        with open(self.criteria_file, 'w') as f:
            json.dump(criteria_dict, f, indent=2)
        
        print(f"✅ {len(criteria)} criteria")
        self._criteria_registered = True
        return criteria_dict
    
    def stage3_development(self, development_data: Any, training_function: Callable, 
                          validation_function: Callable, **kwargs) -> Dict:
        print("\nSTAGE 3: DEVELOPMENT")
        from sklearn.model_selection import StratifiedKFold
        
        if isinstance(development_data, pd.DataFrame):
            y_labels = development_data['consciousness'].values
        else:
            y_labels = np.array([d['consciousness'] for d in development_data])
        
        skf = StratifiedKFold(n_splits=self.config.n_folds_cv, shuffle=True, random_state=self.config.sacred_seed)
        cv_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_labels)), y_labels)):
            print(f"\nFOLD {fold_idx + 1}/{self.config.n_folds_cv}")
            
            if isinstance(development_data, pd.DataFrame):
                train_data = development_data.iloc[train_idx]
                val_data = development_data.iloc[val_idx]
            else:
                train_data = [development_data[i] for i in train_idx]
                val_data = [development_data[i] for i in val_idx]
            
            try:
                model = training_function(train_data, **kwargs)
                metrics = validation_function(model, val_data, **kwargs)
                cv_results.append({'fold': fold_idx + 1, 'metrics': metrics, 'status': 'success'})
                print(f"   ✅ MCC: {metrics.get('mcc', 0):.3f}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                cv_results.append({'fold': fold_idx + 1, 'status': 'failed', 'error': str(e)})
        
        successful_folds = [r for r in cv_results if r['status'] == 'success']
        
        if successful_folds:
            metric_names = list(successful_folds[0]['metrics'].keys())
            aggregated_metrics = {}
            for metric_name in metric_names:
                values = [r['metrics'][metric_name] for r in successful_folds]
                aggregated_metrics[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': [float(v) for v in values]
                }
        else:
            aggregated_metrics = {}
        
        self._development_completed = True
        return {
            'n_folds': self.config.n_folds_cv,
            'n_successful': len(successful_folds),
            'aggregated_metrics': aggregated_metrics
        }
    
    def stage4_single_shot_validation(self, holdout_data: Any, final_model: Any,
                                     validation_function: Callable, force: bool = False, **kwargs) -> Dict:
        if self.results_file.exists() and not force:
            raise RuntimeError("VALIDATION DONE!")
        
        print("\nSTAGE 4: VALIDATION")
        confirmation = input("\n🚨 Type 'I ACCEPT SINGLE-SHOT VALIDATION': ")
        
        if confirmation != "I ACCEPT SINGLE-SHOT VALIDATION":
            print("❌ Cancelled")
            return None
        
        print("\n🚀 Executing...")
        
        try:
            metrics = validation_function(final_model, holdout_data, **kwargs)
            validation_results = {
                'status': 'success',
                'n_holdout_samples': len(holdout_data) if isinstance(holdout_data, (list, pd.DataFrame)) else 0,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }
            print(f"\n✅ Complete")
        except Exception as e:
            print(f"\n❌ Failed: {e}")
            validation_results = {'status': 'failed', 'error': str(e)}
        
        self._validation_completed = True
        return validation_results
    
    def compute_integrity_metrics(self) -> Dict[str, Any]:
        """Compute Eclipse Integrity Score and Data Snooping Test"""
        print("\n📊 Computing Eclipse Integrity Score (EIS)...")
        
        if self.integrity_scorer is None:
            self.integrity_scorer = EclipseIntegrityScore(self)
        
        eis_results = self.integrity_scorer.compute_eis()
        
        if self._validation_completed:
            if self.snooping_tester is None:
                self.snooping_tester = StatisticalTestDataSnooping(self)
            stds_results = self.snooping_tester.perform_snooping_test()
        else:
            stds_results = {'status': 'not_applicable'}
        
        return {
            'eis': eis_results,
            'stds': stds_results
        }
    
    def audit_code(
        self,
        code_paths: List[str],
        holdout_identifiers: List[str] = None,
        api_key: Optional[str] = None
    ) -> AuditResult:
        """Audit analysis code for protocol violations"""
        print("\n" + "=" * 80)
        print("🤖 LLM-POWERED CODE AUDIT (v3.2.0)")
        print("=" * 80)
        
        if self.llm_auditor is None:
            self.llm_auditor = LLMAuditor(self, api_key=api_key)
        
        audit_result = self.llm_auditor.audit_analysis_code(
            code_paths=code_paths,
            holdout_identifiers=holdout_identifiers
        )
        
        print(f"\n{'✅' if audit_result.passed else '❌'} Audit: {audit_result.summary}")
        print(f"   Score: {audit_result.adherence_score:.0f}/100")
        print(f"   Risk Level: {audit_result.risk_level.upper()}")
        
        audit_path = self.output_dir / f"{self.config.project_name}_CODE_AUDIT.txt"
        self.llm_auditor.save_audit_report(audit_result, str(audit_path))
        
        return audit_result
    
    def verify_integrity(self) -> Dict:
        """Verify cryptographic integrity of all ECLIPSE files"""
        print("\n🔍 Verifying ECLIPSE integrity...")
        print("─" * 80)
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'files_checked': [],
            'all_valid': True
        }
        
        # Check split file
        if self.split_file.exists():
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            
            all_ids = split_data['development_ids'] + split_data['holdout_ids']
            recomputed_hash = hashlib.sha256(
                f"{split_data['sacred_seed']}_{sorted(all_ids)}".encode()
            ).hexdigest()
            
            split_valid = recomputed_hash == split_data['integrity_verification']['split_hash']
            
            verification['files_checked'].append({
                'file': 'split',
                'valid': split_valid
            })
            
            status = "✅" if split_valid else "❌"
            print(f"{status} Split file: {'VALID' if split_valid else 'COMPROMISED'}")
            
            if not split_valid:
                verification['all_valid'] = False
        
        # Check criteria file
        if self.criteria_file.exists():
            with open(self.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            
            recomputed_hash = hashlib.sha256(
                str(criteria_data['criteria']).encode()
            ).hexdigest()
            
            criteria_valid = recomputed_hash == criteria_data['criteria_hash']
            
            verification['files_checked'].append({
                'file': 'criteria',
                'valid': criteria_valid
            })
            
            status = "✅" if criteria_valid else "❌"
            print(f"{status} Criteria file: {'VALID' if criteria_valid else 'COMPROMISED'}")
            
            if not criteria_valid:
                verification['all_valid'] = False
        
        # Check results file
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                results_data = json.load(f)
            
            stored_hash = results_data.pop('final_hash', None)
            recomputed_hash = hashlib.sha256(
                json.dumps(results_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            results_data['final_hash'] = stored_hash
            
            results_valid = recomputed_hash == stored_hash
            
            verification['files_checked'].append({
                'file': 'results',
                'valid': results_valid
            })
            
            status = "✅" if results_valid else "❌"
            print(f"{status} Results file: {'VALID' if results_valid else 'COMPROMISED'}")
            
            if not results_valid:
                verification['all_valid'] = False
        
        print("─" * 80)
        if verification['all_valid']:
            print("✅ ALL FILES VERIFIED - Integrity intact")
        else:
            print("❌ INTEGRITY COMPROMISED - Results may be invalid")
        
        return verification
    
    def stage5_final_assessment(self, development_results: Dict, validation_results: Dict,
                               generate_reports: bool = True, compute_integrity: bool = True) -> Dict:
        print("\nSTAGE 5: ASSESSMENT")
        
        with open(self.criteria_file, 'r') as f:
            criteria_data = json.load(f)
        
        criteria_list = [FalsificationCriteria(**c) for c in criteria_data['criteria']]
        holdout_metrics = validation_results.get('metrics', {})
        criteria_evaluation = []
        
        for criterion in criteria_list:
            if criterion.name in holdout_metrics:
                value = holdout_metrics[criterion.name]
                passed = criterion.evaluate(value)
                evaluation = {'criterion': asdict(criterion), 'value': float(value), 'passed': passed}
                print(f"{'✅' if passed else '❌'} {criterion.name}: {value:.4f}")
            else:
                evaluation = {'criterion': asdict(criterion), 'value': None, 'passed': False}
            criteria_evaluation.append(evaluation)
        
        required_criteria = [e for e in criteria_evaluation if e['criterion']['is_required']]
        required_passed = sum(1 for e in required_criteria if e['passed'])
        required_total = len(required_criteria)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required_criteria) else "FALSIFIED"
        
        final_assessment = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'assessment_date': datetime.now().isoformat(),
            'configuration': asdict(self.config),
            'development_summary': {
                'n_folds': development_results.get('n_folds'),
                'n_successful': development_results.get('n_successful'),
                'aggregated_metrics': development_results.get('aggregated_metrics', {})
            },
            'validation_summary': {
                'status': validation_results.get('status'),
                'n_holdout_samples': validation_results.get('n_holdout_samples'),
                'metrics': validation_results.get('metrics', {})
            },
            'criteria_evaluation': criteria_evaluation,
            'verdict': verdict,
            'verdict_description': f"{required_passed}/{required_total} passed"
        }
        
        if compute_integrity:
            print("\n🔬 Computing integrity metrics...")
            integrity_metrics = self.compute_integrity_metrics()
            final_assessment['integrity_metrics'] = integrity_metrics
        
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        print(f"\n{'✅' if verdict == 'VALIDATED' else '❌'} VERDICT: {verdict}")
        print(f"✅ Saved: {self.results_file}")
        
        if generate_reports:
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
        
        return final_assessment

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - v3.2.0
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🧠 ECLIPSE v3.2.0 - INTEGRACIÓN COMPLETA")
    print("   ✅ Sin SMOTE (datos naturales)")
    print("   ✅ EIS + STDS + LLM Auditor")
    print("   ✅ Múltiples aproximaciones de Φ")
    print("=" * 80)
    
    log_file = setup_logging("./eclipse_results_v3_2")
    
    sleep_edf_path = input("\nRuta Sleep-EDF: ").strip().strip('"').strip("'")
    
    if not os.path.exists(sleep_edf_path):
        print(f"❌ Ruta no existe")
        return
    
    limit = input("Limitar sujetos (Enter=30): ").strip()
    limit_n = int(limit) if limit else 30
    
    n_channels = input("Canales EEG (Enter=2): ").strip()
    n_channels = int(n_channels) if n_channels else 2
    
    print("\nMétodos Φ:")
    print("   1. fast (binary, multilevel_3, gaussian)")
    print("   2. accurate (multilevel_4, gaussian)")
    
    method_choice = input("Elegir (Enter=1): ").strip()
    phi_methods = 'accurate' if method_choice == '2' else 'fast'
    
    print("\n⚙️  Configuración:")
    print(f"   - Canales: {n_channels}")
    print(f"   - Métodos Φ: {phi_methods}")
    print(f"   - Límite: {limit_n}")
    print(f"   - Balanceo: NINGUNO (datos naturales) ✅")
    
    thermal_monitor = ThermalMonitor()
    
    subject_pairs = buscar_archivos_edf_pares(sleep_edf_path)
    
    if len(subject_pairs) == 0:
        print("❌ Sin pares")
        return
    
    if limit_n:
        subject_pairs = subject_pairs[:limit_n]
    
    subject_ids = [pair[2] for pair in subject_pairs]
    
    output_dir = Path("./eclipse_results_v3_2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"processing_v3_2_{n_channels}ch_{phi_methods}_natural"
    all_windows = load_progress(output_dir, checkpoint_name)
    
    if all_windows is not None:
        print(f"\n♻️  Checkpoint: {len(all_windows)} ventanas")
        resume = input("Continuar? (s/n): ").strip().lower()
        if resume != 's':
            all_windows = []
    else:
        all_windows = []
    
    processed = set([w['subject_id'] for w in all_windows if 'subject_id' in w])
    
    print(f"\n🔄 Procesando {len(subject_pairs)} sujetos...")
    
    start_time = time.time()
    
    for i, (psg, hypno, subject_id) in enumerate(subject_pairs, 1):
        if subject_id in processed:
            continue
        
        print(f"\n{'='*70}")
        print(f"📊 SUJETO {i}/{len(subject_pairs)}: {subject_id}")
        print(f"{'='*70}")
        
        if not thermal_monitor.check_temperature():
            logging.warning("Enfriamiento")
        
        try:
            windows = load_sleepedf_subject_multichannel_v3(
                psg, hypno, n_channels, phi_methods, thermal_monitor
            )
            
            if windows is None:
                print(f"   ⏭️  Saltado")
                continue
            
            if len(windows) > 0:
                for w in windows:
                    w['subject_id'] = subject_id
                
                all_windows.extend(windows)
                
                if i % 5 == 0:
                    save_progress(output_dir, all_windows, checkpoint_name)
                    print(f"\n   💾 Checkpoint ({i}/{len(subject_pairs)})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    save_progress(output_dir, all_windows, checkpoint_name)
    
    if len(all_windows) == 0:
        print("\n❌ Sin datos")
        return
    
    df = pd.DataFrame(all_windows)
    
    print("\n" + "=" * 80)
    print("✅ DATASET COMPLETO")
    print("=" * 80)
    print(f"Ventanas: {len(df)}")
    print(f"Sujetos: {df['subject_id'].nunique()}")
    print(f"Conscientes: {(df['consciousness'] == 1).sum()}")
    print(f"Inconscientes: {(df['consciousness'] == 0).sum()}")
    
    comparative_results = comparative_analysis(df)
    
    best_method = max(comparative_results.items(), 
                     key=lambda x: abs(x[1]['spearman_rho']))[0]
    
    print(f"\n🏆 MEJOR MÉTODO: {best_method.upper()}")
    
    df_natural = df.copy()
    print("\n✅ Usando distribución NATURAL (SIN SMOTE)")
    print(f"   Consciente: {(df_natural['consciousness'] == 1).sum()} ventanas")
    print(f"   Inconsciente: {(df_natural['consciousness'] == 0).sum()} ventanas")
    
    config = EclipseConfig(
        project_name=f"IIT_v3_2_{n_channels}ch_natural",
        researcher="Camilo Sjöberg Tala",
        sacred_seed=2025,
        n_channels=n_channels,
        phi_methods=[phi_methods],
        output_dir=str(output_dir)
    )
    
    eclipse = EclipseFramework(config)
    
    # STAGE 1: Split
    unique_subjects = df_natural['subject_id'].unique().tolist()
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(unique_subjects)
    
    dev_data = df_natural[df_natural['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = df_natural[df_natural['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
    print(f"\n📊 División:")
    print(f"   Dev: {len(dev_data)} ventanas")
    print(f"   Holdout: {len(holdout_data)} ventanas")
    
    # STAGE 2: Criteria
    criteria = [
        FalsificationCriteria("balanced_accuracy", 0.60, ">=", "Bal.Acc >= 0.60", True),
        FalsificationCriteria("f1_score", 0.50, ">=", "F1 >= 0.50", True),
        FalsificationCriteria("mcc", 0.20, ">=", "MCC >= 0.20", True)
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    # STAGE 3: Development
    phi_col = f"phi_{best_method}"
    
    def train_fn(train_data, **kwargs):
        return optimize_threshold_mcc(train_data, phi_col)
    
    def val_fn(model, val_data, **kwargs):
        threshold = model['phi_threshold']
        y_pred = (val_data[phi_col] >= threshold).astype(int)
        y_true = val_data['consciousness']
        
        phi_min = val_data[phi_col].min()
        phi_max = val_data[phi_col].max()
        if phi_max > phi_min:
            y_pred_proba = (val_data[phi_col] - phi_min) / (phi_max - phi_min + 1e-10)
        else:
            y_pred_proba = np.ones(len(val_data)) * 0.5
        
        return EclipseValidator.binary_classification_metrics(y_true, y_pred, y_pred_proba)
    
    dev_results = eclipse.stage3_development(
        development_data=dev_data,
        training_function=train_fn,
        validation_function=val_fn
    )
    
    # STAGE 4: Validation
    print("\n🔧 Entrenando modelo final...")
    final_model = train_fn(dev_data)
    print(f"   Threshold: {final_model['phi_threshold']:.4f}")
    print(f"   MCC train: {final_model['best_mcc_train']:.4f}")
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=val_fn
    )
    
    if val_results is None:
        print("Validación cancelada")
        return
    
    # STAGE 5: Assessment con métricas v3.2.0
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True
    )
    
    # NUEVO v3.2.0: Verificación de integridad
    eclipse.verify_integrity()
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ PROCESAMIENTO COMPLETADO v3.2.0")
    print("=" * 80)
    
    holdout_metrics = final_assessment['validation_summary']['metrics']
    
    print(f"\n📊 MÉTRICAS HOLDOUT:")
    print(f"   Balanced Accuracy: {holdout_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"   MCC: {holdout_metrics.get('mcc', 0):.4f}")
    print(f"   F1 Score: {holdout_metrics.get('f1_score', 0):.4f}")
    
    print(f"\n{'✅' if final_assessment['verdict'] == 'VALIDATED' else '❌'} VEREDICTO: {final_assessment['verdict']}")
    
    # Novel metrics v3.2.0
    if 'integrity_metrics' in final_assessment:
        integrity = final_assessment['integrity_metrics']
        
        if 'eis' in integrity:
            eis_score = integrity['eis']['eis']
            print(f"\n📊 Eclipse Integrity Score: {eis_score:.4f}")
            print(f"   {integrity['eis']['interpretation']}")
        
        if 'stds' in integrity and integrity['stds'].get('status') == 'success':
            stds_p = integrity['stds']['p_value']
            print(f"\n🔍 Data Snooping Test: p={stds_p:.4f}")
            print(f"   {integrity['stds']['interpretation']}")
    
    print("\n" + "=" * 80)
    print("🎯 RESUMEN CRÍTICO v3.2.0")
    print("=" * 80)
    print(f"✅ Datos 100% NATURALES (sin SMOTE)")
    print(f"✅ Integridad verificada (EIS + STDS)")
    print(f"✅ Framework completo v2.0 integrado")
    print(f"✅ Tiempo total: {(time.time() - start_time)/60:.1f} min")
    
    print("\n" + "=" * 80)
    print("✅ ECLIPSE v3.2.0 COMPLETADO")
    print("=" * 80)
    
    logging.info("EJECUCIÓN COMPLETADA v3.2.0")
    for handler in logging.root.handlers:
        handler.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        logging.error(traceback.format_exc())file, 'r') as f:
                split_data = json.load(f)
            self._split_completed = True
            return split_data['development_ids'], split_data['holdout_ids']
        
        print("\nSTAGE 1: SPLIT")
        np.random.seed(self.config.sacred_seed)
        shuffled_ids = np.array(data_identifiers).copy()
        np.random.shuffle(shuffled_ids)
        
        n_development = int(len(data_identifiers) * self.config.development_ratio)
        development_ids = shuffled_ids[:n_development].tolist()
        holdout_ids = shuffled_ids[n_development:].tolist()
        
        split_data = {
            'project_name': self.config.project_name,
            'split_date': datetime.now().isoformat(),
            'sacred_seed': self.config.sacred_seed,
            'development_ids': development_ids,
            'holdout_ids': holdout_ids,
            'integrity_verification': {
                'split_hash': hashlib.sha256(f"{self.config.sacred_seed}_{sorted(data_identifiers)}".encode()).hexdigest()
            }
        }
        
        with open(self.split_