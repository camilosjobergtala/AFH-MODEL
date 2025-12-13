"""
═══════════════════════════════════════════════════════════════════════════════
ECLIPSE v3.3.1: FALSIFICACIÓN DE IIT - FUSIÓN COMPLETA
═══════════════════════════════════════════════════════════════════════════════

FUSIÓN DE:
  • v3.0: Framework metodológico corregido (STDS, EIS, análisis semántico)
  • v3.2.0: Aplicación IIT con Sleep-EDF (múltiples Φ, monitoreo térmico)

🔧 FIX v3.3.1:
  - STDS ahora filtra métricas de conteo (true_negatives, false_positives, etc.)
  - Estas métricas dependen del tamaño del dataset y producían z-scores artificiales
  - Solo se incluyen métricas normalizadas (accuracy, f1, mcc, etc.)

✅ MEJORAS DE v3.0 INTEGRADAS:
  - STDS con z-scores REALES (no heurísticas)
  - EIS con pesos justificados bibliográficamente
  - Leakage risk contextual (no asume degradación)
  - Análisis semántico de aliasing en Code Auditor
  - Soporte de notebooks (.ipynb)
  - Modo no-interactivo con commitment criptográfico

✅ FUNCIONALIDAD v3.2.0 PRESERVADA:
  - Múltiples aproximaciones de Φ (binary, multilevel, gaussian)
  - Procesamiento Sleep-EDF multicanal
  - Monitoreo térmico
  - Checkpoints de progreso

Autor: Camilo Alejandro Sjöberg Tala + Claude
Version: 3.3.0
Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v3.3.1. DOI: 10.5281/zenodo.15541550

LICENSE: DUAL LICENSE (AGPL v3.0 / Commercial)
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict, field
from itertools import combinations
from collections import defaultdict
import warnings
import sys
import matplotlib
matplotlib.use('Agg')
import os
from scipy.stats import entropy, spearmanr, pearsonr, rankdata, norm, mannwhitneyu, ttest_ind
from scipy import stats as scipy_stats
import scipy.linalg as la
import time
import psutil
import logging
import multiprocessing as mp
import ast
import re
import base64

# MNE para EEG
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE not available. EEG processing disabled.")

# Silenciar warnings
warnings.filterwarnings('ignore')
if MNE_AVAILABLE:
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


def setup_logging(output_dir: str):
    """Configurar logging"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eclipse_v3_3_{timestamp}.log"
    
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
    
    logging.info("=" * 80)
    logging.info("ECLIPSE v3.3.1 - FUSIÓN COMPLETA (v3.0 + v3.2.0)")
    logging.info(f"Log: {log_file}")
    logging.info("=" * 80)
    
    return log_file


# ═══════════════════════════════════════════════════════════════════════════
# MONITOREO TÉRMICO (de v3.2.0)
# ═══════════════════════════════════════════════════════════════════════════

class ThermalMonitor:
    """Monitor de temperatura para procesamiento largo"""
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
                        logging.warning(f"🔥 GPU: {gpu_temp}°C")
                        needs_cooldown = True
            except:
                pass
        
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                max_cpu_temp = max([t.current for t in temps['coretemp']])
                if max_cpu_temp > self.MAX_CPU_TEMP:
                    logging.warning(f"🔥 CPU: {max_cpu_temp}°C")
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
# CÁLCULOS DE Φ (de v3.2.0 - múltiples aproximaciones)
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
# ESTRUCTURAS DE DATOS CORE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FalsificationCriteria:
    """Criterio de falsificación pre-registrado"""
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
            "==": lambda x, y: abs(x - y) < 1e-9,
            "!=": lambda x, y: abs(x - y) >= 1e-9
        }
        return comparisons[self.comparison](value, self.threshold)
    
    def __str__(self) -> str:
        req = "REQUIRED" if self.is_required else "optional"
        return f"{self.name} {self.comparison} {self.threshold} [{req}]"


@dataclass
class EclipseConfig:
    """Configuración ECLIPSE v3.3.1"""
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results_v3_3"
    timestamp: str = field(default=None)
    n_channels: int = 8
    phi_methods: List[str] = field(default_factory=lambda: ['fast'])
    
    # v3.0: Modo no-interactivo
    non_interactive: bool = False
    commitment_phrase: str = None
    
    # v3.0: Umbrales configurables
    eis_weights: Dict[str, float] = None
    stds_alpha: float = 0.05
    audit_pass_threshold: float = 70.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        if abs(self.development_ratio + self.holdout_ratio - 1.0) > 1e-6:
            raise ValueError("Development + Holdout ratios must sum to 1.0")
        
        if self.non_interactive and not self.commitment_phrase:
            raise ValueError(
                "non_interactive=True requires commitment_phrase"
            )
        
        if self.eis_weights is None:
            self.eis_weights = EclipseIntegrityScore.DEFAULT_WEIGHTS.copy()


@dataclass
class CodeViolation:
    """Violación detectada del protocolo ECLIPSE"""
    severity: str
    category: str
    description: str
    file_path: str
    line_number: Optional[int]
    code_snippet: str
    recommendation: str
    confidence: float
    detection_method: str = "ast"


@dataclass
class AuditResult:
    """Resultados completos de auditoría"""
    timestamp: str
    adherence_score: float
    violations: List[CodeViolation]
    risk_level: str
    passed: bool
    summary: str
    detailed_report: str
    files_analyzed: List[str] = field(default_factory=list)
    notebooks_analyzed: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDADOR DE MÉTRICAS
# ═══════════════════════════════════════════════════════════════════════════

class EclipseValidator:
    """Calculador automático de métricas de validación"""
    
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
# ECLIPSE INTEGRITY SCORE (EIS) - v3.0 CON BIBLIOGRAFÍA
# ═══════════════════════════════════════════════════════════════════════════

class EclipseIntegrityScore:
    """
    Eclipse Integrity Score (EIS) v3.0
    
    Pesos con justificación bibliográfica:
    
    ┌─────────────────────┬────────┬────────────────────────────────────────────┐
    │ Component           │ Weight │ Justification                              │
    ├─────────────────────┼────────┼────────────────────────────────────────────┤
    │ Pre-registration    │ 0.25   │ Nosek et al. (2018): Pre-registration     │
    │                     │        │ reduces false positives by 60%             │
    ├─────────────────────┼────────┼────────────────────────────────────────────┤
    │ Protocol adherence  │ 0.25   │ Simmons et al. (2011): Researcher degrees │
    │                     │        │ of freedom inflate Type I error to 60%+    │
    ├─────────────────────┼────────┼────────────────────────────────────────────┤
    │ Split strength      │ 0.20   │ Information-theoretic: Maximum entropy     │
    │                     │        │ split minimizes information leakage        │
    ├─────────────────────┼────────┼────────────────────────────────────────────┤
    │ Leakage risk        │ 0.15   │ Empirical: Kapoor & Narayanan (2022)      │
    │                     │        │ found data leakage in 17/20 ML studies     │
    ├─────────────────────┼────────┼────────────────────────────────────────────┤
    │ Transparency        │ 0.15   │ FAIR principles: Findability and          │
    │                     │        │ reproducibility require documentation      │
    └─────────────────────┴────────┴────────────────────────────────────────────┘
    """
    
    DEFAULT_WEIGHTS = {
        'preregistration': 0.25,
        'protocol_adherence': 0.25,
        'split_strength': 0.20,
        'leakage_risk': 0.15,
        'transparency': 0.15
    }
    
    WEIGHT_JUSTIFICATIONS = {
        'preregistration': "Nosek et al. (2018) PNAS: Pre-registration reduces false positives by ~60%",
        'protocol_adherence': "Simmons et al. (2011): Researcher degrees of freedom inflate Type I error to 60%+",
        'split_strength': "Information theory: Maximum entropy split minimizes information leakage",
        'leakage_risk': "Kapoor & Narayanan (2022): Found data leakage in 17/20 surveyed ML studies",
        'transparency': "FAIR principles: Documentation enables reproducibility and scrutiny"
    }
    
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
        elif self.framework._criteria_registered:
            score += 0.3
        
        if 'registration_date' in criteria_data and 'criteria_hash' in criteria_data:
            score += 0.3
        
        criteria_list = criteria_data.get('criteria', [])
        if criteria_list:
            has_all_thresholds = all('threshold' in c and 'comparison' in c for c in criteria_list)
            has_descriptions = all('description' in c and c['description'] for c in criteria_list)
            
            if has_all_thresholds:
                score += 0.2
            if has_descriptions:
                score += 0.1
        
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
        
        # v3.0: Ajuste por tamaño de muestra
        if total < 30:
            size_penalty = total / 30
        elif total < 100:
            size_penalty = 0.9 + 0.1 * (total - 30) / 70
        else:
            size_penalty = 1.0
        
        has_hash = 'integrity_verification' in split_data
        hash_bonus = 0.15 if has_hash else 0.0
        
        return min(1.0, 0.85 * entropy_score * size_penalty + hash_bonus)
    
    def compute_protocol_adherence(self) -> float:
        score = 0.0
        
        stages = [
            ('split', self.framework._split_completed),
            ('criteria', self.framework._criteria_registered),
            ('development', self.framework._development_completed),
            ('validation', self.framework._validation_completed)
        ]
        
        completed_in_order = True
        for i, (name, completed) in enumerate(stages):
            if completed:
                previous_completed = all(stages[j][1] for j in range(i))
                if previous_completed:
                    score += 0.2
                else:
                    completed_in_order = False
                    score += 0.1
        
        if completed_in_order and all(s[1] for s in stages):
            score += 0.2
        
        return min(1.0, score)
    
    def estimate_leakage_risk(self) -> float:
        """
        v3.0 CORREGIDO: Estimación contextual de riesgo de leakage
        
        No asume que degradación es siempre esperada.
        Usa z-score para detectar holdout "demasiado bueno".
        """
        if not self.framework._validation_completed:
            return 0.5
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
            
            dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
            holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
            
            if not dev_metrics or not holdout_metrics:
                return 0.5
            
            risk_scores = []
            
            for metric_name in dev_metrics:
                if metric_name in holdout_metrics:
                    stats = dev_metrics[metric_name]
                    dev_mean = stats.get('mean', 0)
                    dev_std = stats.get('std', 0)
                    holdout_val = holdout_metrics[metric_name]
                    
                    if hasattr(holdout_val, 'item'):
                        holdout_val = holdout_val.item()
                    
                    if not isinstance(holdout_val, (int, float)) or dev_mean == 0:
                        continue
                    
                    # v3.0: Evaluación contextual con z-score
                    if dev_std > 0:
                        z_score = (holdout_val - dev_mean) / dev_std
                        
                        if z_score > 1.5:
                            # Holdout MEJOR que dev - sospechoso
                            risk = 0.9
                        elif z_score > 0.5:
                            risk = 0.5
                        elif z_score > -1.0:
                            # Dentro del rango esperado
                            risk = 0.2
                        else:
                            # Peor que esperado - posible distribution shift
                            risk = 0.3
                    else:
                        if abs(holdout_val - dev_mean) < 0.01:
                            risk = 0.7
                        else:
                            risk = 0.4
                    
                    risk_scores.append(risk)
            
            if not risk_scores:
                return 0.5
            
            return float(np.mean(risk_scores))
            
        except:
            return 0.5
    
    def compute_transparency_score(self) -> float:
        score = 0.0
        
        files_to_check = [
            (self.framework.split_file, 0.15),
            (self.framework.criteria_file, 0.15),
        ]
        
        for file_path, weight in files_to_check:
            if file_path.exists():
                score += weight
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
            if 'split_date' in split_data:
                score += 0.15
        except:
            pass
        
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            if 'registration_date' in criteria_data:
                score += 0.15
        except:
            pass
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
            if 'integrity_verification' in split_data:
                score += 0.2
        except:
            pass
        
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            if 'criteria_hash' in criteria_data:
                score += 0.1
            if 'binding_declaration' in criteria_data:
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
    
    def compute_eis(self, weights: Dict[str, float] = None) -> Dict[str, Any]:
        if weights is None:
            weights = self.framework.config.eis_weights or self.DEFAULT_WEIGHTS
        
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
        
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
            'weight_justifications': self.WEIGHT_JUSTIFICATIONS,
            'timestamp': datetime.now().isoformat(),
            'interpretation': self._interpret_eis(eis),
            'version': '3.3.1'
        }
        
        return self.scores
    
    def _interpret_eis(self, eis: float) -> str:
        if eis >= 0.90:
            return "EXCELLENT - Exceptional methodological rigor"
        elif eis >= 0.80:
            return "VERY GOOD - High methodological quality"
        elif eis >= 0.70:
            return "GOOD - Adequate methodological standards"
        elif eis >= 0.60:
            return "FAIR - Some methodological concerns"
        elif eis >= 0.50:
            return "POOR - Significant methodological issues"
        else:
            return "VERY POOR - Critical methodological flaws"
    
    def generate_eis_report(self, output_path: Optional[str] = None) -> str:
        if not self.scores:
            self.compute_eis()
        
        lines = []
        lines.append("=" * 80)
        lines.append("ECLIPSE INTEGRITY SCORE (EIS) REPORT v3.3")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Computed: {self.scores['timestamp']}")
        lines.append("")
        
        eis = self.scores['eis']
        lines.append(f"OVERALL EIS: {eis:.4f} / 1.00")
        lines.append(f"Interpretation: {self.scores['interpretation']}")
        lines.append("")
        
        lines.append("COMPONENT SCORES:")
        lines.append("-" * 80)
        components = self.scores['components']
        weights = self.scores['weights']
        
        component_names = [
            ('preregistration_score', 'preregistration', 'Pre-registration completeness'),
            ('split_strength', 'split_strength', 'Split strength (entropy)'),
            ('protocol_adherence', 'protocol_adherence', 'Protocol adherence'),
            ('leakage_score', 'leakage_risk', 'Data leakage score (1-risk)'),
            ('transparency_score', 'transparency', 'Transparency & documentation')
        ]
        
        for score_key, weight_key, display_name in component_names:
            score = components[score_key]
            weight = weights[weight_key]
            contribution = score * weight
            lines.append(f"  {display_name:<35s}: {score:.4f} × {weight:.2f} = {contribution:.4f}")
        
        lines.append("")
        lines.append("WEIGHT JUSTIFICATIONS:")
        lines.append("-" * 80)
        for key, justification in self.WEIGHT_JUSTIFICATIONS.items():
            lines.append(f"  • {key}: {justification}")
        
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ═══════════════════════════════════════════════════════════════════════════
# STDS - v3.0 CORREGIDO CON ESTADÍSTICA REAL
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalTestDataSnooping:
    """
    Statistical Test for Data Snooping (STDS) v3.0 CORREGIDO
    
    METODOLOGÍA: Estadística estándar. Sin heurísticas.
    
    Bajo H0 (sin snooping): El resultado holdout es simplemente otra
    observación de la misma distribución que generó los resultados CV.
    
    Procedimiento:
    1. Calcular z-score: z = (holdout - CV_mean) / CV_std
    2. Calcular percentil del holdout en distribución CV
    3. Reportar resultados - interpretación del investigador
    
    Interpretación:
    - z > 0: Holdout mejor que CV mean
    - z < 0: Holdout peor que CV mean  
    - |z| > 2: Inusual (~p < 0.05 bajo normalidad)
    - |z| > 3: Muy inusual (~p < 0.003 bajo normalidad)
    """
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.test_results = {}
    
    def perform_snooping_test(self, alpha: float = None) -> Dict[str, Any]:
        """
        Test estadístico para data snooping - v3.0 CORREGIDO
        
        Usa z-score estándar, no heurísticas inventadas.
        """
        if alpha is None:
            alpha = self.framework.config.stds_alpha
        
        z_crit = scipy_stats.norm.ppf(1 - alpha/2)
        
        if not self.framework._validation_completed:
            return {
                'status': 'incomplete',
                'message': 'Validation not yet performed'
            }
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
        
        dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
        holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
        
        if not dev_metrics or not holdout_metrics:
            return {'status': 'insufficient_data', 'message': 'Insufficient metrics'}
        
        metric_results = {}
        z_scores = []
        
        # v3.3.1 FIX: Solo métricas normalizadas (no conteos que dependen del tamaño)
        # Las métricas de conteo (true_negatives, false_positives, etc.) varían con
        # el tamaño del dataset, produciendo z-scores artificialmente altos
        NORMALIZED_METRICS = {
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 
            'f1_score', 'mcc', 'specificity', 'npv', 'roc_auc',
            'sensitivity', 'ppv', 'f1', 'auc', 'auroc'
        }
        
        # Métricas a EXCLUIR (conteos que dependen del tamaño del dataset)
        COUNT_METRICS = {
            'true_negatives', 'false_positives', 'false_negatives', 'true_positives',
            'tn', 'fp', 'fn', 'tp', 'n_samples', 'support'
        }
        
        for metric_name in dev_metrics:
            if metric_name not in holdout_metrics:
                continue
            
            # v3.3.1: Filtrar métricas de conteo
            metric_lower = metric_name.lower()
            if metric_lower in COUNT_METRICS:
                continue
            
            # Solo incluir si es métrica normalizada conocida O si está en rango [0,1] o [-1,1]
            cv_mean = dev_metrics[metric_name].get('mean')
            cv_std = dev_metrics[metric_name].get('std')
            cv_values = dev_metrics[metric_name].get('values', [])
            holdout_val = holdout_metrics[metric_name]
            
            if hasattr(holdout_val, 'item'):
                holdout_val = holdout_val.item()
            
            if not isinstance(holdout_val, (int, float)):
                continue
            
            if cv_std is None or cv_std == 0 or cv_mean is None:
                continue
            
            # v3.3.1: Verificación adicional - si no es métrica conocida, verificar rango
            if metric_lower not in NORMALIZED_METRICS:
                # Si el valor está fuera de [-1, 1], probablemente es un conteo
                if abs(cv_mean) > 1.5 or abs(holdout_val) > 1.5:
                    continue
            
            # Z-score estándar
            z = (holdout_val - cv_mean) / cv_std
            z_scores.append(z)
            
            # Percentil
            if cv_values:
                percentile = np.mean(np.array(cv_values) <= holdout_val) * 100
            else:
                percentile = scipy_stats.norm.cdf(z) * 100
            
            is_significant = abs(z) > z_crit
            
            metric_results[metric_name] = {
                'holdout_value': float(holdout_val),
                'cv_mean': float(cv_mean),
                'cv_std': float(cv_std),
                'z_score': float(z),
                'percentile_rank': float(percentile),
                'is_significant': is_significant
            }
        
        if not z_scores:
            return {'status': 'insufficient_data', 'message': 'No comparable metrics'}
        
        mean_z = float(np.mean(z_scores))
        max_z = float(np.max(z_scores))
        min_z = float(np.min(z_scores))
        n_significant = sum(1 for m in metric_results.values() if m['is_significant'])
        n_positive = sum(1 for z in z_scores if z > 0)
        
        # Interpretación basada en umbrales estándar
        if max_z > 3:
            risk_level = "HIGH"
            interpretation = (
                f"🚨 INUSUAL: Al menos una métrica tiene z > 3 (max z = {max_z:.2f}). "
                f"Holdout más de 3 desviaciones estándar mejor que CV mean. "
                f"Esto es estadísticamente raro y requiere escrutinio."
            )
        elif max_z > 2:
            risk_level = "MODERATE"
            interpretation = (
                f"⚠️ NOTABLE: Al menos una métrica tiene z > 2 (max z = {max_z:.2f}). "
                f"Holdout mejor que esperado en algunas métricas."
            )
        elif mean_z > 1:
            risk_level = "LOW-MODERATE"
            interpretation = (
                f"📊 LIGERAMENTE INUSUAL: Mean z-score es {mean_z:.2f}. "
                f"Holdout tiende a ser mejor que CV mean."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"✅ NORMAL: Resultados consistentes con holdout de la "
                f"misma distribución que CV (mean z = {mean_z:.2f})."
            )
        
        self.test_results = {
            'test_name': 'Statistical Test for Data Snooping (STDS) v3.3.1',
            'version': '3.3.1',
            'timestamp': datetime.now().isoformat(),
            
            'mean_z_score': mean_z,
            'max_z_score': max_z,
            'min_z_score': min_z,
            'n_metrics': len(z_scores),
            'n_significant': n_significant,
            'n_positive': n_positive,
            
            'alpha': alpha,
            'z_critical': float(z_crit),
            
            'risk_level': risk_level,
            'interpretation': interpretation,
            'metric_results': metric_results,
            
            'status': 'success',
            
            'methodology': (
                "Z-score estándar: z = (holdout - CV_mean) / CV_std. "
                "v3.3.1: Solo métricas normalizadas (excluye conteos como TP/FP/TN/FN). "
                "Sin heurísticas. |z| > 2 es inusual, |z| > 3 es muy inusual."
            ),
            
            'metrics_excluded': list(COUNT_METRICS)
        }
        
        return self.test_results
    
    def generate_stds_report(self, output_path: Optional[str] = None) -> str:
        if not self.test_results or self.test_results.get('status') != 'success':
            return "No test results available. Run perform_snooping_test() first."
        
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL TEST FOR DATA SNOOPING (STDS) v3.3.1")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Computed: {self.test_results['timestamp']}")
        lines.append("")
        
        lines.append("METODOLOGÍA:")
        lines.append("-" * 80)
        lines.append("Z-score estándar: z = (holdout - CV_mean) / CV_std")
        lines.append("Sin heurísticas. Estadística estándar.")
        lines.append("")
        
        lines.append("RESUMEN:")
        lines.append("-" * 80)
        lines.append(f"Mean z-score: {self.test_results['mean_z_score']:+.4f}")
        lines.append(f"Max z-score:  {self.test_results['max_z_score']:+.4f}")
        lines.append(f"Min z-score:  {self.test_results['min_z_score']:+.4f}")
        lines.append(f"Métricas analizadas: {self.test_results['n_metrics']}")
        lines.append(f"Métricas con holdout > CV mean: {self.test_results['n_positive']}/{self.test_results['n_metrics']}")
        lines.append(f"Métricas con |z| > {self.test_results['z_critical']:.2f}: {self.test_results['n_significant']}")
        lines.append(f"Nivel de riesgo: {self.test_results['risk_level']}")
        lines.append("")
        
        lines.append("INTERPRETACIÓN:")
        lines.append("-" * 80)
        lines.append(self.test_results['interpretation'])
        lines.append("")
        
        lines.append("ANÁLISIS POR MÉTRICA:")
        lines.append("-" * 80)
        
        for metric_name, mr in self.test_results['metric_results'].items():
            z = mr['z_score']
            flag = "⚠️" if mr['is_significant'] else "  "
            direction = "mejor" if z > 0 else "peor"
            
            lines.append(f"\n{flag} {metric_name}:")
            lines.append(f"   Holdout: {mr['holdout_value']:.4f}")
            lines.append(f"   CV mean: {mr['cv_mean']:.4f} ± {mr['cv_std']:.4f}")
            lines.append(f"   z-score: {z:+.2f} ({abs(z):.1f}σ {direction} que CV mean)")
            lines.append(f"   Percentil: {mr['percentile_rank']:.1f}%")
        
        lines.append("")
        lines.append("REFERENCIA:")
        lines.append("-" * 80)
        lines.append("|z| > 2: Inusual (p < 0.05 bajo normalidad)")
        lines.append("|z| > 3: Muy inusual (p < 0.003 bajo normalidad)")
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ═══════════════════════════════════════════════════════════════════════════
# CODE AUDITOR - v3.0 CON ANÁLISIS SEMÁNTICO
# ═══════════════════════════════════════════════════════════════════════════

class NotebookAnalyzer:
    """v3.0: Analizador de Jupyter Notebooks"""
    
    @staticmethod
    def extract_code_cells(notebook_path: str) -> List[Tuple[int, str]]:
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except:
            return []
        
        code_cells = []
        cells = notebook.get('cells', [])
        
        for idx, cell in enumerate(cells):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source
                code_cells.append((idx, code))
        
        return code_cells
    
    @staticmethod
    def notebook_to_script(notebook_path: str) -> str:
        cells = NotebookAnalyzer.extract_code_cells(notebook_path)
        
        script_lines = [f"# Converted from {notebook_path}", ""]
        
        for cell_idx, code in cells:
            script_lines.append(f"# === Cell {cell_idx} ===")
            script_lines.append(code)
            script_lines.append("")
        
        return "\n".join(script_lines)


class SemanticAnalyzer:
    """v3.0: Análisis semántico para detección de aliasing"""
    
    def __init__(self, holdout_identifiers: Set[str]):
        self.holdout_identifiers = holdout_identifiers
        self.alias_graph = defaultdict(set)
    
    def analyze_assignments(self, tree: ast.AST) -> Dict[str, Set[str]]:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                sources = self._extract_all_names(node.value)
                
                for target in node.targets:
                    target_names = self._extract_all_names(target)
                    
                    for target_name in target_names:
                        self.alias_graph[target_name].update(sources)
                        
                        for source in sources:
                            if source in self.alias_graph:
                                self.alias_graph[target_name].update(
                                    self.alias_graph[source]
                                )
        
        return dict(self.alias_graph)
    
    def find_tainted_variables(self) -> Set[str]:
        tainted = set()
        
        for var, sources in self.alias_graph.items():
            if sources & self.holdout_identifiers:
                tainted.add(var)
        
        changed = True
        while changed:
            changed = False
            for var, sources in self.alias_graph.items():
                if var not in tainted:
                    if sources & tainted:
                        tainted.add(var)
                        changed = True
        
        return tainted
    
    def _extract_all_names(self, node: ast.AST) -> Set[str]:
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names


class StaticCodeAnalyzer:
    """Analizador estático de código v3.0 mejorado"""
    
    def __init__(self, holdout_identifiers: List[str]):
        self.holdout_identifiers = set(holdout_identifiers)
        self.semantic_analyzer = SemanticAnalyzer(self.holdout_identifiers)
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            return [{'type': 'file_error', 'line': None, 
                    'description': f"File not found: {file_path}", 'severity': 'high'}]
        except UnicodeDecodeError:
            return [{'type': 'encoding_error', 'line': None,
                    'description': f"Could not decode file", 'severity': 'low'}]
        
        return self._analyze_code(code, file_path)
    
    def analyze_notebook(self, notebook_path: str) -> List[Dict[str, Any]]:
        findings = []
        code_cells = NotebookAnalyzer.extract_code_cells(notebook_path)
        
        if not code_cells:
            return [{'type': 'notebook_empty', 'line': None,
                    'description': 'No code cells found', 'severity': 'low'}]
        
        for cell_idx, code in code_cells:
            cell_findings = self._analyze_code(code, f"{notebook_path}:cell_{cell_idx}")
            for finding in cell_findings:
                finding['notebook_cell'] = cell_idx
            findings.extend(cell_findings)
        
        return findings
    
    def _analyze_code(self, code: str, source_name: str) -> List[Dict[str, Any]]:
        findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [{'type': 'syntax_error', 'line': e.lineno,
                    'description': f"Syntax error: {e.msg}", 'severity': 'high',
                    'detection_method': 'ast'}]
        
        # Nivel 1: AST
        findings.extend(self._detect_holdout_access(tree, code, source_name))
        findings.extend(self._detect_threshold_manipulation(tree, code, source_name))
        findings.extend(self._detect_multiple_testing(code, source_name))
        
        # Nivel 2: Control-flow
        findings.extend(self._control_flow_analysis(tree, source_name))
        
        # Nivel 3: Semántico (v3.0)
        findings.extend(self._semantic_analysis(tree, source_name))
        
        # Nivel 4: Patrones
        findings.extend(self._pattern_matching(code, source_name))
        
        return findings
    
    def _detect_holdout_access(self, tree: ast.AST, code: str, file_path: str) -> List[Dict]:
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in self.holdout_identifiers:
                    findings.append({
                        'type': 'holdout_access',
                        'line': node.lineno,
                        'variable': node.id,
                        'description': f"Direct access to holdout variable: {node.id}",
                        'severity': 'critical',
                        'detection_method': 'ast'
                    })
        
        return findings
    
    def _detect_threshold_manipulation(self, tree: ast.AST, code: str, file_path: str) -> List[Dict]:
        findings = []
        threshold_assignments = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_lower = target.id.lower()
                        if any(kw in name_lower for kw in ['threshold', 'cutoff', 'boundary']):
                            threshold_assignments.append({
                                'line': node.lineno,
                                'name': target.id
                            })
        
        if len(threshold_assignments) > 1:
            lines = [t['line'] for t in threshold_assignments]
            findings.append({
                'type': 'threshold_manipulation',
                'line': lines[0],
                'description': f"Threshold set {len(threshold_assignments)} times (lines: {lines})",
                'severity': 'high',
                'detection_method': 'ast'
            })
        
        return findings
    
    def _detect_multiple_testing(self, code: str, file_path: str) -> List[Dict]:
        findings = []
        
        stat_tests = [r'ttest', r'mannwhitneyu', r'wilcoxon', r'ks_2samp',
                     r'chi2', r'fisher_exact', r'anova', r'kruskal']
        
        loop_patterns = [
            rf'for\s+\w+\s+in.*?({"|".join(stat_tests)})',
            rf'while\s+.*?({"|".join(stat_tests)})',
        ]
        
        for pattern in loop_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
                corrections = ['bonferroni', 'holm', 'fdr', 'benjamini', 'sidak', 'multipletests']
                has_correction = any(c in code.lower() for c in corrections)
                
                if not has_correction:
                    findings.append({
                        'type': 'multiple_testing',
                        'line': None,
                        'description': "Multiple statistical tests without correction",
                        'severity': 'medium',
                        'detection_method': 'pattern'
                    })
                break
        
        return findings
    
    def _control_flow_analysis(self, tree: ast.AST, file_path: str) -> List[Dict]:
        findings = []
        
        performance_keywords = {
            'accuracy', 'f1', 'auc', 'precision', 'recall', 'score',
            'loss', 'error', 'mse', 'mae', 'rmse', 'r2', 'metric'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                condition_names = {
                    n.id.lower() for n in ast.walk(node.test) 
                    if isinstance(n, ast.Name)
                }
                
                involves_performance = bool(condition_names & performance_keywords)
                
                body_names = {
                    n.id for n in ast.walk(node) 
                    if isinstance(n, ast.Name)
                }
                
                accesses_holdout = bool(body_names & self.holdout_identifiers)
                
                if involves_performance and accesses_holdout:
                    findings.append({
                        'type': 'conditional_holdout_access',
                        'line': node.lineno,
                        'description': 'Holdout accessed conditionally based on performance',
                        'severity': 'critical',
                        'detection_method': 'controlflow'
                    })
        
        return findings
    
    def _semantic_analysis(self, tree: ast.AST, file_path: str) -> List[Dict]:
        """v3.0: Análisis semántico para acceso indirecto a holdout"""
        findings = []
        
        self.semantic_analyzer.alias_graph.clear()
        self.semantic_analyzer.analyze_assignments(tree)
        
        tainted = self.semantic_analyzer.find_tainted_variables()
        
        training_contexts = {'fit', 'train', 'optimize', 'tune', 'search'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ''
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr.lower()
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                
                is_training_context = any(ctx in func_name for ctx in training_contexts)
                
                if is_training_context:
                    arg_names = set()
                    for arg in node.args + [kw.value for kw in node.keywords]:
                        for name_node in ast.walk(arg):
                            if isinstance(name_node, ast.Name):
                                arg_names.add(name_node.id)
                    
                    tainted_args = arg_names & tainted
                    
                    if tainted_args:
                        findings.append({
                            'type': 'indirect_holdout_access',
                            'line': node.lineno,
                            'description': (
                                f"Holdout-derived variable(s) {tainted_args} used in "
                                f"training context ({func_name})"
                            ),
                            'severity': 'critical',
                            'detection_method': 'semantic',
                            'tainted_variables': list(tainted_args)
                        })
        
        return findings
    
    def _pattern_matching(self, code: str, file_path: str) -> List[Dict]:
        findings = []
        
        suspicious_patterns = [
            (r'(print|display|show|plot).*?(test|holdout|val).*?(accuracy|f1|score|loss)',
             'result_peeking', 'Test/holdout results displayed', 'high'),
            (r'for\s+\w+\s+in.*?(models?|estimators?).*?(test|holdout)',
             'model_selection_bias', 'Model selection using test/holdout', 'high'),
            (r'(GridSearchCV|RandomizedSearchCV).*?(test|holdout)',
             'test_tuning', 'Hyperparameter search on test set', 'critical'),
            (r'#.*?(hack|cheat|fix.*result|adjust.*threshold|improve.*score)',
             'suspicious_comment', 'Comment suggests manipulation', 'medium'),
            (r'fit_transform.*?(test|holdout|X_test)',
             'transform_leakage', 'fit_transform on test data', 'high'),
        ]
        
        for pattern, violation_type, description, severity in suspicious_patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE | re.DOTALL))
            if matches:
                line_num = code[:matches[0].start()].count('\n') + 1
                findings.append({
                    'type': violation_type,
                    'line': line_num,
                    'description': description,
                    'severity': severity,
                    'detection_method': 'pattern',
                    'n_occurrences': len(matches)
                })
        
        return findings


class CodeAuditor:
    """Auditor de código v3.0 con soporte de notebooks y análisis semántico"""
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.analyzer = None
    
    def audit_analysis_code(
        self, 
        code_paths: List[str] = None,
        notebook_paths: List[str] = None,
        holdout_identifiers: List[str] = None,
        pass_threshold: float = None
    ) -> AuditResult:
        
        if code_paths is None:
            code_paths = []
        if notebook_paths is None:
            notebook_paths = []
        
        if not code_paths and not notebook_paths:
            raise ValueError("Must provide at least one code_path or notebook_path")
        
        if pass_threshold is None:
            pass_threshold = self.framework.config.audit_pass_threshold
        
        if holdout_identifiers is None:
            holdout_identifiers = [
                'holdout', 'test', 'holdout_data', 'test_data',
                'X_test', 'y_test', 'X_holdout', 'y_holdout',
                'test_set', 'holdout_set', 'validation_set',
                'test_df', 'holdout_df', 'df_test', 'df_holdout'
            ]
        
        self.analyzer = StaticCodeAnalyzer(holdout_identifiers)
        
        all_violations = []
        files_analyzed = []
        notebooks_analyzed = []
        
        for code_path in code_paths:
            if not Path(code_path).exists():
                continue
            
            files_analyzed.append(code_path)
            findings = self.analyzer.analyze_file(code_path)
            
            for finding in findings:
                violation = self._finding_to_violation(finding, code_path)
                all_violations.append(violation)
        
        for notebook_path in notebook_paths:
            if not Path(notebook_path).exists():
                continue
            
            notebooks_analyzed.append(notebook_path)
            findings = self.analyzer.analyze_notebook(notebook_path)
            
            for finding in findings:
                violation = self._finding_to_violation(finding, notebook_path)
                all_violations.append(violation)
        
        adherence_score = self._compute_adherence_score(all_violations)
        risk_level = self._determine_risk_level(adherence_score)
        passed = adherence_score >= pass_threshold
        
        summary = self._generate_summary(all_violations, adherence_score, passed)
        detailed_report = self._generate_detailed_report(
            all_violations, adherence_score, files_analyzed, notebooks_analyzed
        )
        
        return AuditResult(
            timestamp=datetime.now().isoformat(),
            adherence_score=adherence_score,
            violations=all_violations,
            risk_level=risk_level,
            passed=passed,
            summary=summary,
            detailed_report=detailed_report,
            files_analyzed=files_analyzed,
            notebooks_analyzed=notebooks_analyzed
        )
    
    def _finding_to_violation(self, finding: Dict, file_path: str) -> CodeViolation:
        code_snippet = ""
        if finding.get('line'):
            code_snippet = self._extract_code_snippet(file_path, finding['line'])
        
        return CodeViolation(
            severity=finding.get('severity', 'medium'),
            category=finding.get('type', 'unknown'),
            description=finding.get('description', ''),
            file_path=file_path,
            line_number=finding.get('line'),
            code_snippet=code_snippet,
            recommendation=self._generate_recommendation(finding.get('type', '')),
            confidence=0.90,
            detection_method=finding.get('detection_method', 'unknown')
        )
    
    def _extract_code_snippet(self, file_path: str, line_num: int) -> str:
        if ':cell_' in file_path:
            return "[Code in notebook cell]"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start = max(0, line_num - 3)
            end = min(len(lines), line_num + 2)
            
            snippet_lines = []
            for i in range(start, end):
                marker = ">>> " if i == line_num - 1 else "    "
                snippet_lines.append(f"{marker}{i+1}: {lines[i].rstrip()}")
            
            return "\n".join(snippet_lines)
        except:
            return ""
    
    def _compute_adherence_score(self, violations: List[CodeViolation]) -> float:
        if not violations:
            return 100.0
        
        severity_penalties = {
            'critical': 30,
            'high': 15,
            'medium': 5,
            'low': 2
        }
        
        total_penalty = sum(
            severity_penalties.get(v.severity, 10) for v in violations
        )
        
        return max(0.0, 100.0 - total_penalty)
    
    def _determine_risk_level(self, score: float) -> str:
        if score >= 90:
            return 'low'
        elif score >= 70:
            return 'medium'
        elif score >= 50:
            return 'high'
        else:
            return 'critical'
    
    def _generate_summary(self, violations: List[CodeViolation], score: float, passed: bool) -> str:
        if passed:
            return f"✅ Audit PASSED (score: {score:.0f}/100). {len(violations)} issue(s) found."
        else:
            critical = sum(1 for v in violations if v.severity == 'critical')
            return f"❌ Audit FAILED (score: {score:.0f}/100). {len(violations)} violations ({critical} critical)."
    
    def _generate_recommendation(self, violation_type: str) -> str:
        recommendations = {
            'holdout_access': "Remove direct holdout references during development.",
            'indirect_holdout_access': "Variable derived from holdout via aliasing. Trace data flow.",
            'conditional_holdout_access': "Never access holdout conditionally based on performance.",
            'threshold_manipulation': "Set threshold ONCE using CV on development data.",
            'multiple_testing': "Apply Bonferroni, Holm, or FDR correction.",
            'result_peeking': "Do not display test results during development.",
            'model_selection_bias': "Use nested CV for model selection.",
            'test_tuning': "CRITICAL: Never tune hyperparameters on test data.",
            'transform_leakage': "Use fit() on train only, then transform() on test.",
            'suspicious_comment': "Review flagged code section."
        }
        
        return recommendations.get(violation_type, "Review code for protocol compliance.")
    
    def _generate_detailed_report(
        self, 
        violations: List[CodeViolation], 
        score: float,
        files_analyzed: List[str],
        notebooks_analyzed: List[str]
    ) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("CODE AUDIT REPORT v3.3")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Adherence Score: {score:.0f}/100")
        lines.append("")
        
        lines.append("FILES ANALYZED:")
        for f in files_analyzed:
            lines.append(f"  📄 {f}")
        for n in notebooks_analyzed:
            lines.append(f"  📓 {n} (notebook)")
        lines.append("")
        
        if not violations:
            lines.append("✅ NO VIOLATIONS DETECTED")
        else:
            lines.append(f"⚠️  {len(violations)} VIOLATION(S) DETECTED")
            
            by_severity = defaultdict(list)
            for v in violations:
                by_severity[v.severity].append(v)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity not in by_severity:
                    continue
                
                lines.append(f"\n{severity.upper()} ({len(by_severity[severity])}):")
                
                for i, v in enumerate(by_severity[severity], 1):
                    lines.append(f"\n{i}. [{v.detection_method.upper()}] {v.category}")
                    lines.append(f"   File: {v.file_path}")
                    if v.line_number:
                        lines.append(f"   Line: {v.line_number}")
                    lines.append(f"   Description: {v.description}")
                    lines.append(f"   Recommendation: {v.recommendation}")
        
        lines.append("\n" + "=" * 80)
        lines.append("v3.3 ANALYSIS METHODS:")
        lines.append("  • AST parsing")
        lines.append("  • Control-flow analysis")
        lines.append("  • Semantic analysis (alias detection)")
        lines.append("  • Pattern matching")
        lines.append("  • Notebook support")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_audit_report(self, audit_result: AuditResult, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(audit_result.detailed_report)
        print(f"✅ Audit report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# COMMITMENT CRIPTOGRÁFICO (v3.0)
# ═══════════════════════════════════════════════════════════════════════════

class CryptographicCommitment:
    """v3.0: Commitment criptográfico para modo no-interactivo"""
    
    @staticmethod
    def generate_commitment(
        project_name: str,
        commitment_phrase: str,
        timestamp: str = None
    ) -> Dict[str, str]:
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        commitment_string = f"{project_name}|{commitment_phrase}|{timestamp}"
        commitment_hash = hashlib.sha256(commitment_string.encode()).hexdigest()
        
        return {
            'project_name': project_name,
            'commitment_timestamp': timestamp,
            'commitment_hash': commitment_hash,
            'algorithm': 'SHA-256',
            'verification_string': base64.b64encode(commitment_string.encode()).decode()
        }
    
    @staticmethod
    def verify_commitment(commitment_data: Dict[str, str], commitment_phrase: str) -> bool:
        project_name = commitment_data['project_name']
        timestamp = commitment_data['commitment_timestamp']
        stored_hash = commitment_data['commitment_hash']
        
        commitment_string = f"{project_name}|{commitment_phrase}|{timestamp}"
        computed_hash = hashlib.sha256(commitment_string.encode()).hexdigest()
        
        return computed_hash == stored_hash


# ═══════════════════════════════════════════════════════════════════════════
# GENERADOR DE REPORTES
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    """Generador de reportes v3.3"""
    
    @staticmethod
    def generate_html_report(final_assessment: Dict, output_path: str = None) -> str:
        project = final_assessment['project_name']
        researcher = final_assessment['researcher']
        verdict = final_assessment['verdict']
        timestamp = final_assessment['assessment_timestamp']
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        eis_data = integrity_metrics.get('eis', {})
        stds_data = integrity_metrics.get('stds', {})
        
        eis_score = eis_data.get('eis', 0)
        eis_interp = eis_data.get('interpretation', 'N/A')
        
        stds_max_z = stds_data.get('max_z_score', 0)
        stds_mean_z = stds_data.get('mean_z_score', 0)
        stds_risk = stds_data.get('risk_level', 'N/A')
        
        verdict_color = {
            'VALIDATED': '#28a745',
            'FALSIFIED': '#dc3545',
        }.get(verdict, '#6c757d')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ECLIPSE v3.3 Report - {project}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: #f5f5f5; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; }}
        .header {{ text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 3px solid #3498db; }}
        .version-badge {{ display: inline-block; background: #3498db; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9em; margin-top: 10px; }}
        .verdict {{ background: {verdict_color}; color: white; padding: 30px; text-align: center; font-size: 2.5em; font-weight: bold; margin: 30px 0; border-radius: 8px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-box {{ background: #f8f9fa; padding: 25px; border-left: 5px solid #3498db; border-radius: 4px; }}
        .metric-box.warning {{ border-left-color: #ffc107; }}
        .metric-box.danger {{ border-left-color: #dc3545; }}
        .metric-box h3 {{ margin-top: 0; color: #2c3e50; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; margin: 10px 0; }}
        h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 40px; }}
        .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .info-item {{ padding: 15px; background: #ecf0f1; border-radius: 4px; }}
        .info-label {{ font-weight: bold; color: #7f8c8d; font-size: 0.9em; }}
        .info-value {{ color: #2c3e50; font-size: 1.1em; margin-top: 5px; }}
        .footer {{ margin-top: 60px; padding-top: 20px; border-top: 2px solid #ecf0f1; color: #7f8c8d; font-size: 0.9em; text-align: center; }}
        .changelog {{ background: #e8f4fd; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .changelog h4 {{ margin-top: 0; color: #2980b9; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ECLIPSE v3.3.1 REPORT</h1>
            <p style="color: #7f8c8d; margin: 10px 0;">IIT Falsification Framework - Fusión Completa</p>
            <span class="version-badge">v3.3.1 - v3.0 Corregido + v3.2.0 IIT</span>
        </div>
        
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">PROJECT</div>
                <div class="info-value">{project}</div>
            </div>
            <div class="info-item">
                <div class="info-label">RESEARCHER</div>
                <div class="info-value">{researcher}</div>
            </div>
            <div class="info-item">
                <div class="info-label">ASSESSMENT DATE</div>
                <div class="info-value">{timestamp}</div>
            </div>
        </div>
        
        <div class="verdict">{verdict}</div>
        
        <div class="changelog">
            <h4>🆕 v3.3.1 - Fusión Completa</h4>
            <ul>
                <li><strong>STDS con z-scores REALES</strong> (corregido de v3.0)</li>
                <li><strong>EIS con bibliografía</strong> (Nosek, Simmons, Kapoor)</li>
                <li><strong>Análisis semántico de aliasing</strong></li>
                <li><strong>Múltiples aproximaciones de Φ</strong> (binary, multilevel, gaussian)</li>
                <li><strong>Sin SMOTE</strong> - Datos naturales</li>
            </ul>
        </div>
        
        <h2>📊 Integrity Metrics</h2>
        
        <div class="metric-grid">
            <div class="metric-box {'warning' if eis_score < 0.7 else ''}">
                <h3>Eclipse Integrity Score (EIS)</h3>
                <div class="metric-value">{eis_score:.4f} / 1.00</div>
                <p><strong>Interpretation:</strong> {eis_interp}</p>
                <p style="color: #7f8c8d; font-size: 0.9em;">
                    Pesos justificados: Nosek et al. (2018), Simmons et al. (2011), 
                    Kapoor & Narayanan (2022).
                </p>
            </div>
            
            <div class="metric-box {'danger' if stds_max_z > 3 else 'warning' if stds_max_z > 2 else ''}">
                <h3>Data Snooping Test (STDS)</h3>
                <div class="metric-value">max z = {stds_max_z:+.2f}</div>
                <p><strong>Mean z-score:</strong> {stds_mean_z:+.4f}</p>
                <p><strong>Risk Level:</strong> {stds_risk}</p>
                <p style="color: #7f8c8d; font-size: 0.9em;">
                    Z-score estándar: z = (holdout - CV_mean) / CV_std.
                    |z| > 2 es inusual, |z| > 3 es muy inusual.
                </p>
            </div>
        </div>
        
        <h2>📋 Pre-Registered Criteria</h2>
        <p>Required criteria passed: {final_assessment.get('verdict_description', 'N/A')}</p>
        
        <div class="footer">
            <p><strong>ECLIPSE v3.3.1</strong> - Fusión v3.0 + v3.2.0</p>
            <p>Citation: Sjöberg Tala, C. A. (2025). ECLIPSE v3.3.1</p>
        </div>
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
        lines = []
        lines.append("=" * 100)
        lines.append("ECLIPSE v3.3.1 FALSIFICATION REPORT")
        lines.append("Fusión: v3.0 (metodología corregida) + v3.2.0 (IIT/Sleep-EDF)")
        lines.append("=" * 100)
        lines.append("")
        
        lines.append(f"Project: {final_assessment['project_name']}")
        lines.append(f"Researcher: {final_assessment['researcher']}")
        lines.append(f"Assessment Date: {final_assessment['assessment_timestamp']}")
        lines.append("")
        
        lines.append("=" * 100)
        lines.append(f"FINAL VERDICT: {final_assessment['verdict']}")
        lines.append("=" * 100)
        lines.append(f"{final_assessment['verdict_description']}")
        lines.append("")
        
        lines.append("-" * 100)
        lines.append("v3.3.1 CARACTERÍSTICAS:")
        lines.append("-" * 100)
        lines.append("  ✓ STDS con z-scores reales (no heurísticas)")
        lines.append("  ✓ EIS con pesos bibliográficos")
        lines.append("  ✓ Análisis semántico de aliasing")
        lines.append("  ✓ Soporte de notebooks")
        lines.append("  ✓ Múltiples aproximaciones de Φ")
        lines.append("  ✓ Sin SMOTE (datos naturales)")
        lines.append("")
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        
        if integrity_metrics:
            lines.append("-" * 100)
            lines.append("INTEGRITY METRICS")
            lines.append("-" * 100)
            
            eis_data = integrity_metrics.get('eis', {})
            if eis_data:
                lines.append(f"\n📊 Eclipse Integrity Score: {eis_data.get('eis', 0):.4f}")
                lines.append(f"   {eis_data.get('interpretation', 'N/A')}")
                
                components = eis_data.get('components', {})
                lines.append(f"\n   Components:")
                lines.append(f"   • Pre-registration: {components.get('preregistration_score', 0):.3f}")
                lines.append(f"   • Split strength: {components.get('split_strength', 0):.3f}")
                lines.append(f"   • Protocol adherence: {components.get('protocol_adherence', 0):.3f}")
                lines.append(f"   • Leakage score: {components.get('leakage_score', 0):.3f}")
                lines.append(f"   • Transparency: {components.get('transparency_score', 0):.3f}")
            
            stds_data = integrity_metrics.get('stds', {})
            if stds_data.get('status') == 'success':
                lines.append(f"\n🔍 Statistical Test for Data Snooping:")
                lines.append(f"   Max z-score: {stds_data.get('max_z_score', 0):+.4f}")
                lines.append(f"   Mean z-score: {stds_data.get('mean_z_score', 0):+.4f}")
                lines.append(f"   Risk Level: {stds_data.get('risk_level', 'N/A')}")
                lines.append(f"   {stds_data.get('interpretation', '')}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("ECLIPSE v3.3.1 - Fusión Completa")
        lines.append("Citation: Sjöberg Tala, C. A. (2025). ECLIPSE v3.3.1")
        lines.append("=" * 100)
        
        text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
        
        return text


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES SLEEP-EDF (de v3.2.0)
# ═══════════════════════════════════════════════════════════════════════════

def load_sleepedf_subject_multichannel_v3(psg_path, hypno_path, n_channels=8, 
                                          phi_methods='all', thermal_monitor=None):
    """Cargar sujeto Sleep-EDF con monitoreo"""
    if not MNE_AVAILABLE:
        raise ImportError("MNE not available")
    
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
        except:
            pass
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
    """Análisis comparativo de métodos Φ"""
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS COMPARATIVO DE MÉTODOS Φ")
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
# FRAMEWORK PRINCIPAL v3.3.1
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    """
    ECLIPSE v3.3.1: Fusión Completa
    
    Combina:
    - v3.0: Framework metodológico corregido (STDS, EIS, análisis semántico)
    - v3.2.0: Aplicación IIT con Sleep-EDF (múltiples Φ, monitoreo térmico)
    """
    
    VERSION = "3.3.1"
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA.json"
        self.results_file = self.output_dir / f"{config.project_name}_RESULT.json"
        self.commitment_file = self.output_dir / f"{config.project_name}_COMMITMENT.json"
        
        self._split_completed = False
        self._criteria_registered = False
        self._development_completed = False
        self._validation_completed = False
        
        self.integrity_scorer = None
        self.snooping_tester = None
        self.code_auditor = None
        
        self._load_existing_state()
        
        print("=" * 80)
        print(f"🔬 ECLIPSE v{self.VERSION} INITIALIZED")
        print("=" * 80)
        print(f"Project: {config.project_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Sacred Seed: {config.sacred_seed}")
        print(f"Mode: {'Non-interactive (CI/CD)' if config.non_interactive else 'Interactive'}")
        print("")
        print("v3.3.1 FUSIÓN:")
        print("  ✅ STDS con z-scores REALES (de v3.0)")
        print("  ✅ EIS con bibliografía (de v3.0)")
        print("  ✅ Análisis semántico de aliasing (de v3.0)")
        print("  ✅ Múltiples Φ: binary, multilevel, gaussian (de v3.2.0)")
        print("  ✅ Soporte Sleep-EDF (de v3.2.0)")
        print("  ✅ Monitoreo térmico (de v3.2.0)")
        print("=" * 80)
    
    def _load_existing_state(self):
        if self.split_file.exists():
            self._split_completed = True
        if self.criteria_file.exists():
            self._criteria_registered = True
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    results = json.load(f)
                if 'validation_summary' in results:
                    self._validation_completed = True
                if 'development_summary' in results:
                    self._development_completed = True
            except:
                pass
    
    def stage1_irreversible_split(
        self, 
        data_identifiers: List[Any],
        force: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """Stage 1: Split irreversible con verificación criptográfica"""
        
        if not data_identifiers:
            raise ValueError("data_identifiers cannot be empty")
        
        if self.split_file.exists() and not force:
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            self._split_completed = True
            return split_data['development_ids'], split_data['holdout_ids']
        
        print("\n" + "=" * 80)
        print("STAGE 1: IRREVERSIBLE DATA SPLIT")
        print("=" * 80)
        
        np.random.seed(self.config.sacred_seed)
        shuffled_ids = np.array(data_identifiers).copy()
        np.random.shuffle(shuffled_ids)
        
        n_development = int(len(data_identifiers) * self.config.development_ratio)
        development_ids = shuffled_ids[:n_development].tolist()
        holdout_ids = shuffled_ids[n_development:].tolist()
        
        split_hash = hashlib.sha256(
            f"{self.config.sacred_seed}_{sorted(data_identifiers)}".encode()
        ).hexdigest()
        
        split_data = {
            'project_name': self.config.project_name,
            'split_date': datetime.now().isoformat(),
            'sacred_seed': self.config.sacred_seed,
            'total_samples': len(data_identifiers),
            'n_development': len(development_ids),
            'n_holdout': len(holdout_ids),
            'development_ratio': self.config.development_ratio,
            'holdout_ratio': self.config.holdout_ratio,
            'development_ids': development_ids,
            'holdout_ids': holdout_ids,
            'integrity_verification': {
                'split_hash': split_hash,
                'algorithm': 'SHA-256'
            },
            'eclipse_version': self.VERSION
        }
        
        with open(self.split_file, 'w') as f:
            json.dump(split_data, f, indent=2, default=str)
        
        print(f"✅ Development: {len(development_ids)} samples ({self.config.development_ratio*100:.0f}%)")
        print(f"✅ Holdout: {len(holdout_ids)} samples ({self.config.holdout_ratio*100:.0f}%)")
        print(f"🔒 Split hash: {split_hash[:16]}...")
        
        self._split_completed = True
        return development_ids, holdout_ids
    
    def stage2_register_criteria(
        self, 
        criteria: List[FalsificationCriteria],
        force: bool = False
    ) -> Dict:
        """Stage 2: Pre-registrar criterios de falsificación"""
        
        if not criteria:
            raise ValueError("criteria cannot be empty")
        
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            self._criteria_registered = True
            return criteria_data
        
        print("\n" + "=" * 80)
        print("STAGE 2: PRE-REGISTERED CRITERIA")
        print("=" * 80)
        
        print(f"\n📋 Registering {len(criteria)} criteria:")
        for i, c in enumerate(criteria, 1):
            print(f"   {i}. {c}")
        
        criteria_list = [asdict(c) for c in criteria]
        criteria_hash = hashlib.sha256(
            json.dumps(criteria_list, sort_keys=True).encode()
        ).hexdigest()
        
        criteria_dict = {
            'project_name': self.config.project_name,
            'registration_date': datetime.now().isoformat(),
            'criteria': criteria_list,
            'n_required': sum(1 for c in criteria if c.is_required),
            'n_optional': sum(1 for c in criteria if not c.is_required),
            'criteria_hash': criteria_hash,
            'binding_declaration': (
                "These criteria are binding and cannot be modified after registration."
            ),
            'eclipse_version': self.VERSION
        }
        
        with open(self.criteria_file, 'w') as f:
            json.dump(criteria_dict, f, indent=2, default=str)
        
        print(f"\n✅ {len(criteria)} criteria registered")
        print(f"🔒 Criteria hash: {criteria_hash[:16]}...")
        
        self._criteria_registered = True
        return criteria_dict
    
    def stage3_development(
        self,
        development_data: Any,
        training_function: Callable,
        validation_function: Callable,
        **kwargs
    ) -> Dict:
        """Stage 3: Desarrollo con CV estratificado"""
        
        if not self._split_completed:
            raise RuntimeError("Must complete Stage 1 first")
        
        print("\n" + "=" * 80)
        print("STAGE 3: CLEAN DEVELOPMENT PROTOCOL")
        print("=" * 80)
        print(f"Cross-validation: {self.config.n_folds_cv} folds")
        
        from sklearn.model_selection import StratifiedKFold
        
        if isinstance(development_data, pd.DataFrame):
            y_labels = development_data['consciousness'].values
        else:
            y_labels = np.array([d['consciousness'] for d in development_data])
        
        skf = StratifiedKFold(
            n_splits=self.config.n_folds_cv, 
            shuffle=True, 
            random_state=self.config.sacred_seed
        )
        
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
                
                cv_results.append({
                    'fold': fold_idx + 1,
                    'n_train': len(train_idx),
                    'n_val': len(val_idx),
                    'metrics': metrics,
                    'status': 'success'
                })
                print(f"   ✅ Complete - MCC: {metrics.get('mcc', 0):.3f}")
                
            except Exception as e:
                logging.error(f"Fold {fold_idx + 1} failed: {e}")
                cv_results.append({
                    'fold': fold_idx + 1,
                    'status': 'failed',
                    'error': str(e)
                })
        
        successful_folds = [r for r in cv_results if r['status'] == 'success']
        
        if not successful_folds:
            raise RuntimeError("All CV folds failed!")
        
        metric_names = list(successful_folds[0]['metrics'].keys())
        aggregated_metrics = {}
        
        for metric_name in metric_names:
            values = [r['metrics'][metric_name] for r in successful_folds]
            aggregated_metrics[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'values': [float(v) for v in values]
            }
            
            print(f"\n{metric_name}: {aggregated_metrics[metric_name]['mean']:.4f} ± {aggregated_metrics[metric_name]['std']:.4f}")
        
        print(f"\n✅ DEVELOPMENT COMPLETE ({len(successful_folds)}/{self.config.n_folds_cv} folds)")
        
        self._development_completed = True
        
        return {
            'n_folds': self.config.n_folds_cv,
            'n_successful': len(successful_folds),
            'fold_results': cv_results,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def stage4_single_shot_validation(
        self,
        holdout_data: Any,
        final_model: Any,
        validation_function: Callable,
        force: bool = False,
        **kwargs
    ) -> Dict:
        """Stage 4: Validación single-shot (v3.0: soporte no-interactivo)"""
        
        if not self._split_completed:
            raise RuntimeError("Must complete Stage 1 first")
        
        if not self._criteria_registered:
            raise RuntimeError("Must complete Stage 2 first")
        
        if self.results_file.exists() and not force:
            raise RuntimeError("VALIDATION ALREADY PERFORMED! Single-shot validation.")
        
        print("\n" + "=" * 80)
        print("🎯 STAGE 4: SINGLE-SHOT VALIDATION")
        print("=" * 80)
        print("⚠️  THIS HAPPENS EXACTLY ONCE - NO SECOND CHANCES")
        
        # v3.0: Modo no-interactivo
        if self.config.non_interactive:
            print("\n🤖 NON-INTERACTIVE MODE")
            
            commitment = CryptographicCommitment.generate_commitment(
                self.config.project_name,
                self.config.commitment_phrase
            )
            
            with open(self.commitment_file, 'w') as f:
                json.dump(commitment, f, indent=2)
            
            print(f"   Commitment hash: {commitment['commitment_hash'][:16]}...")
            confirmation = "AUTOMATED"
        else:
            confirmation = input("\n🚨 Type 'I ACCEPT SINGLE-SHOT VALIDATION' to proceed: ")
            
            if confirmation != "I ACCEPT SINGLE-SHOT VALIDATION":
                print("\n❌ Validation cancelled")
                return None
        
        print("\n🚀 EXECUTING SINGLE-SHOT VALIDATION...")
        
        try:
            metrics = validation_function(final_model, holdout_data, **kwargs)
            
            print("\n📊 HOLDOUT RESULTS:")
            for metric_name, value in metrics.items():
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, (int, float)):
                    print(f"   {metric_name}: {value:.4f}")
            
            def to_native(v):
                if hasattr(v, 'item'):
                    return v.item()
                elif isinstance(v, (int, float)):
                    return float(v)
                else:
                    return v
            
            validation_results = {
                'status': 'success',
                'n_holdout_samples': len(holdout_data) if hasattr(holdout_data, '__len__') else None,
                'metrics': {k: to_native(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat(),
                'confirmation_mode': 'non_interactive' if self.config.non_interactive else 'interactive'
            }
            
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            validation_results = {
                'status': 'failed', 
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self._validation_completed = True
        return validation_results
    
    def compute_integrity_metrics(self) -> Dict[str, Any]:
        """Compute EIS y STDS con metodología v3.0 corregida"""
        
        print("\n📊 Computing Eclipse Integrity Score...")
        
        if self.integrity_scorer is None:
            self.integrity_scorer = EclipseIntegrityScore(self)
        
        eis_results = self.integrity_scorer.compute_eis()
        print(f"   EIS: {eis_results['eis']:.4f} - {eis_results['interpretation']}")
        
        eis_path = self.output_dir / f"{self.config.project_name}_EIS_REPORT.txt"
        self.integrity_scorer.generate_eis_report(str(eis_path))
        
        stds_results = {'status': 'not_applicable'}
        
        if self._validation_completed:
            print("\n🔍 Performing STDS v3.3 (z-scores reales)...")
            
            if self.snooping_tester is None:
                self.snooping_tester = StatisticalTestDataSnooping(self)
            
            stds_results = self.snooping_tester.perform_snooping_test()
            
            if stds_results.get('status') == 'success':
                max_z = stds_results['max_z_score']
                mean_z = stds_results['mean_z_score']
                print(f"   Max z-score: {max_z:+.4f}")
                print(f"   Mean z-score: {mean_z:+.4f}")
                
                if max_z > 3:
                    print("   🚨 WARNING: z-score muy alto detectado!")
                elif max_z > 2:
                    print("   ⚠️ Notable: Alguna métrica tiene z > 2")
                else:
                    print("   ✅ Resultados dentro del rango normal")
                
                stds_path = self.output_dir / f"{self.config.project_name}_STDS_REPORT.txt"
                self.snooping_tester.generate_stds_report(str(stds_path))
        
        return {
            'eis': eis_results,
            'stds': stds_results
        }
    
    def audit_code(
        self,
        code_paths: List[str] = None,
        notebook_paths: List[str] = None,
        holdout_identifiers: List[str] = None
    ) -> AuditResult:
        """Auditar código con análisis semántico v3.0"""
        
        print("\n" + "=" * 80)
        print("🤖 CODE AUDIT v3.3 (con análisis semántico)")
        print("=" * 80)
        
        if self.code_auditor is None:
            self.code_auditor = CodeAuditor(self)
        
        audit_result = self.code_auditor.audit_analysis_code(
            code_paths=code_paths,
            notebook_paths=notebook_paths,
            holdout_identifiers=holdout_identifiers
        )
        
        print(f"\n{'✅' if audit_result.passed else '❌'} {audit_result.summary}")
        
        audit_path = self.output_dir / f"{self.config.project_name}_CODE_AUDIT.txt"
        self.code_auditor.save_audit_report(audit_result, str(audit_path))
        
        return audit_result
    
    def verify_integrity(self) -> Dict:
        """Verificar integridad criptográfica"""
        print("\n🔍 Verifying integrity...")
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'version': self.VERSION,
            'all_valid': True,
            'files': {}
        }
        
        if self.split_file.exists():
            try:
                with open(self.split_file, 'r') as f:
                    data = json.load(f)
                
                all_ids = data['development_ids'] + data['holdout_ids']
                computed = hashlib.sha256(
                    f"{data['sacred_seed']}_{sorted(all_ids)}".encode()
                ).hexdigest()
                
                valid = computed == data['integrity_verification']['split_hash']
                verification['files']['split'] = {'valid': valid}
                
                if not valid:
                    verification['all_valid'] = False
                    
                print(f"{'✅' if valid else '❌'} Split: {'VALID' if valid else 'COMPROMISED'}")
            except Exception as e:
                verification['files']['split'] = {'valid': False, 'error': str(e)}
                verification['all_valid'] = False
        
        if self.criteria_file.exists():
            try:
                with open(self.criteria_file, 'r') as f:
                    data = json.load(f)
                
                computed = hashlib.sha256(
                    json.dumps(data['criteria'], sort_keys=True).encode()
                ).hexdigest()
                
                valid = computed == data['criteria_hash']
                verification['files']['criteria'] = {'valid': valid}
                
                if not valid:
                    verification['all_valid'] = False
                    
                print(f"{'✅' if valid else '❌'} Criteria: {'VALID' if valid else 'COMPROMISED'}")
            except Exception as e:
                verification['files']['criteria'] = {'valid': False, 'error': str(e)}
                verification['all_valid'] = False
        
        return verification
    
    def stage5_final_assessment(
        self,
        development_results: Dict,
        validation_results: Dict,
        generate_reports: bool = True,
        compute_integrity: bool = True
    ) -> Dict:
        """Stage 5: Evaluación final con métricas v3.0"""
        
        if validation_results is None or validation_results.get('status') != 'success':
            raise RuntimeError("Validation failed - cannot assess")
        
        print("\n" + "=" * 80)
        print(f"🎯 STAGE 5: FINAL ASSESSMENT v{self.VERSION}")
        print("=" * 80)
        
        with open(self.criteria_file, 'r') as f:
            criteria_data = json.load(f)
        
        criteria_list = [FalsificationCriteria(**c) for c in criteria_data['criteria']]
        
        holdout_metrics = validation_results.get('metrics', {})
        criteria_evaluation = []
        
        print("\n📋 EVALUATING CRITERIA:")
        
        for criterion in criteria_list:
            if criterion.name in holdout_metrics:
                value = holdout_metrics[criterion.name]
                passed = criterion.evaluate(value)
                evaluation = {
                    'criterion': asdict(criterion),
                    'value': float(value),
                    'passed': passed
                }
                
                status = "✅ PASS" if passed else "❌ FAIL"
                req = "[REQUIRED]" if criterion.is_required else "[optional]"
                print(f"{status} {req} {criterion.name} {criterion.comparison} {criterion.threshold} (got: {value:.4f})")
            else:
                evaluation = {
                    'criterion': asdict(criterion),
                    'value': None,
                    'passed': False
                }
                print(f"⚠️  MISSING: {criterion.name}")
            
            criteria_evaluation.append(evaluation)
        
        required_criteria = [e for e in criteria_evaluation if e['criterion']['is_required']]
        required_passed = sum(1 for e in required_criteria if e['passed'])
        required_total = len(required_criteria)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required_criteria) else "FALSIFIED"
        
        print("\n" + "=" * 80)
        print(f"{'✅' if verdict == 'VALIDATED' else '❌'} VERDICT: {verdict}")
        print(f"Required criteria: {required_passed}/{required_total}")
        print("=" * 80)
        
        final_assessment = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'assessment_date': datetime.now().isoformat(),
            'assessment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eclipse_version': self.VERSION,
            
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
            'verdict_description': f"{required_passed}/{required_total} required criteria passed"
        }
        
        # Guardar preliminar
        final_assessment['final_hash'] = 'pending'
        
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        # Compute integrity metrics v3.0
        if compute_integrity:
            print("\n🔬 COMPUTING v3.3 INTEGRITY METRICS...")
            
            try:
                integrity_metrics = self.compute_integrity_metrics()
                final_assessment['integrity_metrics'] = integrity_metrics
            except Exception as e:
                logging.error(f"Integrity metrics failed: {e}")
                warnings.warn(f"Could not compute integrity metrics: {e}")
        
        # Hash final
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        # Generar reportes
        if generate_reports:
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
        
        return final_assessment
    
    def get_status(self) -> Dict:
        """Get pipeline status"""
        return {
            'version': self.VERSION,
            'stage1_split': self._split_completed,
            'stage2_criteria': self._criteria_registered,
            'stage3_development': self._development_completed,
            'stage4_validation': self._validation_completed,
            'non_interactive_mode': self.config.non_interactive
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN - IIT FALSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🧠 ECLIPSE v3.3.1 - IIT FALSIFICATION")
    print("   Fusión: v3.0 (metodología corregida) + v3.2.0 (IIT/Sleep-EDF)")
    print("=" * 80)
    print("\n✅ CARACTERÍSTICAS:")
    print("   • STDS con z-scores REALES (no heurísticas)")
    print("   • EIS con pesos bibliográficos")
    print("   • Análisis semántico de aliasing")
    print("   • Múltiples Φ (binary, multilevel, gaussian)")
    print("   • Sin SMOTE (datos naturales)")
    print("=" * 80)
    
    if not MNE_AVAILABLE:
        print("❌ MNE no disponible. Instalar con: pip install mne")
        return
    
    output_dir = "./eclipse_results_v3_3"
    log_file = setup_logging(output_dir)
    
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
    print(f"   - Balanceo: NINGUNO (datos naturales)")
    
    thermal_monitor = ThermalMonitor()
    
    subject_pairs = buscar_archivos_edf_pares(sleep_edf_path)
    
    if len(subject_pairs) == 0:
        print("❌ Sin pares")
        return
    
    if limit_n:
        subject_pairs = subject_pairs[:limit_n]
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"processing_v3_3_{n_channels}ch_{phi_methods}_natural"
    all_windows = load_progress(output_dir_path, checkpoint_name)
    
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
        
        thermal_monitor.check_temperature()
        
        try:
            windows = load_sleepedf_subject_multichannel_v3(
                psg, hypno, n_channels, phi_methods, thermal_monitor
            )
            
            if windows is None:
                continue
            
            if len(windows) > 0:
                for w in windows:
                    w['subject_id'] = subject_id
                
                all_windows.extend(windows)
                
                if i % 5 == 0:
                    save_progress(output_dir_path, all_windows, checkpoint_name)
                    print(f"\n   💾 Checkpoint ({i}/{len(subject_pairs)})")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    save_progress(output_dir_path, all_windows, checkpoint_name)
    
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
    
    config = EclipseConfig(
        project_name=f"IIT_v3_3_{n_channels}ch_natural",
        researcher="Camilo Sjöberg Tala",
        sacred_seed=2025,
        n_channels=n_channels,
        phi_methods=[phi_methods],
        output_dir=output_dir
    )
    
    eclipse = EclipseFramework(config)
    
    unique_subjects = df_natural['subject_id'].unique().tolist()
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(unique_subjects)
    
    dev_data = df_natural[df_natural['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = df_natural[df_natural['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
    print(f"\n📊 División:")
    print(f"   Dev: {len(dev_data)} ventanas")
    print(f"   Holdout: {len(holdout_data)} ventanas")
    
    criteria = [
        FalsificationCriteria("balanced_accuracy", 0.60, ">=", "Bal.Acc >= 0.60", True),
        FalsificationCriteria("f1_score", 0.50, ">=", "F1 >= 0.50", True),
        FalsificationCriteria("mcc", 0.20, ">=", "MCC >= 0.20", True)
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
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
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True
    )
    
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ ECLIPSE v3.3.1 COMPLETADO")
    print("=" * 80)
    
    holdout_metrics = final_assessment['validation_summary']['metrics']
    
    print(f"\n📊 MÉTRICAS HOLDOUT:")
    print(f"   Balanced Accuracy: {holdout_metrics.get('balanced_accuracy', 0):.4f}")
    print(f"   MCC: {holdout_metrics.get('mcc', 0):.4f}")
    print(f"   F1 Score: {holdout_metrics.get('f1_score', 0):.4f}")
    
    print(f"\n{'✅' if final_assessment['verdict'] == 'VALIDATED' else '❌'} VEREDICTO: {final_assessment['verdict']}")
    
    if 'integrity_metrics' in final_assessment:
        integrity = final_assessment['integrity_metrics']
        
        if 'eis' in integrity:
            eis_score = integrity['eis']['eis']
            print(f"\n📊 Eclipse Integrity Score: {eis_score:.4f}")
            print(f"   {integrity['eis']['interpretation']}")
        
        if 'stds' in integrity and integrity['stds'].get('status') == 'success':
            stds_max_z = integrity['stds']['max_z_score']
            stds_mean_z = integrity['stds']['mean_z_score']
            print(f"\n🔍 Data Snooping Test (z-scores REALES):")
            print(f"   Max z-score: {stds_max_z:+.4f}")
            print(f"   Mean z-score: {stds_mean_z:+.4f}")
            print(f"   Risk Level: {integrity['stds']['risk_level']}")
    
    print(f"\n⏱️  Tiempo total: {(time.time() - start_time)/60:.1f} min")
    
    print("\n" + "=" * 80)
    print("🎯 ECLIPSE v3.3.1 - FUSIÓN COMPLETA EXITOSA")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        logging.error(traceback.format_exc())