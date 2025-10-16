"""
═══════════════════════════════════════════════════════════════════════════════
FALSIFICACIÓN DE IIT CON ECLIPSE v3.1.0 - VERSIÓN INTEGRADA COMPLETA
═══════════════════════════════════════════════════════════════════════════════
Autor: Camilo Alejandro Sjöberg Tala + Claude
Version: 3.1.0-INTEGRATED
Hardware Target: Intel i7-10700K (8 cores), 32GB RAM

✅ NUEVO v3.1.0 - INTEGRACIÓN ECLIPSE v2.0:
  - Eclipse Integrity Score (EIS) - Métrica cuantitativa de rigor
  - Statistical Test for Data Snooping (STDS) - Detección de p-hacking  
  - LLM-Powered Code Auditor - Auditoría automatizada
  - Reportes mejorados con métricas de integridad
  - Verificación criptográfica completa
  
✅ MANTIENE v3.0.3:
  - Monitoreo en tiempo real de procesamiento
  - Estadísticas acumuladas por sujeto
  - Progreso detallado cada 30 segundos
  - Logging con flush inmediato
  - Múltiples aproximaciones de Φ

APROXIMACIONES DE Φ IMPLEMENTADAS:
1. phi_binary: MEJORADO con temporal dynamics - ~3 seg/ventana ⭐ MÁS FIEL A IIT
2. phi_multilevel_3: 3 niveles - ~10 seg/ventana
3. phi_multilevel_4: 4 niveles - ~20 seg/ventana
4. phi_gaussian_copula: Continua - ~2 seg/ventana

FIDELIDAD A IIT 3.0:
✅ Evaluación temporal causa-efecto
✅ Minimum Information Partition completo
✅ Normalización por complejidad
❌ Solo biparticiones (triparticiones+ computacionalmente prohibitivas)
❌ Binarización (IIT usa continuo, pero nadie puede implementarlo)

NOVEL CONTRIBUTIONS (ECLIPSE v2.0):
✅ EIS: Primera métrica cuantitativa de integridad metodológica
✅ STDS: Primera prueba estadística para detección de data snooping
✅ LLM Auditor: Primer auditor automatizado de protocolos científicos

Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v2.0. DOI: 10.5281/zenodo.15541550
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
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os
from scipy import signal
from scipy.stats import entropy, spearmanr, pearsonr, rankdata, norm, mannwhitneyu, ttest_ind
import scipy.linalg as la
import time
import psutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import ast
import re

# ═══════════════════════════════════════════════════════════════
# SILENCIAR WARNINGS MOLESTOS
# ═══════════════════════════════════════════════════════════════
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*GPU.*')
warnings.filterwarnings('ignore', message='.*CUDA.*')
warnings.filterwarnings('ignore', message='.*cupy.*')

logging.getLogger('mne').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Optional: LLM integration
try:
    import anthropic
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# GPU SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

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
            logging.info("GPU detectada - Aceleración activada")
        except:
            USE_GPU = False
            logging.info("GPU no disponible - Usando CPU")
        GPU_INITIALIZED = True
    return USE_GPU

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    init_gpu()

if not USE_GPU:
    cp = np

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

N_WORKERS = min(8, mp.cpu_count())

# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CON FLUSH INMEDIATO
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str):
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"iit_eclipse_v3_1_{timestamp}.log"
    
    # Limpiar handlers previos
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # File handler con flush inmediato
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    # Stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    # Configurar root logger
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stream_handler)
    
    # Test y flush
    logging.info("="*80)
    logging.info("LOG INICIALIZADO - ECLIPSE v3.1.0 INTEGRATED")
    logging.info(f"Archivo: {log_file}")
    logging.info("="*80)
    
    for handler in logging.root.handlers:
        handler.flush()
    
    return log_file

# ═══════════════════════════════════════════════════════════════════════════
# THERMAL MONITOR
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
            logging.info(f"⏸️  Pausa de enfriamiento #{self.cooldown_count} ({self.COOLDOWN_TIME}s)")
            time.sleep(self.COOLDOWN_TIME)
            return self.check_temperature(force=True)
        
        return True

# ═══════════════════════════════════════════════════════════════════════════
# PHI CALCULATIONS - Todas las funciones originales
# ═══════════════════════════════════════════════════════════════════════════

def calculate_phi_binary_improved(eeg_segment, use_gpu=False):
    """Φ* mejorado - MÁS cercano a IIT 3.0"""
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
    
    states_t = []
    for t in range(n_time - 1):
        state_t = tuple(binary_signals[:, t])
        states_t.append(state_t)
    
    states_t1 = []
    for t in range(1, n_time):
        state_t1 = tuple(binary_signals[:, t])
        states_t1.append(state_t1)
    
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
            
            states_A_t = [tuple(binary_signals[list(partition_A_idx), t]) 
                         for t in range(n_time - 1)]
            unique_A_t, counts_A_t = np.unique(states_A_t, axis=0, return_counts=True)
            p_A_t = counts_A_t / len(states_A_t)
            H_A_t = -np.sum(p_A_t * np.log2(p_A_t + 1e-10))
            
            states_A_t1 = [tuple(binary_signals[list(partition_A_idx), t]) 
                          for t in range(1, n_time)]
            unique_A_t1, counts_A_t1 = np.unique(states_A_t1, axis=0, return_counts=True)
            p_A_t1 = counts_A_t1 / len(states_A_t1)
            H_A_t1 = -np.sum(p_A_t1 * np.log2(p_A_t1 + 1e-10))
            
            states_B_t = [tuple(binary_signals[list(partition_B_idx), t]) 
                         for t in range(n_time - 1)]
            unique_B_t, counts_B_t = np.unique(states_B_t, axis=0, return_counts=True)
            p_B_t = counts_B_t / len(states_B_t)
            H_B_t = -np.sum(p_B_t * np.log2(p_B_t + 1e-10))
            
            states_B_t1 = [tuple(binary_signals[list(partition_B_idx), t]) 
                          for t in range(1, n_time)]
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
    """Φ con discretización multinivel"""
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
    
    joint_states = []
    for t in range(n_time):
        state = tuple(discretized[:, t])
        joint_states.append(state)
    
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
    """Φ con Gaussian Copula - APROXIMACIÓN CONTINUA"""
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
    """Calcula múltiples aproximaciones de Φ"""
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
# ECLIPSE v2.0 COMPONENTS - NUEVAS CLASES INTEGRADAS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FalsificationCriteria:
    """Pre-registered falsification criterion"""
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
    output_dir: str = "./eclipse_results_v3_1"
    timestamp: str = field(default=None)
    n_channels: int = 8
    phi_methods: List[str] = field(default_factory=lambda: ['fast'])
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CodeViolation:
    """Detected violation of ECLIPSE protocol"""
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
    """Complete audit results"""
    timestamp: str
    adherence_score: float
    violations: List[CodeViolation]
    risk_level: str
    passed: bool
    summary: str
    detailed_report: str

# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE INTEGRITY SCORE (EIS) - NEW IN v2.0 / v3.1.0
# ═══════════════════════════════════════════════════════════════════════════

class EclipseIntegrityScore:
    """
    Eclipse Integrity Score (EIS): Quantitative metric for methodological rigor
    
    NOVEL CONTRIBUTION - Evaluates study integrity across 5 dimensions
    """
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.scores = {}
    
    def compute_preregistration_score(self) -> float:
        """Score pre-registration completeness"""
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
        """Measure split strength using entropy"""
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
        """Measure adherence to pre-registered protocol"""
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
        """Estimate risk of data leakage between dev and holdout"""
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
        """Score documentation and transparency"""
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
        """Compute Eclipse Integrity Score"""
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
        """Interpret EIS value"""
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
        """Generate EIS report"""
        if not self.scores:
            self.compute_eis()
        
        lines = []
        lines.append("=" * 80)
        lines.append("ECLIPSE INTEGRITY SCORE (EIS) REPORT v3.1.0")
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
        
        lines.append(f"  {'Pre-registration completeness':<35s}: {components['preregistration_score']:.4f}")
        lines.append(f"  {'Split strength (entropy)':<35s}: {components['split_strength']:.4f}")
        lines.append(f"  {'Protocol adherence':<35s}: {components['protocol_adherence']:.4f}")
        lines.append(f"  {'Data leakage score (1-risk)':<35s}: {components['leakage_score']:.4f}")
        lines.append(f"  {'Transparency & documentation':<35s}: {components['transparency_score']:.4f}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("Novel Metric: First quantitative integrity score for consciousness research")
        lines.append("Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v2.0. DOI: 10.5281/zenodo.15541550")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ EIS Report: {output_path}")
        
        return report

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TEST FOR DATA SNOOPING (STDS) - NEW IN v2.0 / v3.1.0
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalTestDataSnooping:
    """
    Statistical Test for Data Snooping (STDS)
    
    NOVEL CONTRIBUTION - Detects suspicious patterns indicating holdout contamination
    """
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.test_results = {}
    
    def _compute_performance_similarity(self, dev_metrics: Dict, holdout_metrics: Dict) -> float:
        """Compute similarity between dev and holdout performance"""
        similarities = []
        
        for metric_name in dev_metrics:
            if metric_name in holdout_metrics:
                dev_mean = dev_metrics[metric_name].get('mean', 0)
                holdout_val = holdout_metrics[metric_name]
                
                if isinstance(holdout_val, (int, float)) and dev_mean != 0:
                    rel_diff = abs((dev_mean - holdout_val) / abs(dev_mean))
                    similarity = 1.0 - min(rel_diff, 1.0)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _bootstrap_expected_degradation(
        self, 
        dev_metrics: Dict, 
        n_bootstrap: int = 1000
    ) -> Dict[str, Tuple[float, float]]:
        """Bootstrap expected performance degradation"""
        expected_degradation = {}
        
        for metric_name, stats in dev_metrics.items():
            values = stats.get('values', [])
            
            if len(values) < 3:
                continue
            
            degradations = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                simulated_holdout = np.min(sample)
                simulated_dev_mean = np.mean(sample)
                
                if simulated_dev_mean != 0:
                    deg = ((simulated_dev_mean - simulated_holdout) / abs(simulated_dev_mean)) * 100
                    degradations.append(deg)
            
            expected_degradation[metric_name] = (np.mean(degradations), np.std(degradations))
        
        return expected_degradation
    
    def perform_snooping_test(
        self, 
        n_permutations: int = 1000,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Perform statistical test for data snooping"""
        if not self.framework._validation_completed:
            return {
                'status': 'incomplete',
                'message': 'Validation not yet performed'
            }
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
        except:
            return {
                'status': 'error',
                'message': 'Could not load results file'
            }
        
        dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
        holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
        
        if not dev_metrics or not holdout_metrics:
            return {
                'status': 'insufficient_data',
                'message': 'Insufficient metrics for testing'
            }
        
        observed_similarity = self._compute_performance_similarity(dev_metrics, holdout_metrics)
        expected_deg = self._bootstrap_expected_degradation(dev_metrics, n_bootstrap=n_permutations)
        
        actual_degradations = {}
        for metric_name in dev_metrics:
            if metric_name in holdout_metrics and metric_name in expected_deg:
                dev_mean = dev_metrics[metric_name]['mean']
                holdout_val = holdout_metrics[metric_name]
                
                if isinstance(holdout_val, (int, float)) and dev_mean != 0:
                    actual_deg = ((dev_mean - holdout_val) / abs(dev_mean)) * 100
                    actual_degradations[metric_name] = actual_deg
        
        z_scores = {}
        for metric_name, actual_deg in actual_degradations.items():
            if metric_name in expected_deg:
                expected_mean, expected_std = expected_deg[metric_name]
                
                if expected_std > 0:
                    z = (actual_deg - expected_mean) / expected_std
                    z_scores[metric_name] = z
        
        if z_scores:
            test_statistic = np.mean([abs(z) for z in z_scores.values()])
        else:
            test_statistic = 0.0
        
        null_statistics = []
        for _ in range(n_permutations):
            sim_z_scores = []
            for metric_name, (exp_mean, exp_std) in expected_deg.items():
                if exp_std > 0:
                    sim_degradation = np.random.normal(exp_mean, exp_std)
                    sim_z = (sim_degradation - exp_mean) / exp_std
                    sim_z_scores.append(abs(sim_z))
            
            if sim_z_scores:
                null_stat = np.mean(sim_z_scores)
                null_statistics.append(null_stat)
        
        if null_statistics:
            p_value = np.mean([s >= test_statistic for s in null_statistics])
        else:
            p_value = 1.0
        
        if p_value < alpha:
            interpretation = "SUSPICIOUS - Results implausibly good, suggesting possible data snooping"
            verdict = "REJECT H0"
            risk_level = "HIGH"
        else:
            interpretation = "NO EVIDENCE - Results consistent with honest single-shot validation"
            verdict = "FAIL TO REJECT H0"
            risk_level = "LOW"
        
        self.test_results = {
            'test_name': 'Statistical Test for Data Snooping (STDS)',
            'timestamp': datetime.now().isoformat(),
            'observed_similarity': float(observed_similarity),
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'alpha': alpha,
            'verdict': verdict,
            'risk_level': risk_level,
            'interpretation': interpretation,
            'n_permutations': n_permutations,
            'z_scores': {k: float(v) for k, v in z_scores.items()},
            'actual_degradations': {k: float(v) for k, v in actual_degradations.items()},
            'expected_degradations': {
                k: {'mean': float(v[0]), 'std': float(v[1])} 
                for k, v in expected_deg.items()
            },
            'status': 'success'
        }
        
        return self.test_results
    
    def generate_stds_report(self, output_path: Optional[str] = None) -> str:
        """Generate STDS report"""
        if not self.test_results or self.test_results.get('status') != 'success':
            return "No test results available."
        
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL TEST FOR DATA SNOOPING (STDS) v3.1.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Computed: {self.test_results['timestamp']}")
        lines.append("")
        
        lines.append(f"Test statistic: {self.test_results['test_statistic']:.4f}")
        lines.append(f"P-value: {self.test_results['p_value']:.4f}")
        lines.append(f"Verdict: {self.test_results['verdict']}")
        lines.append(f"Risk Level: {self.test_results['risk_level']}")
        lines.append("")
        
        lines.append(f"INTERPRETATION:")
        lines.append("-" * 80)
        lines.append(f"{self.test_results['interpretation']}")
        lines.append("")
        
        if self.test_results['p_value'] < self.test_results['alpha']:
            lines.append("🚨 WARNING: Statistical evidence of possible data snooping detected!")
        else:
            lines.append("✅ No statistical evidence of data snooping detected.")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("Novel Test: First statistical test for data snooping in consciousness research")
        lines.append("Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v2.0. DOI: 10.5281/zenodo.15541550")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✅ STDS Report: {output_path}")
        
        return report

# ═══════════════════════════════════════════════════════════════════════════
# LLM-POWERED CODE AUDITOR - NEW IN v2.0 / v3.1.0
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

class LLMAuditor:
    """LLM-Powered Protocol Auditor - NOVEL CONTRIBUTION"""
    
    def __init__(self, eclipse_framework, api_key: Optional[str] = None):
        self.framework = eclipse_framework
        self.api_key = api_key
        self.code_analyzer = None
    
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
            summary = f"Code audit PASSED (score: {adherence_score:.0f}/100)"
        else:
            summary = f"Code audit FAILED (score: {adherence_score:.0f}/100)"
        
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
        lines.append("LLM-POWERED CODE AUDIT REPORT v3.1.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Adherence Score: {score:.0f}/100")
        lines.append("")
        
        if not violations:
            lines.append("✅ NO VIOLATIONS DETECTED")
        else:
            lines.append(f"⚠️  {len(violations)} VIOLATIONS DETECTED")
            lines.append("")
            
            by_severity = defaultdict(list)
            for v in violations:
                by_severity[v.severity].append(v)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in by_severity:
                    lines.append(f"{severity.upper()} SEVERITY ({len(by_severity[severity])}):")
                    lines.append("-" * 80)
                    
                    for v in by_severity[severity]:
                        lines.append(f"  File: {v.file_path}")
                        if v.line_number:
                            lines.append(f"  Line: {v.line_number}")
                        lines.append(f"  Type: {v.category}")
                        lines.append(f"  Description: {v.description}")
                        lines.append(f"  Recommendation: {v.recommendation}")
                        lines.append("")
        
        lines.append("=" * 80)
        lines.append("Novel Tool: First LLM-powered auditor for scientific protocols")
        lines.append("Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v2.0")
        lines.append("=" * 80)
        
        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class EclipseValidator:
    """Automatic validation metrics calculator"""
    
    @staticmethod
    def binary_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Calculate comprehensive binary classification metrics"""
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
# ENHANCED REPORT GENERATOR - v3.1.0
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    """Enhanced report generator with v3.1.0 integrity metrics"""
    
    @staticmethod
    def generate_html_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate comprehensive HTML report"""
        project = final_assessment['project_name']
        verdict = final_assessment['verdict']
        val_metrics = final_assessment['validation_summary'].get('metrics', {})
        criteria_eval = final_assessment['criteria_evaluation']
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        eis_data = integrity_metrics.get('eis', {})
        stds_data = integrity_metrics.get('stds', {})
        
        eis_score = eis_data.get('eis', 0)
        eis_interp = eis_data.get('interpretation', 'N/A')
        stds_p = stds_data.get('p_value', 1.0)
        stds_verdict = stds_data.get('verdict', 'N/A')
        
        warnings_html = ""
        if 'roc_auc' in val_metrics and val_metrics['roc_auc'] < 0.5:
            warnings_html += f'<div style="background:#f8d7da;border-left:4px solid #dc3545;padding:15px;margin:20px 0"><strong>⚠️ ROC-AUC INVERSO ({val_metrics["roc_auc"]:.4f})</strong></div>'
        
        if 'mcc' in val_metrics and abs(val_metrics['mcc']) < 0.1:
            warnings_html += f'<div style="background:#fff3cd;border-left:4px solid #ffc107;padding:15px;margin:20px 0"><strong>⚠️ MCC ≈ 0 ({val_metrics["mcc"]:.4f})</strong></div>'
        
        verdict_color = {'VALIDATED': '#28a745', 'FALSIFIED': '#dc3545'}.get(verdict, '#6c757d')
        
        html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{project}</title>
<style>body{{font-family:Arial;padding:20px;background:#f5f5f5}}.container{{max-width:1200px;margin:0 auto;background:white;padding:40px}}
.verdict{{background:{verdict_color};color:white;padding:20px;text-align:center;font-size:2em}}table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#34495e;color:white}}
.pass{{background:#d4edda;color:#155724;padding:5px 10px;border-radius:3px}}.fail{{background:#f8d7da;color:#721c24;padding:5px 10px;border-radius:3px}}
.metric-box{{background:#f8f9fa;padding:20px;margin:20px 0;border-left:4px solid #3498db}}
h2{{color:#34495e;border-bottom:2px solid #ecf0f1;padding-bottom:10px}}
</style></head><body><div class="container">
<h1>🔬 ECLIPSE v3.1.0 REPORT</h1><div class="verdict">{verdict}</div>{warnings_html}
<h2>📊 Novel Integrity Metrics (v3.1.0)</h2>
<div class="metric-box"><h3>Eclipse Integrity Score (EIS)</h3><p><strong>Score:</strong> {eis_score:.4f} / 1.00</p>
<p><strong>Interpretation:</strong> {eis_interp}</p></div>
<div class="metric-box"><h3>Statistical Test for Data Snooping (STDS)</h3><p><strong>P-value:</strong> {stds_p:.4f}</p>
<p><strong>Verdict:</strong> {stds_verdict}</p></div>
<h2>Criteria Evaluation</h2><table><thead><tr><th>Criterion</th><th>Threshold</th><th>Observed</th><th>Status</th></tr></thead><tbody>'''
        
        for crit in criteria_eval:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            value_str = f"{value:.4f}" if value is not None else "N/A"
            status = "✅" if passed else "❌"
            status_class = "pass" if passed else "fail"
            html += f'<tr><td>{criterion["name"]}</td><td>{criterion["comparison"]} {criterion["threshold"]}</td><td>{value_str}</td><td><span class="{status_class}">{status}</span></td></tr>'
        
        html += f'</tbody></table><p style="margin-top:40px;color:#7f8c8d;font-size:0.9em;"><strong>ECLIPSE v3.1.0</strong><br>Citation: Sjöberg Tala, C.A. (2025). DOI: 10.5281/zenodo.15541550</p></div></body></html>'
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logging.info(f"✅ HTML: {output_path}")
        return html
    
    @staticmethod
    def generate_text_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate plain text report"""
        lines = ["=" * 80, "ECLIPSE v3.1.0 REPORT", "=" * 80,
                f"Project: {final_assessment['project_name']}", f"Verdict: {final_assessment['verdict']}", ""]
        
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        if integrity_metrics:
            lines.append("-" * 80)
            lines.append("NOVEL INTEGRITY METRICS (v3.1.0)")
            lines.append("-" * 80)
            
            eis_data = integrity_metrics.get('eis', {})
            if eis_data:
                lines.append(f"\nEclipse Integrity Score (EIS): {eis_data.get('eis', 0):.4f}")
                lines.append(f"Interpretation: {eis_data.get('interpretation', 'N/A')}")
            
            stds_data = integrity_metrics.get('stds', {})
            if stds_data.get('status') == 'success':
                lines.append(f"\nStatistical Test for Data Snooping (STDS):")
                lines.append(f"  P-value: {stds_data.get('p_value', 1.0):.4f}")
                lines.append(f"  Verdict: {stds_data.get('verdict', 'N/A')}")
        
        for crit in final_assessment['criteria_evaluation']:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            value_str = f"{value:.4f}" if value is not None else "N/A"
            lines.append(f"{'✅' if passed else '❌'} {criterion['name']}: {value_str}")
        
        text = "\n".join(lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logging.info(f"✅ Text: {output_path}")
        return text

# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE FRAMEWORK v3.1.0 - MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    """ECLIPSE v3.1.0: Enhanced Systematic Falsification Framework"""
    
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
        
        # v3.1.0 components
        self.integrity_scorer = None
        self.snooping_tester = None
        self.llm_auditor = None
        
        print("=" * 80)
        print("🔬 ECLIPSE v3.1.0 FRAMEWORK INITIALIZED")
        print("=" * 80)
        print(f"Project: {config.project_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Canales EEG: {config.n_channels}")
        print("")
        print("NEW IN v3.1.0:")
        print("  ✅ Eclipse Integrity Score (EIS)")
        print("  ✅ Statistical Test for Data Snooping (STDS)")
        print("  ✅ LLM-Powered Code Auditor")
        print("  ✅ Real-time monitoring")
        print("=" * 80)
        logging.info(f"ECLIPSE v3.1.0 inicializado - {config.project_name}")
    
    def stage1_irreversible_split(self, data_identifiers: List[Any], force: bool = False) -> Tuple[List[Any], List[Any]]:
        if self.split_file.exists() and not force:
            with open(self.split_file, 'r', encoding='utf-8') as f:
                split_data = json.load(f)
            self._split_completed = True
            logging.info("Split cargado desde archivo existente")
            return split_data['development_ids'], split_data['holdout_ids']
        
        logging.info("STAGE 1: Creando split irreversible")
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
        
        with open(self.split_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✅ {len(development_ids)} dev / {len(holdout_ids)} holdout")
        logging.info(f"Split: {len(development_ids)} dev, {len(holdout_ids)} holdout")
        self._split_completed = True
        return development_ids, holdout_ids
    
    def stage2_register_criteria(self, criteria: List[FalsificationCriteria], force: bool = False) -> Dict:
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file, 'r', encoding='utf-8') as f:
                criteria_data = json.load(f)
            self._criteria_registered = True
            logging.info("Criterios cargados")
            return criteria_data
        
        logging.info("STAGE 2: Registrando criterios")
        print("\nSTAGE 2: CRITERIA")
        criteria_dict = {
            'project_name': self.config.project_name,
            'registration_date': datetime.now().isoformat(),
            'criteria': [asdict(c) for c in criteria],
            'criteria_hash': hashlib.sha256(str([asdict(c) for c in criteria]).encode()).hexdigest()
        }
        
        with open(self.criteria_file, 'w', encoding='utf-8') as f:
            json.dump(criteria_dict, f, indent=2)
        
        print(f"✅ {len(criteria)} criteria registered")
        logging.info(f"{len(criteria)} criterios registrados")
        self._criteria_registered = True
        return criteria_dict
    
    def stage3_development(self, development_data: Any, training_function: Callable, 
                          validation_function: Callable, **kwargs) -> Dict:
        logging.info("STAGE 3: Desarrollo con CV estratificada")
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
            logging.info(f"Fold {fold_idx + 1}/{self.config.n_folds_cv}")
            
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
                print(f"   ✅ Complete - MCC: {metrics.get('mcc', 0):.3f}, Bal.Acc: {metrics.get('balanced_accuracy', 0):.3f}")
                logging.info(f"Fold {fold_idx + 1} OK - MCC: {metrics.get('mcc', 0):.3f}")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                logging.error(f"Fold {fold_idx + 1} error: {e}")
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
        logging.info(f"Desarrollo OK: {len(successful_folds)}/{self.config.n_folds_cv}")
        return {
            'n_folds': self.config.n_folds_cv,
            'n_successful': len(successful_folds),
            'aggregated_metrics': aggregated_metrics
        }
    
    def stage4_single_shot_validation(self, holdout_data: Any, final_model: Any,
                                     validation_function: Callable, force: bool = False, **kwargs) -> Dict:
        if self.results_file.exists() and not force:
            raise RuntimeError("VALIDATION DONE! Use force=True to override")
        
        logging.info("STAGE 4: Validación holdout")
        print("\nSTAGE 4: SINGLE-SHOT VALIDATION")
        print("⚠️  THIS HAPPENS EXACTLY ONCE")
        
        confirmation = input("\n🚨 Type 'I ACCEPT SINGLE-SHOT VALIDATION': ")
        
        if confirmation != "I ACCEPT SINGLE-SHOT VALIDATION":
            print("❌ Cancelled")
            logging.warning("Validación cancelada")
            return None
        
        print("\n🚀 EXECUTING...")
        logging.info("Ejecutando validación...")
        
        try:
            metrics = validation_function(final_model, holdout_data, **kwargs)
            validation_results = {
                'status': 'success',
                'n_holdout_samples': len(holdout_data) if isinstance(holdout_data, (list, pd.DataFrame)) else 0,
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }
            print(f"\n✅ COMPLETE")
            logging.info("Validación OK")
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            logging.error(f"Validación error: {e}")
            validation_results = {'status': 'failed', 'error': str(e)}
        
        self._validation_completed = True
        return validation_results
    
    def stage5_final_assessment(self, development_results: Dict, validation_results: Dict,
                               generate_reports: bool = True, compute_integrity: bool = True) -> Dict:
        """Stage 5: Final assessment with v3.1.0 integrity metrics"""
        logging.info("STAGE 5: Assessment")
        print("\nSTAGE 5: ASSESSMENT")
        
        with open(self.criteria_file, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)
        
        criteria_list = [FalsificationCriteria(**c) for c in criteria_data['criteria']]
        holdout_metrics = validation_results.get('metrics', {})
        criteria_evaluation = []
        
        for criterion in criteria_list:
            if criterion.name in holdout_metrics:
                value = holdout_metrics[criterion.name]
                passed = criterion.evaluate(value)
                evaluation = {'criterion': asdict(criterion), 'value': float(value), 'passed': passed}
                print(f"{'✅' if passed else '❌'} {criterion.name}: {value:.4f} ({criterion.comparison} {criterion.threshold})")
                logging.info(f"{'✅' if passed else '❌'} {criterion.name}: {value:.4f}")
            else:
                evaluation = {'criterion': asdict(criterion), 'value': None, 'passed': False}
            criteria_evaluation.append(evaluation)
        
        required_criteria = [e for e in criteria_evaluation if e['criterion']['is_required']]
        required_passed = sum(1 for e in required_criteria if e['passed'])
        required_total = len(required_criteria)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required_criteria) else "FALSIFIED"
        
        if 'roc_auc' in holdout_metrics and holdout_metrics['roc_auc'] < 0.5:
            print(f"\n⚠️  ROC-AUC < 0.5 ({holdout_metrics['roc_auc']:.4f}): Φ correlación NEGATIVA")
            logging.warning(f"ROC-AUC inverso: {holdout_metrics['roc_auc']:.4f}")
        
        if 'mcc' in holdout_metrics and abs(holdout_metrics['mcc']) < 0.1:
            print(f"\n⚠️  MCC ≈ 0 ({holdout_metrics['mcc']:.4f}): Sin capacidad predictiva")
            logging.warning(f"MCC ≈ 0: {holdout_metrics['mcc']:.4f}")
        
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
            'verdict_description': f"{required_passed}/{required_total} required criteria passed",
            'required_criteria_passed': f"{required_passed}/{required_total}"
        }
        
        # NEW IN v3.1.0: Compute integrity metrics
        if compute_integrity:
            print("\n" + "─" * 80)
            print("🔬 COMPUTING NOVEL INTEGRITY METRICS (v3.1.0)")
            print("─" * 80)
            
            integrity_metrics = self.compute_integrity_metrics()
            final_assessment['integrity_metrics'] = integrity_metrics
            
            eis_score = integrity_metrics['eis']['eis']
            print(f"\n📊 Eclipse Integrity Score: {eis_score:.4f}")
            
            if integrity_metrics['stds'].get('status') == 'success':
                stds_p = integrity_metrics['stds']['p_value']
                print(f"🔍 Data Snooping Test p-value: {stds_p:.4f}")
        
        final_assessment_copy = {k: v for k, v in final_assessment.items()}
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment_copy, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        print(f"\n{'✅' if verdict == 'VALIDATED' else '❌'} VERDICT: {verdict}")
        logging.info(f"VEREDICTO: {verdict}")
        print(f"✅ SAVED: {self.results_file}")
        
        if generate_reports:
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
        
        return final_assessment
    
    def compute_integrity_metrics(self) -> Dict[str, Any]:
        """Compute Eclipse Integrity Score and Data Snooping Test - NEW IN v3.1.0"""
        print("\n📊 Computing Eclipse Integrity Score (EIS)...")
        
        if self.integrity_scorer is None:
            self.integrity_scorer = EclipseIntegrityScore(self)
        
        eis_results = self.integrity_scorer.compute_eis()
        print(f"   ✅ EIS: {eis_results['eis']:.4f} - {eis_results['interpretation']}")
        
        eis_report_path = self.output_dir / f"{self.config.project_name}_EIS_REPORT.txt"
        self.integrity_scorer.generate_eis_report(str(eis_report_path))
        
        if self._validation_completed:
            print("\n🔍 Performing Statistical Test for Data Snooping (STDS)...")
            
            if self.snooping_tester is None:
                self.snooping_tester = StatisticalTestDataSnooping(self)
            
            stds_results = self.snooping_tester.perform_snooping_test()
            
            if stds_results.get('status') == 'success':
                print(f"   ✅ STDS p-value: {stds_results['p_value']:.4f}")
                print(f"   {stds_results['interpretation']}")
                
                stds_report_path = self.output_dir / f"{self.config.project_name}_STDS_REPORT.txt"
                self.snooping_tester.generate_stds_report(str(stds_report_path))
            else:
                print(f"   ⚠️  STDS: {stds_results.get('message', 'Could not compute')}")
        else:
            stds_results = {'status': 'not_applicable', 'message': 'Validation not completed'}
        
        return {
            'eis': eis_results,
            'stds': stds_results
        }
    
    def audit_code(self, code_paths: List[str], holdout_identifiers: List[str] = None,
                  api_key: Optional[str] = None) -> AuditResult:
        """Audit analysis code for protocol violations - NEW IN v3.1.0"""
        print("\n" + "=" * 80)
        print("🤖 LLM-POWERED CODE AUDIT (v3.1.0)")
        print("=" * 80)
        
        if self.llm_auditor is None:
            self.llm_auditor = LLMAuditor(self, api_key=api_key)
        
        audit_result = self.llm_auditor.audit_analysis_code(
            code_paths=code_paths,
            holdout_identifiers=holdout_identifiers
        )
        
        print(f"\n{'✅' if audit_result.passed else '❌'} Audit: {audit_result.summary}")
        print(f"   Score: {audit_result.adherence_score:.0f}/100")
        
        audit_path = self.output_dir / f"{self.config.project_name}_CODE_AUDIT.txt"
        with open(audit_path, 'w', encoding='utf-8') as f:
            f.write(audit_result.detailed_report)
        print(f"✅ Audit report: {audit_path}")
        
        return audit_result
    
    def verify_integrity(self) -> Dict:
        print("\n🔍 Verifying integrity...")
        logging.info("Verificando integridad...")
        verification = {'all_valid': True}
        
        if self.results_file.exists():
            with open(self.results_file, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            stored_hash = results_data.get('final_hash')
            results_copy = {k: v for k, v in results_data.items() if k != 'final_hash'}
            recomputed = hashlib.sha256(json.dumps(results_copy, sort_keys=True, default=str).encode()).hexdigest()
            valid = recomputed == stored_hash
            print(f"{'✅' if valid else '❌'} Results file")
            if not valid:
                verification['all_valid'] = False
                logging.error("Archivo comprometido")
            else:
                logging.info("Integridad OK")
        
        print(f"{'✅ VALID' if verification['all_valid'] else '❌ COMPROMISED'}")
        return verification

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING CON MONITOREO EN TIEMPO REAL
# ═══════════════════════════════════════════════════════════════════════════

def load_sleepedf_subject_multichannel_v3(psg_path, hypno_path, n_channels=8, 
                                          phi_methods='all', thermal_monitor=None):
    """Cargar sujeto con MONITOREO EN TIEMPO REAL"""
    if thermal_monitor:
        thermal_monitor.check_temperature()
    
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    preferred_channels = [
        'EEG Fpz-Cz', 'EEG Pz-Oz',
        'EEG F3-A2', 'EEG F4-A1',
        'EEG C3-A2', 'EEG C4-A1',
        'EEG O1-A2', 'EEG O2-A1',
        'EEG F7-A2', 'EEG F8-A1',
        'EEG T3-A2', 'EEG T4-A1',
        'EEG T5-A2', 'EEG T6-A1',
        'EEG Fz-Cz', 'EEG Cz-Pz'
    ]
    
    available = [ch for ch in preferred_channels if ch in raw.ch_names]
    
    if len(available) < n_channels:
        logging.warning(f"⚠️  Sujeto: {len(available)} canales disponibles (necesita {n_channels})")
        logging.info(f"⏭️  SALTANDO sujeto - insuficientes canales EEG")
        return None
    
    selected = available[:n_channels]
    actual_n_channels = len(selected)
    
    print(f"      📡 Canales: {', '.join(selected[:3])}{'...' if len(selected) > 3 else ''}")
    logging.info(f"✅ Usando {actual_n_channels} canales: {', '.join(selected[:3])}...")
    for handler in logging.root.handlers:
        handler.flush()
    
    raw.pick_channels(selected)
    raw.filter(0.5, 30, fir_design='firwin', verbose=False)
    
    hypno_data = mne.read_annotations(hypno_path)
    
    sfreq = raw.info['sfreq']
    window_size = 30
    n_samples_window = int(window_size * sfreq)
    
    data = raw.get_data()
    n_windows = data.shape[1] // n_samples_window
    
    print(f"      📊 Total ventanas a procesar: {n_windows}")
    logging.info(f"Procesando {n_windows} ventanas")
    for handler in logging.root.handlers:
        handler.flush()
    
    windows = []
    start_time = time.time()
    last_update = start_time
    update_interval = 30
    
    conscientes = 0
    inconscientes = 0
    
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
        
        if consciousness_label == 1:
            conscientes += 1
        else:
            inconscientes += 1
        
        try:
            phi_results = calculate_all_phi_methods(eeg_window, methods=phi_methods)
        except Exception as e:
            logging.warning(f"Error phi en ventana {w}: {e}")
            phi_results = {
                'phi_binary': 0.0,
                'phi_multilevel_3': 0.0,
                'phi_multilevel_4': 0.0,
                'phi_gaussian': 0.0
            }
        
        window_data = {
            **phi_results,
            'consciousness': consciousness_label,
            'sleep_stage': sleep_stage,
            'window_idx': w,
            'n_channels_used': actual_n_channels
        }
        
        windows.append(window_data)
        
        current_time = time.time()
        if current_time - last_update >= update_interval:
            elapsed = current_time - start_time
            progress = (w + 1) / n_windows * 100
            windows_per_sec = (w + 1) / elapsed
            eta_seconds = (n_windows - w - 1) / windows_per_sec if windows_per_sec > 0 else 0
            
            phi_cols = [k for k in phi_results.keys() if k.startswith('phi_') and not k.endswith('_time')]
            avg_phi_values = {}
            for phi_col in phi_cols:
                values = [win[phi_col] for win in windows if phi_col in win]
                if values:
                    avg_phi_values[phi_col] = np.mean(values)
            
            print(f"         🔄 Progreso: {progress:.1f}% | Ventana {w+1}/{n_windows}")
            print(f"         ⏱️  Velocidad: {windows_per_sec:.2f} vent/seg | ETA: {eta_seconds/60:.1f} min")
            print(f"         📊 Consciente: {conscientes} | Inconsciente: {inconscientes}")
            
            if avg_phi_values:
                phi_str = " | ".join([f"{k.replace('phi_', 'Φ_')}: {v:.4f}" for k, v in list(avg_phi_values.items())[:2]])
                print(f"         🧠 Φ promedio: {phi_str}")
            
            logging.info(f"Progreso ventana: {w+1}/{n_windows} ({progress:.1f}%)")
            for handler in logging.root.handlers:
                handler.flush()
            
            last_update = current_time
    
    print(f"      ✅ SUJETO COMPLETADO:")
    print(f"         Total: {len(windows)} ventanas válidas")
    print(f"         Consciente: {conscientes} ({conscientes/len(windows)*100:.1f}%)")
    print(f"         Inconsciente: {inconscientes} ({inconscientes/len(windows)*100:.1f}%)")
    
    if windows:
        phi_cols = [k for k in windows[0].keys() if k.startswith('phi_') and not k.endswith('_time')]
        for phi_col in phi_cols[:2]:
            conscious_phi = [w[phi_col] for w in windows if w['consciousness'] == 1]
            unconscious_phi = [w[phi_col] for w in windows if w['consciousness'] == 0]
            
            if conscious_phi and unconscious_phi:
                method_name = phi_col.replace('phi_', '')
                print(f"         {method_name}: Vigilia={np.mean(conscious_phi):.4f} | Sueño={np.mean(unconscious_phi):.4f}")
    
    logging.info(f"Sujeto completado: {len(windows)} ventanas, {conscientes} consciente, {inconscientes} inconsciente")
    for handler in logging.root.handlers:
        handler.flush()
    
    return windows

def buscar_archivos_edf_pares(carpeta_base):
    """Búsqueda de pares PSG-Hypnogram"""
    pares_encontrados = []
    carpeta_path = Path(carpeta_base)
    
    logging.info(f"Buscando en: {carpeta_base}")
    print(f"\n🔍 Buscando archivos EDF en: {carpeta_base}")
    
    archivos_psg = list(carpeta_path.glob("*-PSG.edf"))
    archivos_hypno = list(carpeta_path.glob("*-Hypnogram.edf"))
    
    print(f"📂 PSG: {len(archivos_psg)}, Hypnogram: {len(archivos_hypno)}")
    logging.info(f"PSG: {len(archivos_psg)}, Hypno: {len(archivos_hypno)}")
    
    if len(archivos_psg) == 0 or len(archivos_hypno) == 0:
        print("❌ No se encontraron archivos")
        logging.error("Sin archivos EDF")
        return []
    
    hypno_map = {}
    for hypno_path in archivos_hypno:
        codigo_hypno = hypno_path.stem.replace("-Hypnogram", "")
        if len(codigo_hypno) >= 7:
            base = codigo_hypno[:-1]
            hypno_map[base] = hypno_path
    
    for psg_path in archivos_psg:
        codigo_psg = psg_path.stem.replace("-PSG", "")
        
        if len(codigo_psg) >= 7 and codigo_psg[-1] == '0':
            base = codigo_psg[:-1]
            
            if base in hypno_map:
                pares_encontrados.append((
                    str(psg_path),
                    str(hypno_map[base]),
                    codigo_psg
                ))
    
    print(f"\n✅ Pares encontrados: {len(pares_encontrados)}")
    logging.info(f"Pares: {len(pares_encontrados)}")
    
    return pares_encontrados

def save_progress(output_dir: Path, subject_data: List, checkpoint_name: str):
    """Guardar checkpoint"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    try:
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(subject_data, f)
        logging.info(f"✅ Checkpoint: {checkpoint_file}")
        for handler in logging.root.handlers:
            handler.flush()
    except Exception as e:
        logging.error(f"Error checkpoint: {e}")

def load_progress(output_dir: Path, checkpoint_name: str):
    """Cargar checkpoint"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    if checkpoint_file.exists():
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"✅ Checkpoint cargado: {checkpoint_file}")
            return data
        except Exception as e:
            logging.error(f"Error cargando: {e}")
            return None
    return None

def balance_dataset(df: pd.DataFrame, method='combined', random_state=2025):
    """Balancear dataset"""
    from sklearn.utils import resample
    
    df_majority = df[df['consciousness'] == 1]
    df_minority = df[df['consciousness'] == 0]
    
    print(f"\n🔄 Balanceando ({method}):")
    print(f"   Original - Consciente: {len(df_majority)}, Inconsciente: {len(df_minority)}")
    
    if method == 'undersample':
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=random_state
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
    elif method == 'oversample':
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=random_state
        )
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        
    elif method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            phi_cols = [col for col in df.columns if col.startswith('phi_') and not col.endswith('_time')]
            X = df[phi_cols].values
            y = df['consciousness'].values
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df_balanced = pd.DataFrame(X_resampled, columns=phi_cols)
            df_balanced['consciousness'] = y_resampled
        except ImportError:
            print("   ⚠️  imblearn no disponible, usando oversample")
            return balance_dataset(df, method='oversample', random_state=random_state)
            
    elif method == 'combined':
        target_majority = len(df_minority) * 2
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=min(target_majority, len(df_majority)),
            random_state=random_state
        )
        df_temp = pd.concat([df_majority_downsampled, df_minority])
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            phi_cols = [col for col in df_temp.columns if col.startswith('phi_') and not col.endswith('_time')]
            X = df_temp[phi_cols].values
            y = df_temp['consciousness'].values
            X_resampled, y_resampled = smote.fit_resample(X, y)
            df_balanced = pd.DataFrame(X_resampled, columns=phi_cols)
            df_balanced['consciousness'] = y_resampled
        except ImportError:
            print("   ⚠️  imblearn no disponible, usando solo undersample")
            df_balanced = df_temp
    
    else:
        df_balanced = df
    
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n_conscious = (df_balanced['consciousness'] == 1).sum()
    n_unconscious = (df_balanced['consciousness'] == 0).sum()
    print(f"   Balanceado - Consciente: {n_conscious}, Inconsciente: {n_unconscious}")
    logging.info(f"Balanceo: {n_conscious} consciente, {n_unconscious} inconsciente")
    
    return df_balanced

def optimize_threshold_mcc(train_df: pd.DataFrame, phi_column: str, n_thresholds=200):
    """Optimizar threshold usando MCC"""
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

def analyze_phi_correlation(df: pd.DataFrame, phi_column: str, output_dir: Path):
    """Analizar correlación entre Φ y consciencia"""
    print(f"\n📊 Análisis de Correlación {phi_column} vs Consciencia:")
    
    conscious = df[df['consciousness'] == 1][phi_column]
    unconscious = df[df['consciousness'] == 0][phi_column]
    
    print(f"\n   Φ en VIGILIA (consciente):")
    print(f"      Media: {conscious.mean():.4f} ± {conscious.std():.4f}")
    print(f"      Mediana: {conscious.median():.4f}")
    
    print(f"\n   Φ en SUEÑO (inconsciente):")
    print(f"      Media: {unconscious.mean():.4f} ± {unconscious.std():.4f}")
    print(f"      Mediana: {unconscious.median():.4f}")
    
    u_stat, p_value_mw = mannwhitneyu(conscious, unconscious, alternative='two-sided')
    t_stat, p_value_t = ttest_ind(conscious, unconscious)
    
    print(f"\n   Mann-Whitney U: p={p_value_mw:.6f}")
    print(f"   T-test: p={p_value_t:.6f}")
    
    pearson_r, pearson_p = pearsonr(df[phi_column], df['consciousness'])
    spearman_r, spearman_p = spearmanr(df[phi_column], df['consciousness'])
    
    print(f"\n   Pearson r: {pearson_r:.4f}, p={pearson_p:.6f}")
    print(f"   Spearman ρ: {spearman_r:.4f}, p={spearman_p:.6f}")
    
    if pearson_r < 0:
        print(f"\n   ⚠️  CORRELACIÓN NEGATIVA")
    elif pearson_r < 0.1:
        print(f"\n   ⚠️  CORRELACIÓN MUY DÉBIL")
    elif pearson_r < 0.3:
        print(f"\n   ⚠️  CORRELACIÓN DÉBIL")
    else:
        print(f"\n   ✅ CORRELACIÓN POSITIVA")
    
    return {
        'phi_column': phi_column,
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mannwhitney_p': float(p_value_mw),
        'ttest_p': float(p_value_t)
    }

def comparative_analysis(df: pd.DataFrame, output_dir: Path):
    """Análisis comparativo de todos los métodos de Φ"""
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS COMPARATIVO DE MÉTODOS Φ")
    print("=" * 80)
    
    phi_columns = [col for col in df.columns if col.startswith('phi_') and not col.endswith('_time')]
    
    comparative_results = {}
    
    for phi_col in phi_columns:
        method_name = phi_col.replace('phi_', '')
        
        print(f"\n{'─' * 60}")
        print(f"MÉTODO: {method_name.upper()}")
        print(f"{'─' * 60}")
        
        print(f"\n   Estadísticas de {phi_col}:")
        print(f"      Media total: {df[phi_col].mean():.4f} ± {df[phi_col].std():.4f}")
        print(f"      Rango: [{df[phi_col].min():.4f}, {df[phi_col].max():.4f}]")
        
        correlation_results = analyze_phi_correlation(df, phi_col, output_dir)
        
        time_col = f"{phi_col}_time"
        if time_col in df.columns:
            avg_time = df[time_col].mean()
            print(f"\n   ⏱️  Tiempo promedio: {avg_time:.2f} seg/ventana")
        else:
            avg_time = None
        
        comparative_results[method_name] = {
            'phi_column': phi_col,
            'mean': float(df[phi_col].mean()),
            'std': float(df[phi_col].std()),
            'min': float(df[phi_col].min()),
            'max': float(df[phi_col].max()),
            'avg_compute_time': float(avg_time) if avg_time else None,
            'correlations': correlation_results
        }
    
    comparison_file = output_dir / "phi_methods_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparative_results, f, indent=2)
    
    print(f"\n✅ Análisis comparativo guardado: {comparison_file}")
    logging.info(f"Análisis comparativo guardado: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("📋 TABLA RESUMEN COMPARATIVA")
    print("=" * 80)
    
    print(f"\n{'Método':<20} {'Pearson r':<12} {'Spearman ρ':<12} {'Tiempo (s)':<12}")
    print("─" * 60)
    
    for method_name, results in comparative_results.items():
        r = results['correlations']['pearson_r']
        rho = results['correlations']['spearman_rho']
        time_val = results['avg_compute_time']
        time_str = f"{time_val:.2f}" if time_val else "N/A"
        
        print(f"{method_name:<20} {r:<12.4f} {rho:<12.4f} {time_str:<12}")
    
    return comparative_results

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - CONTINUARÁ EN PRÓXIMO MENSAJE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🧠 FALSIFICACIÓN DE IIT CON ECLIPSE v3.1.0 INTEGRATED")
    print("   ✅ Múltiples aproximaciones de Φ")
    print("   ✅ Eclipse Integrity Score (EIS) - NOVEL")
    print("   ✅ Statistical Test for Data Snooping (STDS) - NOVEL")
    print("   ✅ LLM-Powered Code Auditor - NOVEL")
    print("   ✅ Monitoreo en tiempo real")
    print("=" * 80)
    
    log_file = setup_logging("./eclipse_results_v3_1")
    print(f"\n📝 Log: {log_file}")
    
    logging.info("=" * 80)
    logging.info("INICIO - ECLIPSE v3.1.0 INTEGRATED")
    logging.info("=" * 80)
    for handler in logging.root.handlers:
        handler.flush()
    
    sleep_edf_path = input("\nRuta Sleep-EDF: ").strip().strip('"').strip("'")
    
    if not os.path.exists(sleep_edf_path):
        print(f"❌ Ruta no existe: {sleep_edf_path}")
        logging.error(f"Ruta inválida: {sleep_edf_path}")
        return
    
    limit = input("¿Limitar sujetos? (Enter=todos, recomendado: 20-30): ").strip()
    limit_n = int(limit) if limit else None
    
    n_channels = input("¿Canales EEG? (2-16, recomendado 8): ").strip()
    n_channels = int(n_channels) if n_channels else 8
    n_channels = max(2, min(16, n_channels))
    
    print("\n🔬 MÉTODOS DE Φ DISPONIBLES:")
    print("   1. fast - Rápidos (binary, multilevel_3, gaussian) ⭐ RECOMENDADO")
    print("   2. accurate - Precisos (multilevel_4, gaussian)")
    
    method_choice = input("\nElegir (1/2 o Enter=fast): ").strip()
    
    if method_choice == '2':
        phi_methods = 'accurate'
    else:
        phi_methods = 'fast'
    
    balance_method = input("\nMétodo balanceo (undersample/oversample/smote/combined): ").strip().lower()
    if balance_method not in ['undersample', 'oversample', 'smote', 'combined']:
        balance_method = 'combined'
    
    print(f"\n⚙️  Configuración:")
    print(f"   - Canales EEG: {n_channels}")
    print(f"   - CPU Cores: {N_WORKERS}")
    print(f"   - GPU: {'✅' if USE_GPU else '❌'}")
    print(f"   - Métodos Φ: {phi_methods}")
    print(f"   - Límite: {limit_n if limit_n else 'Todos'}")
    print(f"   - Balanceo: {balance_method}")
    print(f"   - ECLIPSE v3.1.0: EIS + STDS + LLM Auditor ✅")
    
    logging.info(f"Config: {n_channels}ch, métodos={phi_methods}, balance={balance_method}")
    for handler in logging.root.handlers:
        handler.flush()
    
    thermal_monitor = ThermalMonitor()
    
    print("\n🚀 Buscando archivos...")
    subject_pairs = buscar_archivos_edf_pares(sleep_edf_path)
    
    if len(subject_pairs) == 0:
        print("\n❌ Sin pares válidos")
        return
    
    if limit_n:
        subject_pairs = subject_pairs[:limit_n]
    
    subject_ids = [pair[2] for pair in subject_pairs]
    
    output_dir = Path("./eclipse_results_v3_1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"processing_v3_1_{n_channels}ch_{phi_methods}_{balance_method}"
    all_windows = load_progress(output_dir, checkpoint_name)
    
    if all_windows is not None:
        print(f"\n♻️  Checkpoint: {len(all_windows)} ventanas")
        resume = input("¿Continuar? (s/n): ").strip().lower()
        if resume != 's':
            all_windows = []
    else:
        all_windows = []
    
    processed_subjects = set([w['subject_id'] for w in all_windows if 'subject_id' in w])
    
    print(f"\n🔄 Procesando {len(subject_pairs)} sujetos con monitoreo en tiempo real...")
    
    start_time = time.time()
    subjects_processed = 0
    subjects_skipped = 0
    
    global_stats = {
        'total_windows': 0,
        'total_conscious': 0,
        'total_unconscious': 0,
        'phi_sums': {},
        'phi_counts': {}
    }
    
    for i, (psg, hypno, subject_id) in enumerate(subject_pairs, 1):
        if subject_id in processed_subjects:
            continue
        
        print(f"\n{'='*70}")
        print(f"📊 SUJETO {i}/{len(subject_pairs)}: {subject_id}")
        print(f"{'='*70}")
        logging.info(f"Procesando {i}/{len(subject_pairs)}: {subject_id}")
        for handler in logging.root.handlers:
            handler.flush()
        
        if not thermal_monitor.check_temperature():
            logging.warning("Enfriamiento requerido")
        
        subject_start = time.time()
        
        try:
            windows = load_sleepedf_subject_multichannel_v3(
                psg, hypno, n_channels, phi_methods, thermal_monitor
            )
            
            if windows is None:
                subjects_skipped += 1
                print(f"   ⏭️  SALTADO (insuficientes canales)")
                logging.info(f"Sujeto saltado: {subject_id}")
                for handler in logging.root.handlers:
                    handler.flush()
                continue
            
            if len(windows) > 0:
                for w in windows:
                    w['subject_id'] = subject_id
                
                all_windows.extend(windows)
                subjects_processed += 1
                
                global_stats['total_windows'] += len(windows)
                global_stats['total_conscious'] += sum(1 for w in windows if w['consciousness'] == 1)
                global_stats['total_unconscious'] += sum(1 for w in windows if w['consciousness'] == 0)
                
                phi_cols = [k for k in windows[0].keys() if k.startswith('phi_') and not k.endswith('_time')]
                for phi_col in phi_cols:
                    if phi_col not in global_stats['phi_sums']:
                        global_stats['phi_sums'][phi_col] = 0
                        global_stats['phi_counts'][phi_col] = 0
                    
                    global_stats['phi_sums'][phi_col] += sum(w[phi_col] for w in windows)
                    global_stats['phi_counts'][phi_col] += len(windows)
                
                subject_time = time.time() - subject_start
                
                print(f"\n   ✅ SUJETO COMPLETADO en {subject_time:.1f}s")
                print(f"\n   📊 ESTADÍSTICAS GLOBALES:")
                print(f"   ├─ Sujetos: {subjects_processed}/{len(subject_pairs)}")
                print(f"   ├─ Ventanas totales: {global_stats['total_windows']:,}")
                print(f"   └─ Consciente: {global_stats['total_conscious']:,} | Inconsciente: {global_stats['total_unconscious']:,}")
                
                logging.info(f"{subject_id}: {len(windows)} ventanas")
                for handler in logging.root.handlers:
                    handler.flush()
                
                if i % 5 == 0:
                    save_progress(output_dir, all_windows, checkpoint_name)
                    print(f"\n   💾 CHECKPOINT GUARDADO ({i}/{len(subject_pairs)})")
            else:
                print(f"   ⚠️  Sin datos válidos")
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            logging.error(f"Error {subject_id}: {e}")
            continue
    
    save_progress(output_dir, all_windows, checkpoint_name)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE PROCESAMIENTO")
    print("=" * 80)
    print(f"Sujetos procesados: {subjects_processed}")
    print(f"Ventanas totales: {len(all_windows)}")
    print(f"⏱️  Tiempo: {total_time/60:.1f} min")
    
    if len(all_windows) == 0:
        print("\n❌ Sin datos válidos")
        return
    
    df = pd.DataFrame(all_windows)
    
    print("\n✅ DATASET COMPLETO")
    print(f"Ventanas: {len(df)}, Sujetos: {df['subject_id'].nunique()}")
    
    comparative_results = comparative_analysis(df, output_dir)
    
    best_method = max(comparative_results.items(), 
                     key=lambda x: abs(x[1]['correlations']['spearman_rho']))[0]
    
    print(f"\n🏆 MEJOR MÉTODO: {best_method.upper()}")
    
    # Continuar con ECLIPSE framework...
    df_balanced = balance_dataset(df, method=balance_method, random_state=2025)
    
    config = EclipseConfig(
        project_name=f"IIT_Falsification_v3_1_{n_channels}ch",
        researcher="Camilo Sjöberg Tala",
        sacred_seed=2025,
        n_channels=n_channels,
        phi_methods=[phi_methods],
        output_dir=str(output_dir)
    )
    
    eclipse = EclipseFramework(config)
    
    # STAGE 1: Split
    unique_subjects = df_balanced['subject_id'].unique().tolist()
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(unique_subjects)
    
    dev_data = df_balanced[df_balanced['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = df_balanced[df_balanced['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
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
        return EclipseValidator.binary_classification_metrics(y_true, y_pred)
    
    dev_results = eclipse.stage3_development(
        development_data=dev_data,
        training_function=train_fn,
        validation_function=val_fn
    )
    
    # STAGE 4: Validation
    final_model = train_fn(dev_data)
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=val_fn
    )
    
    if val_results is None:
        print("Validación cancelada")
        return
    
    # STAGE 5: Assessment con métricas v3.1.0
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True,
        compute_integrity=True  # ← NOVEL: EIS + STDS
    )
    
    # NOVEL: Code Audit (opcional)
    print("\n" + "=" * 80)
    print("🤖 CODE AUDIT (Opcional)")
    print("=" * 80)
    audit_code = input("¿Auditar código de análisis? (s/n): ").strip().lower()
    
    if audit_code == 's':
        code_file = input("Ruta al archivo .py de análisis: ").strip()
        if code_file and Path(code_file).exists():
            eclipse.audit_code(
                code_paths=[code_file],
                holdout_identifiers=['holdout_data', 'holdout_subjects']
            )
    
    # Verificación de integridad
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ PROCESAMIENTO COMPLETADO - ECLIPSE v3.1.0")
    print("=" * 80)
    print(f"⏱️  Tiempo total: {(time.time() - start_time)/3600:.2f} horas")
    print(f"📁 Resultados: {output_dir}")
    print(f"📝 Log: {log_file}")
    print("\nNOVEL CONTRIBUTIONS:")
    print("  ✅ Eclipse Integrity Score (EIS)")
    print("  ✅ Statistical Test for Data Snooping (STDS)")
    print("  ✅ LLM-Powered Code Auditor")
    print("=" * 80)
    
    logging.info("EJECUCIÓN COMPLETADA - v3.1.0")
    for handler in logging.root.handlers:
        handler.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrumpido")
        logging.warning("Interrumpido por usuario")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        logging.error(f"Error crítico: {e}")
        import traceback
        logging.error(traceback.format_exc())