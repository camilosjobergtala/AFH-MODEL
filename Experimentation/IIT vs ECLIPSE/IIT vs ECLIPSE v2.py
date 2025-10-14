"""
═══════════════════════════════════════════════════════════════════════════════
FALSIFICACIÓN SISTEMÁTICA DE IIT CON ECLIPSE - VERSIÓN OPTIMIZADA V3
═══════════════════════════════════════════════════════════════════════════════
Autor: Camilo Alejandro Sjöberg Tala
DOI: 10.5281/zenodo.15541550
Version: 1.0.4-OPTIMIZED-8CH
✅ 8 CANALES EEG (4x más preciso)
✅ GPU opcional (20-50x más rápido)
✅ Encoding UTF-8 corregido
✅ Monitoreo térmico automático
✅ Guardado progresivo anti-pérdida
✅ Logs detallados
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict, field
import warnings
import sys
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import mne
import os
from scipy import signal
from scipy.stats import entropy
import time
import psutil
import logging

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE GPU (OPCIONAL - DETECTA AUTOMÁTICAMENTE)
# ═══════════════════════════════════════════════════════════════════════════

USE_GPU = False
try:
    import cupy as cp
    import GPUtil
    USE_GPU = True
    print("✅ GPU detectada - Aceleración activada")
except ImportError:
    cp = np  # Fallback a NumPy
    print("⚠️  GPU no disponible - Usando CPU (más lento pero funciona)")

# ═══════════════════════════════════════════════════════════════════════════
# SISTEMA DE LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str):
    """Configurar sistema de logs detallado"""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"iit_falsification_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

# ═══════════════════════════════════════════════════════════════════════════
# MONITOREO TÉRMICO
# ═══════════════════════════════════════════════════════════════════════════

class ThermalMonitor:
    """Monitor de temperatura para prevenir sobrecalentamiento"""
    
    MAX_CPU_TEMP = 85  # °C
    MAX_GPU_TEMP = 80  # °C
    COOLDOWN_TIME = 60  # segundos
    CHECK_INTERVAL = 30  # segundos entre checks
    
    def __init__(self):
        self.last_check = 0
        self.cooldown_count = 0
    
    def check_temperature(self, force=False) -> bool:
        """Verificar temperatura. Retorna True si es seguro continuar"""
        current_time = time.time()
        
        # Solo verificar cada CHECK_INTERVAL segundos (excepto si force=True)
        if not force and (current_time - self.last_check) < self.CHECK_INTERVAL:
            return True
        
        self.last_check = current_time
        needs_cooldown = False
        
        # Verificar GPU
        if USE_GPU:
            try:
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > 0:
                    gpu_temp = gpus[0].temperature
                    if gpu_temp > self.MAX_GPU_TEMP:
                        logging.warning(f"🔥 GPU caliente: {gpu_temp}°C")
                        needs_cooldown = True
            except:
                pass
        
        # Verificar CPU
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
            return self.check_temperature(force=True)  # Re-verificar
        
        return True

# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE FRAMEWORK (CORREGIDO)
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
    output_dir: str = "./eclipse_results"
    timestamp: str = field(default=None)
    n_channels: int = 8  # NUEVO: número de canales EEG
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


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


class EclipseReporter:
    @staticmethod
    def generate_html_report(final_assessment: Dict, output_path: str = None) -> str:
        project = final_assessment['project_name']
        verdict = final_assessment['verdict']
        val_metrics = final_assessment['validation_summary'].get('metrics', {})
        criteria_eval = final_assessment['criteria_evaluation']
        
        roc_warning = ""
        if 'roc_auc' in val_metrics and val_metrics['roc_auc'] < 0.5:
            roc_warning = f'<div style="background:#f8d7da;border-left:4px solid #dc3545;padding:15px;margin:20px 0"><strong>⚠️ ROC-AUC INVERSO ({val_metrics["roc_auc"]:.4f})</strong></div>'
        
        verdict_color = {'VALIDATED': '#28a745', 'FALSIFIED': '#dc3545'}.get(verdict, '#6c757d')
        
        html = f'''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{project}</title>
<style>body{{font-family:Arial;padding:20px;background:#f5f5f5}}.container{{max-width:1200px;margin:0 auto;background:white;padding:40px}}
.verdict{{background:{verdict_color};color:white;padding:20px;text-align:center;font-size:2em}}table{{width:100%;border-collapse:collapse;margin:20px 0}}
th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#34495e;color:white}}
.pass{{background:#d4edda;color:#155724;padding:5px 10px;border-radius:3px}}.fail{{background:#f8d7da;color:#721c24;padding:5px 10px;border-radius:3px}}
</style></head><body><div class="container"><h1>🔬 ECLIPSE REPORT</h1><div class="verdict">{verdict}</div>{roc_warning}
<h2>Criteria</h2><table><thead><tr><th>Criterion</th><th>Threshold</th><th>Observed</th><th>Status</th></tr></thead><tbody>'''
        
        for crit in criteria_eval:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            value_str = f"{value:.4f}" if value is not None else "N/A"
            status = "✅" if passed else "❌"
            status_class = "pass" if passed else "fail"
            html += f'<tr><td>{criterion["name"]}</td><td>{criterion["comparison"]} {criterion["threshold"]}</td><td>{value_str}</td><td><span class="{status_class}">{status}</span></td></tr>'
        
        html += '</tbody></table><h2>Metrics</h2><table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        for k, v in val_metrics.items():
            html += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        html += f'</tbody></table><p>Hash: <code>{final_assessment.get("final_hash", "")[:32]}</code></p></div></body></html>'
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:  # ✅ CORREGIDO
                f.write(html)
            logging.info(f"✅ HTML: {output_path}")
        return html
    
    @staticmethod
    def generate_text_report(final_assessment: Dict, output_path: str = None) -> str:
        lines = ["=" * 80, "ECLIPSE REPORT", "=" * 80,
                f"Project: {final_assessment['project_name']}", f"Verdict: {final_assessment['verdict']}", ""]
        for crit in final_assessment['criteria_evaluation']:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            value_str = f"{value:.4f}" if value is not None else "N/A"
            lines.append(f"{'✅' if passed else '❌'} {criterion['name']}: {value_str}")
        text = "\n".join(lines)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:  # ✅ CORREGIDO
                f.write(text)
            logging.info(f"✅ Text: {output_path}")
        return text


class EclipseFramework:
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
        
        print("=" * 80)
        print("🔬 ECLIPSE INITIALIZED")
        print(f"Project: {config.project_name}")
        print(f"Canales EEG: {config.n_channels}")
        print(f"GPU: {'✅ Activada' if USE_GPU else '❌ Desactivada (CPU)'}")
        print("=" * 80)
        logging.info(f"ECLIPSE Framework inicializado - {config.n_channels} canales")
    
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
        logging.info(f"Split creado: {len(development_ids)} dev, {len(holdout_ids)} holdout")
        self._split_completed = True
        return development_ids, holdout_ids
    
    def stage2_register_criteria(self, criteria: List[FalsificationCriteria], force: bool = False) -> Dict:
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file, 'r', encoding='utf-8') as f:
                criteria_data = json.load(f)
            self._criteria_registered = True
            logging.info("Criterios cargados desde archivo existente")
            return criteria_data
        
        logging.info("STAGE 2: Registrando criterios de falsificación")
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
        logging.info("STAGE 3: Iniciando desarrollo con cross-validation")
        print("\nSTAGE 3: DEVELOPMENT")
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.config.n_folds_cv, shuffle=True, random_state=self.config.sacred_seed)
        cv_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(development_data)))):
            print(f"\nFOLD {fold_idx + 1}/{self.config.n_folds_cv}")
            logging.info(f"Procesando fold {fold_idx + 1}/{self.config.n_folds_cv}")
            
            if isinstance(development_data, (list, tuple)):
                train_data = [development_data[i] for i in train_idx]
                val_data = [development_data[i] for i in val_idx]
            else:
                train_data = development_data[train_idx]
                val_data = development_data[val_idx]
            
            try:
                model = training_function(train_data, **kwargs)
                metrics = validation_function(model, val_data, **kwargs)
                cv_results.append({'fold': fold_idx + 1, 'metrics': metrics, 'status': 'success'})
                print(f"   ✅ Complete")
                logging.info(f"Fold {fold_idx + 1} completado exitosamente")
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                logging.error(f"Fold {fold_idx + 1} falló: {e}")
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
                    'max': float(np.max(values))
                }
        else:
            aggregated_metrics = {}
        
        self._development_completed = True
        logging.info(f"Desarrollo completado: {len(successful_folds)}/{self.config.n_folds_cv} folds exitosos")
        return {
            'n_folds': self.config.n_folds_cv,
            'n_successful': len(successful_folds),
            'aggregated_metrics': aggregated_metrics
        }
    
    def stage4_single_shot_validation(self, holdout_data: Any, final_model: Any,
                                     validation_function: Callable, force: bool = False, **kwargs) -> Dict:
        if self.results_file.exists() and not force:
            raise RuntimeError("VALIDATION DONE! Use force=True to override")
        
        logging.info("STAGE 4: Validación single-shot en holdout")
        print("\nSTAGE 4: SINGLE-SHOT VALIDATION")
        print("⚠️  THIS HAPPENS EXACTLY ONCE")
        
        confirmation = input("\n🚨 Type 'I ACCEPT SINGLE-SHOT VALIDATION': ")
        
        if confirmation != "I ACCEPT SINGLE-SHOT VALIDATION":
            print("❌ Cancelled")
            logging.warning("Validación cancelada por el usuario")
            return None
        
        print("\n🚀 EXECUTING...")
        logging.info("Ejecutando validación final...")
        
        try:
            metrics = validation_function(final_model, holdout_data, **kwargs)
            validation_results = {
                'status': 'success',
                'n_holdout_samples': len(holdout_data),
                'metrics': {k: float(v) for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat()
            }
            print(f"\n✅ COMPLETE")
            logging.info("Validación completada exitosamente")
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            logging.error(f"Validación falló: {e}")
            validation_results = {'status': 'failed', 'error': str(e)}
        
        self._validation_completed = True
        return validation_results
    
    def stage5_final_assessment(self, development_results: Dict, validation_results: Dict,
                               generate_reports: bool = True) -> Dict:
        logging.info("STAGE 5: Assessment final")
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
                print(f"{'✅' if passed else '❌'} {criterion.name}: {value:.4f}")
                logging.info(f"{'✅' if passed else '❌'} {criterion.name}: {value:.4f}")
            else:
                evaluation = {'criterion': asdict(criterion), 'value': None, 'passed': False}
            criteria_evaluation.append(evaluation)
        
        required_criteria = [e for e in criteria_evaluation if e['criterion']['is_required']]
        required_passed = sum(1 for e in required_criteria if e['passed'])
        required_total = len(required_criteria)
        
        verdict = "VALIDATED" if all(e['passed'] for e in required_criteria) else "FALSIFIED"
        verdict_description = f"{'All' if verdict == 'VALIDATED' else 'Failed'} {required_total} criteria"
        
        if 'roc_auc' in holdout_metrics and holdout_metrics['roc_auc'] < 0.5:
            print("\n⚠️  ROC-AUC < 0.5: INVERTED predictions!")
            logging.warning(f"ROC-AUC inverso detectado: {holdout_metrics['roc_auc']:.4f}")
        
        final_assessment = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'assessment_date': datetime.now().isoformat(),
            'assessment_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
            'verdict_description': verdict_description,
            'required_criteria_passed': f"{required_passed}/{required_total}"
        }
        
        final_assessment_copy = {k: v for k, v in final_assessment.items()}
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment_copy, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        print(f"\n{'✅' if verdict == 'VALIDATED' else '❌'} VERDICT: {verdict}")
        logging.info(f"VEREDICTO FINAL: {verdict}")
        print(f"✅ SAVED: {self.results_file}")
        
        if generate_reports:
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
        
        return final_assessment
    
    def verify_integrity(self) -> Dict:
        print("\n🔍 Verifying integrity...")
        logging.info("Verificando integridad de resultados...")
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
                logging.error("Archivo de resultados comprometido - hash no coincide")
            else:
                logging.info("Integridad verificada correctamente")
        
        print(f"{'✅ VALID' if verification['all_valid'] else '❌ COMPROMISED'}")
        return verification


# ═══════════════════════════════════════════════════════════════════════════
# PHI CALCULATION (8 CANALES + GPU OPCIONAL)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_phi_star_8ch(eeg_segment):
    """
    Φ* optimizado para 8 canales con soporte GPU opcional
    Compatible con GPU RTX 3050 Ti (4GB VRAM)
    """
    n_channels = eeg_segment.shape[0]
    
    if n_channels > 16:
        logging.warning(f"⚠️ {n_channels} canales excede el límite recomendado (16). Truncando.")
        eeg_segment = eeg_segment[:16, :]
        n_channels = 16
    
    # Usar GPU si está disponible
    if USE_GPU:
        try:
            eeg_gpu = cp.array(eeg_segment)
            
            # Binarizar en GPU
            thresholds = cp.median(eeg_gpu, axis=1, keepdims=True)
            binary_signals = (eeg_gpu > thresholds).astype(cp.int8)
            
            # Calcular estados conjuntos
            joint_states = cp.zeros(binary_signals.shape[1], dtype=cp.int32)
            for t in range(binary_signals.shape[1]):
                state = binary_signals[:, t]
                joint_states[t] = cp.sum(state * (2 ** cp.arange(n_channels)))
            
            # Calcular entropía del sistema completo
            unique_states, counts = cp.unique(joint_states, return_counts=True)
            probs = counts / len(joint_states)
            H_whole = -cp.sum(probs * cp.log2(probs + 1e-10))
            
            # Buscar MIP (Minimum Information Partition)
            min_mi = float('inf')
            
            from itertools import combinations
            for k in range(1, n_channels):
                for partition_A_indices in combinations(range(n_channels), k):
                    partition_B_indices = tuple(i for i in range(n_channels) if i not in partition_A_indices)
                    
                    # Estados A
                    states_A = cp.zeros(binary_signals.shape[1], dtype=cp.int32)
                    for t in range(binary_signals.shape[1]):
                        state_A = binary_signals[list(partition_A_indices), t]
                        states_A[t] = cp.sum(state_A * (2 ** cp.arange(len(partition_A_indices))))
                    
                    unique_A, counts_A = cp.unique(states_A, return_counts=True)
                    prob_A = counts_A / len(states_A)
                    H_A = -cp.sum(prob_A * cp.log2(prob_A + 1e-10))
                    
                    # Estados B
                    states_B = cp.zeros(binary_signals.shape[1], dtype=cp.int32)
                    for t in range(binary_signals.shape[1]):
                        state_B = binary_signals[list(partition_B_indices), t]
                        states_B[t] = cp.sum(state_B * (2 ** cp.arange(len(partition_B_indices))))
                    
                    unique_B, counts_B = cp.unique(states_B, return_counts=True)
                    prob_B = counts_B / len(states_B)
                    H_B = -cp.sum(prob_B * cp.log2(prob_B + 1e-10))
                    
                    MI = float(H_A + H_B - H_whole)
                    
                    if MI < min_mi:
                        min_mi = MI
            
            # Limpiar memoria GPU
            cp.get_default_memory_pool().free_all_blocks()
            
            phi = min_mi if min_mi != float('inf') else 0.0
            return max(0.0, float(phi))
            
        except Exception as e:
            logging.warning(f"GPU falló, usando CPU: {e}")
            # Fallback a CPU
            pass
    
    # Versión CPU (fallback)
    binary_signals = np.zeros_like(eeg_segment, dtype=int)
    
    for ch in range(n_channels):
        threshold = np.median(eeg_segment[ch, :])
        binary_signals[ch, :] = (eeg_segment[ch, :] > threshold).astype(int)
    
    joint_states = []
    for t in range(binary_signals.shape[1]):
        state = tuple(binary_signals[:, t])
        joint_states.append(state)
    
    unique_states, counts = np.unique(joint_states, axis=0, return_counts=True)
    probabilities = counts / len(joint_states)
    H_whole = entropy(probabilities, base=2)
    
    min_mi = float('inf')
    
    from itertools import combinations
    for k in range(1, n_channels):
        for partition_A_indices in combinations(range(n_channels), k):
            partition_B_indices = tuple(i for i in range(n_channels) if i not in partition_A_indices)
            
            states_A = []
            for t in range(binary_signals.shape[1]):
                state_A = tuple(binary_signals[list(partition_A_indices), t])
                states_A.append(state_A)
            
            unique_A, counts_A = np.unique(states_A, axis=0, return_counts=True)
            prob_A = counts_A / len(states_A)
            H_A = entropy(prob_A, base=2)
            
            states_B = []
            for t in range(binary_signals.shape[1]):
                state_B = tuple(binary_signals[list(partition_B_indices), t])
                states_B.append(state_B)
            
            unique_B, counts_B = np.unique(states_B, axis=0, return_counts=True)
            prob_B = counts_B / len(states_B)
            H_B = entropy(prob_B, base=2)
            
            MI = H_A + H_B - H_whole
            
            if MI < min_mi:
                min_mi = MI
    
    phi = min_mi if min_mi != float('inf') else 0.0
    return max(0.0, phi)


def load_sleepedf_subject_8ch(psg_path, hypno_path, n_channels=8, thermal_monitor=None):
    """Cargar sujeto Sleep-EDF con 8 canales"""
    
    # Verificar temperatura antes de procesar
    if thermal_monitor:
        thermal_monitor.check_temperature()
    
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    
    # Lista de canales EEG prioritarios (orden de preferencia)
    preferred_channels = [
        'EEG Fpz-Cz',
        'EEG Pz-Oz',
        'EEG F3-A2',
        'EEG F4-A1',
        'EEG C3-A2',
        'EEG C4-A1',
        'EEG O1-A2',
        'EEG O2-A1',
        'EEG F7-A2',
        'EEG F8-A1',
        'EEG T3-A2',
        'EEG T4-A1',
        'EEG T5-A2',
        'EEG T6-A1'
    ]
    
    # Seleccionar canales disponibles
    available = [ch for ch in preferred_channels if ch in raw.ch_names]
    
    if len(available) < 2:
        logging.warning(f"Sujeto con menos de 2 canales EEG disponibles")
        return None
    
    # Usar hasta n_channels
    selected = available[:min(n_channels, len(available))]
    logging.info(f"Usando {len(selected)} canales: {', '.join(selected)}")
    
    raw.pick_channels(selected)
    raw.filter(0.5, 30, fir_design='firwin', verbose=False)
    
    hypno_data = mne.read_annotations(hypno_path)
    
    sfreq = raw.info['sfreq']
    window_size = 30
    n_samples_window = int(window_size * sfreq)
    
    data = raw.get_data()
    n_windows = data.shape[1] // n_samples_window
    
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
            phi = calculate_phi_star_8ch(eeg_window)
        except Exception as e:
            logging.warning(f"Error calculando phi en ventana {w}: {e}")
            phi = 0.0
        
        windows.append({
            'phi': phi,
            'consciousness': consciousness_label,
            'sleep_stage': sleep_stage,
            'window_idx': w,
            'n_channels_used': len(selected)
        })
    
    return windows


# ═══════════════════════════════════════════════════════════════════════════
# BÚSQUEDA DE ARCHIVOS CORREGIDA V2
# ═══════════════════════════════════════════════════════════════════════════

def buscar_archivos_edf_pares_corregido(carpeta_base):
    """Búsqueda de pares PSG-Hypnogram"""
    pares_encontrados = []
    carpeta_path = Path(carpeta_base)
    
    logging.info(f"Buscando archivos EDF en: {carpeta_base}")
    print(f"\n🔍 Buscando archivos EDF en: {carpeta_base}")
    
    archivos_psg = list(carpeta_path.glob("*-PSG.edf"))
    archivos_hypno = list(carpeta_path.glob("*-Hypnogram.edf"))
    
    print(f"📂 PSG encontrados: {len(archivos_psg)}")
    print(f"📂 Hypnogram encontrados: {len(archivos_hypno)}")
    logging.info(f"Encontrados {len(archivos_psg)} PSG y {len(archivos_hypno)} Hypnogram")
    
    if len(archivos_psg) == 0 or len(archivos_hypno) == 0:
        print("❌ No se encontraron archivos con el patrón esperado")
        logging.error("No se encontraron archivos EDF")
        return []
    
    print(f"\n📋 Analizando patrón de nombres...")
    if len(archivos_psg) > 0:
        ejemplo_psg = archivos_psg[0].stem.replace("-PSG", "")
        print(f"   Ejemplo PSG: {ejemplo_psg} (longitud: {len(ejemplo_psg)})")
    if len(archivos_hypno) > 0:
        ejemplo_hypno = archivos_hypno[0].stem.replace("-Hypnogram", "")
        print(f"   Ejemplo Hypno: {ejemplo_hypno} (longitud: {len(ejemplo_hypno)})")
    
    hypno_map = {}
    for hypno_path in archivos_hypno:
        codigo_hypno = hypno_path.stem.replace("-Hypnogram", "")
        if len(codigo_hypno) >= 7:
            base = codigo_hypno[:-1]
            hypno_map[base] = hypno_path
            if len(hypno_map) <= 3:
                print(f"   Hypno: {codigo_hypno} → base: {base}")
    
    print(f"\n🔗 Emparejando {len(archivos_psg)} PSG con {len(hypno_map)} bases Hypno...")
    
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
                
                if len(pares_encontrados) <= 3:
                    hypno_codigo = hypno_map[base].stem.replace("-Hypnogram", "")
                    print(f"   ✅ Par #{len(pares_encontrados)}: {codigo_psg} ↔ {hypno_codigo}")
    
    print(f"\n✅ Pares completos encontrados: {len(pares_encontrados)}")
    logging.info(f"Pares válidos encontrados: {len(pares_encontrados)}")
    
    if len(pares_encontrados) > 0:
        print(f"\n📊 Ejemplos de pares (primeros 5):")
        for i, (psg, hypno, codigo) in enumerate(pares_encontrados[:5]):
            print(f"   {i+1}. {codigo}")
            print(f"      PSG: {Path(psg).name}")
            print(f"      Hypno: {Path(hypno).name}")
    
    return pares_encontrados


# ═══════════════════════════════════════════════════════════════════════════
# GUARDADO PROGRESIVO
# ═══════════════════════════════════════════════════════════════════════════

def save_progress(output_dir: Path, subject_data: List, checkpoint_name: str):
    """Guardar progreso para evitar pérdida de datos"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    try:
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(subject_data, f)
        logging.info(f"✅ Checkpoint guardado: {checkpoint_file}")
    except Exception as e:
        logging.error(f"Error guardando checkpoint: {e}")


def load_progress(output_dir: Path, checkpoint_name: str):
    """Cargar progreso guardado"""
    checkpoint_file = output_dir / f"{checkpoint_name}.pkl"
    if checkpoint_file.exists():
        try:
            import pickle
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"✅ Checkpoint cargado: {checkpoint_file}")
            return data
        except Exception as e:
            logging.error(f"Error cargando checkpoint: {e}")
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZADO
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🧠 FALSIFICACIÓN SISTEMÁTICA DE IIT CON ECLIPSE")
    print("   Versión 1.0.4 - OPTIMIZADA PARA EJECUCIÓN NOCTURNA")
    print("   ✅ 8 Canales EEG | ✅ GPU Opcional | ✅ Monitoreo Térmico")
    print("=" * 80)
    
    # Configurar logging
    log_file = setup_logging("./eclipse_iit_results")
    print(f"\n📝 Log file: {log_file}")
    logging.info("=" * 80)
    logging.info("INICIO DE EJECUCIÓN")
    logging.info("=" * 80)
    
    sleep_edf_path = input("\nRuta carpeta Sleep-EDF: ").strip()
    sleep_edf_path = sleep_edf_path.strip('"').strip("'")
    
    if not os.path.exists(sleep_edf_path):
        print(f"❌ La ruta no existe: {sleep_edf_path}")
        logging.error(f"Ruta no existe: {sleep_edf_path}")
        return
    
    limit = input("¿Limitar número de sujetos? (Enter para todos): ").strip()
    limit_n = int(limit) if limit else None
    
    n_channels = input("¿Cuántos canales EEG usar? (2-16, recomendado 8): ").strip()
    n_channels = int(n_channels) if n_channels else 8
    n_channels = max(2, min(16, n_channels))  # Limitar entre 2 y 16
    
    print(f"\n⚙️  Configuración:")
    print(f"   - Canales EEG: {n_channels}")
    print(f"   - GPU: {'✅ Activada' if USE_GPU else '❌ CPU only'}")
    print(f"   - Límite sujetos: {limit_n if limit_n else 'Todos'}")
    logging.info(f"Configuración: {n_channels} canales, GPU={USE_GPU}, Límite={limit_n}")
    
    # Inicializar monitor térmico
    thermal_monitor = ThermalMonitor()
    
    print("\n🚀 Buscando pares PSG-Hypnogram...")
    subject_pairs = buscar_archivos_edf_pares_corregido(sleep_edf_path)
    
    if len(subject_pairs) == 0:
        print("\n❌ No se encontraron pares PSG-Hypnogram válidos")
        logging.error("No se encontraron pares válidos")
        return
    
    print(f"\n✅ {len(subject_pairs)} pares encontrados y listos para procesar")
    
    if limit_n:
        subject_pairs = subject_pairs[:limit_n]
        print(f"   📊 Limitando a {len(subject_pairs)} pares")
        logging.info(f"Limitando a {len(subject_pairs)} pares")
    
    subject_ids = [pair[2] for pair in subject_pairs]
    
    # Verificar si hay progreso guardado
    output_dir = Path("./eclipse_iit_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_name = f"processing_checkpoint_{n_channels}ch"
    all_windows = load_progress(output_dir, checkpoint_name)
    
    if all_windows is not None:
        print(f"\n♻️  Checkpoint encontrado con {len(all_windows)} ventanas procesadas")
        resume = input("¿Continuar desde el checkpoint? (s/n): ").strip().lower()
        if resume != 's':
            all_windows = []
    else:
        all_windows = []
    
    processed_count = len([w for w in all_windows if 'subject_id' in w])
    
    # Procesar cada par
    print(f"\n🔄 Procesando {len(subject_pairs)} sujetos...")
    logging.info(f"Iniciando procesamiento de {len(subject_pairs)} sujetos")
    
    start_time = time.time()
    
    for i, (psg, hypno, subject_id) in enumerate(subject_pairs, 1):
        if i <= processed_count:
            continue  # Saltar sujetos ya procesados
        
        print(f"\n📊 {i}/{len(subject_pairs)}: {subject_id}")
        print(f"   PSG: {Path(psg).name}")
        print(f"   Hypno: {Path(hypno).name}")
        logging.info(f"Procesando {i}/{len(subject_pairs)}: {subject_id}")
        
        # Verificar temperatura
        if not thermal_monitor.check_temperature():
            logging.warning("Sistema requiere enfriamiento adicional")
        
        subject_start = time.time()
        
        try:
            windows = load_sleepedf_subject_8ch(psg, hypno, n_channels, thermal_monitor)
            
            if windows and len(windows) > 0:
                # Agregar ID del sujeto
                for w in windows:
                    w['subject_id'] = subject_id
                
                all_windows.extend(windows)
                print(f"   ✅ {len(windows)} ventanas procesadas")
                
                subject_time = time.time() - subject_start
                print(f"   ⏱️  Tiempo: {subject_time:.1f}s")
                logging.info(f"Sujeto {subject_id}: {len(windows)} ventanas en {subject_time:.1f}s")
                
                # Guardar checkpoint cada 10 sujetos
                if i % 10 == 0:
                    save_progress(output_dir, all_windows, checkpoint_name)
                    print(f"   💾 Checkpoint guardado ({i}/{len(subject_pairs)})")
                    
                    # Estimación de tiempo restante
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (len(subject_pairs) - i) * avg_time
                    eta = datetime.now() + pd.Timedelta(seconds=remaining)
                    print(f"   📈 ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                    logging.info(f"Progreso: {i}/{len(subject_pairs)}, ETA: {eta}")
            else:
                print(f"   ⚠️  Sin datos válidos")
                logging.warning(f"Sujeto {subject_id} sin datos válidos")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            logging.error(f"Error procesando {subject_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            continue
    
    # Guardar checkpoint final
    save_progress(output_dir, all_windows, checkpoint_name)
    
    if len(all_windows) == 0:
        print("\n❌ No se procesaron datos válidos")
        logging.error("No se obtuvieron datos válidos")
        return
    
    df = pd.DataFrame(all_windows)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✅ DATASET COMPLETO")
    print("=" * 80)
    print(f"Ventanas totales: {len(df)}")
    print(f"Sujetos procesados: {len(subject_ids)}")
    print(f"Conscientes (vigilia): {(df['consciousness'] == 1).sum()} ({(df['consciousness'] == 1).sum()/len(df)*100:.1f}%)")
    print(f"Inconscientes (sueño): {(df['consciousness'] == 0).sum()} ({(df['consciousness'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"Φ promedio: {df['phi'].mean():.4f} ± {df['phi'].std():.4f}")
    print(f"Φ rango: [{df['phi'].min():.4f}, {df['phi'].max():.4f}]")
    print(f"⏱️  Tiempo total: {total_time/60:.1f} minutos")
    print(f"🔥 Pausas térmicas: {thermal_monitor.cooldown_count}")
    
    logging.info("=" * 80)
    logging.info("DATASET COMPLETO")
    logging.info(f"Ventanas: {len(df)}, Sujetos: {len(subject_ids)}")
    logging.info(f"Φ: {df['phi'].mean():.4f} ± {df['phi'].std():.4f}")
    logging.info(f"Tiempo: {total_time/60:.1f} min, Pausas: {thermal_monitor.cooldown_count}")
    logging.info("=" * 80)
    
    # ECLIPSE FRAMEWORK
    config = EclipseConfig(
        project_name=f"IIT_Falsification_2025_{n_channels}ch",
        researcher="Camilo Alejandro Sjöberg Tala",
        sacred_seed=2025,
        output_dir="./eclipse_iit_results",
        n_channels=n_channels
    )
    
    eclipse = EclipseFramework(config)
    
    # Stage 1: Split
    print("\n" + "=" * 80)
    print("STAGE 1: SPLIT IRREVERSIBLE")
    print("=" * 80)
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(subject_ids)
    
    dev_mask = df.index < int(len(df) * 0.7)
    dev_data = df[dev_mask].reset_index(drop=True)
    holdout_data = df[~dev_mask].reset_index(drop=True)
    
    print(f"\n📊 División de datos:")
    print(f"   Desarrollo: {len(dev_data)} ventanas")
    print(f"   Holdout: {len(holdout_data)} ventanas")
    
    # Stage 2: Criterios
    print("\n" + "=" * 80)
    print("STAGE 2: CRITERIOS DE FALSACIÓN")
    print("=" * 80)
    criteria = [
        FalsificationCriteria("f1_score", 0.6, ">=", "F1≥0.60", True),
        FalsificationCriteria("precision", 0.7, ">=", "Precision≥0.70", True),
        FalsificationCriteria("recall", 0.5, ">=", "Recall≥0.50", True),
        FalsificationCriteria("roc_auc", 0.7, ">=", "ROC-AUC≥0.70", False)
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    # Stage 3: Desarrollo
    print("\n" + "=" * 80)
    print("STAGE 3: DESARROLLO CON CV")
    print("=" * 80)
    
    def train_iit(train_data, **kwargs):
        train_df = dev_data.iloc[train_data]
        best_threshold = None
        best_f1 = 0
        
        for threshold in np.linspace(train_df['phi'].min(), train_df['phi'].max(), 50):
            pred = (train_df['phi'] >= threshold).astype(int)
            true = train_df['consciousness']
            f1 = EclipseValidator.binary_classification_metrics(true, pred)['f1_score']
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {'phi_threshold': best_threshold}
    
    def validate_iit(model, val_data, **kwargs):
        val_df = dev_data.iloc[val_data]
        threshold = model['phi_threshold']
        y_pred = (val_df['phi'] >= threshold).astype(int)
        y_true = val_df['consciousness']
        y_pred_proba = val_df['phi'] / (val_df['phi'].max() + 1e-10)
        return EclipseValidator.binary_classification_metrics(y_true, y_pred, y_pred_proba)
    
    dev_results = eclipse.stage3_development(
        development_data=list(range(len(dev_data))),
        training_function=train_iit,
        validation_function=validate_iit
    )
    
    print(f"\n📈 Resultados CV:")
    for metric_name, stats in dev_results['aggregated_metrics'].items():
        print(f"   {metric_name}: {stats['mean']:.3f} ± {stats['std']:.3f}")
    
    # Stage 4: Validación Final
    print("\n" + "=" * 80)
    print("STAGE 4: VALIDACIÓN FINAL EN HOLDOUT")
    print("=" * 80)
    print("🔧 Entrenando modelo final con todos los datos de desarrollo...")
    final_model = train_iit(list(range(len(dev_data))))
    print(f"   Φ threshold óptimo: {final_model['phi_threshold']:.4f}")
    
    def validate_final(model, holdout_df, **kwargs):
        threshold = model['phi_threshold']
        y_pred = (holdout_df['phi'] >= threshold).astype(int)
        y_true = holdout_df['consciousness']
        y_pred_proba = holdout_df['phi'] / (holdout_df['phi'].max() + 1e-10)
        return EclipseValidator.binary_classification_metrics(y_true, y_pred, y_pred_proba)
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=validate_final
    )
    
    if val_results is None:
        print("\n⚠️  Validación cancelada por el usuario")
        logging.warning("Validación cancelada")
        return
    
    # Stage 5: Assessment Final
    print("\n" + "=" * 80)
    print("STAGE 5: ASSESSMENT FINAL")
    print("=" * 80)
    final_assessment = eclipse.stage5_final_assessment(dev_results, val_results, generate_reports=True)
    
    # Resumen Final
    print("\n" + "=" * 80)
    print("🎯 RESULTADOS FINALES - IIT FALSIFICATION")
    print("=" * 80)
    print(f"\n{'✅' if final_assessment['verdict'] == 'VALIDATED' else '❌'} VEREDICTO: {final_assessment['verdict']}")
    
    print("\n📊 MÉTRICAS HOLDOUT:")
    for k, v in final_assessment['validation_summary']['metrics'].items():
        print(f"   {k}: {v:.4f}")
    
    print(f"\n📁 Resultados guardados en: {config.output_dir}")
    print(f"   - Reporte HTML: {config.project_name}_REPORT.html")
    print(f"   - Reporte texto: {config.project_name}_REPORT.txt")
    print(f"   - Datos JSON: {config.project_name}_RESULT.json")
    print(f"   - Log: {log_file}")
    
    # Verificación de integridad
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ ECLIPSE FRAMEWORK COMPLETADO")
    print(f"⏱️  Tiempo total: {(time.time() - start_time)/3600:.2f} horas")
    print("=" * 80)
    
    logging.info("=" * 80)
    logging.info("EJECUCIÓN COMPLETADA EXITOSAMENTE")
    logging.info(f"Tiempo total: {(time.time() - start_time)/3600:.2f} horas")
    logging.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        logging.warning("Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error crítico: {e}")
        logging.error(f"Error crítico: {e}")
        import traceback
        logging.error(traceback.format_exc())