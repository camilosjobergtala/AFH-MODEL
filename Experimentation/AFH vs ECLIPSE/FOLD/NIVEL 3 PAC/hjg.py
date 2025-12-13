#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NIVEL 3b: Test de Precedencia Temporal H* -> PAC (Sleep -> Wake)
Autopsychic Fold Hypothesis - ECLIPSE v3.0 FULL INTEGRATION

HIPOTESIS: Durante transiciones Sleep->Wake, H* se ACTIVA ANTES que PAC.
Prediccion: Delta_t = onset(H*) - onset(PAC) < 0

Author: Camilo Sjoberg Tala
Date: 2025-12-11
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
import warnings
import logging
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import sys
import json
from datetime import datetime
import hashlib
import math
import base64
import ast
import re

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from tensorpac import Pac

#==============================================================================
# ECLIPSE v3.0 DATA STRUCTURES
#==============================================================================

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
        ops = {">=": lambda x,y: x>=y, "<=": lambda x,y: x<=y, ">": lambda x,y: x>y, "<": lambda x,y: x<y}
        return ops[self.comparison](value, self.threshold)

@dataclass
class EclipseConfig:
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results"
    non_interactive: bool = False
    commitment_phrase: str = None
    eis_weights: Dict[str, float] = None
    stds_alpha: float = 0.05
    
    def __post_init__(self):
        if self.eis_weights is None:
            self.eis_weights = {'preregistration': 0.25, 'protocol_adherence': 0.25, 
                               'split_strength': 0.20, 'leakage_risk': 0.15, 'transparency': 0.15}

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
    detection_method: str = "ast"

@dataclass
class AuditResult:
    timestamp: str
    adherence_score: float
    violations: List[CodeViolation]
    risk_level: str
    passed: bool
    summary: str
    detailed_report: str
    files_analyzed: List[str] = field(default_factory=list)

#==============================================================================
# ECLIPSE v3.0 INTEGRITY SCORE
#==============================================================================

class EclipseIntegrityScore:
    DEFAULT_WEIGHTS = {'preregistration': 0.25, 'protocol_adherence': 0.25, 
                       'split_strength': 0.20, 'leakage_risk': 0.15, 'transparency': 0.15}
    
    JUSTIFICATIONS = {
        'preregistration': "Nosek et al. 2018: Pre-registration reduces false positives ~60%",
        'protocol_adherence': "Simmons et al. 2011: Researcher degrees of freedom inflate Type I",
        'split_strength': "Information theory: Max entropy split minimizes leakage",
        'leakage_risk': "Kapoor & Narayanan 2022: Data leakage in 17/20 ML studies",
        'transparency': "FAIR principles: Documentation enables reproducibility"
    }
    
    def __init__(self, framework):
        self.framework = framework
        self.scores = {}
    
    def compute_preregistration_score(self):
        if not self.framework._criteria_registered:
            return 0.0
        try:
            with open(self.framework.criteria_file, 'r') as f:
                data = json.load(f)
            score = 0.3 if self.framework._criteria_registered else 0.0
            if 'registration_date' in data and 'criteria_hash' in data:
                score += 0.3
            if data.get('criteria'):
                if all('threshold' in c for c in data['criteria']):
                    score += 0.2
                if all(c.get('description') for c in data['criteria']):
                    score += 0.2
            return min(1.0, score)
        except:
            return 0.0
    
    def compute_split_strength(self):
        if not self.framework._split_completed:
            return 0.0
        try:
            with open(self.framework.split_file, 'r') as f:
                data = json.load(f)
            n_dev = data.get('n_development', 0)
            n_hold = data.get('n_holdout', 0)
            total = n_dev + n_hold
            if total == 0:
                return 0.0
            p_dev = n_dev / total
            p_hold = n_hold / total
            if p_dev == 0 or p_hold == 0:
                return 0.0
            entropy = -(p_dev * np.log2(p_dev) + p_hold * np.log2(p_hold))
            size_pen = min(1.0, total / 100)
            hash_bonus = 0.15 if 'integrity_verification' in data else 0.0
            return min(1.0, 0.85 * entropy * size_pen + hash_bonus)
        except:
            return 0.0
    
    def compute_protocol_adherence(self):
        score = 0.0
        if self.framework._split_completed: score += 0.2
        if self.framework._criteria_registered: score += 0.2
        if self.framework._development_completed: score += 0.2
        if self.framework._validation_completed: score += 0.2
        if all([self.framework._split_completed, self.framework._criteria_registered,
                self.framework._development_completed, self.framework._validation_completed]):
            score += 0.2
        return min(1.0, score)
    
    def estimate_leakage_risk(self):
        if not self.framework._validation_completed:
            return 0.5
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
            dev = results.get('development_summary', {}).get('aggregated_metrics', {})
            hold = results.get('validation_summary', {}).get('metrics', {})
            if not dev or not hold:
                return 0.5
            risks = []
            for m in dev:
                if m in hold:
                    cv_mean = dev[m].get('mean', 0)
                    cv_std = dev[m].get('std', 0)
                    h_val = hold[m]
                    if hasattr(h_val, 'item'): h_val = h_val.item()
                    if not isinstance(h_val, (int, float)) or cv_std == 0:
                        continue
                    z = (h_val - cv_mean) / cv_std
                    risk = 0.9 if z > 1.5 else 0.5 if z > 0.5 else 0.2
                    risks.append(risk)
            return float(np.mean(risks)) if risks else 0.5
        except:
            return 0.5
    
    def compute_transparency_score(self):
        score = 0.0
        if self.framework.split_file.exists(): score += 0.2
        if self.framework.criteria_file.exists(): score += 0.2
        try:
            with open(self.framework.criteria_file, 'r') as f:
                data = json.load(f)
            if 'registration_date' in data: score += 0.2
            if 'criteria_hash' in data: score += 0.2
            if 'binding_declaration' in data: score += 0.2
        except:
            pass
        return min(1.0, score)
    
    def compute_eis(self, weights=None):
        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        prereg = self.compute_preregistration_score()
        split = self.compute_split_strength()
        adherence = self.compute_protocol_adherence()
        leakage = self.estimate_leakage_risk()
        transparency = self.compute_transparency_score()
        eis = (weights['preregistration'] * prereg + weights['split_strength'] * split +
               weights['protocol_adherence'] * adherence + weights['leakage_risk'] * (1-leakage) +
               weights['transparency'] * transparency)
        if eis >= 0.90: interp = "EXCELLENT"
        elif eis >= 0.80: interp = "VERY GOOD"
        elif eis >= 0.70: interp = "GOOD"
        elif eis >= 0.60: interp = "FAIR"
        else: interp = "POOR"
        self.scores = {
            'eis': float(eis), 'interpretation': interp,
            'components': {'preregistration': prereg, 'split_strength': split,
                          'protocol_adherence': adherence, 'leakage_risk': leakage,
                          'transparency': transparency},
            'weights': weights, 'justifications': self.JUSTIFICATIONS
        }
        return self.scores
    
    def generate_report(self, path=None):
        if not self.scores: self.compute_eis()
        lines = ["="*80, "ECLIPSE INTEGRITY SCORE (EIS) v3.0", "="*80,
                 f"EIS: {self.scores['eis']:.4f} - {self.scores['interpretation']}", "",
                 "Components:"]
        for k, v in self.scores['components'].items():
            lines.append(f"  {k}: {v:.4f}")
        report = "\n".join(lines)
        if path:
            with open(path, 'w') as f: f.write(report)
        return report

#==============================================================================
# ECLIPSE v3.0 STDS
#==============================================================================

class StatisticalTestDataSnooping:
    def __init__(self, framework):
        self.framework = framework
        self.results = {}
    
    def perform_test(self, alpha=0.05):
        if not self.framework._validation_completed:
            return {'status': 'incomplete'}
        try:
            with open(self.framework.results_file, 'r') as f:
                data = json.load(f)
            dev = data.get('development_summary', {}).get('aggregated_metrics', {})
            hold = data.get('validation_summary', {}).get('metrics', {})
            if not dev or not hold:
                return {'status': 'insufficient_data'}
            z_scores = []
            for m in dev:
                if m not in hold: continue
                cv_mean = dev[m].get('mean')
                cv_std = dev[m].get('std')
                h_val = hold[m]
                if hasattr(h_val, 'item'): h_val = h_val.item()
                if cv_std is None or cv_std == 0: continue
                z = (h_val - cv_mean) / cv_std
                z_scores.append(z)
            if not z_scores:
                return {'status': 'no_metrics'}
            mean_z = float(np.mean(z_scores))
            max_z = float(np.max(z_scores))
            if max_z > 3: risk = "HIGH"
            elif max_z > 2: risk = "MODERATE"
            elif mean_z > 1: risk = "LOW-MODERATE"
            else: risk = "LOW"
            self.results = {'status': 'success', 'mean_z': mean_z, 'max_z': max_z, 'risk': risk}
            return self.results
        except:
            return {'status': 'error'}
    
    def generate_report(self, path=None):
        if not self.results: self.perform_test()
        lines = ["="*80, "STDS v3.0", "="*80]
        if self.results.get('status') == 'success':
            lines.extend([f"Mean z: {self.results['mean_z']:+.4f}",
                         f"Max z: {self.results['max_z']:+.4f}",
                         f"Risk: {self.results['risk']}"])
        report = "\n".join(lines)
        if path:
            with open(path, 'w') as f: f.write(report)
        return report

#==============================================================================
# ECLIPSE v3.0 CODE AUDITOR
#==============================================================================

class SemanticAnalyzer:
    def __init__(self, holdout_ids):
        self.holdout_ids = set(holdout_ids)
        self.alias_graph = defaultdict(set)
    
    def analyze(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                sources = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                for t in node.targets:
                    names = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
                    for name in names:
                        self.alias_graph[name].update(sources)
        return dict(self.alias_graph)
    
    def find_tainted(self):
        tainted = set()
        for var, sources in self.alias_graph.items():
            if sources & self.holdout_ids:
                tainted.add(var)
        changed = True
        while changed:
            changed = False
            for var, sources in self.alias_graph.items():
                if var not in tainted and sources & tainted:
                    tainted.add(var)
                    changed = True
        return tainted

class StaticCodeAnalyzer:
    def __init__(self, holdout_ids):
        self.holdout_ids = set(holdout_ids)
        self.semantic = SemanticAnalyzer(holdout_ids)
    
    def analyze_file(self, path):
        try:
            with open(path, 'r') as f:
                code = f.read()
            tree = ast.parse(code)
        except:
            return []
        findings = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in self.holdout_ids:
                findings.append({'type': 'holdout_access', 'line': node.lineno, 
                               'severity': 'critical', 'description': f'Direct access: {node.id}'})
        self.semantic.alias_graph.clear()
        self.semantic.analyze(tree)
        tainted = self.semantic.find_tainted()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func.attr if isinstance(node.func, ast.Attribute) else \
                       node.func.id if isinstance(node.func, ast.Name) else ''
                if any(k in func.lower() for k in ['fit', 'train', 'optimize']):
                    args = {n.id for a in node.args for n in ast.walk(a) if isinstance(n, ast.Name)}
                    if args & tainted:
                        findings.append({'type': 'indirect_access', 'line': node.lineno,
                                       'severity': 'critical', 'description': f'Tainted in {func}'})
        return findings

class CodeAuditor:
    def __init__(self, framework):
        self.framework = framework
    
    def audit(self, code_paths=None, holdout_ids=None):
        code_paths = code_paths or []
        holdout_ids = holdout_ids or ['holdout', 'test', 'X_test', 'y_test', 'holdout_data']
        analyzer = StaticCodeAnalyzer(holdout_ids)
        violations = []
        for path in code_paths:
            if Path(path).exists():
                for f in analyzer.analyze_file(path):
                    violations.append(CodeViolation(
                        severity=f.get('severity', 'medium'),
                        category=f.get('type', 'unknown'),
                        description=f.get('description', ''),
                        file_path=path, line_number=f.get('line'),
                        code_snippet='', recommendation='Review',
                        confidence=0.9, detection_method='ast'))
        penalties = {'critical': 30, 'high': 15, 'medium': 5, 'low': 2}
        score = max(0, 100 - sum(penalties.get(v.severity, 10) for v in violations))
        risk = 'low' if score >= 90 else 'medium' if score >= 70 else 'high' if score >= 50 else 'critical'
        return AuditResult(
            timestamp=datetime.now().isoformat(), adherence_score=score,
            violations=violations, risk_level=risk, passed=score >= 70,
            summary=f"{'PASS' if score >= 70 else 'FAIL'} ({score}/100)",
            detailed_report=f"{len(violations)} issues", files_analyzed=code_paths)

#==============================================================================
# ECLIPSE v3.0 MAIN FRAMEWORK
#==============================================================================

class EclipseFramework:
    VERSION = "3.0.0"
    
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA.json"
        self.results_file = self.output_dir / f"{config.project_name}_RESULTS.json"
        self._split_completed = self.split_file.exists()
        self._criteria_registered = self.criteria_file.exists()
        self._development_completed = False
        self._validation_completed = False
        self.integrity_scorer = None
        self.snooping_tester = None
        self.code_auditor = None
        print("="*80)
        print(f"ECLIPSE v{self.VERSION} INITIALIZED")
        print(f"Project: {config.project_name} | Researcher: {config.researcher}")
        print("="*80)
    
    def stage1_split(self, ids, force=False):
        if self.split_file.exists() and not force:
            with open(self.split_file, 'r') as f:
                data = json.load(f)
            self._split_completed = True
            return data['development_ids'], data['holdout_ids']
        np.random.seed(self.config.sacred_seed)
        shuffled = np.array(ids).copy()
        np.random.shuffle(shuffled)
        n_dev = int(len(ids) * self.config.development_ratio)
        dev_ids = shuffled[:n_dev].tolist()
        hold_ids = shuffled[n_dev:].tolist()
        split_hash = hashlib.sha256(f"{self.config.sacred_seed}_{sorted(ids)}".encode()).hexdigest()
        data = {
            'project_name': self.config.project_name, 'split_date': datetime.now().isoformat(),
            'sacred_seed': self.config.sacred_seed, 'n_development': len(dev_ids),
            'n_holdout': len(hold_ids), 'development_ids': dev_ids, 'holdout_ids': hold_ids,
            'integrity_verification': {'split_hash': split_hash, 'algorithm': 'SHA-256'}
        }
        with open(self.split_file, 'w') as f:
            json.dump(data, f, indent=2)
        self._split_completed = True
        print(f"Split: {len(dev_ids)} dev, {len(hold_ids)} holdout")
        return dev_ids, hold_ids
    
    def stage2_register_criteria(self, criteria, force=False):
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file, 'r') as f:
                return json.load(f)
        print("\n" + "="*80)
        print("STAGE 2: PRE-REGISTERED CRITERIA")
        print("="*80)
        for i, c in enumerate(criteria, 1):
            req = "[REQ]" if c.is_required else "[opt]"
            print(f"  {i}. {req} {c.name} {c.comparison} {c.threshold}")
        criteria_list = [asdict(c) for c in criteria]
        criteria_hash = hashlib.sha256(json.dumps(criteria_list, sort_keys=True).encode()).hexdigest()
        data = {
            'project_name': self.config.project_name,
            'registration_date': datetime.now().isoformat(),
            'criteria': criteria_list, 'criteria_hash': criteria_hash,
            'binding_declaration': "BINDING - Cannot modify after registration"
        }
        with open(self.criteria_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Registered: hash={criteria_hash[:16]}...")
        self._criteria_registered = True
        return data
    
    def stage3_development(self, data, train_fn, val_fn, **kwargs):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=self.config.n_folds_cv, shuffle=True, random_state=self.config.sacred_seed)
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(data)))):
            try:
                model = train_fn(train_idx, **kwargs)
                metrics = val_fn(model, val_idx, **kwargs)
                results.append({'fold': fold+1, 'metrics': metrics, 'status': 'success'})
            except Exception as e:
                results.append({'fold': fold+1, 'status': 'failed', 'error': str(e)})
        success = [r for r in results if r['status'] == 'success']
        if not success:
            raise RuntimeError("All folds failed")
        agg = {}
        for m in success[0]['metrics']:
            vals = [r['metrics'][m] for r in success]
            agg[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
        self._development_completed = True
        return {'n_folds': self.config.n_folds_cv, 'aggregated_metrics': agg, 'fold_results': results}
    
    def stage4_validation(self, holdout, model, val_fn, force=False, **kwargs):
        if self.results_file.exists() and not force:
            raise RuntimeError("Validation already performed!")
        if self.config.non_interactive:
            commitment = {'hash': hashlib.sha256(
                f"{self.config.project_name}|{self.config.commitment_phrase}".encode()).hexdigest()}
        else:
            confirm = input("Type 'I ACCEPT SINGLE-SHOT': ")
            if confirm != "I ACCEPT SINGLE-SHOT":
                return None
        metrics = val_fn(model, holdout, **kwargs)
        self._validation_completed = True
        return {'status': 'success', 'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                                  for k, v in metrics.items()}}
    
    def stage5_assessment(self, dev_results, val_results):
        with open(self.criteria_file, 'r') as f:
            crit_data = json.load(f)
        criteria = [FalsificationCriteria(**c) for c in crit_data['criteria']]
        metrics = val_results.get('metrics', {})
        evals = []
        for c in criteria:
            val = metrics.get(c.name)
            passed = c.evaluate(val) if val is not None else False
            evals.append({'criterion': asdict(c), 'value': val, 'passed': passed})
        req = [e for e in evals if e['criterion']['is_required']]
        verdict = "VALIDATED" if all(e['passed'] for e in req) else "FALSIFIED"
        result = {
            'project_name': self.config.project_name,
            'assessment_date': datetime.now().isoformat(),
            'development_summary': dev_results,
            'validation_summary': val_results,
            'criteria_evaluation': evals,
            'verdict': verdict
        }
        with open(self.results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        return result
    
    def compute_integrity(self):
        self.integrity_scorer = EclipseIntegrityScore(self)
        eis = self.integrity_scorer.compute_eis()
        self.snooping_tester = StatisticalTestDataSnooping(self)
        stds = self.snooping_tester.perform_test()
        return {'eis': eis, 'stds': stds}
    
    def audit_code(self, paths=None, holdout_ids=None):
        self.code_auditor = CodeAuditor(self)
        return self.code_auditor.audit(paths, holdout_ids)
    
    def verify_integrity(self):
        result = {'all_valid': True}
        if self.split_file.exists():
            with open(self.split_file, 'r') as f:
                data = json.load(f)
            all_ids = data['development_ids'] + data['holdout_ids']
            computed = hashlib.sha256(f"{data['sacred_seed']}_{sorted(all_ids)}".encode()).hexdigest()
            valid = computed == data['integrity_verification']['split_hash']
            result['split'] = valid
            if not valid: result['all_valid'] = False
        if self.criteria_file.exists():
            with open(self.criteria_file, 'r') as f:
                data = json.load(f)
            computed = hashlib.sha256(json.dumps(data['criteria'], sort_keys=True).encode()).hexdigest()
            valid = computed == data['criteria_hash']
            result['criteria'] = valid
            if not valid: result['all_valid'] = False
        return result

#==============================================================================
# NIVEL 3b ANALYSIS CONFIG
#==============================================================================

@dataclass
class AnalysisConfig:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./nivel3b_eclipse_results')
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    window_before: float = 300.0
    window_after: float = 300.0
    sliding_window: float = 30.0
    sliding_step: float = 5.0
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 45.0)
    onset_threshold_sd: float = 1.5
    wake_states: List[str] = None
    sleep_states: List[str] = None
    sacred_seed: int = 42
    project_name: str = "AFH_Nivel3b_SleepToWake"
    researcher: str = "Camilo Sjoberg Tala"
    
    def __post_init__(self):
        self.wake_states = ['Sleep stage W']
        self.sleep_states = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']

def get_preregistered_criteria():
    return [
        FalsificationCriteria("mean_delta_negative", 0.0, "<", "Mean delta < 0 (H* first)", True),
        FalsificationCriteria("pct_hstar_first", 70.0, ">=", ">=70% H* first", True),
        FalsificationCriteria("p_value", 0.01, "<", "p < 0.01 binomial", True),
        FalsificationCriteria("pct_pac_first", 50.0, "<", "<50% PAC first", True)
    ]

#==============================================================================
# NIVEL 3b METRICS CALCULATOR
#==============================================================================

class MetricsCalculator:
    def __init__(self, config):
        self.config = config
        self.fs = config.sampling_rate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pac = Pac(idpac=(1,0,0), f_pha=list(config.theta_band), 
                          f_amp=list(config.gamma_band), dcomplex='wavelet', width=7, verbose=False)
    
    def preprocess(self, data):
        if len(data) < 10: return data
        sos = signal.butter(4, [self.config.lowcut, self.config.highcut], 
                           btype='bandpass', fs=self.fs, output='sos')
        filt = signal.sosfiltfilt(sos, data)
        return (filt - np.mean(filt)) / (np.std(filt) + 1e-10)
    
    def compute_pac(self, data):
        if len(data) < int(2*self.fs): return np.nan
        clean = self.preprocess(data)
        if np.ptp(clean) > 8.0: return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(self.pac.filterfit(self.fs, clean[np.newaxis,:], clean[np.newaxis,:])[0,0,0])
        except: return np.nan
    
    def compute_spectral_entropy(self, data):
        if len(data) < int(2*self.fs): return np.nan
        clean = self.preprocess(data)
        _, psd = signal.welch(clean, fs=self.fs, nperseg=min(256, len(clean)//2))
        psd_n = psd / (np.sum(psd) + 1e-10)
        ent = -np.sum(psd_n * np.log2(psd_n + 1e-10))
        return ent / np.log2(len(psd))
    
    def compute_perm_entropy(self, data, order=3, delay=1):
        if len(data) < (order-1)*delay+1: return np.nan
        clean = self.preprocess(data)
        n = len(clean)
        patterns = [tuple(np.argsort([clean[i+j*delay] for j in range(order)])) for i in range(n-(order-1)*delay)]
        counts = Counter(patterns)
        total = len(patterns)
        ent = -sum((c/total)*np.log2(c/total) for c in counts.values())
        return ent / np.log2(math.factorial(order))
    
    def compute_lziv(self, data):
        if len(data) < 100: return np.nan
        clean = self.preprocess(data)
        binary = ''.join(['1' if x > np.median(clean) else '0' for x in clean])
        n = len(binary)
        c, pl, i = 1, 1, 0
        while i + pl <= n:
            if binary[i:i+pl] not in binary[:i+pl-1]:
                c += 1; i += pl; pl = 1
            else:
                pl += 1
                if i + pl > n: break
        return c / (n / np.log2(n)) if n > 1 else 0

#==============================================================================
# NIVEL 3b TRANSITION DETECTOR
#==============================================================================

class TransitionDetector:
    def __init__(self, config):
        self.config = config
    
    def find_transitions(self, annotations):
        trans = []
        desc, onsets = annotations.description, annotations.onset
        for i in range(len(desc)-1):
            if desc[i] in self.config.sleep_states and desc[i+1] in self.config.wake_states:
                trans.append({'index': i, 'time': onsets[i+1], 'from': desc[i], 'to': desc[i+1]})
        return trans

#==============================================================================
# NIVEL 3b PRECEDENCE ANALYZER
#==============================================================================

class PrecedenceAnalyzer:
    def __init__(self, config):
        self.config = config
        self.metrics = MetricsCalculator(config)
    
    def extract_window(self, raw, trans_time):
        fs = raw.info['sfreq']
        start = max(0, trans_time - self.config.window_before)
        end = min(raw.times[-1], trans_time + self.config.window_after)
        s_samp, e_samp = int(start*fs), int(end*fs)
        if e_samp - s_samp < int(60*fs): return None
        for ch in ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']:
            if ch in raw.ch_names:
                return raw.copy().pick_channels([ch])[:, s_samp:e_samp][0][0,:]
        return None
    
    def compute_sliding(self, data, trans_in_win):
        fs = self.config.sampling_rate
        win_s = int(self.config.sliding_window * fs)
        step_s = int(self.config.sliding_step * fs)
        results = []
        for i in range((len(data)-win_s)//step_s + 1):
            start = i * step_s
            end = start + win_s
            if end > len(data): break
            w = data[start:end]
            center = (start + win_s//2) / fs
            results.append({
                'relative_time': center - trans_in_win,
                'pac': self.metrics.compute_pac(w),
                'spectral_entropy': self.metrics.compute_spectral_entropy(w),
                'perm_entropy': self.metrics.compute_perm_entropy(w),
                'lziv': self.metrics.compute_lziv(w)
            })
        return pd.DataFrame(results)
    
    def detect_onset(self, df, metric):
        valid = df[['relative_time', metric]].dropna()
        if len(valid) < 10: return None
        baseline = valid.loc[valid['relative_time'] < -60, metric]
        if len(baseline) < 3: baseline = valid[metric].iloc[:len(valid)//3]
        mean, std = baseline.mean(), baseline.std()
        if std < 1e-10: return None
        thresh = mean + self.config.onset_threshold_sd * std
        for _, row in valid[valid['relative_time'] >= -60].iterrows():
            if row[metric] > thresh:
                return row['relative_time']
        return None
    
    def analyze(self, raw, trans):
        data = self.extract_window(raw, trans['time'])
        if data is None: return None
        trans_in_win = min(self.config.window_before, trans['time'])
        df = self.compute_sliding(data, trans_in_win)
        if len(df) < 10: return None
        onset_pac = self.detect_onset(df, 'pac')
        onset_spec = self.detect_onset(df, 'spectral_entropy')
        onset_perm = self.detect_onset(df, 'perm_entropy')
        onset_lziv = self.detect_onset(df, 'lziv')
        result = {'transition_time': trans['time'], 'from': trans['from'], 'to': trans['to'],
                  'onset_pac': onset_pac, 'onset_spec': onset_spec, 'onset_perm': onset_perm, 'onset_lziv': onset_lziv}
        if onset_pac is not None:
            if onset_spec is not None: result['delta_spec'] = onset_spec - onset_pac
            if onset_perm is not None: result['delta_perm'] = onset_perm - onset_pac
            if onset_lziv is not None: result['delta_lziv'] = onset_lziv - onset_pac
        return result

#==============================================================================
# NIVEL 3b PROCESSOR
#==============================================================================

class SleepEDFProcessor:
    def __init__(self, config):
        self.config = config
        self.analyzer = PrecedenceAnalyzer(config)
        self.detector = TransitionDetector(config)
    
    def find_files(self):
        psg = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        hmap = {h.stem.replace("-Hypnogram","")[:-1]: h for h in hypno if len(h.stem) >= 6}
        files = []
        for p in psg:
            c = p.stem.replace("-PSG","")
            if len(c) >= 7 and c.endswith('0') and c[:-1] in hmap:
                files.append({'psg': p, 'hypno': hmap[c[:-1]], 'subject_id': c})
        return files
    
    def process(self, info):
        try:
            raw = mne.io.read_raw_edf(info['psg'], preload=True, verbose=False)
            ann = mne.read_annotations(info['hypno'])
        except Exception as e:
            logger.error(f"Error {info['subject_id']}: {e}")
            return []
        trans = self.detector.find_transitions(ann)
        results = []
        for t in trans:
            r = self.analyzer.analyze(raw, t)
            if r:
                r['subject_id'] = info['subject_id']
                results.append(r)
        return results

#==============================================================================
# MAIN ANALYSIS
#==============================================================================

def run_analysis():
    config = AnalysisConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("NIVEL 3b: H* -> PAC PRECEDENCE (Sleep -> Wake)")
    print("ECLIPSE v3.0 FULL")
    print("="*100)
    
    # ECLIPSE init
    ecfg = EclipseConfig(project_name=config.project_name, researcher=config.researcher,
                         sacred_seed=config.sacred_seed, output_dir=str(config.output_dir),
                         non_interactive=True, commitment_phrase="COMMIT_NIVEL3B")
    eclipse = EclipseFramework(ecfg)
    eclipse.stage2_register_criteria(get_preregistered_criteria())
    
    print("\nHYPOTHESIS: H* activates BEFORE PAC during awakening")
    print("PREDICTION: Delta < 0 in >=70% of transitions\n")
    
    # Process data
    processor = SleepEDFProcessor(config)
    files = processor.find_files()
    print(f"[DATA] {len(files)} subjects")
    
    if not files:
        print("[ERROR] No files found")
        return None, None
    
    all_results = []
    n_subj = 0
    for i, f in enumerate(files, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{i}/{len(files)}] {f['subject_id']}")
        r = processor.process(f)
        if r:
            n_subj += 1
            all_results.extend(r)
    
    print(f"\n[RESULTS] {len(all_results)} transitions from {n_subj} subjects")
    
    if not all_results:
        print("[ERROR] No valid transitions")
        return None, None
    
    # Compile
    df = pd.DataFrame(all_results)
    df.to_csv(config.output_dir / 'raw_results.csv', index=False)
    
    # Combine H* proxies
    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    df['delta_combined'] = df[delta_cols].mean(axis=1, skipna=True)
    valid = df['delta_combined'].dropna()
    
    if len(valid) < 5:
        print(f"[ERROR] Insufficient data ({len(valid)})")
        return df, None
    
    n_total = len(valid)
    n_hstar = (valid < 0).sum()
    n_pac = (valid > 0).sum()
    pct_hstar = 100 * n_hstar / n_total
    pct_pac = 100 * n_pac / n_total
    mean_d = valid.mean()
    std_d = valid.std()
    
    from scipy.stats import binomtest
    p_bin = binomtest(n_hstar, n_hstar+n_pac, 0.5, alternative='greater').pvalue if n_hstar+n_pac > 0 else np.nan
    
    print(f"\n[STATS]")
    print(f"  n = {n_total}")
    print(f"  H* first: {n_hstar} ({pct_hstar:.1f}%)")
    print(f"  PAC first: {n_pac} ({pct_pac:.1f}%)")
    print(f"  Mean delta: {mean_d:.2f} +/- {std_d:.2f} s")
    print(f"  p-value: {p_bin:.6f}")
    
    # Evaluate criteria
    print("\n" + "="*100)
    print("ECLIPSE v3.0 EVALUATION")
    print("="*100)
    
    with open(eclipse.criteria_file, 'r') as f:
        crit_data = json.load(f)
    
    criteria = [FalsificationCriteria(**c) for c in crit_data['criteria']]
    metrics = {'mean_delta_negative': mean_d, 'pct_hstar_first': pct_hstar, 
               'p_value': p_bin, 'pct_pac_first': pct_pac}
    
    evals = []
    all_pass = True
    for c in criteria:
        v = metrics.get(c.name)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            passed = c.evaluate(v)
            if c.is_required and not passed: all_pass = False
            icon = "[OK]" if passed else "[X]"
            print(f"  {icon} {c.name} {c.comparison} {c.threshold} (got: {v:.4f})")
            evals.append({'criterion': asdict(c), 'value': float(v), 'passed': passed})
        else:
            print(f"  [?] MISSING: {c.name}")
            evals.append({'criterion': asdict(c), 'value': None, 'passed': False})
            if c.is_required: all_pass = False
    
    # Verdict
    if all_pass:
        verdict = "VALIDATED"
        desc = "All required criteria passed - AFH hypothesis SUPPORTED"
    elif pct_pac >= 50:
        verdict = "FALSIFIED"
        desc = f"PAC first in {pct_pac:.1f}% - AFH architecture FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"
        desc = "Some criteria failed but not enough for falsification"
    
    print(f"\n>>> VERDICT: {verdict} <<<")
    print(f"    {desc}")
    
    # Save results
    stats = {'n_total': int(n_total), 'n_hstar': int(n_hstar), 'n_pac': int(n_pac),
             'pct_hstar': float(pct_hstar), 'pct_pac': float(pct_pac),
             'mean_delta': float(mean_d), 'std_delta': float(std_d),
             'p_binomial': float(p_bin) if not np.isnan(p_bin) else None}
    
    # Create ECLIPSE results file
    eclipse_results = {
        'project_name': config.project_name,
        'assessment_date': datetime.now().isoformat(),
        'development_summary': {'aggregated_metrics': {
            'pct_hstar_first': {'mean': pct_hstar, 'std': 0, 'values': [pct_hstar]},
            'mean_delta': {'mean': mean_d, 'std': 0, 'values': [mean_d]}}},
        'validation_summary': {'status': 'success', 'metrics': metrics},
        'criteria_evaluation': evals,
        'verdict': verdict
    }
    with open(eclipse.results_file, 'w') as f:
        json.dump(eclipse_results, f, indent=2, default=str)
    eclipse._validation_completed = True
    
    # Compute integrity
    print("\n[INTEGRITY METRICS]")
    eclipse.integrity_scorer = EclipseIntegrityScore(eclipse)
    eis = eclipse.integrity_scorer.compute_eis()
    print(f"  EIS: {eis['eis']:.4f} - {eis['interpretation']}")
    eclipse.integrity_scorer.generate_report(str(config.output_dir / 'EIS_REPORT.txt'))
    
    eclipse.snooping_tester = StatisticalTestDataSnooping(eclipse)
    stds = eclipse.snooping_tester.perform_test()
    if stds.get('status') == 'success':
        print(f"  STDS: max_z={stds['max_z']:+.2f}, risk={stds['risk']}")
        eclipse.snooping_tester.generate_report(str(config.output_dir / 'STDS_REPORT.txt'))
    
    # Final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'project': config.project_name,
        'statistics': stats,
        'verdict': verdict,
        'description': desc,
        'integrity': {'eis': eis, 'stds': stds},
        'evaluations': evals
    }
    
    with open(config.output_dir / 'FINAL_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OUTPUT] {config.output_dir}")
    print("="*100)
    
    return df, summary

if __name__ == "__main__":
    try:
        df, summary = run_analysis()
        if summary:
            print("\n" + "="*100)
            if summary['verdict'] == 'VALIDATED':
                print(">>> AFH ARCHITECTURE (Sleep->Wake): SUPPORTED <<<")
                print("  H* activates BEFORE PAC during awakening")
            elif summary['verdict'] == 'FALSIFIED':
                print(">>> AFH ARCHITECTURE (Sleep->Wake): FALSIFIED <<<")
            else:
                print(">>> INCONCLUSIVE <<<")
            print("="*100)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)