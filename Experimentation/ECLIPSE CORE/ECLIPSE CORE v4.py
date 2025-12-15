#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECLIPSE v4.0: Enhanced Systematic Falsification Framework
With Integrated External Timestamping

Version: 4.0.0
Author: Camilo Alejandro Sjöberg Tala
License: AGPL v3.0 / Commercial
"""

import json, hashlib, os, sys, ast, re, logging, argparse, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

__version__ = "4.0.0"
__author__ = "Camilo Alejandro Sjöberg Tala"

# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

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
        ops = {">=": lambda x,y: x>=y, "<=": lambda x,y: x<=y, ">": lambda x,y: x>y,
               "<": lambda x,y: x<y, "==": lambda x,y: abs(x-y)<1e-9}
        return ops.get(self.comparison, lambda x,y: False)(value, self.threshold)
    
    def __str__(self):
        return f"{self.name} {self.comparison} {self.threshold} [{'REQUIRED' if self.is_required else 'optional'}]"
    
    def to_dict(self): return asdict(self)

@dataclass
class TimestampProof:
    method: str
    timestamp_utc: str
    criteria_hash: str
    verification_url: Optional[str]
    proof_file: Optional[str]
    raw_response: Optional[Dict]
    status: str
    message: str
    def to_dict(self): return asdict(self)

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
    notebooks_analyzed: List[str] = field(default_factory=list)

@dataclass
class EclipseConfig:
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results"
    enable_timestamping: bool = False
    timestamp_method: str = 'opentimestamps'
    github_token: Optional[str] = None
    smtp_user: Optional[str] = None
    smtp_pass: Optional[str] = None
    non_interactive: bool = False
    commitment_phrase: Optional[str] = None
    eis_weights: Optional[Dict[str, float]] = None
    stds_alpha: float = 0.05
    audit_pass_threshold: float = 70.0
    
    def __post_init__(self):
        if abs(self.development_ratio + self.holdout_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        if self.n_folds_cv < 2:
            raise ValueError("n_folds_cv must be >= 2")
        if self.non_interactive and not self.commitment_phrase:
            raise ValueError("non_interactive requires commitment_phrase")
        self.github_token = self.github_token or os.environ.get('GITHUB_TOKEN')
        self.smtp_user = self.smtp_user or os.environ.get('SMTP_USER')
        self.smtp_pass = self.smtp_pass or os.environ.get('SMTP_PASS')

# ═══════════════════════════════════════════════════════════════════════════
# TIMESTAMPING
# ═══════════════════════════════════════════════════════════════════════════

class EclipseTimestamp:
    VERSION = "1.0.0"
    
    def __init__(self, output_dir="./eclipse_timestamps"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _utcnow(self): return datetime.now(timezone.utc).isoformat()
    
    def stamp_criteria(self, criteria, method='opentimestamps', project_name="project", **kwargs):
        if not HAS_REQUESTS:
            return TimestampProof(method, self._utcnow(), 'N/A', None, None, None, 'failed', 'requests not installed')
        
        prereg = {
            'project_name': project_name,
            'timestamp_utc': self._utcnow(),
            'criteria': criteria,
            'eclipse_version': __version__,
            'binding_declaration': "BINDING criteria defined BEFORE analysis."
        }
        
        criteria_hash = hashlib.sha256(json.dumps(prereg, sort_keys=True).encode()).hexdigest()
        prereg['document_hash'] = criteria_hash
        
        local_file = self.output_dir / f"{project_name}_preregistration.json"
        with open(local_file, 'w') as f:
            json.dump(prereg, f, indent=2)
        
        print(f"\n{'='*60}\n🔏 TIMESTAMP: {project_name}\nHash: {criteria_hash}\n{'='*60}")
        
        if method == 'opentimestamps':
            return self._ots(prereg, criteria_hash, project_name)
        elif method == 'github_gist':
            return self._github(prereg, criteria_hash, project_name, **kwargs)
        else:
            return TimestampProof(method, self._utcnow(), criteria_hash, None, None, None, 'failed', f'Unknown method: {method}')
    
    def _ots(self, prereg, h, proj):
        try:
            r = requests.post('https://a.pool.opentimestamps.org/stamp', 
                            data=bytes.fromhex(h),
                            headers={'Content-Type': 'application/octet-stream'}, timeout=30)
            if r.status_code == 200:
                f = self.output_dir / f"{proj}.ots"
                with open(f, 'wb') as fp: fp.write(r.content)
                return TimestampProof('opentimestamps', self._utcnow(), h, 'https://opentimestamps.org',
                                     str(f), {'size': len(r.content)}, 'pending', f'✅ Submitted. Verify: ots verify {f}')
        except Exception as e:
            pass
        return TimestampProof('opentimestamps', self._utcnow(), h, None, None, None, 'failed', 'OTS failed')
    
    def _github(self, prereg, h, proj, github_token=None, **kw):
        token = github_token or os.environ.get('GITHUB_TOKEN')
        if not token:
            return TimestampProof('github_gist', self._utcnow(), h, None, None, None, 'failed', 'No token')
        try:
            r = requests.post('https://api.github.com/gists',
                            headers={'Authorization': f'token {token}'},
                            json={'description': f'ECLIPSE: {proj}', 'public': True,
                                  'files': {f'{proj}.json': {'content': json.dumps(prereg, indent=2)}}}, timeout=30)
            if r.status_code == 201:
                url = r.json()['html_url']
                return TimestampProof('github_gist', r.json()['created_at'], h, url, None, {'url': url}, 'success', f'✅ {url}')
        except:
            pass
        return TimestampProof('github_gist', self._utcnow(), h, None, None, None, 'failed', 'GitHub failed')

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

class EclipseValidator:
    @staticmethod
    def binary_classification_metrics(y_true, y_pred, y_pred_proba=None):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score, matthews_corrcoef
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        m = {'accuracy': accuracy_score(y_true, y_pred), 'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
             'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0),
             'f1_score': f1_score(y_true, y_pred, zero_division=0), 'mcc': matthews_corrcoef(y_true, y_pred)}
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2,2):
            tn,fp,fn,tp = cm.ravel()
            m.update({'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                      'specificity': tn/(tn+fp) if tn+fp>0 else 0})
        return m

# ═══════════════════════════════════════════════════════════════════════════
# EIS & STDS
# ═══════════════════════════════════════════════════════════════════════════

class EclipseIntegrityScore:
    DEFAULT_WEIGHTS = {'preregistration': 0.25, 'protocol_adherence': 0.25, 'split_strength': 0.20, 'leakage_risk': 0.15, 'transparency': 0.15}
    
    def __init__(self, framework):
        self.fw = framework
        self.scores = {}
    
    def compute_eis(self):
        w = self.fw.config.eis_weights or self.DEFAULT_WEIGHTS
        prereg = self._prereg_score()
        split = self._split_score()
        protocol = self._protocol_score()
        leak = self._leakage_risk()
        trans = self._transparency()
        
        eis = w['preregistration']*prereg + w['split_strength']*split + w['protocol_adherence']*protocol + w['leakage_risk']*(1-leak) + w['transparency']*trans
        interp = "EXCELLENT" if eis>=0.9 else "VERY GOOD" if eis>=0.8 else "GOOD" if eis>=0.7 else "FAIR" if eis>=0.6 else "POOR"
        self.scores = {'eis': float(eis), 'interpretation': interp, 'components': {'prereg': prereg, 'split': split, 'protocol': protocol, 'leakage': leak, 'transparency': trans}}
        return self.scores
    
    def _prereg_score(self):
        if not self.fw._criteria_registered: return 0
        try:
            with open(self.fw.criteria_file) as f: d = json.load(f)
            s = 0.5 if 'criteria_hash' in d else 0.3
            if 'external_timestamp' in d and d['external_timestamp'].get('status') in ['success','pending']: s += 0.3
            return min(1, s + 0.2)
        except: return 0.3
    
    def _split_score(self):
        if not self.fw._split_completed: return 0
        try:
            with open(self.fw.split_file) as f: d = json.load(f)
            n = d['n_development'] + d['n_holdout']
            p = d['n_development'] / n
            entropy = -(p*np.log2(p) + (1-p)*np.log2(1-p)) if 0<p<1 else 0
            return min(1, entropy * (1 if n>=100 else n/100) + (0.15 if 'integrity_verification' in d else 0))
        except: return 0
    
    def _protocol_score(self):
        stages = [self.fw._split_completed, self.fw._criteria_registered, self.fw._development_completed, self.fw._validation_completed]
        return sum(0.25 for s in stages if s)
    
    def _leakage_risk(self):
        if not self.fw._validation_completed: return 0.5
        try:
            with open(self.fw.results_file) as f: r = json.load(f)
            dev = r.get('development_summary',{}).get('aggregated_metrics',{})
            val = r.get('validation_summary',{}).get('metrics',{})
            risks = []
            for k,v in dev.items():
                if k in val and v.get('std',0) > 0:
                    z = (val[k] - v['mean']) / v['std']
                    risks.append(0.9 if z>1.5 else 0.5 if z>0.5 else 0.2)
            return np.mean(risks) if risks else 0.5
        except: return 0.5
    
    def _transparency(self):
        s = 0
        if self.fw.split_file.exists(): s += 0.3
        if self.fw.criteria_file.exists(): s += 0.3
        try:
            with open(self.fw.criteria_file) as f:
                if 'criteria_hash' in json.load(f): s += 0.2
        except: pass
        try:
            with open(self.fw.split_file) as f:
                if 'integrity_verification' in json.load(f): s += 0.2
        except: pass
        return min(1, s)
    
    def generate_report(self, path=None):
        if not self.scores: self.compute_eis()
        r = f"EIS REPORT\n{'='*40}\nEIS: {self.scores['eis']:.4f}\n{self.scores['interpretation']}\n"
        if path:
            with open(path, 'w') as f: f.write(r)
        return r

class StatisticalTestDataSnooping:
    def __init__(self, framework):
        self.fw = framework
        self.results = {}
    
    def test(self, alpha=None):
        alpha = alpha or self.fw.config.stds_alpha
        if not self.fw._validation_completed:
            return {'status': 'incomplete'}
        try:
            with open(self.fw.results_file) as f: r = json.load(f)
            dev = r.get('development_summary',{}).get('aggregated_metrics',{})
            val = r.get('validation_summary',{}).get('metrics',{})
            zs = []
            for k,v in dev.items():
                if k in val and v.get('std',0)>0:
                    z = (val[k] - v['mean']) / v['std']
                    zs.append(z)
            if not zs: return {'status': 'insufficient_data'}
            maxz = max(zs)
            risk = "HIGH" if maxz>3 else "MODERATE" if maxz>2 else "LOW"
            self.results = {'status': 'success', 'max_z': maxz, 'mean_z': np.mean(zs), 'risk_level': risk}
            return self.results
        except:
            return {'status': 'error'}
    
    def generate_report(self, path=None):
        if self.results.get('status') != 'success': return ""
        r = f"STDS REPORT\n{'='*40}\nMax z: {self.results['max_z']:.2f}\nRisk: {self.results['risk_level']}\n"
        if path:
            with open(path, 'w') as f: f.write(r)
        return r

# ═══════════════════════════════════════════════════════════════════════════
# CODE AUDITOR
# ═══════════════════════════════════════════════════════════════════════════

class SemanticAnalyzer:
    def __init__(self, holdout_vars):
        self.holdout = holdout_vars
        self.aliases = defaultdict(set)
    
    def analyze(self, tree):
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                src = {x.id for x in ast.walk(n.value) if isinstance(x, ast.Name)}
                for t in n.targets:
                    tgt = {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
                    for v in tgt:
                        self.aliases[v].update(src)
                        for s in src:
                            if s in self.aliases: self.aliases[v].update(self.aliases[s])
        tainted = {v for v,s in self.aliases.items() if s & self.holdout}
        changed = True
        while changed:
            changed = False
            for v,s in self.aliases.items():
                if v not in tainted and s & tainted:
                    tainted.add(v)
                    changed = True
        return tainted

class CodeAuditor:
    HOLDOUT_NAMES = {'holdout','test','holdout_data','test_data','X_test','y_test','X_holdout','y_holdout'}
    
    def __init__(self, framework):
        self.fw = framework
    
    def audit(self, code_paths=None, notebook_paths=None, holdout_names=None):
        paths = code_paths or []
        notebooks = notebook_paths or []
        names = set(holdout_names or self.HOLDOUT_NAMES)
        violations = []
        
        for p in paths:
            if Path(p).exists():
                violations.extend(self._analyze_file(p, names))
        
        for p in notebooks:
            if Path(p).exists():
                try:
                    with open(p) as f: nb = json.load(f)
                    for i,c in enumerate(nb.get('cells',[])):
                        if c.get('cell_type')=='code':
                            code = ''.join(c.get('source',[]))
                            violations.extend(self._analyze_code(code, f"{p}:cell{i}", names))
                except: pass
        
        penalty = sum({'critical':30,'high':15,'medium':5,'low':2}.get(v.severity,10) for v in violations)
        score = max(0, 100 - penalty)
        passed = score >= self.fw.config.audit_pass_threshold
        
        return AuditResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            adherence_score=score,
            violations=violations,
            risk_level='low' if score>=90 else 'medium' if score>=70 else 'high',
            passed=passed,
            summary=f"{'✅' if passed else '❌'} Score: {score:.0f}/100",
            detailed_report=f"Audit: {len(violations)} violations",
            files_analyzed=paths,
            notebooks_analyzed=notebooks
        )
    
    def _analyze_file(self, p, names):
        try:
            with open(p, encoding='utf-8') as f: return self._analyze_code(f.read(), p, names)
        except: return []
    
    def _analyze_code(self, code, src, names):
        violations = []
        try:
            tree = ast.parse(code)
        except: return violations
        
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id in names:
                violations.append(CodeViolation('critical','holdout_access',f"Direct access: {n.id}",src,getattr(n,'lineno',None),"","Remove holdout refs",0.95))
        
        tainted = SemanticAnalyzer(names).analyze(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func.attr.lower() if isinstance(n.func, ast.Attribute) else n.func.id.lower() if isinstance(n.func, ast.Name) else ''
                if any(t in fn for t in ['fit','train','optimize']):
                    args = {x.id for a in n.args+[k.value for k in n.keywords] for x in ast.walk(a) if isinstance(x,ast.Name)}
                    leaked = args & tainted
                    if leaked:
                        violations.append(CodeViolation('critical','indirect_access',f"Tainted var {leaked} in {fn}",src,getattr(n,'lineno',None),"","Trace data flow",0.85,'semantic'))
        return violations

# ═══════════════════════════════════════════════════════════════════════════
# REPORTER
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    @staticmethod
    def html_report(a, path=None):
        v = a.get('verdict','UNKNOWN')
        color = {'VALIDATED':'#28a745','FALSIFIED':'#dc3545'}.get(v,'#6c757d')
        html = f"""<!DOCTYPE html><html><head><title>ECLIPSE Report</title>
<style>body{{font-family:Arial;max-width:800px;margin:auto;padding:20px}}
.verdict{{background:{color};color:white;padding:20px;text-align:center;font-size:2em;border-radius:8px}}</style></head>
<body><h1>ECLIPSE v{__version__}</h1><p>Project: {a.get('project_name')}</p><p>Researcher: {a.get('researcher')}</p>
<div class="verdict">{v}</div><p>Criteria: {a.get('required_criteria_passed')}</p></body></html>"""
        if path:
            with open(path,'w') as f: f.write(html)
        return html
    
    @staticmethod
    def text_report(a, path=None):
        txt = f"ECLIPSE v{__version__} REPORT\n{'='*40}\nVerdict: {a.get('verdict')}\nCriteria: {a.get('required_criteria_passed')}\n"
        if path:
            with open(path,'w') as f: f.write(txt)
        return txt

# ═══════════════════════════════════════════════════════════════════════════
# MAIN FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    VERSION = __version__
    
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
        
        if self.results_file.exists():
            try:
                with open(self.results_file) as f: d = json.load(f)
                self._development_completed = 'development_summary' in d
                self._validation_completed = 'validation_summary' in d
            except: pass
        
        self.timestamper = EclipseTimestamp(str(self.output_dir)) if config.enable_timestamping else None
        self._timestamp_proof = None
        
        print(f"{'='*60}\n🔬 ECLIPSE v{self.VERSION}\nProject: {config.project_name}\nTimestamping: {'ON' if config.enable_timestamping else 'OFF'}\n{'='*60}")
    
    def stage1_split(self, ids, force=False):
        if self.split_file.exists() and not force:
            with open(self.split_file) as f: d = json.load(f)
            self._split_completed = True
            return d['development_ids'], d['holdout_ids']
        
        np.random.seed(self.config.sacred_seed)
        shuffled = np.array(ids).copy()
        np.random.shuffle(shuffled)
        n = int(len(ids) * self.config.development_ratio)
        dev, hold = shuffled[:n].tolist(), shuffled[n:].tolist()
        
        h = hashlib.sha256(f"{self.config.sacred_seed}_{sorted(ids)}".encode()).hexdigest()
        d = {'project_name': self.config.project_name, 'split_date': datetime.now(timezone.utc).isoformat(),
             'sacred_seed': self.config.sacred_seed, 'n_development': len(dev), 'n_holdout': len(hold),
             'development_ids': dev, 'holdout_ids': hold, 'integrity_verification': {'hash': h}}
        
        with open(self.split_file, 'w') as f: json.dump(d, f, indent=2)
        print(f"✅ Split: {len(dev)} dev, {len(hold)} holdout")
        self._split_completed = True
        return dev, hold
    
    def stage2_register_criteria(self, criteria, force=False):
        if self.criteria_file.exists() and not force:
            with open(self.criteria_file) as f: d = json.load(f)
            self._criteria_registered = True
            return d
        
        clist = [c.to_dict() for c in criteria]
        h = hashlib.sha256(json.dumps(clist, sort_keys=True).encode()).hexdigest()
        d = {'project_name': self.config.project_name, 'registration_date': datetime.now(timezone.utc).isoformat(),
             'criteria': clist, 'criteria_hash': h, 'binding_declaration': 'BINDING'}
        
        if self.config.enable_timestamping and self.timestamper:
            self._timestamp_proof = self.timestamper.stamp_criteria(clist, self.config.timestamp_method, self.config.project_name,
                                                                    github_token=self.config.github_token)
            if self._timestamp_proof.status in ['success','pending']:
                d['external_timestamp'] = self._timestamp_proof.to_dict()
        
        with open(self.criteria_file, 'w') as f: json.dump(d, f, indent=2)
        print(f"✅ {len(criteria)} criteria registered. Hash: {h[:16]}...")
        self._criteria_registered = True
        return d
    
    def stage3_development(self, dev_data, train_fn, val_fn, **kw):
        from sklearn.model_selection import KFold
        n = len(dev_data) if hasattr(dev_data, '__len__') else dev_data.shape[0]
        kf = KFold(n_splits=self.config.n_folds_cv, shuffle=True, random_state=self.config.sacred_seed)
        
        results = []
        for fold, (tr, va) in enumerate(kf.split(range(n)), 1):
            try:
                model = train_fn(tr, **kw)
                metrics = val_fn(model, va, **kw)
                results.append({'fold': fold, 'metrics': metrics, 'status': 'success'})
                print(f"Fold {fold} ✅")
            except Exception as e:
                results.append({'fold': fold, 'status': 'failed', 'error': str(e)})
        
        ok = [r for r in results if r['status']=='success']
        if not ok: raise RuntimeError("All folds failed")
        
        agg = {}
        for k in ok[0]['metrics']:
            vals = [r['metrics'][k] for r in ok]
            agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals}
            print(f"{k}: {agg[k]['mean']:.4f} ± {agg[k]['std']:.4f}")
        
        self._development_completed = True
        return {'n_folds': self.config.n_folds_cv, 'aggregated_metrics': agg, 'fold_results': results}
    
    def stage4_validation(self, holdout_data, model, val_fn, force=False, **kw):
        if self._validation_completed and not force:
            raise RuntimeError("ALREADY VALIDATED! Single-shot only.")
        
        print(f"\n{'='*60}\n🎯 SINGLE-SHOT VALIDATION\n{'='*60}")
        
        if not self.config.non_interactive:
            c = input("Type 'I ACCEPT SINGLE-SHOT VALIDATION': ")
            if c != 'I ACCEPT SINGLE-SHOT VALIDATION':
                print("Cancelled")
                return None
        
        try:
            metrics = val_fn(model, holdout_data, **kw)
            clean = {k: (v.item() if hasattr(v,'item') else float(v)) for k,v in metrics.items()}
            print("Results:", clean)
            self._validation_completed = True
            return {'status': 'success', 'metrics': clean}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def stage5_assessment(self, dev_results, val_results, generate_reports=True):
        if not val_results or val_results.get('status') != 'success':
            raise RuntimeError("Validation failed")
        
        with open(self.criteria_file) as f: cd = json.load(f)
        criteria = [FalsificationCriteria(**c) for c in cd['criteria']]
        metrics = val_results.get('metrics', {})
        
        evals = []
        for c in criteria:
            v = metrics.get(c.name)
            passed = c.evaluate(v) if v is not None else False
            evals.append({'criterion': c.to_dict(), 'value': v, 'passed': passed})
            print(f"{'✅' if passed else '❌'} {c.name}: {v}")
        
        req = [e for e in evals if e['criterion']['is_required']]
        n_pass = sum(1 for e in req if e['passed'])
        verdict = "VALIDATED" if all(e['passed'] for e in req) else "FALSIFIED"
        
        print(f"\n{'='*60}\n{'✅' if verdict=='VALIDATED' else '❌'} VERDICT: {verdict} ({n_pass}/{len(req)})\n{'='*60}")
        
        assessment = {
            'project_name': self.config.project_name, 'researcher': self.config.researcher,
            'assessment_timestamp': datetime.now(timezone.utc).isoformat(), 'eclipse_version': self.VERSION,
            'development_summary': dev_results, 'validation_summary': val_results,
            'criteria_evaluation': evals, 'verdict': verdict, 'required_criteria_passed': f"{n_pass}/{len(req)}"
        }
        
        # Integrity metrics
        eis = EclipseIntegrityScore(self)
        eis_r = eis.compute_eis()
        stds = StatisticalTestDataSnooping(self)
        stds_r = stds.test()
        assessment['integrity_metrics'] = {'eis': eis_r, 'stds': stds_r}
        
        print(f"EIS: {eis_r['eis']:.4f} ({eis_r['interpretation']})")
        if stds_r.get('status')=='success':
            print(f"STDS: {stds_r['risk_level']} (max z={stds_r['max_z']:.2f})")
        
        with open(self.results_file, 'w') as f: json.dump(assessment, f, indent=2, default=str)
        
        if generate_reports:
            EclipseReporter.html_report(assessment, str(self.output_dir / f"{self.config.project_name}_REPORT.html"))
            EclipseReporter.text_report(assessment, str(self.output_dir / f"{self.config.project_name}_REPORT.txt"))
            eis.generate_report(str(self.output_dir / f"{self.config.project_name}_EIS.txt"))
            if stds_r.get('status')=='success':
                stds.generate_report(str(self.output_dir / f"{self.config.project_name}_STDS.txt"))
        
        return assessment
    
    def audit_code(self, code_paths=None, notebook_paths=None):
        return CodeAuditor(self).audit(code_paths, notebook_paths)
    
    def get_status(self):
        return {'version': self.VERSION, 'split': self._split_completed, 'criteria': self._criteria_registered,
                'development': self._development_completed, 'validation': self._validation_completed}

# ═══════════════════════════════════════════════════════════════════════════
# TESTS & DEMO
# ═══════════════════════════════════════════════════════════════════════════

def run_tests():
    print(f"\n{'='*60}\n🧪 UNIT TESTS\n{'='*60}")
    p, f = 0, 0
    
    # Test 1: z-score direction
    try:
        cv = [0.70, 0.72, 0.68, 0.71, 0.69]
        z = (0.85 - np.mean(cv)) / np.std(cv)
        assert z > 3, "Better should be positive"
        print("✅ Test 1: z-score direction")
        p += 1
    except Exception as e:
        print(f"❌ Test 1: {e}")
        f += 1
    
    # Test 2: criteria eval
    try:
        c = FalsificationCriteria("x", 0.5, ">=", "d")
        assert c.evaluate(0.6) and not c.evaluate(0.4)
        print("✅ Test 2: criteria evaluation")
        p += 1
    except Exception as e:
        print(f"❌ Test 2: {e}")
        f += 1
    
    # Test 3: config validation
    try:
        try:
            EclipseConfig("t", "r", 42, development_ratio=0.8, holdout_ratio=0.3)
            assert False
        except ValueError: pass
        print("✅ Test 3: config validation")
        p += 1
    except Exception as e:
        print(f"❌ Test 3: {e}")
        f += 1
    
    print(f"\n{'='*60}\nResults: {p} passed, {f} failed\n{'='*60}")
    return f == 0

def demo():
    print(f"\n{'='*60}\n🧠 DEMO\n{'='*60}")
    np.random.seed(42)
    n = 100
    ids = [f"s{i}" for i in range(n)]
    data = []
    for s in ids:
        for w in range(50):
            c = 1 if np.random.random() < 0.2 else 0
            phi = np.random.gamma(2,2) + (1 if c else 0)
            data.append({'id': s, 'c': c, 'phi': phi})
    df = pd.DataFrame(data)
    
    ts = datetime.now().strftime("%H%M%S")
    config = EclipseConfig(f"Demo_{ts}", "Demo", 2025, output_dir=f"./demo_{ts}", enable_timestamping=False)
    eclipse = EclipseFramework(config)
    
    dev_ids, hold_ids = eclipse.stage1_split(ids)
    dev_df = df[df['id'].isin(dev_ids)]
    hold_df = df[df['id'].isin(hold_ids)]
    
    criteria = [FalsificationCriteria("f1_score", 0.5, ">=", "F1>=0.5", True)]
    eclipse.stage2_register_criteria(criteria)
    
    def train(idx, **kw):
        d = dev_df.iloc[idx]
        best_t, best_f1 = 0, 0
        for t in np.linspace(d['phi'].min(), d['phi'].max(), 50):
            pred = (d['phi'] >= t).astype(int)
            tp = ((pred==1)&(d['c']==1)).sum()
            fp = ((pred==1)&(d['c']==0)).sum()
            fn = ((pred==0)&(d['c']==1)).sum()
            pr = tp/(tp+fp) if tp+fp else 0
            re = tp/(tp+fn) if tp+fn else 0
            f1 = 2*pr*re/(pr+re) if pr+re else 0
            if f1 > best_f1: best_t, best_f1 = t, f1
        return {'threshold': best_t}
    
    def val(m, idx, **kw):
        d = dev_df.iloc[idx]
        pred = (d['phi'] >= m['threshold']).astype(int)
        return EclipseValidator.binary_classification_metrics(d['c'], pred)
    
    dev_r = eclipse.stage3_development(list(range(len(dev_df))), train, val)
    
    final_m = train(list(range(len(dev_df))))
    
    def holdout_val(m, data, **kw):
        pred = (data['phi'] >= m['threshold']).astype(int)
        return EclipseValidator.binary_classification_metrics(data['c'], pred)
    
    val_r = eclipse.stage4_validation(hold_df, final_m, holdout_val)
    if val_r:
        eclipse.stage5_assessment(dev_r, val_r)
        print(f"\n✅ Demo complete! Output: {config.output_dir}")

def main():
    parser = argparse.ArgumentParser(description=f"ECLIPSE v{__version__}")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--version', action='store_true')
    args = parser.parse_args()
    
    if args.version:
        print(f"ECLIPSE v{__version__}")
    elif args.test:
        sys.exit(0 if run_tests() else 1)
    elif args.demo:
        demo()
    else:
        print(f"""
ECLIPSE v{__version__} - Systematic Falsification Framework

Usage:
  python eclipse_v4.py --test     Run tests
  python eclipse_v4.py --demo     Run demo
  python eclipse_v4.py --version  Show version

As library:
  from eclipse_v4 import EclipseFramework, EclipseConfig, FalsificationCriteria
""")

if __name__ == "__main__":
    main()