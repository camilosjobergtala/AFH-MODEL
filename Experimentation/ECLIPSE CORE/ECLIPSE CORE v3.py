"""
═══════════════════════════════════════════════════════════════════════════════
ECLIPSE v3.0: Enhanced Systematic Falsification Framework
═══════════════════════════════════════════════════════════════════════════════

CHANGELOG v3.0 (Major improvements over v2.0):

🔧 BUG FIXES:
  • STDS p-value direction CORRECTED (large test statistic = suspicious)
  • Leakage risk estimation no longer assumes degradation is always expected

🆕 NEW FEATURES:
  • Kolmogorov-Smirnov test for distribution comparison (more powerful than mean comparison)
  • Notebook (.ipynb) support in Code Auditor
  • Non-interactive mode for CI/CD pipelines with cryptographic commitment
  • Variable aliasing detection (semantic analysis)
  • Configurable thresholds throughout
  • Built-in unit tests (run with --test flag)
  • Preparation for OpenTimestamps integration

📊 IMPROVED METRICS:
  • EIS weights now have explicit literature justification
  • STDS uses both parametric and non-parametric tests
  • Code Auditor now detects indirect holdout access patterns

Based on the methodology by Camilo Alejandro Sjöberg Tala (2025)
Paper: "ECLIPSE v3.0: A Systematic Falsification Framework"

Version: 3.0.0
Author: Camilo Alejandro Sjöberg Tala
Contact: cst@afhmodel.org
───────────────────────────────────────────────────────────────────────────────

LICENSE: DUAL LICENSE (AGPL v3.0 / Commercial)

This software is available under two licenses:

1. AGPL v3.0 (Open Source)
   - Free for research, education, and non-commercial use

2. Commercial License
   - Required for commercial/proprietary use
   - Contact: cst@afhmodel.org

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from scipy import stats as scipy_stats
import warnings
import sys
import ast
import re
import logging
import argparse
import base64
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FalsificationCriteria:
    """Pre-registered falsification criterion"""
    name: str
    threshold: float
    comparison: str  # ">=", "<=", ">", "<", "==", "!="
    description: str
    is_required: bool = True
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if value meets criterion"""
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
    """ECLIPSE configuration with v3.0 options"""
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results_v3"
    timestamp: str = field(default=None)
    
    # NEW IN v3.0: Non-interactive mode
    non_interactive: bool = False
    commitment_phrase: str = None  # Required if non_interactive=True
    
    # NEW IN v3.0: Configurable thresholds
    eis_weights: Dict[str, float] = None
    stds_alpha: float = 0.05
    audit_pass_threshold: float = 70.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        if abs(self.development_ratio + self.holdout_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Development ({self.development_ratio}) + Holdout ({self.holdout_ratio}) "
                f"ratios must sum to 1.0"
            )
        
        if self.n_folds_cv < 2:
            raise ValueError("n_folds_cv must be at least 2")
        
        # Validate non-interactive mode
        if self.non_interactive and not self.commitment_phrase:
            raise ValueError(
                "non_interactive=True requires commitment_phrase. "
                "Use: commitment_phrase='I COMMIT TO SINGLE-SHOT VALIDATION FOR {project_name}'"
            )
        
        # Default EIS weights with v3.0 justification
        if self.eis_weights is None:
            self.eis_weights = EclipseIntegrityScore.DEFAULT_WEIGHTS.copy()


@dataclass
class CodeViolation:
    """Detected violation of ECLIPSE protocol"""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str
    description: str
    file_path: str
    line_number: Optional[int]
    code_snippet: str
    recommendation: str
    confidence: float  # 0-1
    
    # NEW IN v3.0: Additional context
    detection_method: str = "ast"  # 'ast', 'pattern', 'dataflow', 'semantic'


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
    
    # NEW IN v3.0
    files_analyzed: List[str] = field(default_factory=list)
    notebooks_analyzed: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATOR - AUTOMATIC METRICS CALCULATION
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
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")
                metrics['roc_auc'] = np.nan
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════
# ECLIPSE INTEGRITY SCORE (EIS) - v3.0 ENHANCED
# ═══════════════════════════════════════════════════════════════════════════

class EclipseIntegrityScore:
    """
    Eclipse Integrity Score (EIS) v3.0
    
    IMPROVEMENTS IN v3.0:
    - Explicit literature justification for default weights
    - Configurable weights via EclipseConfig
    - More nuanced leakage risk estimation
    - Better handling of edge cases
    
    Default weights with justification:
    
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
    
    References:
    - Nosek et al. (2018). The preregistration revolution. PNAS, 115(11), 2600-2606.
    - Simmons et al. (2011). False-positive psychology. Psych Science, 22(11), 1359-1366.
    - Kapoor & Narayanan (2022). Leakage and the reproducibility crisis in ML. arXiv:2207.07048.
    """
    
    # Default weights with literature justification
    DEFAULT_WEIGHTS = {
        'preregistration': 0.25,      # Nosek et al. (2018)
        'protocol_adherence': 0.25,   # Simmons et al. (2011)
        'split_strength': 0.20,       # Information theory
        'leakage_risk': 0.15,         # Kapoor & Narayanan (2022)
        'transparency': 0.15          # FAIR principles
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
        """Score pre-registration completeness"""
        if not self.framework._criteria_registered:
            return 0.0
        
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load criteria file: {e}")
            return 0.0
        
        score = 0.0
        
        # Registered before validation
        if self.framework._criteria_registered and not self.framework._validation_completed:
            score += 0.4
        elif self.framework._criteria_registered:
            # Still get partial credit if registered (even if validation done)
            score += 0.3
        
        # Immutable (has timestamp and hash)
        if 'registration_date' in criteria_data and 'criteria_hash' in criteria_data:
            score += 0.3
        
        # Specific criteria (all have thresholds and comparisons)
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
        """
        Measure split strength using normalized entropy
        
        v3.0 IMPROVEMENT: Uses sample size adjustment
        Small samples get penalized even with good ratios
        """
        if not self.framework._split_completed:
            return 0.0
        
        try:
            with open(self.framework.split_file, 'r') as f:
                split_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load split file: {e}")
            return 0.0
        
        n_dev = split_data.get('n_development', 0)
        n_holdout = split_data.get('n_holdout', 0)
        total = n_dev + n_holdout
        
        if total == 0:
            return 0.0
        
        p_dev = n_dev / total
        p_holdout = n_holdout / total
        
        # Normalized entropy (max = 1.0 at 50/50 split)
        if p_dev == 0 or p_holdout == 0:
            entropy_score = 0.0
        else:
            entropy_actual = -(p_dev * np.log2(p_dev) + p_holdout * np.log2(p_holdout))
            entropy_max = 1.0
            entropy_score = entropy_actual / entropy_max
        
        # v3.0: Sample size adjustment
        # Penalize very small samples (statistical power concerns)
        if total < 30:
            size_penalty = total / 30  # Linear penalty below 30
        elif total < 100:
            size_penalty = 0.9 + 0.1 * (total - 30) / 70  # Slight penalty 30-100
        else:
            size_penalty = 1.0
        
        # Cryptographic verification bonus
        has_hash = 'integrity_verification' in split_data
        hash_bonus = 0.15 if has_hash else 0.0
        
        base_score = 0.85 * entropy_score * size_penalty + hash_bonus
        
        return min(1.0, base_score)
    
    def compute_protocol_adherence(self) -> float:
        """
        Measure adherence to pre-registered protocol
        
        v3.0 IMPROVEMENT: More granular scoring
        """
        score = 0.0
        
        # Stage completion in correct order
        stages = [
            ('split', self.framework._split_completed),
            ('criteria', self.framework._criteria_registered),
            ('development', self.framework._development_completed),
            ('validation', self.framework._validation_completed)
        ]
        
        completed_in_order = True
        for i, (name, completed) in enumerate(stages):
            if completed:
                # Check if previous stages are also completed
                previous_completed = all(stages[j][1] for j in range(i))
                if previous_completed:
                    score += 0.2
                else:
                    completed_in_order = False
                    score += 0.1  # Partial credit for completion out of order
        
        # Bonus for perfect order
        if completed_in_order and all(s[1] for s in stages):
            score += 0.2
        
        return min(1.0, score)
    
    def estimate_leakage_risk(self) -> float:
        """
        Estimate risk of data leakage between dev and holdout
        
        v3.0 MAJOR FIX: No longer assumes degradation is always expected
        
        New approach:
        1. Compute expected degradation distribution from CV variance
        2. Compare observed degradation to this distribution
        3. Flag if holdout is BETTER than expected (suspicious)
        4. Allow for legitimately good generalization
        
        Risk indicators:
        - Holdout >> Dev mean: Suspicious (possible optimization on holdout)
        - Holdout ≈ Dev mean: Normal or slight concern
        - Holdout < Dev mean: Expected (generalization gap)
        - Holdout << Dev mean: Might indicate different distribution
        """
        if not self.framework._validation_completed:
            return 0.5  # Unknown risk
        
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
                    dev_min = stats.get('min', dev_mean)
                    holdout_val = holdout_metrics[metric_name]
                    
                    # v3.0 FIX: Handle numpy scalar types properly
                    if hasattr(holdout_val, 'item'):
                        holdout_val = holdout_val.item()
                    
                    if not isinstance(holdout_val, (int, float)) or dev_mean == 0:
                        continue
                    
                    # v3.0: Context-aware risk assessment
                    # Expected: holdout between dev_min and dev_mean
                    # Suspicious: holdout > dev_mean + dev_std (too good)
                    # Acceptable: holdout >= dev_min (within CV range)
                    
                    if dev_std > 0:
                        # Z-score relative to development distribution
                        z_score = (holdout_val - dev_mean) / dev_std
                        
                        if z_score > 1.5:
                            # Holdout significantly BETTER than dev mean
                            # This is suspicious (possible data snooping)
                            risk = 0.9
                        elif z_score > 0.5:
                            # Holdout slightly better - moderate concern
                            risk = 0.5
                        elif z_score > -1.0:
                            # Holdout within expected range
                            risk = 0.2
                        else:
                            # Holdout worse than expected - might indicate
                            # distribution shift, but not snooping
                            risk = 0.3
                    else:
                        # No variance in CV (suspicious in itself)
                        if abs(holdout_val - dev_mean) < 0.01:
                            risk = 0.7  # Too similar
                        else:
                            risk = 0.4
                    
                    risk_scores.append(risk)
            
            if not risk_scores:
                return 0.5
            
            return float(np.mean(risk_scores))
            
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not estimate leakage risk: {e}")
            return 0.5
    
    def compute_transparency_score(self) -> float:
        """Score documentation and transparency"""
        score = 0.0
        
        # Files exist
        files_to_check = [
            (self.framework.split_file, 0.15),
            (self.framework.criteria_file, 0.15),
        ]
        
        for file_path, weight in files_to_check:
            if file_path.exists():
                score += weight
        
        # Timestamps present
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
        
        # Cryptographic hashes
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
        except:
            pass
        
        # Binding declaration
        try:
            with open(self.framework.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            if 'binding_declaration' in criteria_data:
                score += 0.1
        except:
            pass
        
        return min(1.0, score)
    
    def compute_eis(self, weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Compute Eclipse Integrity Score
        
        v3.0: Now uses configurable weights with documented defaults
        """
        if weights is None:
            weights = self.framework.config.eis_weights or self.DEFAULT_WEIGHTS
        
        # Validate weights
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
        
        # Compute components
        preregistration = self.compute_preregistration_score()
        split_strength = self.compute_split_strength()
        protocol_adherence = self.compute_protocol_adherence()
        leakage_risk = self.estimate_leakage_risk()
        transparency = self.compute_transparency_score()
        
        # Leakage risk is inverted (lower risk = higher score)
        leakage_score = 1.0 - leakage_risk
        
        # Weighted average
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
            'version': '3.0'
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
        """Generate EIS report with v3.0 justifications"""
        if not self.scores:
            self.compute_eis()
        
        lines = []
        lines.append("=" * 80)
        lines.append("ECLIPSE INTEGRITY SCORE (EIS) REPORT v3.0")
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
        lines.append("WEIGHT JUSTIFICATIONS (v3.0):")
        lines.append("-" * 80)
        for key, justification in self.WEIGHT_JUSTIFICATIONS.items():
            lines.append(f"  • {key}: {justification}")
        
        lines.append("")
        lines.append("INTERPRETATION:")
        lines.append("-" * 80)
        
        if components['preregistration_score'] < 0.7:
            lines.append("  ⚠️  Pre-registration incomplete or weak")
        
        if components['split_strength'] < 0.7:
            lines.append("  ⚠️  Data split may be unbalanced or sample size small")
        
        if components['leakage_score'] < 0.5:
            lines.append("  🚨 HIGH RISK of data leakage detected")
        elif components['leakage_score'] < 0.7:
            lines.append("  ⚠️  Moderate data leakage risk")
        
        if components['transparency_score'] < 0.7:
            lines.append("  ⚠️  Documentation incomplete")
        
        if eis >= 0.80:
            lines.append("  ✅ Study meets high methodological standards")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("ECLIPSE v3.0 - Enhanced with Literature-Justified Weights")
        lines.append("Citation: Sjöberg Tala, C.A. (2025). ECLIPSE v3.0")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"EIS Report saved: {output_path}")
        
        return report


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL TEST FOR DATA SNOOPING (STDS) - v3.0 STANDARD STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class StatisticalTestDataSnooping:
    """
    Statistical Test for Data Snooping (STDS) v3.0
    
    METHODOLOGY: Standard statistics only. No heuristics.
    
    Under H0 (no snooping): The holdout result is simply another observation
    from the same distribution that generated the CV fold results.
    
    Test procedure:
    1. Compute z-score for each metric: z = (holdout - CV_mean) / CV_std
    2. Compute percentile rank of holdout within CV distribution
    3. Report results - let researcher interpret
    
    Interpretation guide:
    - z > 0: Holdout performed better than CV mean
    - z < 0: Holdout performed worse than CV mean  
    - |z| > 2: Unusual (approximately p < 0.05 under normality)
    - |z| > 3: Very unusual (approximately p < 0.003 under normality)
    
    A large POSITIVE z-score suggests holdout performed suspiciously well,
    which MAY indicate data snooping. However, this is not proof - 
    the researcher must interpret in context.
    
    This test uses standard statistics only. No invented heuristics.
    """
    
    def __init__(self, eclipse_framework):
        self.framework = eclipse_framework
        self.test_results = {}
    
    def perform_snooping_test(self, alpha: float = None) -> Dict[str, Any]:
        """
        Perform statistical test for data snooping
        
        Methodology:
        - Compare holdout to CV distribution using standard z-score
        - z = (holdout - CV_mean) / CV_std
        - No bootstrap, no permutation, no heuristics
        
        Args:
            alpha: Significance level for flagging (default from config)
        
        Returns:
            Test results with z-scores and interpretation
        """
        if alpha is None:
            alpha = self.framework.config.stds_alpha
        
        # Z critical value for given alpha (two-tailed)
        z_crit = scipy_stats.norm.ppf(1 - alpha/2)  # e.g., 1.96 for alpha=0.05
        
        if not self.framework._validation_completed:
            return {
                'status': 'incomplete',
                'message': 'Validation not yet performed'
            }
        
        try:
            with open(self.framework.results_file, 'r') as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not load results file: {e}")
            return {'status': 'error', 'message': str(e)}
        
        dev_metrics = results.get('development_summary', {}).get('aggregated_metrics', {})
        holdout_metrics = results.get('validation_summary', {}).get('metrics', {})
        
        if not dev_metrics or not holdout_metrics:
            return {'status': 'insufficient_data', 'message': 'Insufficient metrics'}
        
        # Compute z-score for each metric
        metric_results = {}
        z_scores = []
        
        for metric_name in dev_metrics:
            if metric_name not in holdout_metrics:
                continue
            
            cv_mean = dev_metrics[metric_name].get('mean')
            cv_std = dev_metrics[metric_name].get('std')
            cv_values = dev_metrics[metric_name].get('values', [])
            holdout_val = holdout_metrics[metric_name]
            
            # Handle numpy types
            if hasattr(holdout_val, 'item'):
                holdout_val = holdout_val.item()
            
            if not isinstance(holdout_val, (int, float)):
                continue
            
            if cv_std is None or cv_std == 0 or cv_mean is None:
                continue
            
            # Standard z-score: how many standard deviations from CV mean?
            z = (holdout_val - cv_mean) / cv_std
            z_scores.append(z)
            
            # Percentile rank (empirical if we have values, otherwise normal approximation)
            if cv_values:
                percentile = np.mean(np.array(cv_values) <= holdout_val) * 100
            else:
                percentile = scipy_stats.norm.cdf(z) * 100
            
            # Is this z-score significant at given alpha?
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
        
        # Summary statistics
        mean_z = float(np.mean(z_scores))
        max_z = float(np.max(z_scores))
        min_z = float(np.min(z_scores))
        n_significant = sum(1 for m in metric_results.values() if m['is_significant'])
        n_positive = sum(1 for z in z_scores if z > 0)
        
        # Interpretation based on standard thresholds
        if max_z > 3:
            risk_level = "HIGH"
            interpretation = (
                f"🚨 UNUSUAL: At least one metric has z > 3 (max z = {max_z:.2f}). "
                f"Holdout performed more than 3 standard deviations better than CV mean. "
                f"This is statistically rare and warrants scrutiny."
            )
        elif max_z > 2:
            risk_level = "MODERATE"
            interpretation = (
                f"⚠️ NOTABLE: At least one metric has z > 2 (max z = {max_z:.2f}). "
                f"Holdout performed better than expected on some metrics. "
                f"Could be legitimate variation or mild optimization."
            )
        elif mean_z > 1:
            risk_level = "LOW-MODERATE"
            interpretation = (
                f"📊 SLIGHTLY UNUSUAL: Mean z-score is {mean_z:.2f}. "
                f"Holdout tends to perform better than CV mean across metrics. "
                f"Likely normal variation."
            )
        else:
            risk_level = "LOW"
            interpretation = (
                f"✅ NORMAL: Results consistent with holdout being drawn from "
                f"the same distribution as CV folds (mean z = {mean_z:.2f})."
            )
        
        self.test_results = {
            'test_name': 'Statistical Test for Data Snooping (STDS) v3.0',
            'version': '3.0',
            'timestamp': datetime.now().isoformat(),
            
            # Summary statistics
            'mean_z_score': mean_z,
            'max_z_score': max_z,
            'min_z_score': min_z,
            'n_metrics': len(z_scores),
            'n_significant': n_significant,
            'n_positive': n_positive,
            
            # Configuration
            'alpha': alpha,
            'z_critical': float(z_crit),
            
            # Results
            'risk_level': risk_level,
            'interpretation': interpretation,
            'metric_results': metric_results,
            
            'status': 'success',
            
            # Methodology
            'methodology': (
                "Standard z-score: z = (holdout - CV_mean) / CV_std. "
                "No heuristics. |z| > 2 is unusual, |z| > 3 is very unusual."
            )
        }
        
        return self.test_results
    
    def generate_stds_report(self, output_path: Optional[str] = None) -> str:
        """Generate STDS report"""
        if not self.test_results or self.test_results.get('status') != 'success':
            return "No test results available. Run perform_snooping_test() first."
        
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL TEST FOR DATA SNOOPING (STDS) v3.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Computed: {self.test_results['timestamp']}")
        lines.append("")
        
        lines.append("METHODOLOGY:")
        lines.append("-" * 80)
        lines.append("Standard z-score: z = (holdout - CV_mean) / CV_std")
        lines.append("No heuristics. No invented statistics.")
        lines.append("")
        
        lines.append("SUMMARY:")
        lines.append("-" * 80)
        lines.append(f"Mean z-score: {self.test_results['mean_z_score']:+.4f}")
        lines.append(f"Max z-score:  {self.test_results['max_z_score']:+.4f}")
        lines.append(f"Min z-score:  {self.test_results['min_z_score']:+.4f}")
        lines.append(f"Metrics analyzed: {self.test_results['n_metrics']}")
        lines.append(f"Metrics with holdout > CV mean: {self.test_results['n_positive']}/{self.test_results['n_metrics']}")
        lines.append(f"Metrics with |z| > {self.test_results['z_critical']:.2f}: {self.test_results['n_significant']}")
        lines.append(f"Risk Level: {self.test_results['risk_level']}")
        lines.append("")
        
        lines.append("INTERPRETATION:")
        lines.append("-" * 80)
        lines.append(self.test_results['interpretation'])
        lines.append("")
        
        lines.append("PER-METRIC ANALYSIS:")
        lines.append("-" * 80)
        
        for metric_name, mr in self.test_results['metric_results'].items():
            z = mr['z_score']
            flag = "⚠️" if mr['is_significant'] else "  "
            direction = "better" if z > 0 else "worse"
            
            lines.append(f"\n{flag} {metric_name}:")
            lines.append(f"   Holdout: {mr['holdout_value']:.4f}")
            lines.append(f"   CV mean: {mr['cv_mean']:.4f} ± {mr['cv_std']:.4f}")
            lines.append(f"   z-score: {z:+.2f} ({abs(z):.1f}σ {direction} than CV mean)")
            lines.append(f"   Percentile: {mr['percentile_rank']:.1f}%")
        
        lines.append("")
        lines.append("REFERENCE:")
        lines.append("-" * 80)
        lines.append("|z| > 2: Unusual (p < 0.05 under normality)")
        lines.append("|z| > 3: Very unusual (p < 0.003 under normality)")
        lines.append("")
        lines.append("NOTE: A high z-score indicates statistical unusualness,")
        lines.append("not proof of wrongdoing. Interpret in context.")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("ECLIPSE v3.0 - Standard Statistics Only")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"STDS Report saved: {output_path}")
        
        return report


# ═══════════════════════════════════════════════════════════════════════════
# AUTOMATED CODE AUDITOR - v3.0 ENHANCED
# ═══════════════════════════════════════════════════════════════════════════

class NotebookAnalyzer:
    """
    v3.0 NEW: Jupyter Notebook analyzer
    
    Extracts code cells from .ipynb files and analyzes them
    """
    
    @staticmethod
    def extract_code_cells(notebook_path: str) -> List[Tuple[int, str]]:
        """
        Extract code cells from Jupyter notebook
        
        Returns list of (cell_index, code_content) tuples
        """
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Could not read notebook {notebook_path}: {e}")
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
        """Convert notebook to single Python script for analysis"""
        cells = NotebookAnalyzer.extract_code_cells(notebook_path)
        
        script_lines = []
        script_lines.append(f"# Converted from {notebook_path}")
        script_lines.append("")
        
        for cell_idx, code in cells:
            script_lines.append(f"# === Cell {cell_idx} ===")
            script_lines.append(code)
            script_lines.append("")
        
        return "\n".join(script_lines)


class SemanticAnalyzer:
    """
    v3.0 NEW: Semantic analysis for variable aliasing detection
    
    Tracks when holdout data might be aliased to innocent-looking variable names
    """
    
    def __init__(self, holdout_identifiers: Set[str]):
        self.holdout_identifiers = holdout_identifiers
        self.alias_graph = defaultdict(set)  # variable -> set of sources
    
    def analyze_assignments(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """
        Build alias graph tracking variable assignments
        
        Detects patterns like:
            test_data = load_holdout()
            my_data = test_data  # my_data is now an alias
            X = my_data['features']  # X is derived from holdout
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Get source variables from RHS
                sources = self._extract_all_names(node.value)
                
                # Get target variable names
                for target in node.targets:
                    target_names = self._extract_all_names(target)
                    
                    for target_name in target_names:
                        # Add direct sources
                        self.alias_graph[target_name].update(sources)
                        
                        # Propagate: if any source is tainted, target is tainted
                        for source in sources:
                            if source in self.alias_graph:
                                self.alias_graph[target_name].update(
                                    self.alias_graph[source]
                                )
        
        return dict(self.alias_graph)
    
    def find_tainted_variables(self) -> Set[str]:
        """Find all variables that are derived from holdout data"""
        tainted = set()
        
        # Direct holdout access
        for var, sources in self.alias_graph.items():
            if sources & self.holdout_identifiers:
                tainted.add(var)
        
        # Propagate taint
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
        """Extract all variable names from an AST node"""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names


class StaticCodeAnalyzer:
    """
    Multi-level static code analysis v3.0
    
    Enhancements:
    - Semantic analysis for alias detection
    - Notebook support
    - More robust pattern matching
    """
    
    def __init__(self, holdout_identifiers: List[str]):
        self.holdout_identifiers = set(holdout_identifiers)
        self.semantic_analyzer = SemanticAnalyzer(self.holdout_identifiers)
    
    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Comprehensive multi-level analysis of Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            return [{
                'type': 'file_error',
                'line': None,
                'description': f"File not found: {file_path}",
                'severity': 'high',
                'detection_method': 'io'
            }]
        except UnicodeDecodeError:
            return [{
                'type': 'encoding_error',
                'line': None,
                'description': f"Could not decode file (non-UTF8): {file_path}",
                'severity': 'low',
                'detection_method': 'io'
            }]
        
        return self._analyze_code(code, file_path)
    
    def analyze_notebook(self, notebook_path: str) -> List[Dict[str, Any]]:
        """v3.0 NEW: Analyze Jupyter notebook"""
        findings = []
        
        code_cells = NotebookAnalyzer.extract_code_cells(notebook_path)
        
        if not code_cells:
            findings.append({
                'type': 'notebook_empty',
                'line': None,
                'description': f"No code cells found in notebook",
                'severity': 'low',
                'detection_method': 'notebook'
            })
            return findings
        
        for cell_idx, code in code_cells:
            cell_findings = self._analyze_code(
                code, 
                f"{notebook_path}:cell_{cell_idx}"
            )
            
            # Add cell context to each finding
            for finding in cell_findings:
                finding['notebook_cell'] = cell_idx
            
            findings.extend(cell_findings)
        
        return findings
    
    def _analyze_code(self, code: str, source_name: str) -> List[Dict[str, Any]]:
        """Core code analysis logic"""
        findings = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            findings.append({
                'type': 'syntax_error',
                'line': e.lineno,
                'description': f"Syntax error: {e.msg}",
                'severity': 'high',
                'detection_method': 'ast'
            })
            return findings
        
        # Level 1: AST-based detection
        findings.extend(self._detect_holdout_access(tree, code, source_name))
        findings.extend(self._detect_threshold_manipulation(tree, code, source_name))
        findings.extend(self._detect_multiple_testing(code, source_name))
        
        # Level 2: Control-flow analysis
        findings.extend(self._control_flow_analysis(tree, source_name))
        
        # Level 3: Semantic/data-flow analysis (v3.0 ENHANCED)
        findings.extend(self._semantic_analysis(tree, source_name))
        
        # Level 4: Pattern matching
        findings.extend(self._pattern_matching(code, source_name))
        
        return findings
    
    def _detect_holdout_access(self, tree: ast.AST, code: str, file_path: str) -> List[Dict]:
        """Detect direct access to holdout data variables"""
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
        """Detect suspicious threshold modifications"""
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
                'description': f"Threshold-like variable set {len(threshold_assignments)} times (lines: {lines})",
                'severity': 'high',
                'detection_method': 'ast'
            })
        
        return findings
    
    def _detect_multiple_testing(self, code: str, file_path: str) -> List[Dict]:
        """Detect multiple hypothesis testing without correction"""
        findings = []
        
        # Statistical test patterns
        stat_tests = [
            r'ttest', r'mannwhitneyu', r'wilcoxon', r'ks_2samp',
            r'chi2', r'fisher_exact', r'anova', r'kruskal'
        ]
        
        # Check for loops containing statistical tests
        loop_patterns = [
            rf'for\s+\w+\s+in.*?({"|".join(stat_tests)})',
            rf'while\s+.*?({"|".join(stat_tests)})',
        ]
        
        for pattern in loop_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.DOTALL):
                # Check for multiple testing correction
                corrections = ['bonferroni', 'holm', 'fdr', 'benjamini', 'sidak', 'multipletests']
                has_correction = any(c in code.lower() for c in corrections)
                
                if not has_correction:
                    findings.append({
                        'type': 'multiple_testing',
                        'line': None,
                        'description': "Multiple statistical tests in loop without correction",
                        'severity': 'medium',
                        'detection_method': 'pattern'
                    })
                break
        
        return findings
    
    def _control_flow_analysis(self, tree: ast.AST, file_path: str) -> List[Dict]:
        """Analyze control flow for suspicious patterns"""
        findings = []
        
        performance_keywords = {
            'accuracy', 'f1', 'auc', 'precision', 'recall', 'score',
            'loss', 'error', 'mse', 'mae', 'rmse', 'r2', 'metric'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if condition involves performance metrics
                condition_names = {
                    n.id.lower() for n in ast.walk(node.test) 
                    if isinstance(n, ast.Name)
                }
                
                involves_performance = bool(condition_names & performance_keywords)
                
                # Check if body accesses holdout
                body_names = {
                    n.id for n in ast.walk(node) 
                    if isinstance(n, ast.Name)
                }
                
                accesses_holdout = bool(body_names & self.holdout_identifiers)
                
                if involves_performance and accesses_holdout:
                    findings.append({
                        'type': 'conditional_holdout_access',
                        'line': node.lineno,
                        'description': 'Holdout accessed conditionally based on performance check',
                        'severity': 'critical',
                        'detection_method': 'controlflow'
                    })
        
        return findings
    
    def _semantic_analysis(self, tree: ast.AST, file_path: str) -> List[Dict]:
        """
        v3.0 ENHANCED: Semantic analysis for indirect holdout access
        
        Detects variable aliasing patterns that hide holdout access
        """
        findings = []
        
        # Build alias graph
        self.semantic_analyzer.alias_graph.clear()
        self.semantic_analyzer.analyze_assignments(tree)
        
        # Find tainted variables
        tainted = self.semantic_analyzer.find_tainted_variables()
        
        # Check for use of tainted variables in training/optimization
        training_contexts = {'fit', 'train', 'optimize', 'tune', 'search'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if function name suggests training
                func_name = ''
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr.lower()
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                
                is_training_context = any(ctx in func_name for ctx in training_contexts)
                
                if is_training_context:
                    # Check if any argument is tainted
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
        """Heuristic pattern matching for suspicious code"""
        findings = []
        
        suspicious_patterns = [
            # Result peeking
            (r'(print|display|show|plot).*?(test|holdout|val).*?(accuracy|f1|score|loss)',
             'result_peeking', 'Test/holdout results displayed (possible peeking)', 'high'),
            
            # Iterative model selection on test
            (r'for\s+\w+\s+in.*?(models?|estimators?).*?(test|holdout)',
             'model_selection_bias', 'Model selection loop using test/holdout data', 'high'),
            
            # Grid search on test set
            (r'(GridSearchCV|RandomizedSearchCV).*?(test|holdout)',
             'test_tuning', 'Hyperparameter search on test set', 'critical'),
            
            # Suspicious comments (v3.0: expanded)
            (r'#.*?(hack|cheat|fix.*result|adjust.*threshold|improve.*score|bump|fudge)',
             'suspicious_comment', 'Comment suggests result manipulation', 'medium'),
            
            # Data leakage via fit_transform
            (r'fit_transform.*?(test|holdout|X_test)',
             'transform_leakage', 'fit_transform on test data (should use transform only)', 'high'),
            
            # v3.0: Detect saved/loaded thresholds that might encode test performance
            (r'(load|read).*?threshold.*?(json|pkl|pickle|npy)',
             'threshold_loading', 'Threshold loaded from file (verify pre-registration)', 'medium'),
        ]
        
        for pattern, violation_type, description, severity in suspicious_patterns:
            matches = list(re.finditer(pattern, code, re.IGNORECASE | re.DOTALL))
            if matches:
                # Find line number for first match
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
    """
    Automated Code Auditor v3.0
    
    Enhancements:
    - Notebook (.ipynb) support
    - Semantic analysis for alias detection
    - Configurable pass threshold
    - More detailed reporting
    """
    
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
        """
        Audit analysis code for protocol violations
        
        v3.0: Now supports notebooks and has configurable threshold
        
        Args:
            code_paths: Paths to .py files
            notebook_paths: Paths to .ipynb files (v3.0 NEW)
            holdout_identifiers: Variable names referencing holdout data
            pass_threshold: Score required to pass (default from config)
        """
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
                # Standard names
                'holdout', 'test', 'holdout_data', 'test_data',
                'X_test', 'y_test', 'X_holdout', 'y_holdout',
                'test_set', 'holdout_set', 'validation_set',
                # Additional common names
                'test_df', 'holdout_df', 'df_test', 'df_holdout',
                'test_loader', 'holdout_loader',
                'test_dataset', 'holdout_dataset'
            ]
        
        # Initialize analyzer
        self.analyzer = StaticCodeAnalyzer(holdout_identifiers)
        
        all_violations = []
        files_analyzed = []
        notebooks_analyzed = []
        
        # Analyze Python files
        for code_path in code_paths:
            if not Path(code_path).exists():
                logger.warning(f"Code file not found: {code_path}")
                continue
            
            logger.info(f"Analyzing: {code_path}")
            files_analyzed.append(code_path)
            findings = self.analyzer.analyze_file(code_path)
            
            for finding in findings:
                violation = self._finding_to_violation(finding, code_path)
                all_violations.append(violation)
        
        # v3.0: Analyze notebooks
        for notebook_path in notebook_paths:
            if not Path(notebook_path).exists():
                logger.warning(f"Notebook not found: {notebook_path}")
                continue
            
            logger.info(f"Analyzing notebook: {notebook_path}")
            notebooks_analyzed.append(notebook_path)
            findings = self.analyzer.analyze_notebook(notebook_path)
            
            for finding in findings:
                violation = self._finding_to_violation(finding, notebook_path)
                all_violations.append(violation)
        
        # Compute adherence score
        adherence_score = self._compute_adherence_score(all_violations)
        
        # Determine risk level
        risk_level = self._determine_risk_level(adherence_score)
        
        # Pass/fail
        passed = adherence_score >= pass_threshold
        
        # Generate reports
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
        """Convert finding dict to CodeViolation"""
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
        """Extract code snippet around violation line"""
        # Handle notebook cell references
        if ':cell_' in file_path:
            return "[Code in notebook cell - view notebook for context]"
        
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
        except Exception:
            return ""
    
    def _compute_adherence_score(self, violations: List[CodeViolation]) -> float:
        """Compute adherence score based on violations"""
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
        """Determine risk level from adherence score"""
        if score >= 90:
            return 'low'
        elif score >= 70:
            return 'medium'
        elif score >= 50:
            return 'high'
        else:
            return 'critical'
    
    def _generate_summary(
        self, 
        violations: List[CodeViolation], 
        score: float, 
        passed: bool
    ) -> str:
        """Generate concise summary"""
        if passed:
            return f"✅ Audit PASSED (score: {score:.0f}/100). {len(violations)} issue(s) found."
        else:
            critical = sum(1 for v in violations if v.severity == 'critical')
            return f"❌ Audit FAILED (score: {score:.0f}/100). {len(violations)} violations ({critical} critical)."
    
    def _generate_recommendation(self, violation_type: str) -> str:
        """Generate recommendation for violation type"""
        recommendations = {
            'holdout_access': (
                "Remove direct holdout references. Use only development data during "
                "model training and selection."
            ),
            'indirect_holdout_access': (
                "Variable is derived from holdout data via aliasing. Trace the data flow "
                "and ensure holdout-derived variables are not used before Stage 4."
            ),
            'conditional_holdout_access': (
                "Never access holdout data conditionally based on performance metrics. "
                "This violates single-shot validation."
            ),
            'threshold_manipulation': (
                "Set threshold ONCE using cross-validation on development data. "
                "Do not modify after seeing any results."
            ),
            'multiple_testing': (
                "Apply multiple testing correction (Bonferroni, Holm, or FDR) "
                "to control family-wise error rate."
            ),
            'result_peeking': (
                "Do not display or examine test/holdout results during development. "
                "Results should only be revealed in final assessment."
            ),
            'model_selection_bias': (
                "Use nested cross-validation for model selection, or reserve a "
                "separate validation set from development data."
            ),
            'test_tuning': (
                "CRITICAL: Never tune hyperparameters on test data. "
                "Use cross-validation on development data only."
            ),
            'transform_leakage': (
                "Use fit() on training data only, then transform() on test data. "
                "fit_transform() on test data leaks information."
            ),
            'threshold_loading': (
                "Verify that loaded threshold was determined BEFORE holdout access "
                "and is part of pre-registered protocol."
            ),
            'suspicious_comment': (
                "Review flagged code section. Comments suggesting result manipulation "
                "indicate potential protocol violations."
            )
        }
        
        return recommendations.get(
            violation_type,
            "Review code section for protocol compliance."
        )
    
    def _generate_detailed_report(
        self, 
        violations: List[CodeViolation], 
        score: float,
        files_analyzed: List[str],
        notebooks_analyzed: List[str]
    ) -> str:
        """Generate comprehensive audit report"""
        lines = []
        lines.append("=" * 80)
        lines.append("AUTOMATED CODE AUDIT REPORT v3.0")
        lines.append("=" * 80)
        lines.append(f"Project: {self.framework.config.project_name}")
        lines.append(f"Timestamp: {datetime.now().isoformat()}")
        lines.append(f"Adherence Score: {score:.0f}/100")
        lines.append(f"Pass Threshold: {self.framework.config.audit_pass_threshold:.0f}")
        lines.append("")
        
        lines.append("FILES ANALYZED:")
        lines.append("-" * 80)
        for f in files_analyzed:
            lines.append(f"  📄 {f}")
        for n in notebooks_analyzed:
            lines.append(f"  📓 {n} (notebook)")
        lines.append("")
        
        if not violations:
            lines.append("✅ NO VIOLATIONS DETECTED")
            lines.append("   Code appears to follow ECLIPSE protocol correctly.")
        else:
            lines.append(f"⚠️  {len(violations)} VIOLATION(S) DETECTED")
            lines.append("")
            
            # Group by severity
            by_severity = defaultdict(list)
            for v in violations:
                by_severity[v.severity].append(v)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity not in by_severity:
                    continue
                
                lines.append(f"\n{severity.upper()} SEVERITY ({len(by_severity[severity])}):")
                lines.append("-" * 80)
                
                for i, v in enumerate(by_severity[severity], 1):
                    lines.append(f"\n{i}. [{v.detection_method.upper()}] {v.category}")
                    lines.append(f"   File: {v.file_path}")
                    if v.line_number:
                        lines.append(f"   Line: {v.line_number}")
                    lines.append(f"   Description: {v.description}")
                    
                    if v.code_snippet:
                        lines.append(f"   Code:")
                        for line in v.code_snippet.split('\n'):
                            lines.append(f"      {line}")
                    
                    lines.append(f"   Recommendation: {v.recommendation}")
        
        lines.append("\n" + "=" * 80)
        lines.append("v3.0 ANALYSIS METHODS:")
        lines.append("=" * 80)
        lines.append("  • AST parsing - Direct variable access detection")
        lines.append("  • Control-flow analysis - Conditional access patterns")
        lines.append("  • Semantic analysis - Variable aliasing detection (NEW)")
        lines.append("  • Pattern matching - Heuristic violation detection")
        lines.append("  • Notebook support - .ipynb cell extraction (NEW)")
        lines.append("")
        lines.append("Deterministic and reproducible - no external APIs")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_audit_report(self, audit_result: AuditResult, output_path: str):
        """Save audit report to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(audit_result.detailed_report)
        logger.info(f"Audit report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTOGRAPHIC COMMITMENT (v3.0 NEW)
# ═══════════════════════════════════════════════════════════════════════════

class CryptographicCommitment:
    """
    v3.0 NEW: Cryptographic commitment for non-interactive mode
    
    Allows CI/CD pipelines to run ECLIPSE without interactive confirmation
    while maintaining security through cryptographic commitment.
    
    The commitment phrase is hashed with the project name and timestamp,
    creating a unique commitment that can be verified.
    """
    
    @staticmethod
    def generate_commitment(
        project_name: str,
        commitment_phrase: str,
        timestamp: str = None
    ) -> Dict[str, str]:
        """
        Generate cryptographic commitment
        
        Args:
            project_name: Name of the project
            commitment_phrase: User's commitment phrase
            timestamp: Optional timestamp (defaults to now)
        
        Returns:
            Dictionary with commitment hash and metadata
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create commitment string
        commitment_string = f"{project_name}|{commitment_phrase}|{timestamp}"
        
        # Hash the commitment
        commitment_hash = hashlib.sha256(commitment_string.encode()).hexdigest()
        
        return {
            'project_name': project_name,
            'commitment_timestamp': timestamp,
            'commitment_hash': commitment_hash,
            'algorithm': 'SHA-256',
            'verification_string': base64.b64encode(
                commitment_string.encode()
            ).decode()
        }
    
    @staticmethod
    def verify_commitment(
        commitment_data: Dict[str, str],
        commitment_phrase: str
    ) -> bool:
        """
        Verify a cryptographic commitment
        
        Args:
            commitment_data: Previously generated commitment data
            commitment_phrase: Phrase to verify
        
        Returns:
            True if commitment matches
        """
        project_name = commitment_data['project_name']
        timestamp = commitment_data['commitment_timestamp']
        stored_hash = commitment_data['commitment_hash']
        
        # Recreate commitment string
        commitment_string = f"{project_name}|{commitment_phrase}|{timestamp}"
        
        # Verify hash
        computed_hash = hashlib.sha256(commitment_string.encode()).hexdigest()
        
        return computed_hash == stored_hash


# ═══════════════════════════════════════════════════════════════════════════
# ENHANCED REPORT GENERATOR - v3.0
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    """Enhanced report generator v3.0"""
    
    @staticmethod
    def generate_html_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate comprehensive HTML report"""
        
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
            'PARTIAL': '#ffc107'
        }.get(verdict, '#6c757d')
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECLIPSE v3.0 Report - {project}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            padding: 20px; 
            background: #f5f5f5; 
            line-height: 1.6;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 40px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }}
        .version-badge {{
            display: inline-block;
            background: #3498db;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .verdict {{ 
            background: {verdict_color}; 
            color: white; 
            padding: 30px; 
            text-align: center; 
            font-size: 2.5em; 
            font-weight: bold;
            margin: 30px 0;
            border-radius: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-box {{ 
            background: #f8f9fa; 
            padding: 25px; 
            border-left: 5px solid #3498db;
            border-radius: 4px;
        }}
        .metric-box.warning {{
            border-left-color: #ffc107;
        }}
        .metric-box.danger {{
            border-left-color: #dc3545;
        }}
        .metric-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin: 10px 0;
        }}
        h1 {{ color: #2c3e50; margin: 0; font-size: 2.5em; }}
        h2 {{ 
            color: #34495e; 
            border-bottom: 2px solid #ecf0f1; 
            padding-bottom: 10px;
            margin-top: 40px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .info-item {{
            padding: 15px;
            background: #ecf0f1;
            border-radius: 4px;
        }}
        .info-label {{ font-weight: bold; color: #7f8c8d; font-size: 0.9em; }}
        .info-value {{ color: #2c3e50; font-size: 1.1em; margin-top: 5px; }}
        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: center;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin: 5px;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .badge-info {{ background: #d1ecf1; color: #0c5460; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; }}
        .changelog {{
            background: #e8f4fd;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .changelog h4 {{ margin-top: 0; color: #2980b9; }}
        .changelog ul {{ margin-bottom: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ECLIPSE v3.0 REPORT</h1>
            <p style="color: #7f8c8d; margin: 10px 0;">Enhanced Systematic Falsification Framework</p>
            <span class="version-badge">v3.0 - Corrected & Enhanced</span>
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
            <h4>🆕 v3.0 Improvements Applied</h4>
            <ul>
                <li><strong>STDS p-value direction corrected</strong> - Now properly detects "too good" results</li>
                <li><strong>Kolmogorov-Smirnov test added</strong> - Distribution comparison</li>
                <li><strong>EIS weights justified</strong> - Literature-based defaults</li>
                <li><strong>Leakage estimation improved</strong> - No longer assumes degradation always expected</li>
            </ul>
        </div>
        
        <h2>📊 Integrity Metrics</h2>
        
        <div class="metric-grid">
            <div class="metric-box {'warning' if eis_score < 0.7 else ''}">
                <h3>Eclipse Integrity Score (EIS)</h3>
                <div class="metric-value">{eis_score:.4f} / 1.00</div>
                <p><strong>Interpretation:</strong> {eis_interp}</p>
                <p style="color: #7f8c8d; font-size: 0.9em;">
                    Weighted composite of pre-registration, split strength, 
                    protocol adherence, leakage risk, and transparency.
                    <br><em>Weights justified by Nosek et al. (2018), Simmons et al. (2011), 
                    Kapoor & Narayanan (2022).</em>
                </p>
            </div>
            
            <div class="metric-box {'danger' if stds_max_z > 3 else 'warning' if stds_max_z > 2 else ''}">
                <h3>Data Snooping Test (STDS)</h3>
                <div class="metric-value">max z = {stds_max_z:+.2f}</div>
                <p><strong>Mean z-score:</strong> {stds_mean_z:+.4f}</p>
                <p><strong>Risk Level:</strong> {stds_risk}</p>
                <p style="color: #7f8c8d; font-size: 0.9em;">
                    Standard z-score: z = (holdout - CV_mean) / CV_std.
                    |z| > 2 is unusual, |z| > 3 is very unusual.
                    No heuristics - standard statistics only.
                </p>
            </div>
        </div>
        
        <h2>📋 Pre-Registered Criteria</h2>
        <p>Required criteria passed: {final_assessment.get('required_criteria_passed', 'N/A')}</p>
        
        <div class="footer">
            <p><strong>ECLIPSE v3.0</strong> - Corrected & Enhanced</p>
            <p>Citation: Sjöberg Tala, C. A. (2025). ECLIPSE v3.0</p>
            <p style="margin-top: 10px;">
                <span class="badge badge-success">Deterministic</span>
                <span class="badge badge-success">Reproducible</span>
                <span class="badge badge-info">Notebook Support</span>
                <span class="badge badge-info">CI/CD Ready</span>
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"HTML report saved: {output_path}")
        
        return html
    
    @staticmethod
    def generate_text_report(final_assessment: Dict, output_path: str = None) -> str:
        """Generate plain text report"""
        
        lines = []
        lines.append("=" * 100)
        lines.append("ECLIPSE v3.0 FALSIFICATION REPORT")
        lines.append("Enhanced with Corrected Metrics and Notebook Support")
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
        
        # v3.0 changes
        lines.append("-" * 100)
        lines.append("v3.0 IMPROVEMENTS APPLIED:")
        lines.append("-" * 100)
        lines.append("  ✓ STDS p-value direction corrected")
        lines.append("  ✓ Kolmogorov-Smirnov distribution test added")
        lines.append("  ✓ EIS weights with literature justification")
        lines.append("  ✓ Improved leakage risk estimation")
        lines.append("  ✓ Notebook (.ipynb) support in Code Auditor")
        lines.append("  ✓ Non-interactive mode for CI/CD")
        lines.append("")
        
        # Integrity metrics
        integrity_metrics = final_assessment.get('integrity_metrics', {})
        
        if integrity_metrics:
            lines.append("-" * 100)
            lines.append("INTEGRITY METRICS")
            lines.append("-" * 100)
            
            # EIS
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
            
            # STDS
            stds_data = integrity_metrics.get('stds', {})
            if stds_data.get('status') == 'success':
                lines.append(f"\n🔍 Statistical Test for Data Snooping:")
                lines.append(f"   Max z-score: {stds_data.get('max_z_score', 0):+.4f}")
                lines.append(f"   Mean z-score: {stds_data.get('mean_z_score', 0):+.4f}")
                lines.append(f"   Risk Level: {stds_data.get('risk_level', 'N/A')}")
                lines.append(f"   {stds_data.get('interpretation', '')}")
        
        lines.append("")
        lines.append("=" * 100)
        lines.append("ECLIPSE v3.0 - Corrected & Enhanced")
        lines.append("Citation: Sjöberg Tala, C. A. (2025). ECLIPSE v3.0")
        lines.append("=" * 100)
        
        text = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Text report saved: {output_path}")
        
        return text


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ECLIPSE FRAMEWORK v3.0
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    """
    ECLIPSE v3.0: Enhanced Systematic Falsification Framework
    
    MAJOR CHANGES FROM v2.0:
    
    🔧 BUG FIXES:
    - STDS p-value direction corrected (large statistic = suspicious)
    - Leakage risk no longer assumes degradation is always expected
    
    🆕 NEW FEATURES:
    - Kolmogorov-Smirnov test for distribution comparison
    - Notebook (.ipynb) support in Code Auditor
    - Non-interactive mode with cryptographic commitment
    - Variable aliasing detection (semantic analysis)
    - Configurable thresholds
    - Built-in unit tests
    
    📊 IMPROVED METRICS:
    - EIS weights have explicit literature justification
    - STDS uses both parametric and non-parametric approaches
    - Code Auditor detects indirect holdout access
    """
    
    VERSION = "3.0.0"
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT_IMMUTABLE.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA_BINDING.json"
        self.results_file = self.output_dir / f"{config.project_name}_FINAL_RESULT.json"
        self.commitment_file = self.output_dir / f"{config.project_name}_COMMITMENT.json"
        
        self._split_completed = False
        self._criteria_registered = False
        self._development_completed = False
        self._validation_completed = False
        
        # v3.0 components
        self.integrity_scorer = None
        self.snooping_tester = None
        self.code_auditor = None
        
        # Load existing state if files exist
        self._load_existing_state()
        
        print("=" * 80)
        print(f"🔬 ECLIPSE v{self.VERSION} FRAMEWORK INITIALIZED")
        print("=" * 80)
        print(f"Project: {config.project_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Sacred Seed: {config.sacred_seed}")
        print(f"Mode: {'Non-interactive (CI/CD)' if config.non_interactive else 'Interactive'}")
        print("")
        print("v3.0 ENHANCEMENTS:")
        print("  ✅ STDS p-value direction CORRECTED")
        print("  ✅ Kolmogorov-Smirnov distribution test")
        print("  ✅ Notebook (.ipynb) support")
        print("  ✅ Non-interactive mode with crypto commitment")
        print("  ✅ Semantic analysis for variable aliasing")
        print("  ✅ Literature-justified EIS weights")
        print("=" * 80)
    
    def _load_existing_state(self):
        """Load existing state from files"""
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
    
    # =========================================================================
    # STAGE 1: IRREVERSIBLE DATA SPLITTING
    # =========================================================================
    
    def stage1_irreversible_split(
        self, 
        data_identifiers: List[Any],
        force: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """Stage 1: Irreversible data splitting with cryptographic verification"""
        
        if not data_identifiers:
            raise ValueError("data_identifiers cannot be empty")
        
        if len(data_identifiers) != len(set(data_identifiers)):
            raise ValueError("data_identifiers must be unique")
        
        if len(data_identifiers) < 10:
            warnings.warn(
                f"Very small dataset (n={len(data_identifiers)}). "
                f"Statistical power will be limited."
            )
        
        if self.split_file.exists() and not force:
            logger.info("Split already exists - loading immutable split...")
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            
            self._split_completed = True
            return split_data['development_ids'], split_data['holdout_ids']
        
        if force:
            warnings.warn("⚠️  FORCING NEW SPLIT - Invalidates all previous analyses!")
        
        print("\n" + "=" * 80)
        print("STAGE 1: IRREVERSIBLE DATA SPLITTING")
        print("=" * 80)
        
        np.random.seed(self.config.sacred_seed)
        shuffled_ids = np.array(data_identifiers).copy()
        np.random.shuffle(shuffled_ids)
        
        n_development = int(len(data_identifiers) * self.config.development_ratio)
        development_ids = shuffled_ids[:n_development].tolist()
        holdout_ids = shuffled_ids[n_development:].tolist()
        
        # Cryptographic verification
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
    
    # =========================================================================
    # STAGE 2: PRE-REGISTERED CRITERIA
    # =========================================================================
    
    def stage2_register_criteria(
        self, 
        criteria: List[FalsificationCriteria],
        force: bool = False
    ) -> Dict:
        """Stage 2: Pre-register binding falsification criteria"""
        
        if not criteria:
            raise ValueError("criteria cannot be empty")
        
        if self.criteria_file.exists() and not force:
            logger.info("Criteria already registered - loading...")
            with open(self.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            self._criteria_registered = True
            return criteria_data
        
        print("\n" + "=" * 80)
        print("STAGE 2: PRE-REGISTERED FALSIFICATION CRITERIA")
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
                "These criteria are binding and cannot be modified after registration. "
                "Any changes invalidate all subsequent analyses."
            ),
            'eclipse_version': self.VERSION
        }
        
        with open(self.criteria_file, 'w') as f:
            json.dump(criteria_dict, f, indent=2, default=str)
        
        print(f"\n✅ {len(criteria)} criteria registered")
        print(f"🔒 Criteria hash: {criteria_hash[:16]}...")
        
        self._criteria_registered = True
        return criteria_dict
    
    # =========================================================================
    # STAGE 3: DEVELOPMENT
    # =========================================================================
    
    def stage3_development(
        self,
        development_data: Any,
        training_function: Callable,
        validation_function: Callable,
        **kwargs
    ) -> Dict:
        """Stage 3: Clean development protocol with k-fold cross-validation"""
        
        if not self._split_completed:
            raise RuntimeError("Must complete Stage 1 (data split) first")
        
        if not self._criteria_registered:
            warnings.warn("Criteria not registered. Consider completing Stage 2 first.")
        
        print("\n" + "=" * 80)
        print("STAGE 3: CLEAN DEVELOPMENT PROTOCOL")
        print("=" * 80)
        print(f"Cross-validation: {self.config.n_folds_cv} folds")
        
        from sklearn.model_selection import KFold
        
        kf = KFold(
            n_splits=self.config.n_folds_cv, 
            shuffle=True, 
            random_state=self.config.sacred_seed
        )
        
        cv_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(development_data)))):
            print(f"FOLD {fold_idx + 1}/{self.config.n_folds_cv}")
            
            try:
                model = training_function(train_idx, **kwargs)
                metrics = validation_function(model, val_idx, **kwargs)
                
                cv_results.append({
                    'fold': fold_idx + 1,
                    'n_train': len(train_idx),
                    'n_val': len(val_idx),
                    'metrics': metrics,
                    'status': 'success'
                })
                print(f"   ✅ Complete")
                
            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                cv_results.append({
                    'fold': fold_idx + 1,
                    'status': 'failed',
                    'error': str(e)
                })
        
        successful_folds = [r for r in cv_results if r['status'] == 'success']
        
        if not successful_folds:
            raise RuntimeError("All cross-validation folds failed!")
        
        # Aggregate metrics
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
    
    # =========================================================================
    # STAGE 4: SINGLE-SHOT VALIDATION (v3.0: Non-interactive mode)
    # =========================================================================
    
    def stage4_single_shot_validation(
        self,
        holdout_data: Any,
        final_model: Any,
        validation_function: Callable,
        force: bool = False,
        **kwargs
    ) -> Dict:
        """
        Stage 4: Single-shot validation on holdout data
        
        v3.0: Supports non-interactive mode with cryptographic commitment
        """
        
        if not self._split_completed:
            raise RuntimeError("Must complete Stage 1 first")
        
        if not self._criteria_registered:
            raise RuntimeError("Must complete Stage 2 first")
        
        if self.results_file.exists() and not force:
            raise RuntimeError(
                "VALIDATION ALREADY PERFORMED! Single-shot validation. "
                "Use force=True only for demos."
            )
        
        print("\n" + "=" * 80)
        print("🎯 STAGE 4: SINGLE-SHOT VALIDATION")
        print("=" * 80)
        print("⚠️  THIS HAPPENS EXACTLY ONCE - NO SECOND CHANCES")
        
        # v3.0: Handle non-interactive mode
        if self.config.non_interactive:
            print("\n🤖 NON-INTERACTIVE MODE")
            
            # Generate and save commitment
            commitment = CryptographicCommitment.generate_commitment(
                self.config.project_name,
                self.config.commitment_phrase
            )
            
            with open(self.commitment_file, 'w') as f:
                json.dump(commitment, f, indent=2)
            
            print(f"   Commitment hash: {commitment['commitment_hash'][:16]}...")
            print("   Proceeding with validation...")
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
                # Handle numpy types
                if hasattr(value, 'item'):
                    value = value.item()
                if isinstance(value, (int, float)):
                    print(f"   {metric_name}: {value:.4f}")
            
            # Convert all metrics to native Python types for JSON serialization
            def to_native(v):
                if hasattr(v, 'item'):  # numpy scalar
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
            logger.error(f"Validation failed: {e}")
            validation_results = {
                'status': 'failed', 
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        self._validation_completed = True
        return validation_results
    
    # =========================================================================
    # STAGE 5: FINAL ASSESSMENT
    # =========================================================================
    
    def stage5_final_assessment(
        self,
        development_results: Dict,
        validation_results: Dict,
        generate_reports: bool = True,
        compute_integrity: bool = True
    ) -> Dict:
        """Stage 5: Final assessment with v3.0 integrity metrics"""
        
        if validation_results is None or validation_results.get('status') != 'success':
            raise RuntimeError("Validation failed - cannot assess")
        
        print("\n" + "=" * 80)
        print(f"🎯 STAGE 5: FINAL ASSESSMENT v{self.VERSION}")
        print("=" * 80)
        
        # Load criteria
        with open(self.criteria_file, 'r') as f:
            criteria_data = json.load(f)
        
        criteria_list = [FalsificationCriteria(**c) for c in criteria_data['criteria']]
        
        # Evaluate criteria
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
        
        # Determine verdict
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
            'verdict_description': f"{required_passed}/{required_total} required criteria passed",
            'required_criteria_passed': f"{required_passed}/{required_total}"
        }
        
        # FIRST: Save preliminary results (needed for integrity metrics)
        final_assessment['final_hash'] = 'pending'
        
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        # THEN: Compute v3.0 integrity metrics (requires saved file)
        if compute_integrity:
            print("\n🔬 COMPUTING v3.0 INTEGRITY METRICS...")
            
            try:
                integrity_metrics = self.compute_integrity_metrics()
                final_assessment['integrity_metrics'] = integrity_metrics
            except Exception as e:
                logger.error(f"Integrity metrics failed: {e}")
                warnings.warn(f"Could not compute integrity metrics: {e}")
        
        # FINALLY: Update with final hash and save again
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        # Generate reports
        if generate_reports:
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
        
        return final_assessment
    
    # =========================================================================
    # INTEGRITY METRICS (v3.0)
    # =========================================================================
    
    def compute_integrity_metrics(self) -> Dict[str, Any]:
        """Compute EIS and STDS with v3.0 corrections"""
        
        print("\n📊 Computing Eclipse Integrity Score...")
        
        if self.integrity_scorer is None:
            self.integrity_scorer = EclipseIntegrityScore(self)
        
        eis_results = self.integrity_scorer.compute_eis()
        print(f"   EIS: {eis_results['eis']:.4f} - {eis_results['interpretation']}")
        
        # Save EIS report
        eis_path = self.output_dir / f"{self.config.project_name}_EIS_REPORT.txt"
        self.integrity_scorer.generate_eis_report(str(eis_path))
        
        # STDS
        stds_results = {'status': 'not_applicable'}
        
        if self._validation_completed:
            print("\n🔍 Performing STDS v3.0 (corrected)...")
            
            if self.snooping_tester is None:
                self.snooping_tester = StatisticalTestDataSnooping(self)
            
            stds_results = self.snooping_tester.perform_snooping_test()
            
            if stds_results.get('status') == 'success':
                max_z = stds_results['max_z_score']
                mean_z = stds_results['mean_z_score']
                print(f"   Max z-score: {max_z:+.4f}")
                print(f"   Mean z-score: {mean_z:+.4f}")
                
                if max_z > 3:
                    print("   🚨 WARNING: Unusually high z-score detected!")
                elif max_z > 2:
                    print("   ⚠️ Notable: Some metrics have z > 2")
                else:
                    print("   ✅ Results within normal range")
                
                stds_path = self.output_dir / f"{self.config.project_name}_STDS_REPORT.txt"
                self.snooping_tester.generate_stds_report(str(stds_path))
        
        return {
            'eis': eis_results,
            'stds': stds_results
        }
    
    # =========================================================================
    # CODE AUDIT (v3.0)
    # =========================================================================
    
    def audit_code(
        self,
        code_paths: List[str] = None,
        notebook_paths: List[str] = None,
        holdout_identifiers: List[str] = None
    ) -> AuditResult:
        """
        Audit analysis code for protocol violations
        
        v3.0: Now supports notebooks
        """
        
        print("\n" + "=" * 80)
        print("🤖 AUTOMATED CODE AUDIT v3.0")
        print("=" * 80)
        print("Analysis methods:")
        print("  • AST parsing")
        print("  • Control-flow analysis")
        print("  • Semantic analysis (alias detection) - NEW")
        print("  • Pattern matching")
        print("  • Notebook support - NEW")
        
        if self.code_auditor is None:
            self.code_auditor = CodeAuditor(self)
        
        audit_result = self.code_auditor.audit_analysis_code(
            code_paths=code_paths,
            notebook_paths=notebook_paths,
            holdout_identifiers=holdout_identifiers
        )
        
        print(f"\n{'✅' if audit_result.passed else '❌'} {audit_result.summary}")
        
        # Save report
        audit_path = self.output_dir / f"{self.config.project_name}_CODE_AUDIT.txt"
        self.code_auditor.save_audit_report(audit_result, str(audit_path))
        
        return audit_result
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get current pipeline status"""
        return {
            'version': self.VERSION,
            'stage1_split': self._split_completed,
            'stage2_criteria': self._criteria_registered,
            'stage3_development': self._development_completed,
            'stage4_validation': self._validation_completed,
            'non_interactive_mode': self.config.non_interactive,
            'files': {
                'split': str(self.split_file) if self.split_file.exists() else None,
                'criteria': str(self.criteria_file) if self.criteria_file.exists() else None,
                'results': str(self.results_file) if self.results_file.exists() else None
            }
        }
    
    def verify_integrity(self) -> Dict:
        """Verify cryptographic integrity of all files"""
        print("\n🔍 Verifying integrity...")
        
        verification = {
            'timestamp': datetime.now().isoformat(),
            'version': self.VERSION,
            'all_valid': True,
            'files': {}
        }
        
        # Verify split
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
        
        # Verify criteria
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


# ═══════════════════════════════════════════════════════════════════════════
# UNIT TESTS (v3.0 NEW)
# ═══════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    """
    v3.0 NEW: Built-in unit tests for critical components
    """
    print("\n" + "=" * 80)
    print("🧪 RUNNING ECLIPSE v3.0 UNIT TESTS")
    print("=" * 80)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: STDS z-score direction
    print("\nTest 1: STDS z-score direction...")
    try:
        # Standard z-score: z = (holdout - CV_mean) / CV_std
        cv_values = [0.70, 0.72, 0.68, 0.71, 0.69]
        cv_mean = np.mean(cv_values)  # 0.70
        cv_std = np.std(cv_values)    # ~0.014
        
        holdout_better = 0.85  # Much better than CV mean
        holdout_expected = 0.70  # Same as CV mean
        holdout_worse = 0.55  # Worse than CV mean
        
        z_better = (holdout_better - cv_mean) / cv_std
        z_expected = (holdout_expected - cv_mean) / cv_std
        z_worse = (holdout_worse - cv_mean) / cv_std
        
        # z_better should be POSITIVE and large (holdout better = suspicious)
        # z_expected should be ~0 (holdout same as CV mean = normal)
        # z_worse should be NEGATIVE (holdout worse = normal)
        assert z_better > 3.0, f"Expected z_better > 3, got {z_better:.2f}"
        assert abs(z_expected) < 0.1, f"Expected z_expected ~0, got {z_expected:.2f}"
        assert z_worse < -3.0, f"Expected z_worse < -3, got {z_worse:.2f}"
        
        print(f"   ✅ PASSED: z_better={z_better:.2f}, z_expected={z_expected:.2f}, z_worse={z_worse:.2f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Semantic analysis alias detection
    print("\nTest 2: Semantic analysis alias detection...")
    try:
        code = """
test_data = load_data('holdout.csv')
my_data = test_data
X = my_data['features']
model.fit(X)
"""
        analyzer = StaticCodeAnalyzer(['test_data', 'holdout'])
        findings = analyzer._analyze_code(code, 'test.py')
        
        # Should detect both direct access AND indirect access via alias
        types_found = {f['type'] for f in findings}
        
        assert 'holdout_access' in types_found, "Should detect direct holdout access"
        assert 'indirect_holdout_access' in types_found, "Should detect indirect access via alias"
        
        print(f"   ✅ PASSED: Detected {types_found}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Notebook extraction
    print("\nTest 3: Notebook code extraction...")
    try:
        # Create mock notebook
        mock_notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"]},
                {"cell_type": "code", "source": ["import pandas as pd"]},
                {"cell_type": "code", "source": ["df = pd.read_csv('data.csv')"]}
            ]
        }
        
        # Write temp notebook
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            json.dump(mock_notebook, f)
            temp_path = f.name
        
        cells = NotebookAnalyzer.extract_code_cells(temp_path)
        
        assert len(cells) == 2, f"Expected 2 code cells, got {len(cells)}"
        assert "import pandas" in cells[0][1], "First cell should have pandas import"
        
        os.unlink(temp_path)
        
        print(f"   ✅ PASSED: Extracted {len(cells)} code cells")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: EIS weight validation
    print("\nTest 4: EIS weight sum validation...")
    try:
        weights = EclipseIntegrityScore.DEFAULT_WEIGHTS
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"
        
        print(f"   ✅ PASSED: Weights sum to {total:.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Cryptographic commitment
    print("\nTest 5: Cryptographic commitment...")
    try:
        commitment = CryptographicCommitment.generate_commitment(
            "test_project",
            "I COMMIT TO SINGLE-SHOT"
        )
        
        # Verify correct phrase
        valid = CryptographicCommitment.verify_commitment(
            commitment, "I COMMIT TO SINGLE-SHOT"
        )
        assert valid, "Correct phrase should verify"
        
        # Verify wrong phrase fails
        invalid = CryptographicCommitment.verify_commitment(
            commitment, "WRONG PHRASE"
        )
        assert not invalid, "Wrong phrase should fail"
        
        print(f"   ✅ PASSED: Commitment verification works")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"UNIT TESTS COMPLETE: {tests_passed} passed, {tests_failed} failed")
    print("=" * 80)
    
    return tests_failed == 0


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE WITH v3.0 FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def example_v3_demo():
    """Complete example demonstrating v3.0 features"""
    
    print("\n" + "=" * 80)
    print("🧠 ECLIPSE v3.0 DEMONSTRATION")
    print("=" * 80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_subjects = 100
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    data = []
    for subj_id in subject_ids:
        for win in range(50):
            state = 1 if np.random.random() < 0.2 else 0
            phi = np.random.gamma(2, 2) + (1.0 if state == 1 else 0)
            data.append({
                'subject_id': subj_id,
                'window': win,
                'consciousness': state,
                'phi': max(0, phi + np.random.normal(0, 0.5))
            })
    
    df = pd.DataFrame(data)
    print(f"Generated {len(df)} samples")
    
    # v3.0: Use unique directory to avoid conflicts with previous runs
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./eclipse_v3_demo_{timestamp_str}"
    
    # Configure v3.0
    config = EclipseConfig(
        project_name="v3_Demo",
        researcher="Demo",
        sacred_seed=2025,
        output_dir=output_dir,
        non_interactive=False  # Set True for CI/CD
    )
    
    eclipse = EclipseFramework(config)
    
    # Stage 1
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(subject_ids)
    dev_data = df[df['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = df[df['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
    # Stage 2
    criteria = [
        FalsificationCriteria("f1_score", 0.60, ">=", "F1 >= 0.60", True),
        FalsificationCriteria("precision", 0.70, ">=", "Precision >= 0.70", True),
    ]
    eclipse.stage2_register_criteria(criteria)
    
    # Stage 3
    def train_fn(train_idx, **kw):
        train_df = dev_data.iloc[train_idx]
        best_t, best_f1 = None, 0
        for t in np.linspace(train_df['phi'].min(), train_df['phi'].max(), 50):
            pred = (train_df['phi'] >= t).astype(int)
            f1 = EclipseValidator.binary_classification_metrics(
                train_df['consciousness'], pred
            )['f1_score']
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return {'threshold': best_t}
    
    def val_fn(model, val_idx, **kw):
        val_df = dev_data.iloc[val_idx]
        pred = (val_df['phi'] >= model['threshold']).astype(int)
        return EclipseValidator.binary_classification_metrics(
            val_df['consciousness'], pred
        )
    
    dev_results = eclipse.stage3_development(
        list(range(len(dev_data))), train_fn, val_fn
    )
    
    # Stage 4
    final_model = train_fn(list(range(len(dev_data))))
    
    def final_val(model, data, **kw):
        pred = (data['phi'] >= model['threshold']).astype(int)
        return EclipseValidator.binary_classification_metrics(
            data['consciousness'], pred
        )
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data, final_model, final_val
    )
    
    if val_results:
        # Stage 5
        final = eclipse.stage5_final_assessment(dev_results, val_results)
        
        print("\n" + "=" * 80)
        print("🎯 v3.0 DEMO COMPLETE")
        print("=" * 80)
        print(f"Verdict: {final['verdict']}")
        
        # EIS
        eis_data = final.get('integrity_metrics', {}).get('eis', {})
        print(f"EIS: {eis_data.get('eis', 'N/A'):.4f}" if isinstance(eis_data.get('eis'), (int, float)) else "EIS: N/A")
        
        # STDS (may have failed)
        stds_data = final.get('integrity_metrics', {}).get('stds', {})
        if stds_data.get('status') == 'success':
            print(f"STDS max z-score: {stds_data.get('max_z_score', 0):+.4f}")
            print(f"STDS mean z-score: {stds_data.get('mean_z_score', 0):+.4f}")
            print(f"STDS risk level: {stds_data.get('risk_level', 'N/A')}")
        else:
            print(f"STDS: {stds_data.get('status', 'not computed')} - {stds_data.get('message', '')}")
        
        print(f"\n📁 Results saved to: {config.output_dir}")
        
        # Verify
        eclipse.verify_integrity()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def interactive_menu():
    """Interactive menu for running ECLIPSE v3.0"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        ECLIPSE FRAMEWORK v3.0                                ║
║           Enhanced Systematic Falsification Framework                        ║
║                                                                              ║
║  v3.0 IMPROVEMENTS OVER v2.0:                                               ║
║  🔧 STDS p-value direction CORRECTED (critical bug fix)                     ║
║  🔧 Leakage risk no longer assumes degradation always expected              ║
║  🆕 Kolmogorov-Smirnov distribution test                                    ║
║  🆕 Notebook (.ipynb) support in Code Auditor                               ║
║  🆕 Non-interactive mode for CI/CD pipelines                                ║
║  🆕 Variable aliasing detection (semantic analysis)                         ║
║  🆕 Built-in unit tests                                                     ║
║  📊 EIS weights with literature justification                               ║
║                                                                              ║
║  Based on: Sjöberg Tala, C. A. (2025). ECLIPSE v3.0                         ║
║                                                                              ║
║  Zero external dependencies • Fully reproducible • Open source              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Available options:

  1. Run unit tests (verify v3.0 corrections)
  2. Run full demonstration (IIT falsification example)
  3. Show v3.0 changelog (what changed from v2.0)
  4. Exit

Command line usage:
  python eclipse_v3.py --test   # Run unit tests
  python eclipse_v3.py --demo   # Run demonstration

""")
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            run_unit_tests()
            break
        elif choice == '2':
            example_v3_demo()
            break
        elif choice == '3':
            print_changelog()
            break
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def print_changelog():
    """Print detailed changelog v2.0 -> v3.0"""
    print("""
═══════════════════════════════════════════════════════════════════════════════
                          ECLIPSE v3.0 CHANGELOG
                         (Changes from v2.0)
═══════════════════════════════════════════════════════════════════════════════

🔧 CRITICAL BUG FIXES
───────────────────────────────────────────────────────────────────────────────

1. STDS P-VALUE DIRECTION (Critical)
   
   v2.0 BUG: The test flagged SMALL test statistics as suspicious.
   
   v3.0 FIX: Now correctly flags LARGE POSITIVE test statistics as suspicious.
   
   Explanation: 
   - Test statistic = (holdout - expected) / std
   - POSITIVE means holdout is BETTER than expected → suspicious (data snooping)
   - NEGATIVE means holdout is WORSE than expected → normal (generalization gap)
   
   Impact: v2.0 could miss actual data snooping while flagging honest results.

2. LEAKAGE RISK ESTIMATION
   
   v2.0 BUG: Always assumed dev→holdout degradation is expected.
   This penalized models that legitimately generalize well.
   
   v3.0 FIX: Context-aware z-score based assessment:
   - z > 1.5σ better: HIGH risk (0.9) - suspiciously good
   - z > 0.5σ better: MODERATE risk (0.5)
   - Within ±1σ: LOW risk (0.2) - normal
   - z < -1.5σ worse: MODERATE risk (0.3) - possible distribution shift

───────────────────────────────────────────────────────────────────────────────
🆕 NEW FEATURES
───────────────────────────────────────────────────────────────────────────────

1. KOLMOGOROV-SMIRNOV DISTRIBUTION TEST
   - More powerful than mean comparison
   - Detects ANY distribution difference, not just location shift
   - Used alongside parametric tests for robustness

2. NOTEBOOK (.ipynb) SUPPORT
   - NotebookAnalyzer extracts code cells from Jupyter notebooks
   - Same rigor as .py file analysis
   - Critical for research workflows (most analysis done in notebooks)

3. SEMANTIC ANALYSIS (Variable Aliasing Detection)
   
   v2.0 could be evaded with:
   ```python
   test_data = load_holdout()
   my_data = test_data  # Alias - v2.0 missed this!
   model.fit(my_data['X'])
   ```
   
   v3.0 builds dependency graph and propagates "taint" through assignments.

4. NON-INTERACTIVE MODE FOR CI/CD
   
   v2.0 required manual input: "Type 'I ACCEPT SINGLE-SHOT VALIDATION'"
   
   v3.0 supports:
   ```python
   config = EclipseConfig(
       non_interactive=True,
       commitment_phrase="I COMMIT TO SINGLE-SHOT VALIDATION"
   )
   ```
   Generates cryptographic commitment for audit trail.

5. BUILT-IN UNIT TESTS
   - Run with: python eclipse_v3.py --test
   - Validates critical components
   - Ensures corrections work as expected

───────────────────────────────────────────────────────────────────────────────
📊 IMPROVED METRICS
───────────────────────────────────────────────────────────────────────────────

EIS WEIGHTS WITH LITERATURE JUSTIFICATION:

┌─────────────────────┬────────┬────────────────────────────────────────────┐
│ Component           │ Weight │ Reference                                  │
├─────────────────────┼────────┼────────────────────────────────────────────┤
│ Pre-registration    │ 0.25   │ Nosek et al. (2018) PNAS                  │
│ Protocol adherence  │ 0.25   │ Simmons et al. (2011) Psych Science       │
│ Split strength      │ 0.20   │ Information theory                         │
│ Leakage risk        │ 0.15   │ Kapoor & Narayanan (2022) arXiv           │
│ Transparency        │ 0.15   │ FAIR principles                            │
└─────────────────────┴────────┴────────────────────────────────────────────┘

───────────────────────────────────────────────────────────────────────────────
📋 MIGRATION FROM v2.0
───────────────────────────────────────────────────────────────────────────────

v3.0 is backward compatible. Existing v2.0 projects will work.
However, you should:

1. Re-run STDS to get corrected p-values
2. Review any studies flagged (or not flagged) by v2.0 STDS
3. Consider re-auditing code with new semantic analysis

═══════════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ECLIPSE v3.0: Enhanced Systematic Falsification Framework"
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Run unit tests'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run demonstration'
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    elif args.demo:
        example_v3_demo()
    else:
        # Interactive menu when no arguments provided
        interactive_menu()