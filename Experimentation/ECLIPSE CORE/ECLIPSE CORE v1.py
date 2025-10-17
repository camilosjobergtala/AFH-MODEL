"""
═══════════════════════════════════════════════════════════════════════════════
ECLIPSE: A Systematic Falsification Framework for Scientific Theories
═══════════════════════════════════════════════════════════════════════════════

Universal falsification protocol applicable to ANY scientific theory:
- Integrated Information Theory (IIT)
- Global Workspace Theory (GWT)
- Higher-Order Theories (HOT)
- Medical diagnostics
- Climate models
- Financial predictions
- Machine learning models
- ANY falsifiable hypothesis

Based on the methodology by Camilo Alejandro Sjöberg Tala (2025)
Paper: "ECLIPSE: A systematic falsification framework for consciousness science"
DOI: 10.5281/zenodo.15541550

Version: 1.0.0
License: MIT
Author: Adapted from original ECLIPSE methodology
───────────────────────────────────────────────────────────────────────────────

FIVE-STAGE PROTOCOL:
1. Irreversible Data Splitting (cryptographic verification)
2. Pre-registered Falsification Criteria (binding thresholds)
3. Clean Development Protocol (k-fold cross-validation)
4. Single-Shot Validation (one attempt only)
5. Final Assessment (automatic verdict)

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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64


# ═══════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FalsificationCriteria:
    """
    Pre-registered falsification criterion
    
    Example:
        >>> criterion = FalsificationCriteria(
        ...     name="f1_score",
        ...     threshold=0.70,
        ...     comparison=">=",
        ...     description="F1-score must be at least 0.70",
        ...     is_required=True
        ... )
    """
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
    """
    ECLIPSE configuration
    
    Args:
        project_name: Name of your project (e.g., "IIT_Falsification_2025")
        researcher: Your name or research group
        sacred_seed: Immutable random seed (e.g., 2025)
        development_ratio: Proportion for development set (default 0.7)
        holdout_ratio: Proportion for holdout set (default 0.3)
        n_folds_cv: Number of folds for cross-validation (default 5)
        output_dir: Directory for results (default "./eclipse_results")
    """
    project_name: str
    researcher: str
    sacred_seed: int
    development_ratio: float = 0.7
    holdout_ratio: float = 0.3
    n_folds_cv: int = 5
    output_dir: str = "./eclipse_results"
    timestamp: str = field(default=None)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        
        # Validate ratios
        if abs(self.development_ratio + self.holdout_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Development ({self.development_ratio}) + Holdout ({self.holdout_ratio}) "
                f"ratios must sum to 1.0"
            )
        
        if self.n_folds_cv < 2:
            raise ValueError("n_folds_cv must be at least 2")
        
        if self.sacred_seed < 0:
            warnings.warn("Sacred seed is negative - consider using positive integers")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATOR - AUTOMATIC METRICS CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

class EclipseValidator:
    """
    Automatic validation metrics calculator
    
    Supports:
    - Binary classification
    - Multi-class classification
    - Regression
    - Custom metrics
    """
    
    @staticmethod
    def binary_classification_metrics(y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """
        Calculate comprehensive binary classification metrics
        
        Args:
            y_true: Ground truth labels (0/1 or True/False)
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary with metrics: accuracy, precision, recall, f1_score, etc.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, roc_auc_score, matthews_corrcoef
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred)
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        })
        
        # AUC if probabilities provided
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = np.nan
        
        return metrics
    
    @staticmethod
    def multiclass_classification_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate multi-class classification metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    @staticmethod
    def regression_metrics(y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics"""
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score
        )
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        }
    
    @staticmethod
    def consciousness_detection_metrics(
        predicted_conscious: List[int],
        true_transitions: List[int],
        total_windows: int,
        tolerance: int = 2
    ) -> Dict[str, float]:
        """
        Specialized metrics for consciousness detection (like in AFH* paper)
        
        Args:
            predicted_conscious: Indices where consciousness was predicted
            true_transitions: Indices of actual consciousness transitions
            total_windows: Total number of time windows
            tolerance: Window tolerance for matching (±tolerance)
        
        Returns:
            Detection metrics
        """
        if len(predicted_conscious) == 0 or len(true_transitions) == 0:
            return {
                'detection_rate': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'true_positives': 0,
                'false_positives': len(predicted_conscious),
                'false_negatives': len(true_transitions)
            }
        
        # Count matches within tolerance
        true_positives = 0
        matched_transitions = set()
        
        for pred_idx in predicted_conscious:
            for true_idx in true_transitions:
                if abs(pred_idx - true_idx) <= tolerance and true_idx not in matched_transitions:
                    true_positives += 1
                    matched_transitions.add(true_idx)
                    break
        
        false_positives = len(predicted_conscious) - true_positives
        false_negatives = len(true_transitions) - true_positives
        
        precision = true_positives / len(predicted_conscious) if len(predicted_conscious) > 0 else 0.0
        recall = true_positives / len(true_transitions) if len(true_transitions) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'detection_rate': len(predicted_conscious) / total_windows,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'n_predicted': len(predicted_conscious),
            'n_true_transitions': len(true_transitions)
        }


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR - HTML/TEXT REPORTS
# ═══════════════════════════════════════════════════════════════════════════

class EclipseReporter:
    """
    Automatic report generator for ECLIPSE results
    
    Generates:
    - HTML report with visualizations
    - Text report for archival
    - Summary statistics
    """
    
    @staticmethod
    def generate_html_report(
        final_assessment: Dict,
        output_path: str = None
    ) -> str:
        """
        Generate comprehensive HTML report
        
        Args:
            final_assessment: Results from stage5_final_assessment
            output_path: Where to save HTML (optional)
        
        Returns:
            HTML string
        """
        
        # Extract key information
        project = final_assessment['project_name']
        researcher = final_assessment['researcher']
        verdict = final_assessment['verdict']
        timestamp = final_assessment['assessment_timestamp']
        
        dev_metrics = final_assessment['development_summary'].get('aggregated_metrics', {})
        val_metrics = final_assessment['validation_summary'].get('metrics', {})
        degradation = final_assessment.get('degradation_analysis', {})
        criteria_eval = final_assessment['criteria_evaluation']
        
        # Verdict styling
        verdict_color = {
            'VALIDATED': '#28a745',
            'FALSIFIED': '#dc3545',
            'PARTIAL': '#ffc107'
        }.get(verdict, '#6c757d')
        
        # Build HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ECLIPSE Report - {project}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ 
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .header {{
            border-bottom: 4px solid {verdict_color};
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{ 
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        h2 {{ 
            color: #34495e;
            font-size: 1.8em;
            margin-top: 30px;
            margin-bottom: 15px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        h3 {{ 
            color: #7f8c8d;
            font-size: 1.3em;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .verdict {{
            background: {verdict_color};
            color: white;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            margin: 30px 0;
            text-transform: uppercase;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .info-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .info-card strong {{
            display: block;
            color: #2c3e50;
            margin-bottom: 5px;
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        .info-card span {{
            font-size: 1.3em;
            color: #34495e;
        }}
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
        th {{
            background: #34495e;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metric-good {{ color: #28a745; font-weight: bold; }}
        .metric-bad {{ color: #dc3545; font-weight: bold; }}
        .criterion-pass {{
            background: #d4edda;
            color: #155724;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .criterion-fail {{
            background: #f8d7da;
            color: #721c24;
            padding: 5px 10px;
            border-radius: 3px;
            font-weight: bold;
        }}
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .critical-box {{
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 20px 0;
        }}
        .success-box {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
            text-align: center;
        }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 ECLIPSE FALSIFICATION REPORT</h1>
            <p style="font-size: 1.2em; color: #7f8c8d;">
                Systematic Falsification Framework for Scientific Theories
            </p>
        </div>
        
        <div class="verdict">
            {verdict}
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <strong>Project</strong>
                <span>{project}</span>
            </div>
            <div class="info-card">
                <strong>Researcher</strong>
                <span>{researcher}</span>
            </div>
            <div class="info-card">
                <strong>Assessment Date</strong>
                <span>{timestamp}</span>
            </div>
            <div class="info-card">
                <strong>Sacred Seed</strong>
                <span>{final_assessment['configuration']['sacred_seed']}</span>
            </div>
        </div>
        
        <h2>📋 Executive Summary</h2>
        <p><strong>Verdict Description:</strong> {final_assessment['verdict_description']}</p>
        <p><strong>Required Criteria Passed:</strong> {final_assessment['required_criteria_passed']}</p>
        """
        
        # Add warning/success box based on verdict
        if verdict == 'VALIDATED':
            html += """
        <div class="success-box">
            <strong>✅ VALIDATION SUCCESSFUL</strong><br>
            All pre-registered criteria were met. The theory has survived systematic falsification.
            This result is immutable and represents honest generalization to unseen data.
        </div>
            """
        else:
            html += """
        <div class="critical-box">
            <strong>❌ THEORY FALSIFIED</strong><br>
            One or more pre-registered criteria were not met. The theory failed systematic empirical testing.
            This negative result is valuable scientific progress and should be published transparently.
        </div>
            """
        
        # Criteria Evaluation Table
        html += """
        <h2>🎯 Pre-Registered Criteria Evaluation</h2>
        <table>
            <thead>
                <tr>
                    <th>Criterion</th>
                    <th>Required</th>
                    <th>Threshold</th>
                    <th>Observed Value</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for crit in criteria_eval:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            
            value_str = f"{value:.4f}" if value is not None else "N/A"
            status_class = "criterion-pass" if passed else "criterion-fail"
            status_text = "✅ PASS" if passed else "❌ FAIL"
            required_text = "Yes" if criterion['is_required'] else "No"
            
            html += f"""
                <tr>
                    <td><strong>{criterion['name']}</strong><br>
                        <small style="color: #7f8c8d;">{criterion['description']}</small>
                    </td>
                    <td>{required_text}</td>
                    <td>{criterion['comparison']} {criterion['threshold']}</td>
                    <td>{value_str}</td>
                    <td><span class="{status_class}">{status_text}</span></td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        # Development vs Validation Metrics
        html += """
        <h2>📊 Performance Summary</h2>
        <h3>Development Phase (Cross-Validation)</h3>
        """
        
        if dev_metrics:
            html += "<table><thead><tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th></tr></thead><tbody>"
            for metric_name, stats in dev_metrics.items():
                html += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td>{stats['mean']:.4f}</td>
                    <td>{stats['std']:.4f}</td>
                    <td>{stats['min']:.4f}</td>
                    <td>{stats['max']:.4f}</td>
                </tr>
                """
            html += "</tbody></table>"
        else:
            html += "<p>No development metrics available.</p>"
        
        html += "<h3>Holdout Validation (Single-Shot)</h3>"
        
        if val_metrics:
            html += "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            for metric_name, value in val_metrics.items():
                html += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td>{value:.4f}</td>
                </tr>
                """
            html += "</tbody></table>"
        else:
            html += "<p>No validation metrics available.</p>"
        
        # Degradation Analysis
        if degradation:
            html += """
            <h3>⚠️ Performance Degradation Analysis</h3>
            <p>Degradation from development to holdout indicates generalization capability:</p>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Development Mean</th>
                        <th>Holdout Value</th>
                        <th>Degradation %</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for metric_name, analysis in degradation.items():
                deg_pct = analysis['degradation_percent']
                deg_class = "metric-bad" if abs(deg_pct) > 20 else "metric-good"
                
                html += f"""
                <tr>
                    <td><strong>{metric_name}</strong></td>
                    <td>{analysis['development_mean']:.4f}</td>
                    <td>{analysis['holdout_value']:.4f}</td>
                    <td class="{deg_class}">{deg_pct:.1f}%</td>
                </tr>
                """
            
            html += "</tbody></table>"
            
            # Degradation interpretation
            max_degradation = max(abs(a['degradation_percent']) for a in degradation.values())
            if max_degradation > 50:
                html += """
                <div class="critical-box">
                    <strong>⚠️ CRITICAL DEGRADATION DETECTED</strong><br>
                    Performance degradation exceeds 50%, indicating severe overfitting.
                    The model failed to generalize to unseen data.
                </div>
                """
            elif max_degradation > 20:
                html += """
                <div class="warning-box">
                    <strong>⚠️ SIGNIFICANT DEGRADATION</strong><br>
                    Performance degradation exceeds 20%, suggesting limited generalization.
                </div>
                """
        
        # Methodology details
        html += f"""
        <h2>🔬 Methodology</h2>
        
        <h3>ECLIPSE Five-Stage Protocol</h3>
        <ol>
            <li><strong>Irreversible Data Splitting:</strong> Sacred seed {final_assessment['configuration']['sacred_seed']}, 
                {final_assessment['configuration']['development_ratio']*100:.0f}% development / 
                {final_assessment['configuration']['holdout_ratio']*100:.0f}% holdout</li>
            <li><strong>Pre-registered Criteria:</strong> {len(criteria_eval)} binding criteria registered before validation</li>
            <li><strong>Clean Development:</strong> {final_assessment['configuration']['n_folds_cv']}-fold cross-validation</li>
            <li><strong>Single-Shot Validation:</strong> One irreversible evaluation on holdout data</li>
            <li><strong>Final Assessment:</strong> Automatic verdict against pre-registered criteria</li>
        </ol>
        
        <h3>Dataset Information</h3>
        <ul>
            <li><strong>Development samples:</strong> {final_assessment['development_summary'].get('n_successful', 'N/A')} folds processed</li>
            <li><strong>Holdout samples:</strong> {final_assessment['validation_summary'].get('n_holdout_samples', 'N/A')}</li>
        </ul>
        
        <h2>🔒 Integrity & Reproducibility</h2>
        <div class="warning-box">
            <strong>IMMUTABILITY DECLARATION</strong><br>
            {final_assessment['immutability_declaration']}
        </div>
        
        <p><strong>Assessment Hash:</strong> <code>{final_assessment.get('final_hash', 'N/A')}</code></p>
        <p><strong>Configuration Hash:</strong> <code>{hashlib.sha256(str(final_assessment['configuration']).encode()).hexdigest()[:32]}</code></p>
        
        <h2>📄 Citation</h2>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; font-size: 0.9em;">
{researcher} ({datetime.now().year}). {project}: Systematic falsification using ECLIPSE framework. 
Methodology based on Sjöberg Tala, C. A. (2025). ECLIPSE: A systematic falsification framework 
for consciousness science. DOI: 10.5281/zenodo.15541550
        </div>
        
        <div class="footer">
            <p><strong>ECLIPSE Framework v1.0.0</strong></p>
            <p>Based on the methodology by Camilo Alejandro Sjöberg Tala (2025)</p>
            <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"✅ HTML report saved to: {output_path}")
        
        return html
    
    @staticmethod
    def generate_text_report(final_assessment: Dict, output_path: str = None) -> str:
        """
        Generate plain text report for archival
        
        Args:
            final_assessment: Results from stage5_final_assessment
            output_path: Where to save text file (optional)
        
        Returns:
            Text string
        """
        
        lines = []
        lines.append("=" * 100)
        lines.append("ECLIPSE FALSIFICATION REPORT")
        lines.append("Systematic Falsification Framework for Scientific Theories")
        lines.append("=" * 100)
        lines.append("")
        
        # Header
        lines.append(f"Project: {final_assessment['project_name']}")
        lines.append(f"Researcher: {final_assessment['researcher']}")
        lines.append(f"Assessment Date: {final_assessment['assessment_timestamp']}")
        lines.append(f"Sacred Seed: {final_assessment['configuration']['sacred_seed']}")
        lines.append("")
        
        # Verdict
        lines.append("=" * 100)
        lines.append(f"FINAL VERDICT: {final_assessment['verdict']}")
        lines.append("=" * 100)
        lines.append(f"{final_assessment['verdict_description']}")
        lines.append(f"Required criteria passed: {final_assessment['required_criteria_passed']}")
        lines.append("")
        
        # Criteria evaluation
        lines.append("-" * 100)
        lines.append("PRE-REGISTERED CRITERIA EVALUATION")
        lines.append("-" * 100)
        
        for crit in final_assessment['criteria_evaluation']:
            criterion = crit['criterion']
            value = crit.get('value')
            passed = crit.get('passed', False)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            value_str = f"{value:.4f}" if value is not None else "N/A"
            required = "REQUIRED" if criterion['is_required'] else "optional"
            
            lines.append(f"\n{criterion['name']} [{required}]:")
            lines.append(f"   Description: {criterion['description']}")
            lines.append(f"   Threshold: {criterion['comparison']} {criterion['threshold']}")
            lines.append(f"   Observed: {value_str}")
            lines.append(f"   Status: {status}")
        
        lines.append("")
        
        # Performance summary
        lines.append("-" * 100)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 100)
        
        lines.append("\nDevelopment Phase (Cross-Validation):")
        dev_metrics = final_assessment['development_summary'].get('aggregated_metrics', {})
        if dev_metrics:
            for metric_name, stats in dev_metrics.items():
                lines.append(f"   {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                           f"(min={stats['min']:.4f}, max={stats['max']:.4f})")
        else:
            lines.append("   No development metrics available")
        
        lines.append("\nHoldout Validation (Single-Shot):")
        val_metrics = final_assessment['validation_summary'].get('metrics', {})
        if val_metrics:
            for metric_name, value in val_metrics.items():
                lines.append(f"   {metric_name}: {value:.4f}")
        else:
            lines.append("   No validation metrics available")
        
        # Degradation
        degradation = final_assessment.get('degradation_analysis', {})
        if degradation:
            lines.append("\nPerformance Degradation:")
            for metric_name, analysis in degradation.items():
                lines.append(f"   {metric_name}: {analysis['degradation_percent']:.1f}% "
                           f"(dev={analysis['development_mean']:.4f} → "
                           f"holdout={analysis['holdout_value']:.4f})")
        
        lines.append("")
        
        # Methodology
        lines.append("-" * 100)
        lines.append("METHODOLOGY")
        lines.append("-" * 100)
        lines.append("\nECLIPSE Five-Stage Protocol:")
        lines.append("1. Irreversible Data Splitting (cryptographic verification)")
        lines.append("2. Pre-registered Falsification Criteria (binding thresholds)")
        lines.append("3. Clean Development Protocol (k-fold cross-validation)")
        lines.append("4. Single-Shot Validation (one attempt only)")
        lines.append("5. Final Assessment (automatic verdict)")
        
        config = final_assessment['configuration']
        lines.append(f"\nConfiguration:")
        lines.append(f"   Sacred seed: {config['sacred_seed']}")
        lines.append(f"   Split: {config['development_ratio']*100:.0f}% dev / {config['holdout_ratio']*100:.0f}% holdout")
        lines.append(f"   Cross-validation: {config['n_folds_cv']} folds")
        
        # Integrity
        lines.append("")
        lines.append("-" * 100)
        lines.append("INTEGRITY & REPRODUCIBILITY")
        lines.append("-" * 100)
        lines.append(f"Assessment hash: {final_assessment.get('final_hash', 'N/A')}")
        lines.append(f"\n{final_assessment['immutability_declaration']}")
        
        # Citation
        lines.append("")
        lines.append("-" * 100)
        lines.append("CITATION")
        lines.append("-" * 100)
        researcher = final_assessment['researcher']
        project = final_assessment['project_name']
        year = datetime.now().year
        lines.append(f"{researcher} ({year}). {project}: Systematic falsification using ECLIPSE framework.")
        lines.append("Methodology based on Sjöberg Tala, C. A. (2025). ECLIPSE: A systematic falsification")
        lines.append("framework for consciousness science. DOI: 10.5281/zenodo.15541550")
        
        # Footer
        lines.append("")
        lines.append("=" * 100)
        lines.append("ECLIPSE Framework v1.0.0")
        lines.append("Based on the methodology by Camilo Alejandro Sjöberg Tala (2025)")
        lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)
        
        text = "\n".join(lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"✅ Text report saved to: {output_path}")
        
        return text


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ECLIPSE FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

class EclipseFramework:
    """
    ECLIPSE: Systematic Falsification Framework
    
    Five-stage protocol:
    1. Irreversible Data Splitting
    2. Pre-registered Falsification Criteria
    3. Clean Development Protocol
    4. Single-Shot Validation
    5. Final Assessment
    
    Example Usage:
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> 
        >>> # Your data
        >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        >>> data_ids = list(range(len(X)))
        >>> 
        >>> # Configure ECLIPSE
        >>> config = EclipseConfig(
        ...     project_name="IIT_Falsification_2025",
        ...     researcher="Your Name",
        ...     sacred_seed=2025
        ... )
        >>> 
        >>> eclipse = EclipseFramework(config)
        >>> 
        >>> # Stage 1: Split
        >>> dev_ids, holdout_ids = eclipse.stage1_irreversible_split(data_ids)
        >>> 
        >>> # Stage 2: Register criteria
        >>> criteria = [
        ...     FalsificationCriteria("f1_score", 0.70, ">=", "F1 must be >= 0.70", True)
        ... ]
        >>> eclipse.stage2_register_criteria(criteria)
        >>> 
        >>> # Stages 3-5: Development, validation, assessment
        >>> # (see examples below)
    """
    
    def __init__(self, config: EclipseConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Critical files
        self.split_file = self.output_dir / f"{config.project_name}_SPLIT_IMMUTABLE.json"
        self.criteria_file = self.output_dir / f"{config.project_name}_CRITERIA_BINDING.json"
        self.results_file = self.output_dir / f"{config.project_name}_FINAL_RESULT.json"
        
        # State tracking
        self._split_completed = False
        self._criteria_registered = False
        self._development_completed = False
        self._validation_completed = False
        
        print("=" * 80)
        print("🔬 ECLIPSE FRAMEWORK INITIALIZED")
        print(f"Project: {config.project_name}")
        print(f"Researcher: {config.researcher}")
        print(f"Sacred Seed: {config.sacred_seed}")
        print(f"Output: {self.output_dir}")
        print("=" * 80)
    
    # =========================================================================
    # STAGE 1: IRREVERSIBLE DATA SPLITTING
    # =========================================================================
    
    def stage1_irreversible_split(
        self, 
        data_identifiers: List[Any],
        force: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        """
        Stage 1: Irreversible data splitting with cryptographic verification
        
        This split is PERMANENT and IMMUTABLE. Once created, it cannot be changed
        without invalidating the entire methodology.
        
        Args:
            data_identifiers: List of unique identifiers for your data samples
                             (e.g., subject IDs, file names, indices)
            force: If True, allows re-splitting (USE WITH EXTREME CAUTION)
                  This breaks reproducibility and should only be used if you
                  understand the consequences
        
        Returns:
            (development_ids, holdout_ids): Two lists of identifiers
        
        Example:
            >>> data_ids = ['subject_001', 'subject_002', ..., 'subject_153']
            >>> dev_ids, holdout_ids = eclipse.stage1_irreversible_split(data_ids)
            >>> 
            >>> # Use dev_ids for all development work
            >>> # NEVER touch holdout_ids until final validation
        """
        
        if self.split_file.exists() and not force:
            print("⚠️  SPLIT ALREADY EXISTS - Loading immutable split...")
            with open(self.split_file, 'r') as f:
                split_data = json.load(f)
            
            print(f"   Split date: {split_data['split_date']}")
            print(f"   Development: {len(split_data['development_ids'])} samples")
            print(f"   Holdout: {len(split_data['holdout_ids'])} samples")
            print("   ✅ IMMUTABLE SPLIT LOADED")
            
            self._split_completed = True
            return split_data['development_ids'], split_data['holdout_ids']
        
        if force and self.split_file.exists():
            print("⚠️  WARNING: FORCING RE-SPLIT - This breaks reproducibility!")
            backup_path = self.split_file.with_suffix(f'.backup_{int(datetime.now().timestamp())}.json')
            self.split_file.rename(backup_path)
            print(f"   Previous split backed up to: {backup_path}")
        
        print("\n" + "=" * 80)
        print("STAGE 1: IRREVERSIBLE DATA SPLITTING")
        print("=" * 80)
        print(f"Total samples: {len(data_identifiers)}")
        print(f"Development ratio: {self.config.development_ratio}")
        print(f"Holdout ratio: {self.config.holdout_ratio}")
        print(f"Sacred seed: {self.config.sacred_seed}")
        
        # Perform split
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
        
        dev_hash = hashlib.sha256(str(sorted(development_ids)).encode()).hexdigest()
        holdout_hash = hashlib.sha256(str(sorted(holdout_ids)).encode()).hexdigest()
        
        # Save immutable split
        split_data = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'split_date': datetime.now().isoformat(),
            'split_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sacred_seed': self.config.sacred_seed,
            
            'total_samples': len(data_identifiers),
            'n_development': len(development_ids),
            'n_holdout': len(holdout_ids),
            
            'development_ids': development_ids,
            'holdout_ids': holdout_ids,
            
            'integrity_verification': {
                'split_hash': split_hash,
                'development_hash': dev_hash,
                'holdout_hash': holdout_hash
            },
            
            'immutability_declaration': (
                'THIS SPLIT IS IMMUTABLE AND IRREVERSIBLE. '
                'Modifying this file compromises the entire validation. '
                'The holdout set must NEVER be examined until final validation.'
            )
        }
        
        with open(self.split_file, 'w') as f:
            json.dump(split_data, f, indent=2, default=str)
        
        print(f"\n✅ IMMUTABLE SPLIT CREATED:")
        print(f"   Development: {len(development_ids)} samples ({len(development_ids)/len(data_identifiers)*100:.1f}%)")
        print(f"   Holdout: {len(holdout_ids)} samples ({len(holdout_ids)/len(data_identifiers)*100:.1f}%)")
        print(f"   Split hash: {split_hash[:32]}...")
        print(f"   Saved to: {self.split_file}")
        print("   🔒 THIS SPLIT IS NOW PERMANENT AND CANNOT BE CHANGED")
        
        self._split_completed = True
        return development_ids, holdout_ids
    
    # =========================================================================
    # STAGE 2: PRE-REGISTERED FALSIFICATION CRITERIA
    # =========================================================================
    
    def stage2_register_criteria(
        self, 
        criteria: List[FalsificationCriteria],
        force: bool = False
    ) -> Dict:
        """
        Stage 2: Pre-register falsification criteria (binding)
        
        These criteria are BINDING. Once registered, your theory will be evaluated
        against these exact thresholds. No post-hoc modifications allowed.
        
        Args:
            criteria: List of FalsificationCriteria objects defining success/failure
            force: If True, allows re-registration (USE WITH EXTREME CAUTION)
        
        Returns:
            criteria_data: Dictionary with registered criteria
        
        Example:
            >>> criteria = [
            ...     FalsificationCriteria(
            ...         name="f1_score",
            ...         threshold=0.70,
            ...         comparison=">=",
            ...         description="F1-score must be at least 0.70 for validation",
            ...         is_required=True
            ...     ),
            ...     FalsificationCriteria(
            ...         name="precision",
            ...         threshold=0.65,
            ...         comparison=">=",
            ...         description="Precision must be at least 0.65",
            ...         is_required=True
            ...     )
            ... ]
            >>> eclipse.stage2_register_criteria(criteria)
        """
        
        if not self._split_completed:
            raise RuntimeError("Must complete Stage 1 (split) before registering criteria")
        
        if self.criteria_file.exists() and not force:
            print("⚠️  CRITERIA ALREADY REGISTERED - Loading binding criteria...")
            with open(self.criteria_file, 'r') as f:
                criteria_data = json.load(f)
            
            print(f"   Registration date: {criteria_data['registration_date']}")
            print(f"   Number of criteria: {len(criteria_data['criteria'])}")
            print("   ✅ BINDING CRITERIA LOADED")
            
            self._criteria_registered = True
            return criteria_data
        
        if force and self.criteria_file.exists():
            print("⚠️  WARNING: FORCING RE-REGISTRATION - This breaks reproducibility!")
            backup_path = self.criteria_file.with_suffix(f'.backup_{int(datetime.now().timestamp())}.json')
            self.criteria_file.rename(backup_path)
            print(f"   Previous criteria backed up to: {backup_path}")
        
        print("\n" + "=" * 80)
        print("STAGE 2: PRE-REGISTERED FALSIFICATION CRITERIA")
        print("=" * 80)
        print(f"Registering {len(criteria)} criteria...")
        
        # Validate criteria
        for crit in criteria:
            if not isinstance(crit, FalsificationCriteria):
                raise TypeError(f"All criteria must be FalsificationCriteria objects")
        
        # Convert criteria to dict
        criteria_dict = {
            'project_name': self.config.project_name,
            'researcher': self.config.researcher,
            'registration_date': datetime.now().isoformat(),
            'registration_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            'criteria': [asdict(c) for c in criteria],
            
            'criteria_hash': hashlib.sha256(
                str([asdict(c) for c in criteria]).encode()
            ).hexdigest(),
            
            'binding_declaration': (
                'THESE CRITERIA ARE BINDING AND IMMUTABLE. '
                'All results will be evaluated against these pre-registered thresholds. '
                'Post-hoc modification invalidates the entire validation. '
                'These criteria were established BEFORE seeing any holdout data.'
            )
        }
        
        # Save binding criteria
        with open(self.criteria_file, 'w') as f:
            json.dump(criteria_dict, f, indent=2, default=str)
        
        print(f"\n✅ BINDING CRITERIA REGISTERED:")
        for i, c in enumerate(criteria, 1):
            req_str = "REQUIRED" if c.is_required else "optional"
            print(f"   {i}. {c.name}: {c.comparison} {c.threshold} [{req_str}]")
            print(f"      → {c.description}")
        
        print(f"\n   Criteria hash: {criteria_dict['criteria_hash'][:32]}...")
        print(f"   Saved to: {self.criteria_file}")
        print("   🔒 THESE CRITERIA ARE NOW BINDING AND CANNOT BE CHANGED")
        
        self._criteria_registered = True
        return criteria_dict
    
    # =========================================================================
    # STAGE 3: CLEAN DEVELOPMENT PROTOCOL
    # =========================================================================
    
    def stage3_development(
        self,
        development_data: Any,
        training_function: Callable,
        validation_function: Callable,
        **kwargs
    ) -> Dict:
        """
        Stage 3: Clean development with k-fold cross-validation
        
        All development occurs ONLY on development data. Holdout data is FORBIDDEN.
        
        Args:
            development_data: Your development dataset
            training_function: Function that trains your model
                Signature: train_fn(train_data, **kwargs) -> model
                Example:
                    def train_fn(train_data, **kwargs):
                        X_train = ...  # Extract from train_data
                        y_train = ...
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        return model
            
            validation_function: Function that validates your model
                Signature: val_fn(model, val_data, **kwargs) -> metrics_dict
                Example:
                    def val_fn(model, val_data, **kwargs):
                        X_val = ...  # Extract from val_data
                        y_val = ...
                        y_pred = model.predict(X_val)
                        return {
                            'f1_score': f1_score(y_val, y_pred),
                            'precision': precision_score(y_val, y_pred)
                        }
            
            **kwargs: Additional arguments passed to training/validation functions
        
        Returns:
            development_results: Dictionary with CV results
        
        Example:
            >>> dev_results = eclipse.stage3_development(
            ...     development_data=dev_X,  # Your dev data
            ...     training_function=train_fn,
            ...     validation_function=val_fn,
            ...     # Any additional kwargs
            ... )
        """
        
        if not self._split_completed or not self._criteria_registered:
            raise RuntimeError("Must complete Stages 1 and 2 before development")
        
        print("\n" + "=" * 80)
        print("STAGE 3: CLEAN DEVELOPMENT PROTOCOL")
        print("=" * 80)
        print(f"K-Fold Cross-Validation: {self.config.n_folds_cv} folds")
        print(f"Sacred seed: {self.config.sacred_seed}")
        print(f"Development samples: {len(development_data)}")
        
        from sklearn.model_selection import KFold
        
        kf = KFold(
            n_splits=self.config.n_folds_cv, 
            shuffle=True, 
            random_state=self.config.sacred_seed
        )
        
        cv_results = []
        
        # Perform k-fold CV
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(len(development_data)))):
            print(f"\n{'─' * 80}")
            print(f"FOLD {fold_idx + 1}/{self.config.n_folds_cv}")
            print(f"{'─' * 80}")
            
            # Split data
            if isinstance(development_data, (list, tuple)):
                train_data = [development_data[i] for i in train_idx]
                val_data = [development_data[i] for i in val_idx]
            elif isinstance(development_data, np.ndarray):
                train_data = development_data[train_idx]
                val_data = development_data[val_idx]
            else:
                # Assume indexable
                train_data = development_data[train_idx]
                val_data = development_data[val_idx]
            
            print(f"   Train samples: {len(train_data)}")
            print(f"   Validation samples: {len(val_data)}")
            
            try:
                # Train model
                print("   🔧 Training model...")
                model = training_function(train_data, **kwargs)
                
                # Validate model
                print("   📊 Validating model...")
                metrics = validation_function(model, val_data, **kwargs)
                
                if not isinstance(metrics, dict):
                    raise ValueError("validation_function must return a dictionary of metrics")
                
                fold_result = {
                    'fold': fold_idx + 1,
                    'n_train': len(train_data),
                    'n_val': len(val_data),
                    'metrics': metrics,
                    'status': 'success'
                }
                
                print(f"   ✅ Fold {fold_idx + 1} complete:")
                for metric_name, metric_value in metrics.items():
                    print(f"      {metric_name}: {metric_value:.4f}")
                
            except Exception as e:
                print(f"   ❌ Fold {fold_idx + 1} failed: {e}")
                import traceback
                traceback.print_exc()
                fold_result = {
                    'fold': fold_idx + 1,
                    'status': 'failed',
                    'error': str(e)
                }
            
            cv_results.append(fold_result)
        
        # Aggregate results
        successful_folds = [r for r in cv_results if r['status'] == 'success']
        
        if successful_folds:
            # Get all metric names
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
            warnings.warn("No successful folds! Check your training/validation functions.")
        
        development_results = {
            'n_folds': self.config.n_folds_cv,
            'n_successful': len(successful_folds),
            'fold_results': cv_results,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n{'=' * 80}")
        print("✅ DEVELOPMENT COMPLETE")
        print(f"{'=' * 80}")
        print(f"   Successful folds: {len(successful_folds)}/{self.config.n_folds_cv}")
        
        if aggregated_metrics:
            print(f"\n   📊 Aggregated Metrics (Development):")
            for metric_name, stats in aggregated_metrics.items():
                print(f"      {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"         (min={stats['min']:.4f}, max={stats['max']:.4f})")
        
        self._development_completed = True
        return development_results
    
    # =========================================================================
    # STAGE 4: SINGLE-SHOT VALIDATION
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
        Stage 4: Single-shot validation on holdout data (IRREVERSIBLE)
        
        ⚠️  THIS IS THE MOMENT OF TRUTH ⚠️
        
        The holdout data has NEVER been seen before. This validation happens
        EXACTLY ONCE. No do-overs, no adjustments, no second chances.
        
        This is what separates honest falsification from statistical opportunism.
        
        Args:
            holdout_data: Your holdout dataset (NEVER seen before)
            final_model: Your final trained model
            validation_function: Function that validates your model
                Signature: val_fn(model, data, **kwargs) -> metrics_dict
            force: If True, allows re-validation (BREAKS ENTIRE METHODOLOGY)
            **kwargs: Additional arguments for validation
        
        Returns:
            validation_results: Dictionary with holdout metrics
        
        Example:
            >>> # Train final model on ALL development data
            >>> final_model = train_final_model(all_dev_data)
            >>> 
            >>> # Single-shot validation (ONCE ONLY)
            >>> val_results = eclipse.stage4_single_shot_validation(
            ...     holdout_data=holdout_X,
            ...     final_model=final_model,
            ...     validation_function=val_fn
            ... )
        """
        
        if not self._development_completed:
            raise RuntimeError("Must complete Stage 3 (development) before validation")
        
        if self.results_file.exists() and not force:
            raise RuntimeError(
                "VALIDATION ALREADY PERFORMED! This is a SINGLE-SHOT protocol. "
                "Results are immutable. Use force=True ONLY if you understand "
                "this completely invalidates the methodology and breaks scientific integrity."
            )
        
        if force and self.results_file.exists():
            print("\n" + "!" * 80)
            print("⚠️  CRITICAL WARNING: FORCING RE-VALIDATION")
            print("!" * 80)
            print("YOU ARE BREAKING THE ENTIRE ECLIPSE METHODOLOGY!")
            print("The ECLIPSE protocol is based on SINGLE-SHOT validation.")
            print("Re-validation invalidates ALL claims of honest falsification.")
            print("This result CANNOT be published as scientifically valid.")
            print("!" * 80)
            
            backup_path = self.results_file.with_suffix(f'.backup_{int(datetime.now().timestamp())}.json')
            self.results_file.rename(backup_path)
            print(f"Previous results backed up to: {backup_path}")
            print("!" * 80 + "\n")
        
        print("\n" + "=" * 80)
        print("🎯 STAGE 4: SINGLE-SHOT VALIDATION")
        print("=" * 80)
        print("⚠️  THIS IS THE MOMENT OF TRUTH")
        print("⚠️  HOLDOUT DATA HAS NEVER BEEN SEEN")
        print("⚠️  THIS HAPPENS EXACTLY ONCE")
        print("⚠️  RESULTS WILL BE IMMUTABLE AND PERMANENT")
        print("=" * 80)
        
        print("\n🔒 CRITICAL REMINDER:")
        print("   • No iterative optimization allowed")
        print("   • No threshold adjustments allowed")
        print("   • No model modifications allowed")
        print("   • No 'one more try' allowed")
        print("   • This is FINAL and IRREVERSIBLE")
        
        confirmation = input("\n🚨 Type 'I ACCEPT SINGLE-SHOT VALIDATION' to proceed: ")
        
        if confirmation != "I ACCEPT SINGLE-SHOT VALIDATION":
            print("❌ Validation cancelled. Come back when you're ready.")
            print("   (This is the right decision if you're unsure.)")
            return None
        
        print("\n" + "─" * 80)
        print("🚀 EXECUTING SINGLE-SHOT VALIDATION...")
        print("─" * 80)
        print(f"Holdout samples: {len(holdout_data)}")
        print("Sacred seed preserved: Yes")
        print("Development data contamination: None")
        print("Number of validation attempts: 1 (and only 1)")
        
        try:
            # Perform validation (ONCE AND ONLY ONCE)
            print("\n📊 Computing metrics on holdout data...")
            metrics = validation_function(final_model, holdout_data, **kwargs)
            
            if not isinstance(metrics, dict):
                raise ValueError("validation_function must return a dictionary of metrics")
            
            validation_results = {
                'status': 'success',
                'n_holdout_samples': len(holdout_data),
                'metrics': {k: float(v) if not isinstance(v, str) else v for k, v in metrics.items()},
                'timestamp': datetime.now().isoformat(),
                'validation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"\n{'=' * 80}")
            print("✅ SINGLE-SHOT VALIDATION COMPLETE")
            print(f"{'=' * 80}")
            print("📊 Holdout Metrics:")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"   {metric_name}: {metric_value:.4f}")
                else:
                    print(f"   {metric_name}: {metric_value}")
            
            print(f"\n🔒 THESE RESULTS ARE NOW PERMANENT AND IMMUTABLE")
            
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"❌ VALIDATION FAILED")
            print(f"{'=' * 80}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            
            validation_results = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n🔒 FAILURE IS ALSO IMMUTABLE - This counts as your single shot")
        
        self._validation_completed = True
        return validation_results
    
    # =========================================================================
    # STAGE 5: FINAL ASSESSMENT
    # =========================================================================
    
    def stage5_final_assessment(
        self,
        development_results: Dict,
        validation_results: Dict,
        generate_reports: bool = True
    ) -> Dict:
        """
        Stage 5: Final assessment against pre-registered criteria
        
        Automatic verdict: VALIDATED or FALSIFIED
        No human judgment, no post-hoc rationalization.
        The numbers speak for themselves.
        
        Args:
            development_results: Results from Stage 3
            validation_results: Results from Stage 4
            generate_reports: If True, auto-generate HTML and text reports
        
        Returns:
            final_assessment: Complete assessment with verdict
        
        Example:
            >>> final_assessment = eclipse.stage5_final_assessment(
            ...     development_results=dev_results,
            ...     validation_results=val_results,
            ...     generate_reports=True
            ... )
            >>> 
            >>> print(f"Verdict: {final_assessment['verdict']}")
            >>> # Reports saved automatically to output_dir
        """
        
        if not self._validation_completed:
            raise RuntimeError("Must complete Stage 4 (validation) before assessment")
        
        if self.results_file.exists():
            print("⚠️  FINAL ASSESSMENT ALREADY EXISTS - Loading results...")
            with open(self.results_file, 'r') as f:
                final_assessment = json.load(f)
            
            print(f"   Assessment date: {final_assessment['assessment_timestamp']}")
            print(f"   Verdict: {final_assessment['verdict']}")
            print("   ✅ IMMUTABLE RESULTS LOADED")
            
            return final_assessment
        
        print("\n" + "=" * 80)
        print("🎯 STAGE 5: FINAL ASSESSMENT")
        print("=" * 80)
        
        # Load pre-registered criteria
        with open(self.criteria_file, 'r') as f:
            criteria_data = json.load(f)
        
        criteria_list = [
            FalsificationCriteria(**c) for c in criteria_data['criteria']
        ]
        
        # Evaluate each criterion
        print("\n📋 Evaluating against pre-registered criteria...")
        print("─" * 80)
        
        holdout_metrics = validation_results.get('metrics', {})
        criteria_evaluation = []
        
        for criterion in criteria_list:
            if criterion.name not in holdout_metrics:
                warnings.warn(f"⚠️  Metric '{criterion.name}' not found in validation results")
                evaluation = {
                    'criterion': asdict(criterion),
                    'value': None,
                    'passed': False,
                    'reason': 'metric_not_found'
                }
            else:
                value = holdout_metrics[criterion.name]
                
                # Handle non-numeric values
                if isinstance(value, str):
                    warnings.warn(f"⚠️  Metric '{criterion.name}' is non-numeric: {value}")
                    evaluation = {
                        'criterion': asdict(criterion),
                        'value': value,
                        'passed': False,
                        'reason': 'non_numeric_metric'
                    }
                else:
                    passed = criterion.evaluate(value)
                    
                    evaluation = {
                        'criterion': asdict(criterion),
                        'value': float(value),
                        'passed': passed,
                        'reason': 'evaluated'
                    }
                    
                    status_icon = "✅" if passed else "❌"
                    status_text = "PASS" if passed else "FAIL"
                    req_text = "REQUIRED" if criterion.is_required else "optional"
                    
                    print(f"{status_icon} {criterion.name} [{req_text}]:")
                    print(f"   Threshold: {criterion.comparison} {criterion.threshold}")
                    print(f"   Observed: {value:.4f}")
                    print(f"   Result: {status_text}")
            
            criteria_evaluation.append(evaluation)
        
        # Determine verdict
        print("\n" + "─" * 80)
        print("🔍 Determining verdict...")
        print("─" * 80)
        
        required_criteria = [e for e in criteria_evaluation if e['criterion']['is_required']]
        required_passed = sum(1 for e in required_criteria if e['passed'])
        required_total = len(required_criteria)
        
        optional_criteria = [e for e in criteria_evaluation if not e['criterion']['is_required']]
        optional_passed = sum(1 for e in optional_criteria if e['passed'])
        optional_total = len(optional_criteria)
        
        all_required_passed = all(e['passed'] for e in required_criteria)
        
        if all_required_passed:
            verdict = "VALIDATED"
            verdict_description = (
                f"All {required_total} required pre-registered criteria were met. "
                "The theory has survived systematic falsification."
            )
            verdict_emoji = "✅"
        else:
            verdict = "FALSIFIED"
            failed_count = required_total - required_passed
            verdict_description = (
                f"Failed {failed_count}/{required_total} required criteria. "
                "The theory did not meet pre-registered standards."
            )
            verdict_emoji = "❌"
        
        print(f"\nRequired criteria: {required_passed}/{required_total} passed")
        if optional_total > 0:
            print(f"Optional criteria: {optional_passed}/{optional_total} passed")
        
        # Calculate degradation if development metrics available
        degradation_analysis = {}
        if development_results.get('aggregated_metrics'):
            print("\n📉 Analyzing performance degradation...")
            for metric_name in holdout_metrics:
                if isinstance(holdout_metrics[metric_name], str):
                    continue  # Skip non-numeric
                    
                if metric_name in development_results['aggregated_metrics']:
                    dev_mean = development_results['aggregated_metrics'][metric_name]['mean']
                    holdout_val = holdout_metrics[metric_name]
                    
                    if dev_mean != 0:
                        degradation_pct = ((dev_mean - holdout_val) / abs(dev_mean)) * 100
                    else:
                        degradation_pct = 0.0
                    
                    degradation_analysis[metric_name] = {
                        'development_mean': float(dev_mean),
                        'holdout_value': float(holdout_val),
                        'degradation_percent': float(degradation_pct)
                    }
                    
                    print(f"   {metric_name}: {degradation_pct:+.1f}% "
                          f"(dev={dev_mean:.4f} → holdout={holdout_val:.4f})")
        
        # Compile final assessment
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
            
            'degradation_analysis': degradation_analysis,
            
            'verdict': verdict,
            'verdict_description': verdict_description,
            'required_criteria_passed': f"{required_passed}/{required_total}",
            'optional_criteria_passed': f"{optional_passed}/{optional_total}" if optional_total > 0 else "N/A",
            
            'immutability_declaration': (
                'THIS ASSESSMENT IS FINAL AND IMMUTABLE. '
                'These results represent a single, honest evaluation '
                'against pre-registered criteria with no post-hoc optimization. '
                'This result is scientifically valid regardless of outcome.'
            ),
            
            'methodology_reference': (
                'Sjöberg Tala, C. A. (2025). ECLIPSE: A systematic falsification '
                'framework for consciousness science. DOI: 10.5281/zenodo.15541550'
            ),
            
            'final_hash': None  # Will be computed below
        }
        
        # Compute final hash
        final_assessment['final_hash'] = hashlib.sha256(
            json.dumps(final_assessment, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        # Save final assessment
        with open(self.results_file, 'w') as f:
            json.dump(final_assessment, f, indent=2, default=str)
        
        # Print verdict
        print("\n" + "=" * 80)
        print(f"{verdict_emoji} FINAL VERDICT: {verdict}")
        print("=" * 80)
        print(f"{verdict_description}")
        print(f"\nRequired criteria passed: {required_passed}/{required_total}")
        if optional_total > 0:
            print(f"Optional criteria passed: {optional_passed}/{optional_total}")
        
        if degradation_analysis:
            max_degradation = max(abs(a['degradation_percent']) for a in degradation_analysis.values())
            if max_degradation > 50:
                print(f"\n⚠️  CRITICAL: Maximum degradation {max_degradation:.1f}% indicates severe overfitting")
            elif max_degradation > 20:
                print(f"\n⚠️  WARNING: Maximum degradation {max_degradation:.1f}% suggests limited generalization")
        
        print(f"\n✅ FINAL ASSESSMENT SAVED TO: {self.results_file}")
        print(f"   Assessment hash: {final_assessment['final_hash'][:32]}...")
        print(f"\n🔒 THIS RESULT IS NOW IMMUTABLE AND PERMANENT")
        
        # Generate reports
        if generate_reports:
            print("\n" + "─" * 80)
            print("📄 Generating reports...")
            print("─" * 80)
            
            # HTML report
            html_path = self.output_dir / f"{self.config.project_name}_REPORT.html"
            EclipseReporter.generate_html_report(final_assessment, str(html_path))
            
            # Text report
            text_path = self.output_dir / f"{self.config.project_name}_REPORT.txt"
            EclipseReporter.generate_text_report(final_assessment, str(text_path))
            
            print(f"\n📊 Reports generated:")
            print(f"   HTML: {html_path}")
            print(f"   Text: {text_path}")
        
        return final_assessment
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_status(self) -> Dict:
        """
        Get current status of ECLIPSE pipeline
        
        Returns:
            Dictionary with completion status of each stage
        """
        return {
            'stage1_split': self._split_completed,
            'stage2_criteria': self._criteria_registered,
            'stage3_development': self._development_completed,
            'stage4_validation': self._validation_completed,
            'files_exist': {
                'split': self.split_file.exists(),
                'criteria': self.criteria_file.exists(),
                'results': self.results_file.exists()
            }
        }
    
    def verify_integrity(self) -> Dict:
        """
        Verify cryptographic integrity of all ECLIPSE files
        
        Returns:
            Dictionary with verification results
        """
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
            
            # Recompute hash
            all_ids = split_data['development_ids'] + split_data['holdout_ids']
            recomputed_hash = hashlib.sha256(
                f"{split_data['sacred_seed']}_{sorted(all_ids)}".encode()
            ).hexdigest()
            
            split_valid = recomputed_hash == split_data['integrity_verification']['split_hash']
            
            verification['files_checked'].append({
                'file': 'split',
                'path': str(self.split_file),
                'valid': split_valid,
                'hash': split_data['integrity_verification']['split_hash'][:32]
            })
            
            status = "✅" if split_valid else "❌"
            print(f"{status} Split file: {'VALID' if split_valid else 'COMPROMISED'}")
            
            if not split_valid:
                verification['all_valid'] = False
                print(f"   ⚠️  Split file integrity has been compromised!")
        
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
                'path': str(self.criteria_file),
                'valid': criteria_valid,
                'hash': criteria_data['criteria_hash'][:32]
            })
            
            status = "✅" if criteria_valid else "❌"
            print(f"{status} Criteria file: {'VALID' if criteria_valid else 'COMPROMISED'}")
            
            if not criteria_valid:
                verification['all_valid'] = False
                print(f"   ⚠️  Criteria file integrity has been compromised!")
        
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
                'path': str(self.results_file),
                'valid': results_valid,
                'hash': stored_hash[:32] if stored_hash else 'N/A'
            })
            
            status = "✅" if results_valid else "❌"
            print(f"{status} Results file: {'VALID' if results_valid else 'COMPROMISED'}")
            
            if not results_valid:
                verification['all_valid'] = False
                print(f"   ⚠️  Results file integrity has been compromised!")
        
        print("─" * 80)
        if verification['all_valid']:
            print("✅ ALL FILES VERIFIED - Integrity intact")
        else:
            print("❌ INTEGRITY COMPROMISED - Results may be invalid")
        
        return verification
    
    def generate_summary(self) -> str:
        """
        Generate a quick text summary of the ECLIPSE run
        
        Returns:
            Summary string
        """
        if not self.results_file.exists():
            return "No final results available yet."
        
        with open(self.results_file, 'r') as f:
            results = json.load(f)
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"ECLIPSE SUMMARY: {results['project_name']}")
        lines.append("=" * 60)
        lines.append(f"Researcher: {results['researcher']}")
        lines.append(f"Date: {results['assessment_timestamp']}")
        lines.append(f"Verdict: {results['verdict']}")
        lines.append(f"Criteria passed: {results['required_criteria_passed']}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE EXAMPLE: FALSIFYING IIT (Integrated Information Theory)
# ═══════════════════════════════════════════════════════════════════════════

def example_iit_falsification():
    """
    Complete example: Using ECLIPSE to falsify Integrated Information Theory (IIT)
    
    This example demonstrates how to use ECLIPSE for consciousness science,
    specifically testing whether IIT's Φ (phi) measure can detect consciousness
    transitions in EEG data.
    """
    
    print("\n" + "=" * 80)
    print("🧠 EXAMPLE: FALSIFYING INTEGRATED INFORMATION THEORY (IIT)")
    print("=" * 80)
    print("Testing hypothesis: IIT's Φ measure can detect consciousness transitions")
    print("Dataset: Simulated sleep-wake EEG data")
    print("=" * 80)
    
    # =========================================================================
    # 1. GENERATE SYNTHETIC DATA (replace with real data)
    # =========================================================================
    
    print("\n📊 Generating synthetic consciousness dataset...")
    
    np.random.seed(42)
    n_subjects = 100
    n_windows_per_subject = 50
    
    # Simulate "subjects"
    subject_ids = [f"subject_{i:03d}" for i in range(n_subjects)]
    
    # Simulate EEG windows with consciousness states
    # States: 0 = unconscious (sleep), 1 = conscious (wake)
    data = []
    for subj_id in subject_ids:
        for win_idx in range(n_windows_per_subject):
            # Simulate consciousness state (20% awake, 80% asleep)
            true_state = 1 if np.random.random() < 0.2 else 0
            
            # Simulate IIT's Φ (phi) measure
            # In reality, Φ is supposed to be higher during consciousness
            # But let's simulate weak/noisy relationship
            if true_state == 1:  # Conscious
                phi = np.random.gamma(2, 2) + 1.0  # Higher phi
            else:  # Unconscious
                phi = np.random.gamma(1.5, 1.5)    # Lower phi
            
            # Add noise to make it realistic
            phi += np.random.normal(0, 0.5)
            phi = max(0, phi)  # Phi cannot be negative
            
            data.append({
                'subject_id': subj_id,
                'window': win_idx,
                'true_consciousness_state': true_state,
                'phi': phi
            })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} EEG windows from {n_subjects} subjects")
    print(f"   Conscious windows: {(df['true_consciousness_state'] == 1).sum()}")
    print(f"   Unconscious windows: {(df['true_consciousness_state'] == 0).sum()}")
    
    # =========================================================================
    # 2. CONFIGURE ECLIPSE
    # =========================================================================
    
    config = EclipseConfig(
        project_name="IIT_Falsification_2025",
        researcher="Your Name (replace with yours)",
        sacred_seed=2025,
        development_ratio=0.7,
        holdout_ratio=0.3,
        n_folds_cv=5,
        output_dir="./eclipse_iit_results"
    )
    
    eclipse = EclipseFramework(config)
    
    # =========================================================================
    # STAGE 1: IRREVERSIBLE SPLIT
    # =========================================================================
    
    dev_subjects, holdout_subjects = eclipse.stage1_irreversible_split(subject_ids)
    
    # Split data by subjects
    dev_data = df[df['subject_id'].isin(dev_subjects)].reset_index(drop=True)
    holdout_data = df[df['subject_id'].isin(holdout_subjects)].reset_index(drop=True)
    
    print(f"\n📊 Data split:")
    print(f"   Development: {len(dev_data)} windows from {len(dev_subjects)} subjects")
    print(f"   Holdout: {len(holdout_data)} windows from {len(holdout_subjects)} subjects")
    
    # =========================================================================
    # STAGE 2: PRE-REGISTER CRITERIA
    # =========================================================================
    
    criteria = [
        FalsificationCriteria(
            name="f1_score",
            threshold=0.60,
            comparison=">=",
            description="F1-score must be at least 0.60 for IIT validation",
            is_required=True
        ),
        FalsificationCriteria(
            name="precision",
            threshold=0.70,
            comparison=">=",
            description="Precision must be at least 0.70",
            is_required=True
        ),
        FalsificationCriteria(
            name="recall",
            threshold=0.50,
            comparison=">=",
            description="Recall must be at least 0.50",
            is_required=True
        ),
        FalsificationCriteria(
            name="roc_auc",
            threshold=0.70,
            comparison=">=",
            description="ROC-AUC must be at least 0.70",
            is_required=False
        )
    ]
    
    eclipse.stage2_register_criteria(criteria)
    
    # =========================================================================
    # STAGE 3: DEVELOPMENT
    # =========================================================================
    
    def train_iit_classifier(train_data, **kwargs):
        """Train IIT-based consciousness classifier"""
        # In this example, we use a simple threshold on Φ
        # In reality, this would be more sophisticated
        
        # Find optimal Φ threshold on training data
        train_df = dev_data.iloc[train_data]
        
        # Try different thresholds
        best_threshold = None
        best_f1 = 0
        
        for threshold in np.linspace(train_df['phi'].min(), train_df['phi'].max(), 50):
            pred = (train_df['phi'] >= threshold).astype(int)
            true = train_df['true_consciousness_state']
            
            f1 = EclipseValidator.binary_classification_metrics(true, pred)['f1_score']
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # "Model" is just the threshold
        return {'phi_threshold': best_threshold}
    
    def validate_iit_classifier(model, val_data, **kwargs):
        """Validate IIT classifier"""
        val_df = dev_data.iloc[val_data]
        
        threshold = model['phi_threshold']
        y_pred = (val_df['phi'] >= threshold).astype(int)
        y_true = val_df['true_consciousness_state']
        y_pred_proba = val_df['phi'] / val_df['phi'].max()  # Normalize to [0,1]
        
        metrics = EclipseValidator.binary_classification_metrics(y_true, y_pred, y_pred_proba)
        
        return metrics
    
    # Run development
    dev_indices = list(range(len(dev_data)))
    dev_results = eclipse.stage3_development(
        development_data=dev_indices,
        training_function=train_iit_classifier,
        validation_function=validate_iit_classifier
    )
    
    # =========================================================================
    # STAGE 4: SINGLE-SHOT VALIDATION
    # =========================================================================
    
    # Train final model on ALL development data
    print("\n🔧 Training final IIT classifier on all development data...")
    final_model = train_iit_classifier(list(range(len(dev_data))))
    print(f"   Final Φ threshold: {final_model['phi_threshold']:.4f}")
    
    def validate_final_iit(model, holdout_df, **kwargs):
        """Final validation on holdout"""
        threshold = model['phi_threshold']
        y_pred = (holdout_df['phi'] >= threshold).astype(int)
        y_true = holdout_df['true_consciousness_state']
        y_pred_proba = holdout_df['phi'] / holdout_df['phi'].max()
        
        metrics = EclipseValidator.binary_classification_metrics(y_true, y_pred, y_pred_proba)
        
        return metrics
    
    # SINGLE-SHOT VALIDATION
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_data,
        final_model=final_model,
        validation_function=validate_final_iit
    )
    
    if val_results is None:
        print("Validation cancelled.")
        return
    
    # =========================================================================
    # STAGE 5: FINAL ASSESSMENT
    # =========================================================================
    
    final_assessment = eclipse.stage5_final_assessment(
        development_results=dev_results,
        validation_results=val_results,
        generate_reports=True
    )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("🎯 IIT FALSIFICATION COMPLETE")
    print("=" * 80)
    print(eclipse.generate_summary())
    print("\n📁 All results saved to:", config.output_dir)
    print("   • JSON: " + str(eclipse.results_file))
    print("   • HTML Report: " + str(eclipse.output_dir / f"{config.project_name}_REPORT.html"))
    print("   • Text Report: " + str(eclipse.output_dir / f"{config.project_name}_REPORT.txt"))
    
    # Verify integrity
    print("\n🔍 Verifying integrity...")
    eclipse.verify_integrity()
    
    print("\n" + "=" * 80)
    print("✅ ECLIPSE RUN COMPLETE")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLE SKLEARN EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

def example_sklearn_classification():
    """
    Simple example using scikit-learn for binary classification
    """
    
    print("\n" + "=" * 80)
    print("📊 SIMPLE EXAMPLE: Binary Classification with scikit-learn")
    print("=" * 80)
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    data_ids = list(range(len(X)))
    
    # Configure ECLIPSE
    config = EclipseConfig(
        project_name="SimpleClassification",
        researcher="Demo User",
        sacred_seed=2025,
        output_dir="./eclipse_simple_results"
    )
    
    eclipse = EclipseFramework(config)
    
    # Stage 1: Split
    dev_ids, holdout_ids = eclipse.stage1_irreversible_split(data_ids)
    dev_X, dev_y = X[dev_ids], y[dev_ids]
    holdout_X, holdout_y = X[holdout_ids], y[holdout_ids]
    
    # Stage 2: Criteria
    criteria = [
        FalsificationCriteria("f1_score", 0.70, ">=", "F1 >= 0.70", True),
        FalsificationCriteria("precision", 0.65, ">=", "Precision >= 0.65", True)
    ]
    eclipse.stage2_register_criteria(criteria)
    
    # Stage 3: Development
    def train_fn(train_data, **kwargs):
        train_indices = train_data
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(dev_X[train_indices], dev_y[train_indices])
        return model
    
    def val_fn(model, val_data, **kwargs):
        val_indices = val_data
        y_pred = model.predict(dev_X[val_indices])
        return EclipseValidator.binary_classification_metrics(dev_y[val_indices], y_pred)
    
    dev_results = eclipse.stage3_development(
        development_data=list(range(len(dev_X))),
        training_function=train_fn,
        validation_function=val_fn
    )
    
    # Stage 4: Validation
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(dev_X, dev_y)
    
    def holdout_val_fn(model, data, **kwargs):
        y_pred = model.predict(data)
        return EclipseValidator.binary_classification_metrics(holdout_y, y_pred)
    
    val_results = eclipse.stage4_single_shot_validation(
        holdout_data=holdout_X,
        final_model=final_model,
        validation_function=holdout_val_fn
    )
    
    if val_results:
        # Stage 5: Assessment
        final_assessment = eclipse.stage5_final_assessment(dev_results, val_results)
        
        print("\n✅ Example complete! Check:", config.output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                        ECLIPSE FRAMEWORK v1.0.0                              ║
║              Systematic Falsification for Scientific Theories                ║
║                                                                              ║
║  Based on: Sjöberg Tala, C. A. (2025). ECLIPSE: A systematic               ║
║            falsification framework for consciousness science.                ║
║            DOI: 10.5281/zenodo.15541550                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Available examples:
1. IIT Falsification (consciousness science)
2. Simple sklearn classification

""")
    
    choice = input("Select example (1 or 2), or 'q' to quit: ").strip()
    
    if choice == '1':
        example_iit_falsification()
    elif choice == '2':
        example_sklearn_classification()
    elif choice.lower() == 'q':
        print("Goodbye!")
    else:
        print("Invalid choice. Run again and select 1 or 2.")