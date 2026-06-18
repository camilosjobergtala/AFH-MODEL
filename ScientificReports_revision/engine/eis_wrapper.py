"""
eis_wrapper.py
═══════════════════════════════════════════════════════════════════════════════
In-memory bridge to the canonical ECLIPSE v3.0 EIS engine.

WHY THIS EXISTS:
  Study 1 (internal coherence + weight stability) must compute the five EIS
  component scores for MANY protocols. The canonical EclipseIntegrityScore in
  eclipse_core.py reads its inputs from three JSON files on disk via a framework
  object. Reimplementing that scoring logic here would create a SECOND source of
  truth and risk silent divergence — exactly the failure mode the reconstruction
  is meant to eliminate.

  Instead, this wrapper materializes the exact JSON structures the engine expects
  in a temporary directory, points a minimal framework-like adapter at them, and
  calls the REAL EclipseIntegrityScore.compute_eis(). Every component value
  therefore comes from the canonical engine, not from a copy of its logic.

USAGE:
    from eis_wrapper import ProtocolSpec, score_protocol
    spec = ProtocolSpec(n_development=70, n_holdout=30, ...)
    result = score_protocol(spec)                       # default v3.0 weights
    result = score_protocol(spec, weights={...})        # custom weighting
    # result -> {'eis', 'components', 'weights', 'interpretation', ...}

DEPENDENCY:
    Requires the full ECLIPSE v3.0 saved as eclipse_core.py in the same package
    (engine/). The wrapper imports EclipseIntegrityScore from it.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from eclipse_core import EclipseIntegrityScore


# ─────────────────────────────────────────────────────────────────────────────
# Protocol specification: the in-memory description of a single study protocol.
# Each field maps to something the canonical engine reads from disk. Varying
# these knobs lets Study 1 build a corpus that spans the component space.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProtocolSpec:
    # Framework stage-completion flags (drive preregistration + protocol adherence)
    split_completed: bool = True
    criteria_registered: bool = True
    development_completed: bool = True
    validation_completed: bool = True

    # Preregistration quality
    has_regdate_and_hash: bool = True        # registration_date + criteria_hash present
    criteria_have_thresholds: bool = True     # each criterion has threshold + comparison
    criteria_have_descriptions: bool = True   # each criterion has a non-empty description
    n_criteria: int = 3

    # Split
    n_development: int = 70
    n_holdout: int = 30
    has_integrity_verification: bool = True   # cryptographic hash present in split
    has_split_date: bool = True

    # Transparency
    has_binding_declaration: bool = True

    # Leakage inputs: development CV metrics and holdout values.
    # dev_metrics: {metric_name: {'mean','std','min','values'}}
    # holdout_metrics: {metric_name: value}
    dev_metrics: Dict[str, Dict[str, Any]] = None
    holdout_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.dev_metrics is None:
            self.dev_metrics = {
                'f1_score': {'mean': 0.70, 'std': 0.05, 'min': 0.62,
                             'values': [0.72, 0.68, 0.70, 0.66, 0.74]}
            }
        if self.holdout_metrics is None:
            self.holdout_metrics = {'f1_score': 0.65}


# ─────────────────────────────────────────────────────────────────────────────
# Minimal adapter that looks enough like an EclipseFramework for the EIS engine.
# The engine only touches: the three file-path attributes, the four stage flags,
# and config.eis_weights. Nothing else.
# ─────────────────────────────────────────────────────────────────────────────

class _Config:
    def __init__(self):
        self.eis_weights = None          # forces DEFAULT_WEIGHTS fallback when no explicit weights
        self.project_name = "study1_protocol"


class _FrameworkAdapter:
    def __init__(self, work_dir: Path, spec: ProtocolSpec):
        self.config = _Config()
        self.split_file = work_dir / "SPLIT_IMMUTABLE.json"
        self.criteria_file = work_dir / "CRITERIA_BINDING.json"
        self.results_file = work_dir / "FINAL_RESULT.json"
        self._split_completed = spec.split_completed
        self._criteria_registered = spec.criteria_registered
        self._development_completed = spec.development_completed
        self._validation_completed = spec.validation_completed


def _materialize_files(work_dir: Path, spec: ProtocolSpec) -> None:
    """Write the three JSON artifacts the engine expects, per the spec."""

    # --- criteria file ---
    criteria_list = []
    for i in range(spec.n_criteria):
        c = {}
        if spec.criteria_have_thresholds:
            c['threshold'] = 0.70
            c['comparison'] = '>='
        if spec.criteria_have_descriptions:
            c['description'] = f"Criterion {i+1} rationale"
        else:
            c['description'] = ""
        criteria_list.append(c)

    criteria_data: Dict[str, Any] = {'criteria': criteria_list}
    if spec.has_regdate_and_hash:
        criteria_data['registration_date'] = "2025-08-05T00:00:00"
        criteria_data['criteria_hash'] = "0" * 64
    if spec.has_binding_declaration:
        criteria_data['binding_declaration'] = (
            "These criteria are binding and cannot be modified after registration."
        )
    (work_dir / "CRITERIA_BINDING.json").write_text(
        json.dumps(criteria_data, indent=2), encoding="utf-8")

    # --- split file ---
    split_data: Dict[str, Any] = {
        'n_development': spec.n_development,
        'n_holdout': spec.n_holdout,
    }
    if spec.has_split_date:
        split_data['split_date'] = "2025-08-05T00:00:00"
    if spec.has_integrity_verification:
        split_data['integrity_verification'] = {'split_hash': "0" * 64, 'algorithm': 'SHA-256'}
    (work_dir / "SPLIT_IMMUTABLE.json").write_text(
        json.dumps(split_data, indent=2), encoding="utf-8")

    # --- results file (drives leakage) ---
    results_data = {
        'development_summary': {'aggregated_metrics': spec.dev_metrics},
        'validation_summary': {'metrics': spec.holdout_metrics},
    }
    (work_dir / "FINAL_RESULT.json").write_text(
        json.dumps(results_data, indent=2), encoding="utf-8")


def score_protocol(spec: ProtocolSpec,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Compute EIS and its five components for a protocol spec using the CANONICAL
    v3.0 engine. Returns the full result dict from EclipseIntegrityScore.compute_eis().

    weights: optional weighting scheme (must sum to 1.0). If None, the engine's
             DEFAULT_WEIGHTS are used.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="eis_protocol_"))
    try:
        _materialize_files(work_dir, spec)
        adapter = _FrameworkAdapter(work_dir, spec)
        scorer = EclipseIntegrityScore(adapter)
        return scorer.compute_eis(weights=weights)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def components_of(result: Dict[str, Any]) -> Dict[str, float]:
    """Convenience: extract the five scored dimensions used in the correlation matrix.
    Note: 'leakage_score' (= 1 - leakage_risk) is the value that enters EIS, so it
    is the one used for the component analysis."""
    c = result['components']
    return {
        'preregistration': c['preregistration_score'],
        'split_strength': c['split_strength'],
        'protocol_adherence': c['protocol_adherence'],
        'leakage_score': c['leakage_score'],
        'transparency': c['transparency_score'],
    }