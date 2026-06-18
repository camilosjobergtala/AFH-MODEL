"""
stds_wrapper.py
In-memory bridge to the canonical ECLIPSE v3.0 STDS engine.

PERFORMANCE: a single reusable temp file is created ONCE (at import) and overwritten
on each call, instead of creating/deleting a temp directory per call. On Windows this
avoids per-call filesystem create/delete operations (each otherwise scanned by
antivirus), making large simulations run in well under a minute.

THE TEST (per the v3.0 code): z = (holdout - CV_mean) / CV_std per metric.
DEPENDENCY: full ECLIPSE v3.0 saved as eclipse_core.py in the same folder (engine/).
"""

import json
import atexit
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from eclipse_core import StatisticalTestDataSnooping


@dataclass
class StudyMetrics:
    """In-memory description of one study's CV-fold and holdout metrics."""
    cv_fold_values: Dict[str, List[float]]   # {metric_name: [fold_1, ..., fold_K]}
    holdout_values: Dict[str, float]          # {metric_name: holdout_value}
    alpha: float = 0.05


# --- single reusable temp file, created once, cleaned up at process exit ---
_WORK_DIR = Path(tempfile.mkdtemp(prefix="stds_reuse_"))
_RESULTS_PATH = _WORK_DIR / "FINAL_RESULT.json"
atexit.register(lambda: shutil.rmtree(_WORK_DIR, ignore_errors=True))


class _Config:
    def __init__(self, alpha: float):
        self.stds_alpha = alpha
        self.project_name = "study2_protocol"


class _FrameworkAdapter:
    """Minimal stand-in: the STDS engine only touches results_file,
    _validation_completed, and config.stds_alpha."""
    def __init__(self, alpha: float):
        self.config = _Config(alpha)
        self.results_file = _RESULTS_PATH
        self._validation_completed = True


def _materialize(sm: StudyMetrics) -> None:
    """Overwrite the single reusable results JSON, aggregating CV folds exactly as
    the engine's own Stage-3 pipeline does (np.std is population std, ddof=0)."""
    agg = {}
    for m, vals in sm.cv_fold_values.items():
        arr = np.asarray(vals, dtype=float)
        agg[m] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),          # ddof=0, matches engine Stage 3
            'min': float(arr.min()),
            'max': float(arr.max()),
            'median': float(np.median(arr)),
            'values': [float(v) for v in arr],
        }
    results = {
        'development_summary': {'aggregated_metrics': agg},
        'validation_summary': {'metrics': sm.holdout_values},
    }
    _RESULTS_PATH.write_text(json.dumps(results), encoding="utf-8")


def run_stds(sm: StudyMetrics) -> Dict[str, Any]:
    """Run the CANONICAL v3.0 STDS on a study's metrics."""
    _materialize(sm)
    adapter = _FrameworkAdapter(sm.alpha)
    tester = StatisticalTestDataSnooping(adapter)
    return tester.perform_snooping_test()