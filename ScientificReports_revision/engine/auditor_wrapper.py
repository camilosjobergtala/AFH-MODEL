"""
auditor_wrapper.py
In-memory bridge to the canonical ECLIPSE v3.0 CodeAuditor.

The CodeAuditor reads .py files from disk and returns an AuditResult. This wrapper
supplies the minimal framework adapter the auditor expects and exposes a single
function, audit_script(path), returning the reported violation categories,
severities, adherence score, and pass/fail — all from the REAL engine.

DEPENDENCY: ECLIPSE v3.0 saved as eclipse_core.py in the same folder (engine/).
"""

from typing import List, Dict, Any, Optional

from eclipse_core import CodeAuditor


class _Config:
    def __init__(self, pass_threshold: float = 70.0):
        self.audit_pass_threshold = pass_threshold
        self.project_name = "study3_protocol"


class _FrameworkAdapter:
    def __init__(self, pass_threshold: float = 70.0):
        self.config = _Config(pass_threshold)


_AUDITOR = CodeAuditor(_FrameworkAdapter())


def audit_script(path: str, holdout_identifiers: Optional[List[str]] = None,
                 pass_threshold: float = 70.0) -> Dict[str, Any]:
    """Run the canonical CodeAuditor on one .py file; return a compact result dict."""
    res = _AUDITOR.audit_analysis_code(
        code_paths=[path],
        holdout_identifiers=holdout_identifiers,
        pass_threshold=pass_threshold,
    )
    return {
        "categories": sorted({v.category for v in res.violations}),
        "severities": sorted({v.severity for v in res.violations}),
        "n_violations": len(res.violations),
        "adherence_score": res.adherence_score,
        "passed": res.passed,
    }