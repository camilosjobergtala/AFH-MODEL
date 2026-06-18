#!/usr/bin/env python3
"""
clean_engine_residuals.py
═══════════════════════════════════════════════════════════════════════════════
Surgical second-pass cleanup for eclipse_core.py.

Your file is already ~95% clean (Apache license, z-score STDS description, etc.).
This script touches ONLY the few remaining inaccuracies, using the EXACT text that
is in your current file, so it will match this time.

  (A) THE LAST FALSE CLAIM:
      - "Kolmogorov-Smirnov test for distribution comparison" still appears in the
        EclipseFramework class docstring (NEW FEATURES). It is not implemented.
        -> deleted.

  (B) WORDING CONSISTENCY ("p-value" -> z-score):
      The STDS produces a z-score, not a p-value. A few lines still say "p-value"
      when describing the direction fix. The directional fix itself is real
      (positive z = suspicious); only the word "p-value" is wrong. -> reworded.

Conservative: makes a .bak backup, applies each change only if the exact text is
present, reports anything NOT FOUND, and scans for residual markers afterward.
Never deletes working code — only descriptive text.

USAGE:
    python clean_engine_residuals.py --selftest        # verify the logic
    python clean_engine_residuals.py eclipse_core.py   # clean your real file
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import shutil
from pathlib import Path

# (label, old_exact_text, new_text). new_text="" deletes the line.
REPLACEMENTS = [
    # --- (A) last false claim: Kolmogorov-Smirnov ---
    ("EclipseFramework docstring: KS feature line",
     "    - Kolmogorov-Smirnov test for distribution comparison\n",
     ""),

    # --- (B) p-value -> z-score wording (STDS produces a z-score, not a p-value) ---
    ("module docstring: BUG FIXES p-value line",
     "  \u2022 STDS p-value direction CORRECTED (large test statistic = suspicious)",
     "  \u2022 STDS direction corrected (large positive z-score = suspicious)"),
    ("EclipseFramework docstring: BUG FIXES p-value line",
     "    - STDS p-value direction corrected (large statistic = suspicious)",
     "    - STDS direction corrected (large positive z-score = suspicious)"),
    ("print_changelog: P-VALUE DIRECTION heading",
     "1. STDS P-VALUE DIRECTION (Critical)",
     "1. STDS DIRECTION (Critical)"),
    ("print_changelog: MIGRATION p-values line",
     "1. Re-run STDS to get corrected p-values",
     "1. Re-run STDS to get corrected z-scores"),
]

RESIDUAL_MARKERS = ["Kolmogorov", "kolmogorov", "p-value", "p-values", "P-VALUE",
                    "non-parametric", "AGPL", "Commercial License"]


def clean_text(text: str):
    report = []
    for label, old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new, 1)
            report.append(("OK", label))
        else:
            report.append(("NOT FOUND", label))
    residual = sorted({m for m in RESIDUAL_MARKERS if m in text})
    return text, report, residual


SELF_TEST_FIXTURE = (
    "\U0001f527 BUG FIXES:\n"
    "  \u2022 STDS p-value direction CORRECTED (large test statistic = suspicious)\n"
    "  \u2022 Leakage risk estimation no longer assumes degradation is always expected\n"
    "\n"
    "    \U0001f527 BUG FIXES:\n"
    "    - STDS p-value direction corrected (large statistic = suspicious)\n"
    "    - Leakage risk no longer assumes degradation is always expected\n"
    "    \n"
    "    \U0001f195 NEW FEATURES:\n"
    "    - Kolmogorov-Smirnov test for distribution comparison\n"
    "    - Notebook (.ipynb) support in Code Auditor\n"
    "\n"
    "1. STDS P-VALUE DIRECTION (Critical)\n"
    "\n"
    "1. Re-run STDS to get corrected p-values\n"
)


def main():
    args = list(sys.argv[1:])
    if "--selftest" in args:
        cleaned, report, residual = clean_text(SELF_TEST_FIXTURE)
        print("SELF-TEST against embedded fixture:")
        for status, label in report:
            print(f"  [{status}] {label}")
        print(f"\nResidual markers after cleanup: {residual if residual else 'NONE'}")
        ok = all(s == "OK" for s, _ in report) and not residual
        print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    target = Path(args[0]) if args else Path("eclipse_core.py")
    if not target.exists():
        print(f"ERROR: {target} not found. Pass the path, e.g.:\n  python clean_engine_residuals.py eclipse_core.py")
        return 1

    original = target.read_text(encoding="utf-8")
    backup = target.with_suffix(target.suffix + ".bak2")
    shutil.copy2(target, backup)

    cleaned, report, residual = clean_text(original)
    target.write_text(cleaned, encoding="utf-8")

    print(f"Backup saved: {backup}")
    print(f"Cleaned file written: {target}\n")
    for status, label in report:
        print(f"  [{status}] {label}")
    print(f"\nResidual markers still present: {residual if residual else 'NONE'}")
    if residual:
        print("  ^ If any remain, paste those lines and they can be handled explicitly.")
    print("\nNext: confirm the engine still imports and its tests pass:")
    print(f"  python {target.name} --test")
    return 0


if __name__ == "__main__":
    sys.exit(main())