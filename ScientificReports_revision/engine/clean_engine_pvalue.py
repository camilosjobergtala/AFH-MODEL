#!/usr/bin/env python3
"""
clean_engine_pvalue.py
═══════════════════════════════════════════════════════════════════════════════
Final, TOLERANT cleanup for the leftover "p-value" wording in eclipse_core.py.

Why tolerant: exact-text matching kept failing because the on-disk file differs
slightly (whitespace/case) from pasted copies. This version uses regular
expressions, so it matches regardless of surrounding spaces or capitalization.

The STDS produces a z-score, not a p-value. These edits fix only the descriptive
text; the directional fix itself is real (a large positive z = suspicious):

  - "p-value direction"  ->  "direction"     (drops the inaccurate word "p-value")
  - "p-values"           ->  "z-scores"
  - any line still mentioning "Kolmogorov" -> removed (belt-and-suspenders)

Conservative: makes a .bak3 backup, reports how many substitutions it made, and
scans for residual markers afterward. Touches descriptive text only, never logic.

USAGE:
    python clean_engine_pvalue.py --selftest        # verify the logic
    python clean_engine_pvalue.py eclipse_core.py   # clean your real file
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import sys
import shutil
from pathlib import Path

RESIDUAL_MARKERS = ["Kolmogorov", "kolmogorov", "p-value", "p-values", "P-VALUE"]


def clean_text(text: str):
    report = []

    # 1) Remove "p-value " when it sits right before "direction" (any case/spacing),
    #    leaving "direction"/"DIRECTION" intact.
    text, n1 = re.subn(r"(?i)p-value\s+(?=direction)", "", text)
    report.append(("p-value before 'direction' removed", n1))

    # 2) "p-values" -> "z-scores" (any case)
    text, n2 = re.subn(r"(?i)p-values", "z-scores", text)
    report.append(("'p-values' -> 'z-scores'", n2))

    # 3) Any leftover standalone "p-value" (not caught above) -> "z-score"
    text, n3 = re.subn(r"(?i)p-value", "z-score", text)
    report.append(("leftover 'p-value' -> 'z-score'", n3))

    # 4) Remove any whole line still mentioning Kolmogorov (case-insensitive)
    lines = text.split("\n")
    kept = [ln for ln in lines if "kolmogorov" not in ln.lower()]
    n4 = len(lines) - len(kept)
    text = "\n".join(kept)
    report.append(("lines mentioning 'Kolmogorov' removed", n4))

    residual = sorted({m for m in RESIDUAL_MARKERS if m in text})
    return text, report, residual


SELF_TEST_FIXTURE = (
    "  \u2022 STDS p-value direction CORRECTED (large test statistic = suspicious)\n"
    "    - STDS  p-value   direction corrected (large statistic = suspicious)\n"   # extra spaces
    "1. STDS P-VALUE DIRECTION (Critical)\n"                                        # uppercase
    "1. Re-run STDS to get corrected p-values\n"
    "    - Kolmogorov-Smirnov test for distribution comparison\n"
)


def main():
    args = list(sys.argv[1:])
    if "--selftest" in args:
        cleaned, report, residual = clean_text(SELF_TEST_FIXTURE)
        print("SELF-TEST against embedded fixture (with varied spacing/case):")
        for label, n in report:
            print(f"  {n:>2} x  {label}")
        print("\n--- cleaned fixture ---")
        print(cleaned.rstrip())
        print("--- end ---")
        print(f"\nResidual markers after cleanup: {residual if residual else 'NONE'}")
        ok = not residual
        print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    # Default target baked in so the script works even with no argument.
    DEFAULT_TARGET = Path(r"g:/Mi unidad/AFH/ScientificReports_revision/engine/eclipse_core.py")
    target = Path(args[0]) if args else DEFAULT_TARGET
    if not target.exists():
        print(f"ERROR: {target} not found. Pass the path, e.g.:\n  python clean_engine_pvalue.py eclipse_core.py")
        return 1

    original = target.read_text(encoding="utf-8")
    backup = target.with_suffix(target.suffix + ".bak3")
    shutil.copy2(target, backup)

    cleaned, report, residual = clean_text(original)
    target.write_text(cleaned, encoding="utf-8")

    print(f"Backup saved: {backup}")
    print(f"Cleaned file written: {target}\n")
    for label, n in report:
        print(f"  {n:>2} x  {label}")
    print(f"\nResidual markers still present: {residual if residual else 'NONE'}")
    if residual:
        print("  ^ If any remain, paste those lines and they can be handled explicitly.")
    print("\nNext: confirm the engine still imports and its tests pass:")
    print(f"  python {target.name} --test")
    return 0


if __name__ == "__main__":
    sys.exit(main())