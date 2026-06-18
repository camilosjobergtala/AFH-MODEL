#!/usr/bin/env python3
"""
clean_engine_claims.py
═══════════════════════════════════════════════════════════════════════════════
One-shot honesty cleanup for eclipse_core.py.

Removes claims the code does NOT actually implement, and aligns the license header:

  (A) FALSE-CLAIM REMOVAL (the reason for this script):
      - "Kolmogorov-Smirnov test" — never implemented; STDS uses a z-score only.
      - "both parametric and non-parametric tests/approaches" — only a z-score exists.
      These are replaced with an honest description of the z-score method.

  (B) LICENSE HEADER:
      - The AGPL v3.0 / Commercial dual-license block is replaced with Apache-2.0,
        matching the manuscript and the repository LICENSE file.

The script is conservative: it makes a .bak backup, applies each replacement only
if the exact target text is present, and reports anything it could NOT find so you
can flag it. It never deletes working code — only descriptive text.

USAGE:
    python clean_engine_claims.py --selftest          # verify the logic (no file needed)
    python clean_engine_claims.py eclipse_core.py     # clean your real engine file
    python clean_engine_claims.py                     # defaults to ./eclipse_core.py
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import shutil
from pathlib import Path

# Each entry: (label, old_exact_text, new_text). new_text="" deletes.
REPLACEMENTS = [
    # --- (A) FALSE CLAIMS: Kolmogorov-Smirnov / non-parametric ---
    ("module docstring: KS feature bullet",
     "  \u2022 Kolmogorov-Smirnov test for distribution comparison (more powerful than mean comparison)\n",
     ""),
    ("module docstring: parametric/non-parametric line",
     "  \u2022 STDS uses both parametric and non-parametric tests",
     "  \u2022 STDS reimplemented as a standard z-score (replaces v2.0 bootstrap/permutation)"),
    ("print_changelog: KS block",
     "1. KOLMOGOROV-SMIRNOV DISTRIBUTION TEST\n"
     "   - More powerful than mean comparison\n"
     "   - Detects ANY distribution difference, not just location shift\n"
     "   - Used alongside parametric tests for robustness",
     "1. STDS REIMPLEMENTED AS A STANDARD Z-SCORE\n"
     "   - z = (holdout - CV_mean) / CV_std\n"
     "   - Replaces the v2.0 bootstrap/permutation procedure\n"
     "   - A large positive z flags a suspiciously good holdout"),
    ("HTML report: KS list item",
     "                <li><strong>Kolmogorov-Smirnov test added</strong> - Distribution comparison</li>",
     "                <li><strong>STDS reimplemented as standard z-score</strong> - Replaces bootstrap/permutation</li>"),
    ("text report: KS line",
     '        lines.append("  \u2713 Kolmogorov-Smirnov distribution test added")',
     '        lines.append("  \u2713 STDS reimplemented as a standard z-score")'),
    ("EclipseFramework docstring: parametric/non-parametric",
     "    - STDS uses both parametric and non-parametric approaches",
     "    - STDS reimplemented as a standard z-score"),
    ("__init__ print: KS line",
     '        print("  \u2705 Kolmogorov-Smirnov distribution test")',
     '        print("  \u2705 STDS: standard z-score (no bootstrap/permutation)")'),
    ("interactive_menu banner: KS line",
     "\u2551  \U0001f195 Kolmogorov-Smirnov distribution test                                    \u2551",
     "\u2551  \U0001f195 STDS: standard z-score (no bootstrap/permutation)                       \u2551"),

    # --- (B) LICENSE HEADER: AGPL/Commercial -> Apache-2.0 ---
    ("license header block",
     "LICENSE: DUAL LICENSE (AGPL v3.0 / Commercial)\n"
     "\n"
     "This software is available under two licenses:\n"
     "\n"
     "1. AGPL v3.0 (Open Source)\n"
     "   - Free for research, education, and non-commercial use\n"
     "\n"
     "2. Commercial License\n"
     "   - Required for commercial/proprietary use\n"
     "   - Contact: cst@afhmodel.org",
     "LICENSE: Apache License 2.0\n"
     "\n"
     "This software is released under the Apache License, Version 2.0.\n"
     "See the LICENSE file in the repository root for full terms.\n"
     "http://www.apache.org/licenses/LICENSE-2.0"),
]

RESIDUAL_MARKERS = ["Kolmogorov", "kolmogorov", "non-parametric", "AGPL", "Commercial License"]


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
    "\U0001f195 NEW FEATURES:\n"
    "  \u2022 Kolmogorov-Smirnov test for distribution comparison (more powerful than mean comparison)\n"
    "  \u2022 Notebook (.ipynb) support in Code Auditor\n"
    "\n"
    "\U0001f4ca IMPROVED METRICS:\n"
    "  \u2022 STDS uses both parametric and non-parametric tests\n"
    "\n"
    "1. KOLMOGOROV-SMIRNOV DISTRIBUTION TEST\n"
    "   - More powerful than mean comparison\n"
    "   - Detects ANY distribution difference, not just location shift\n"
    "   - Used alongside parametric tests for robustness\n"
    "\n"
    "                <li><strong>Kolmogorov-Smirnov test added</strong> - Distribution comparison</li>\n"
    '        lines.append("  \u2713 Kolmogorov-Smirnov distribution test added")\n'
    "    - STDS uses both parametric and non-parametric approaches\n"
    '        print("  \u2705 Kolmogorov-Smirnov distribution test")\n'
    "\u2551  \U0001f195 Kolmogorov-Smirnov distribution test                                    \u2551\n"
    "LICENSE: DUAL LICENSE (AGPL v3.0 / Commercial)\n"
    "\n"
    "This software is available under two licenses:\n"
    "\n"
    "1. AGPL v3.0 (Open Source)\n"
    "   - Free for research, education, and non-commercial use\n"
    "\n"
    "2. Commercial License\n"
    "   - Required for commercial/proprietary use\n"
    "   - Contact: cst@afhmodel.org\n"
)


def main():
    args = [a for a in sys.argv[1:]]
    if "--selftest" in args:
        cleaned, report, residual = clean_text(SELF_TEST_FIXTURE)
        print("SELF-TEST against embedded fixture:")
        for status, label in report:
            print(f"  [{status}] {label}")
        print(f"\nResidual false markers after cleanup: {residual if residual else 'NONE'}")
        ok = all(s == "OK" for s, _ in report) and not residual
        print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    target = Path(args[0]) if args else Path("eclipse_core.py")
    if not target.exists():
        print(f"ERROR: {target} not found. Pass the path, e.g.:\n  python clean_engine_claims.py path/to/eclipse_core.py")
        return 1

    original = target.read_text(encoding="utf-8")
    backup = target.with_suffix(target.suffix + ".bak")
    shutil.copy2(target, backup)

    cleaned, report, residual = clean_text(original)
    target.write_text(cleaned, encoding="utf-8")

    print(f"Backup saved: {backup}")
    print(f"Cleaned file written: {target}\n")
    for status, label in report:
        print(f"  [{status}] {label}")
    print(f"\nResidual false markers still present: {residual if residual else 'NONE'}")
    if residual:
        print("  ^ If any remain, paste those lines and they can be handled explicitly.")
    print("\nNext: confirm the engine still imports and its tests pass:")
    print(f"  python {target.name} --test")
    return 0


if __name__ == "__main__":
    sys.exit(main())