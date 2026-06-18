"""
00_environment_check.py
ECLIPSE v2.0 reconstruction — environment verification and provenance baseline.

Purpose:
  - Verify Python version and that required packages are present.
  - Record exact package versions for reproducibility.
  - Establish and document the master random seed used by all study scripts.
  - Write a provenance log so every later result is traceable to a known environment.

Run this FIRST, before any study script.

Outputs:
  - logs/environment_<timestamp>.json   (full environment snapshot)
  - requirements_frozen.txt              (pinned versions, for the repository)

Place this file in:  ScientificReports_revision/code/00_environment_check.py
Run from anywhere:    python code/00_environment_check.py
"""

import sys
import json
import platform
from datetime import datetime
from pathlib import Path

# --- Master seed: declared ONCE here, imported/reused by all study scripts ---
# Convention (from the ECLIPSE protocol): the seed is declared before any analysis
# and justified. Here it derives from the original submission date (2025-08-05).
MASTER_SEED = 20250805

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib"]


def check_environment():
    report = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "python_full": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "master_seed": MASTER_SEED,
        "packages": {},
        "missing": [],
    }
    for pkg in REQUIRED:
        try:
            mod = __import__(pkg)
            report["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            report["missing"].append(pkg)
    return report


def main():
    # base = ScientificReports_revision/  (parent of code/)
    base = Path(__file__).resolve().parent.parent
    logs = base / "logs"
    logs.mkdir(exist_ok=True)

    report = check_environment()

    # --- Console summary ---
    print("=" * 60)
    print("ECLIPSE reconstruction — environment check")
    print("=" * 60)
    print(f"Python:      {report['python_version']}")
    print(f"Platform:    {report['platform']}")
    print(f"Executable:  {report['executable']}")
    print(f"Master seed: {report['master_seed']}")
    print("-" * 60)
    for pkg, ver in report["packages"].items():
        print(f"  {pkg:14s} {ver}")
    if report["missing"]:
        print("-" * 60)
        print("MISSING packages (install before continuing):")
        for pkg in report["missing"]:
            name = "scikit-learn" if pkg == "sklearn" else pkg
            print(f"  - {name}")
    print("=" * 60)

    # --- Provenance log ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs / f"environment_{ts}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Provenance log written:   {log_path}")

    # --- Frozen requirements (pinned versions for the repo) ---
    req_path = base / "requirements_frozen.txt"
    with open(req_path, "w", encoding="utf-8") as f:
        for pkg, ver in report["packages"].items():
            name = "scikit-learn" if pkg == "sklearn" else pkg
            if ver != "unknown":
                f.write(f"{name}=={ver}\n")
    print(f"Frozen requirements:      {req_path}")

    if report["missing"]:
        print("\nFAIL: install the missing packages, then re-run this script.")
        sys.exit(1)

    print("\nEnvironment OK. Next: 01_study1_eis_coherence.py")


if __name__ == "__main__":
    main()