"""
═══════════════════════════════════════════════════════════════════════════════
STUDY 2 — STDS CONSTRUCT CHECK (operating characteristics)
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS IS (and is NOT):
  This is a CONTROLLED construct check of the v3.0 Statistical Test for Data
  Snooping (STDS), run on SIMULATED studies with KNOWN ground truth. It
  characterizes the test's operating behaviour — how often it flags a holdout as
  suspicious as a function of (a) how inflated the holdout is and (b) how noisy
  the CV folds are. It is a proof-of-concept under simulated conditions and
  carries NO external evidential weight. The external burden remains on Study 4.

THE TEST (v3.0, used verbatim via the engine):
  For each metric, z = (holdout - CV_mean) / CV_std. A holdout is "flagged"
  here when max_z > 2 (one-sided: holdout > 2 SD above the CV mean on at least
  one metric). This matches the engine's MODERATE/HIGH risk thresholds.

SIMULATION MODEL (per the test's own H0 assumption):
  - CV folds for a metric ~ Normal(mu, sigma_fold), K folds.
  - H0 (honest): holdout ~ Normal(mu, sigma_fold)         -> measures FALSE POSITIVES
  - H1 (snooped): holdout ~ Normal(mu*(1+delta), sigma_fold) -> measures SENSITIVITY
  The z-score, CV mean and CV std all come from the real engine via stds_wrapper.

ANALYSES:
  A. False-positive rate (specificity) under H0, vs #folds K and #metrics |M|.
  B. Sensitivity vs inflation delta, at several fold-noise levels sigma_fold.
  C. Boundary condition: when the holdout's noise differs from the folds'
     (an assumption violation), the false-positive rate inflates.

All randomness uses MASTER_SEED for exact reproducibility.

OUTPUTS (written next to this script, under ./outputs/study2/):
  - study2_fpr_by_K_M.csv
  - study2_sensitivity_by_noise.csv
  - study2_boundary_shift.csv
  - study2_summary.json
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np

# --- location-independent imports: find stds_wrapper.py / eclipse_core.py next to this file ---
HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "engine", HERE / "engine"):
    if (cand / "stds_wrapper.py").exists():
        sys.path.insert(0, str(cand))
        break
from stds_wrapper import StudyMetrics, run_stds   # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MASTER_SEED = 20250805
MU = 0.70                 # baseline metric level (e.g., an F1 around 0.70)
FLAG_THRESHOLD = 2.0      # max_z > 2 => flagged (one-sided, holdout suspiciously high)
N_SIM = 2000              # simulated studies per cell
ALPHA = 0.05              # engine's per-metric significance (z_crit ~ 1.96); FLAG uses max_z>2

OUT_DIR = HERE / "outputs" / "study2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(MASTER_SEED)


def simulate_study(k_folds, n_metrics, sigma_fold, delta, holdout_sigma_mult=1.0):
    """Build one simulated study and return the engine's max_z.
    delta: holdout inflation as a fraction of MU (0 = honest/H0).
    holdout_sigma_mult: multiply holdout draw SD vs fold SD (1.0 = assumption holds)."""
    cv_fold_values = {}
    holdout_values = {}
    for m in range(n_metrics):
        name = f"metric_{m}"
        folds = rng.normal(MU, sigma_fold, size=k_folds)
        cv_fold_values[name] = folds.tolist()
        holdout_mean = MU * (1.0 + delta)
        holdout = rng.normal(holdout_mean, sigma_fold * holdout_sigma_mult)
        holdout_values[name] = float(holdout)
    res = run_stds(StudyMetrics(cv_fold_values=cv_fold_values,
                                holdout_values=holdout_values, alpha=ALPHA))
    # cv_std can be 0 in degenerate draws -> engine skips metric; guard:
    return res.get("max_z_score", float("nan"))


def flag_rate(k_folds, n_metrics, sigma_fold, delta, holdout_sigma_mult=1.0, n_sim=N_SIM):
    flagged = 0
    valid = 0
    for _ in range(n_sim):
        mz = simulate_study(k_folds, n_metrics, sigma_fold, delta, holdout_sigma_mult)
        if mz == mz:  # not NaN
            valid += 1
            if mz > FLAG_THRESHOLD:
                flagged += 1
    return flagged / valid if valid else float("nan"), valid


def main():
    t0 = time.time()
    print("=" * 72)
    print("STUDY 2 — STDS CONSTRUCT CHECK (operating characteristics)")
    print(f"MASTER_SEED={MASTER_SEED}  MU={MU}  flag rule: max_z>{FLAG_THRESHOLD}  N_SIM={N_SIM}")
    print("=" * 72)

    summary = {"config": {"master_seed": MASTER_SEED, "mu": MU,
                          "flag_threshold_max_z": FLAG_THRESHOLD,
                          "n_sim": N_SIM, "alpha": ALPHA}}

    # ── A. False-positive rate under H0 (delta=0), vs K and |M| ────────────────
    print("\n[A] False-positive rate under H0 (honest holdout), by #folds K and #metrics |M|")
    print("    (sigma_fold fixed at 0.05; FPR is fraction of HONEST studies wrongly flagged)")
    sigma_A = 0.05
    Ks = [3, 5, 10]
    Ms = [1, 3, 5]
    rows_A = [("K_folds", "n_metrics", "sigma_fold", "fpr")]
    print(f"\n    {'K':>3} | " + " | ".join(f"|M|={m}" for m in Ms))
    for k in Ks:
        cells = []
        for m in Ms:
            fpr, _ = flag_rate(k, m, sigma_A, delta=0.0)
            rows_A.append((k, m, sigma_A, round(fpr, 4)))
            cells.append(f"{fpr*100:5.1f}%")
        print(f"    {k:>3} | " + " | ".join(cells))
    summary["A_fpr_by_K_M"] = [dict(zip(rows_A[0], r)) for r in rows_A[1:]]

    # ── B. Sensitivity vs inflation delta, at several fold-noise levels ────────
    print("\n[B] Sensitivity (power) vs holdout inflation, at K=5, |M|=1, by fold noise sigma_fold")
    print("    (sensitivity = fraction of SNOOPED studies correctly flagged)")
    sigmas_B = [0.02, 0.05, 0.10]
    deltas = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows_B = [("sigma_fold", "delta_inflation", "sensitivity")]
    header = "    sigma | " + " | ".join(f"+{int(d*100):>2}%" for d in deltas)
    print("\n" + header)
    for s in sigmas_B:
        cells = []
        for d in deltas:
            rate, _ = flag_rate(5, 1, s, delta=d)
            rows_B.append((s, d, round(rate, 4)))
            cells.append(f"{rate*100:4.0f}%")
        print(f"    {s:5.2f} | " + " | ".join(cells))
    summary["B_sensitivity_by_noise"] = [dict(zip(rows_B[0], r)) for r in rows_B[1:]]
    print("\n    (delta=0 column = false-positive rate under H0 for that noise level)")

    # ── C. Boundary condition: holdout noisier than folds (assumption violated) ─
    print("\n[C] Boundary condition: holdout drawn with LARGER noise than folds (H0, delta=0)")
    print("    Assumption 'holdout ~ same distribution as folds' is violated.")
    print("    K=5, |M|=1, sigma_fold=0.05; FPR should inflate above the matched-noise case.")
    rows_C = [("holdout_sigma_mult", "fpr_under_H0")]
    print(f"\n    {'holdout SD x':>14} | FPR")
    for mult in [1.0, 1.5, 2.0, 3.0]:
        fpr, _ = flag_rate(5, 1, 0.05, delta=0.0, holdout_sigma_mult=mult)
        rows_C.append((mult, round(fpr, 4)))
        print(f"    {mult:>14.1f} | {fpr*100:5.1f}%")
    summary["C_boundary_shift"] = [dict(zip(rows_C[0], r)) for r in rows_C[1:]]

    # ── write CSVs ─────────────────────────────────────────────────────────────
    def write_csv(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")
    write_csv(OUT_DIR / "study2_fpr_by_K_M.csv", rows_A)
    write_csv(OUT_DIR / "study2_sensitivity_by_noise.csv", rows_B)
    write_csv(OUT_DIR / "study2_boundary_shift.csv", rows_C)
    (OUT_DIR / "study2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Done in {time.time()-t0:.1f}s. Outputs in: {OUT_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()