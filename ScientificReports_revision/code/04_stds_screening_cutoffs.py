"""
═══════════════════════════════════════════════════════════════════════════════
STUDY 2b — DERIVATION OF THE STDS EMPIRICAL SCREENING CUTOFFS
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS IS:
  Reviewer 1 (Major 2) and Reviewer 2 (point 1) both objected that STDS retained
  a nominal-significance framing (|z| > 2 ~ p < 0.05) that its own Study 2
  operating characteristics contradict. Reviewer 2 asked specifically for
  "empirically characterized screening thresholds ... [with] no formal
  hypothesis-test interpretation."

  This script produces those thresholds. For each (K folds, M metrics) it
  simulates the Study 2 HONEST scenario -- holdout drawn from the same
  distribution as the folds -- and records the 95th and 99th percentiles of
  max_z. Screening at those values therefore flags 5% and 1% of honest
  simulated studies BY CONSTRUCTION.

WHAT THIS IS NOT:
  These are NOT p-value thresholds and imply NO Type I error guarantee for real
  cross-validation. The simulation draws folds independently; real CV folds are
  overlapping fits on shared data, so the honest-scenario rate obtained here is
  a lower bound on the flag rate a real honest study would see. The cutoffs are
  descriptive screening values whose behaviour under one explicit generative
  model is known -- nothing more. See §2.6.3 for the assumption set.

  Because the engine screens at these cutoffs, verifying the honest-scenario
  rate against them is a self-consistency check, not independent validation.

CANONICAL SOURCE:
  All z-scores come from the real engine via stds_wrapper.run_stds(). The
  vectorised inner loop is used only after asserting, on this run's own seed,
  that it reproduces the engine's max_z exactly (see verify_against_engine()).

OUTPUT:
  outputs/study2b/stds_screening_cutoffs.csv
  outputs/study2b/stds_screening_cutoffs.json   <- pasted into
                                                   EclipseIntegrityScore-side
                                                   StatisticalTestDataSnooping
                                                   .EMPIRICAL_SCREEN
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent / "engine", HERE / "engine"):
    if (cand / "stds_wrapper.py").exists():
        sys.path.insert(0, str(cand))
        break
from stds_wrapper import StudyMetrics, run_stds  # noqa: E402

MASTER_SEED = 20250805
MU = 0.70
SIGMA = 0.05
N_SIM = 60000
K_GRID = [3, 5, 10, 20]
M_GRID = [1, 2, 3, 5]

OUT_DIR = HERE.parent / "outputs" / "study2b"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def verify_against_engine(rng, n_check: int = 2000) -> float:
    """Assert the vectorised max_z equals the engine's, case by case."""
    worst = 0.0
    for _ in range(n_check):
        m = int(rng.integers(1, 6))
        k = int(rng.choice(K_GRID))
        folds = {f"m{j}": list(rng.normal(MU, SIGMA, k)) for j in range(m)}
        hold = {f"m{j}": float(rng.normal(MU, SIGMA)) for j in range(m)}
        engine = run_stds(StudyMetrics(folds, hold))["max_z_score"]
        arr = np.array([folds[key] for key in folds])
        hv = np.array([hold[key] for key in folds])
        vec = float(((hv - arr.mean(1)) / arr.std(1)).max())
        worst = max(worst, abs(engine - vec))
    return worst


def honest_max_z(rng, k: int, m: int, n_sim: int) -> np.ndarray:
    """max_z over m metrics, holdout drawn from the fold distribution."""
    folds = rng.normal(MU, SIGMA, size=(n_sim, m, k))
    hold = rng.normal(MU, SIGMA, size=(n_sim, m))
    z = (hold - folds.mean(axis=2)) / folds.std(axis=2)
    return z.max(axis=1)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    print("=" * 72)
    print("STDS EMPIRICAL SCREENING CUTOFFS (Study 2b)")
    print("=" * 72)

    worst = verify_against_engine(rng)
    print(f"\n[0] Vectorised path vs canonical engine: max |diff| = {worst:.3e}")
    if worst > 1e-12:
        raise SystemExit("Vectorised path diverges from the engine; aborting.")
    print("    Verified identical. Proceeding.")

    rows, table = [], {}
    print(f"\n[1] Honest-scenario percentiles of max_z (N = {N_SIM:,} per cell)\n")
    print(f"    {'K':>4} {'|M|':>5} {'p95':>7} {'p99':>7}   {'rate at fixed 2.0':>18}")
    for k in K_GRID:
        for m in M_GRID:
            mx = honest_max_z(rng, k, m, N_SIM)
            p95, p99 = float(np.quantile(mx, 0.95)), float(np.quantile(mx, 0.99))
            at2 = float(np.mean(mx > 2.0))
            rows.append({"k_folds": k, "n_metrics": m,
                         "cutoff_p95": round(p95, 2), "cutoff_p99": round(p99, 2),
                         "honest_rate_at_fixed_2": round(at2, 4)})
            table[f"{k},{m}"] = [round(p95, 2), round(p99, 2)]
            print(f"    {k:>4} {m:>5} {p95:>7.2f} {p99:>7.2f}   {at2*100:>17.1f}%")

    import csv
    with open(OUT_DIR / "stds_screening_cutoffs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "stds_screening_cutoffs.json").write_text(
        json.dumps({"master_seed": MASTER_SEED, "n_sim": N_SIM,
                    "mu": MU, "sigma_fold": SIGMA, "cutoffs": table}, indent=2))

    print(f"\n[2] Self-consistency check: screening at p95 should flag ~5%.")
    for k, m in [(3, 1), (5, 3), (10, 1), (20, 5)]:
        mx = honest_max_z(rng, k, m, 20000)
        cut = table[f"{k},{m}"][0]
        print(f"    K={k:>2} |M|={m}  -> {np.mean(mx > cut)*100:5.2f}% flagged "
              f"(target 5.00%)")

    print(f"\nOutputs -> {OUT_DIR}")
    print(f"Done in {time.time() - t0:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
