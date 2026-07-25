# -*- coding: utf-8 -*-
"""
Reproduces Table 6 and the data behind Figure 3 (§4.1 of "An admissibility
test for constitutive claims about phenomenal presence: necessary
conditions, asymmetric falsification, and a conjunctive return-compatible
criterion", C. A. Sjöberg Tala).

For each of the twelve architectures of Table S1, generates 200 replicates
of 1200 trials (six-dimensional populations, four stimulus categories, a
standard normal arousal covariate, unit-scale Gaussian noise -- §3.8) and
evaluates, per replicate:

  - the first-segment estimand dR2_{A1->B}, positive if p<0.05 against its
    conditional-null residual-permutation test (200 permutations, §3.5);
  - the second-segment estimand dR2_{B->A2|A1,X}, same procedure, nuisance
    block [X, A1] (200 permutations);
  - the chain component, positive only if BOTH episodic links individually
    reach p<0.05 against their own within-stratum target-permutation null
    (100 permutations per link, §3.4);
  - the conjunction, positive iff all three are positive (the same-source
    component is a design requirement, satisfied by construction here).

Positive rates over replicates are reported with 95% Wilson intervals, as
in Table 6. Per-architecture JSON summaries are written for make_fig3.py.

Seeds follow Table S4 exactly: generative 10000+i, first segment 20000+i,
second segment 21000+i, episodic 30000+i, where i is the replicate index
within an architecture (0-indexed).

RUN: python3 run_chain.py [start:end] [n_replicates]
     python3 run_chain.py 0:12 200
TIME: ~20-30 minutes for the full 12 x 200 grid on a single core.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from chain_return import (
    ARCHITECTURES, split_half, segment1_r2, segment2_r2, segment1_null,
    segment2_null, link1_query_target, link2_query_target, match_accuracy,
    link_permutation_null, p_value,
)
from sim_return import strata_labels, wilson_interval, DISPLAY_NAME

N_TRIALS = 1200
ALPHA = 0.05
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def evaluate_replicate(gen_fn, i):
    rng_gen = np.random.default_rng(10000 + i)
    A1, B, A2, Xc, Xk, has_return = gen_fn(N_TRIALS, rng_gen)

    rng_seg1 = np.random.default_rng(20000 + i)
    tr1, te1 = split_half(N_TRIALS, rng_seg1)
    dR2_1, _ = segment1_r2(A1, B, Xc, Xk, tr1, te1)
    null1 = segment1_null(A1, B, Xc, Xk, tr1, te1, rng_seg1, n_perm=200)
    p1 = p_value(dR2_1, null1)

    rng_seg2 = np.random.default_rng(21000 + i)
    tr2, te2 = split_half(N_TRIALS, rng_seg2)
    dR2_2, _ = segment2_r2(A1, B, A2, Xc, Xk, tr2, te2)
    null2 = segment2_null(A1, B, A2, Xc, Xk, tr2, te2, rng_seg2, n_perm=200)
    p2 = p_value(dR2_2, null2)

    rng_ep = np.random.default_rng(30000 + i)
    tr_e, te_e = split_half(N_TRIALS, rng_ep)
    strata_te = strata_labels(Xc, Xk)[te_e]
    q1, t1 = link1_query_target(A1, B, Xc, Xk, tr_e, te_e)
    a1, c1 = match_accuracy(q1, t1, strata_te)
    theta1 = a1 - c1
    null_a1 = link_permutation_null(q1, t1, strata_te, rng_ep, n_perm=100)
    p_link1 = p_value(a1, null_a1)

    q2, t2 = link2_query_target(A1, B, A2, Xc, Xk, tr_e, te_e)
    a2, c2 = match_accuracy(q2, t2, strata_te)
    theta2 = a2 - c2
    null_a2 = link_permutation_null(q2, t2, strata_te, rng_ep, n_perm=100)
    p_link2 = p_value(a2, null_a2)

    chain_positive = (p_link1 < ALPHA) and (p_link2 < ALPHA)
    seg1_positive = p1 < ALPHA
    seg2_positive = p2 < ALPHA
    conjunction_positive = seg1_positive and seg2_positive and chain_positive

    return {
        "has_return": has_return,
        "dR2_1": dR2_1, "p1": p1, "seg1_positive": seg1_positive,
        "dR2_2": dR2_2, "p2": p2, "seg2_positive": seg2_positive,
        "theta_chain": min(theta1, theta2),
        "p_link1": p_link1, "p_link2": p_link2, "chain_positive": chain_positive,
        "conjunction_positive": conjunction_positive,
    }


def rate_ci(flags):
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = wilson_interval(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


def main():
    arch_slice = sys.argv[1] if len(sys.argv) > 1 else "0:12"
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    start, end = (int(x) for x in arch_slice.split(":"))
    archs = ARCHITECTURES[start:end]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdr = (f"{'architecture':32s} {'return':6s} | {'dR2 A1->B':>10s} | "
           f"{'dR2 B->A2|A1,X':>15s} | {'chain':>8s} | {'conjunction':>11s}")
    print(hdr)
    print("-" * len(hdr))

    for key, gen_fn in archs:
        t0 = time.time()
        rows = [evaluate_replicate(gen_fn, i) for i in range(n_rep)]
        has_return = rows[0]["has_return"]

        r1, lo1, hi1 = rate_ci([r["seg1_positive"] for r in rows])
        r2, lo2, hi2 = rate_ci([r["seg2_positive"] for r in rows])
        rc, loc, hic = rate_ci([r["chain_positive"] for r in rows])
        rj, loj, hij = rate_ci([r["conjunction_positive"] for r in rows])

        print(f"{DISPLAY_NAME[key]:32s} {str(has_return):6s} | "
              f"{r1:6.1f} [{lo1:4.1f},{hi1:5.1f}] | "
              f"{r2:6.1f} [{lo2:4.1f},{hi2:5.1f}] | "
              f"{rc:5.1f} [{loc:4.1f},{hic:5.1f}] | "
              f"{rj:5.1f} [{loj:4.1f},{hij:5.1f}]  ({time.time()-t0:.0f}s)")

        summary = {
            "architecture": key, "display_name": DISPLAY_NAME[key],
            "has_return": has_return, "n_replicates": n_rep,
            "seg1_rate": r1, "seg1_ci": [lo1, hi1],
            "seg2_rate": r2, "seg2_ci": [lo2, hi2],
            "chain_rate": rc, "chain_ci": [loc, hic],
            "conjunction_rate": rj, "conjunction_ci": [loj, hij],
            "mean_theta_chain": float(np.mean([r["theta_chain"] for r in rows])),
        }
        with open(OUT_DIR / f"chain_{key}.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nJSON summaries written to {OUT_DIR}/chain_*.json")


if __name__ == "__main__":
    main()
