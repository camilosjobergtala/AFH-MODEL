# -*- coding: utf-8 -*-
"""
Reproduces Table 8 (§4.3, S6 of "An admissibility test for constitutive
claims about phenomenal presence...", C. A. Sjöberg Tala): the behaviour of
the conjunction and its components as trial count grows, under one
weak-confounding generator with no return through B (S6):

    U ~ N(0, I_3),  L entries ~ N(0, 0.28^2)
    A1 = dr + U L1 + eps
    B  = dr + A1 W(0.7) + U L2 + eps
    A2 = dr + U L3 + eps

(a single shared covariate driver dr, per S6's own notation, distinct from
the per-population delta_A1/delta_B/delta_A2 convention of Table S1). At
each trial count, 200 replicates are drawn (80 at 4000 trials, per S9's
documented computational deviation) and each is scored exactly as in
run_chain.py: the two segment estimands positive at p<0.05 against their
100-permutation conditional null, and the chain positive only if both
episodic links reach p<0.05 against their own 50-permutation null (S6
reduces the permutation counts relative to run_chain.py's 200/100 for
compute; conjunction = all three positive).

Seeds follow Table S4: generative 5000+i; segment-1 test 6000+i; segment-2
test 6100+i; episodic test 6200+i.

RUN: python3 run_sweep.py [trial_counts] [n_replicates]
     python3 run_sweep.py 250,500,1000,2000,4000 200
TIME: ~10-15 minutes for the full sweep on a single core.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from sim_return import D, make_covariates, dr, eps, coupling, loadings, strata_labels, wilson_interval
from chain_return import (
    split_half, segment1_r2, segment1_null, segment2_r2, segment2_null,
    link1_query_target, link2_query_target, match_accuracy, link_permutation_null, p_value,
)

ALPHA = 0.05
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def gen_sweep(n, rng, d=D, sigma=0.28):
    Xc, Xk = make_covariates(n, rng)
    shared_dr = dr(Xc, Xk, d, rng)
    U = rng.standard_normal((n, 3))
    L1, L2, L3 = loadings(3, d, sigma, rng), loadings(3, d, sigma, rng), loadings(3, d, sigma, rng)
    A1 = shared_dr + U @ L1 + eps(n, d, 1.0, rng)
    B = shared_dr + A1 @ coupling(d, 0.7, rng) + U @ L2 + eps(n, d, 1.0, rng)
    A2 = shared_dr + U @ L3 + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk


def evaluate_replicate(n_trials, i):
    rng_gen = np.random.default_rng(5000 + i)
    A1, B, A2, Xc, Xk = gen_sweep(n_trials, rng_gen)

    rng_seg1 = np.random.default_rng(6000 + i)
    tr1, te1 = split_half(n_trials, rng_seg1)
    dR2_1, _ = segment1_r2(A1, B, Xc, Xk, tr1, te1)
    null1 = segment1_null(A1, B, Xc, Xk, tr1, te1, rng_seg1, n_perm=100)
    p1 = p_value(dR2_1, null1)

    rng_seg2 = np.random.default_rng(6100 + i)
    tr2, te2 = split_half(n_trials, rng_seg2)
    dR2_2, _ = segment2_r2(A1, B, A2, Xc, Xk, tr2, te2)
    null2 = segment2_null(A1, B, A2, Xc, Xk, tr2, te2, rng_seg2, n_perm=100)
    p2 = p_value(dR2_2, null2)

    rng_ep = np.random.default_rng(6200 + i)
    tr_e, te_e = split_half(n_trials, rng_ep)
    strata_te = strata_labels(Xc, Xk)[te_e]
    q1, t1 = link1_query_target(A1, B, Xc, Xk, tr_e, te_e)
    a1, _ = match_accuracy(q1, t1, strata_te)
    null_a1 = link_permutation_null(q1, t1, strata_te, rng_ep, n_perm=50)
    p_link1 = p_value(a1, null_a1)

    q2, t2 = link2_query_target(A1, B, A2, Xc, Xk, tr_e, te_e)
    a2, _ = match_accuracy(q2, t2, strata_te)
    null_a2 = link_permutation_null(q2, t2, strata_te, rng_ep, n_perm=50)
    p_link2 = p_value(a2, null_a2)

    chain_positive = (p_link1 < ALPHA) and (p_link2 < ALPHA)
    seg1_positive = p1 < ALPHA
    seg2_positive = p2 < ALPHA
    conjunction_positive = seg1_positive and seg2_positive and chain_positive
    return seg1_positive, seg2_positive, chain_positive, conjunction_positive


def rate_ci(flags):
    k, n = int(np.sum(flags)), len(flags)
    lo, hi = wilson_interval(k, n)
    return 100.0 * k / n, 100.0 * lo, 100.0 * hi


def main():
    trial_counts = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [250, 500, 1000, 2000, 4000]
    default_n = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdr = f"{'trials':>7s} {'conjunction [95% CI]':>22s} {'dR2 A1->B':>10s} {'dR2 B->A2|A1,X':>15s} {'chain':>7s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for n_trials in trial_counts:
        n_rep = 80 if n_trials == 4000 else default_n  # S9: documented deviation
        t0 = time.time()
        results = [evaluate_replicate(n_trials, i) for i in range(n_rep)]
        seg1 = [r[0] for r in results]
        seg2 = [r[1] for r in results]
        chain = [r[2] for r in results]
        conj = [r[3] for r in results]

        rj, loj, hij = rate_ci(conj)
        r1, _, _ = rate_ci(seg1)
        r2, _, _ = rate_ci(seg2)
        rc, _, _ = rate_ci(chain)

        print(f"{n_trials:7d} {rj:6.1f}% [{loj:4.1f}, {hij:5.1f}] {r1:9.1f}% {r2:14.1f}% {rc:6.1f}%  "
              f"(n={n_rep}, {time.time()-t0:.0f}s)")
        rows.append({
            "trials": n_trials, "n_replicates": n_rep,
            "conjunction_rate": rj, "conjunction_ci": [loj, hij],
            "seg1_rate": r1, "seg2_rate": r2, "chain_rate": rc,
        })

    with open(OUT_DIR / "sweep_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nJSON summary written to {OUT_DIR}/sweep_results.json")


if __name__ == "__main__":
    main()
