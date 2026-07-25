# -*- coding: utf-8 -*-
"""
Reproduces Table 7 and the data behind Figure 4 (§4.2, S3, S7 of "An
admissibility test for constitutive claims about phenomenal presence...",
C. A. Sjöberg Tala).

For each of the twelve four-outcome scenarios of Table S2, simulates 200
studies (a study = several subjects, each contributing several trials, per
Table S2's subject/trial counts). For each study:

  1. Generate every subject's trial data under the scenario's modification
     of the baseline generative model A1=dA1+eps, B=dB+A1 W(0.7)+eps,
     A2=dA2+eps (S3).
  2. Per subject, estimate the two segment effects and theta_chain on a
     50/50 split, each bias-corrected against a 20-permutation null of its
     own conditional-null scheme (Table S3).
  3. Apply the validity gates (reliability: out-of-sample R2(B~X+A1)>0.10;
     positive control: recovery on an independent synthetic calibration
     channel; estimability: >=5 training observations per free parameter,
     always satisfied at the trial counts used here). Any gate failing
     yields "not evaluable" with no component assessed (§2.8).
  4. Otherwise bootstrap each component's per-subject effects over subjects
     (800 resamples, 90% percentile interval, Table S3) and apply the
     component decision rule of S10 to each of the three components.
  5. Aggregate to one of contradicted / survived / inconclusive per the
     conjunction rule of §3.2: any component contradicted -> contradicted;
     all components survived -> survived testing; otherwise (none
     contradicted, at least one inconclusive) -> inconclusive.

The positive-control channel is a synthetic calibration pair (P, Q)
independent of A1/B/A2, standing in for whatever hardware- or task-level
positive control an empirical study would report; it is not specified by
Table S1 (which covers only the return architectures), so this script
documents the choice here rather than silently inventing it: P, Q share the
same covariate/coupling machinery as every other population in this
package, with a fixed, always-recoverable coupling of 0.5, and six
independent estimates (Table S4: seeds study_seed+3000+k, k=0..5) are
averaged into one recovery statistic. "Failed positive control" permutes Q
against P (and the covariates) before recovery is estimated, simulating a
misaligned control channel.

Seeds follow Table S4: subject s of study i, 100000+37i+1000s; effect
estimation, study_seed+50000+s; positive control, study_seed+3000+k;
bootstrap, study_seed+7 (study_seed = 100000+37i).

RUN: python3 run_outcomes.py [start:end] [n_studies]
     python3 run_outcomes.py 0:12 200
TIME: ~15-30 minutes for the full 12 x 200 grid on a single core.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from sim_return import (
    D, design, ridge_fit, ridge_predict, r2_oos, strata_labels, make_covariates,
    dr, eps, coupling, loadings,
)
from chain_return import (
    split_half, segment1_r2, segment1_null, segment2_r2, segment2_null,
    link1_query_target, link2_query_target, match_accuracy, link_permutation_null,
    component_decision, bootstrap_ci,
)

N_BIAS_PERM = 20      # Table S3: permutations, bias correction (S4.2-S4.4)
N_BOOT = 800          # Table S3: bootstrap resamples over subjects
BOOT_PCT = 90         # Table S3: bootstrap interval, percentile
DELTA_MIN = 0.01      # Table S3: R2-scale components
DELTA_EPI = 0.02      # Table S3: chain component
RELIABILITY_THRESHOLD = 0.10
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


# ---------------------------------------------------------------------------
# scenario generator (Table S2): baseline + optional modifications
# ---------------------------------------------------------------------------

def gen_scenario(n, rng, return_w=None, extra_meas_noise=None,
                  corrupt_B_scale=None, misspecify=False, latent_confound=False, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)

    U = None
    if latent_confound:
        U = rng.standard_normal((n, 3))
        L_A1 = loadings(3, d, 0.9, rng)
        A1 = A1 + U @ L_A1

    B = dB + A1 @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    if latent_confound:
        L_B = loadings(3, d, 0.9, rng)
        B = B + U @ L_B

    if misspecify:
        X2 = rng.standard_normal(n)
        beta_B = rng.standard_normal(d)
        B = B + np.outer(X2, beta_B)

    A2 = dA2 + eps(n, d, 1.0, rng)
    if latent_confound:
        L_A2 = loadings(3, d, 0.9, rng)
        A2 = A2 + U @ L_A2
    if misspecify:
        beta_A2 = rng.standard_normal(d)
        A2 = A2 + np.outer(X2, beta_A2)
    if return_w is not None:
        A2 = A2 + B @ coupling(d, return_w, rng)

    if extra_meas_noise is not None:
        B = B + eps(n, d, extra_meas_noise, rng)
        A2 = A2 + eps(n, d, extra_meas_noise, rng)
    if corrupt_B_scale is not None:
        B = eps(n, d, corrupt_B_scale, rng)  # recording replaced by pure noise

    return A1, B, A2, Xc, Xk


SCENARIOS = {
    "absent_return_adequate": dict(subjects=20, trials=200, gen=dict()),
    "absent_return_reduced": dict(subjects=6, trials=200, gen=dict()),
    "effect_within_zone": dict(subjects=20, trials=200, gen=dict(return_w=0.040)),
    "effect_within_zone_larger": dict(subjects=40, trials=400, gen=dict(return_w=0.040)),
    "effect_near_boundary": dict(subjects=20, trials=200, gen=dict(return_w=0.085)),
    "effect_above_delta": dict(subjects=20, trials=200, gen=dict(return_w=0.300)),
    "return_present": dict(subjects=20, trials=200, gen=dict(return_w=0.600)),
    "measurement_error": dict(subjects=20, trials=200,
                               gen=dict(return_w=0.600, extra_meas_noise=2.2)),
    "unreliable_recording": dict(subjects=20, trials=200,
                                  gen=dict(return_w=0.600, corrupt_B_scale=6.0)),
    "misspecified_covariates": dict(subjects=20, trials=200, gen=dict(misspecify=True)),
    "latent_confounding": dict(subjects=20, trials=200, gen=dict(latent_confound=True)),
    "failed_positive_control": dict(subjects=20, trials=200, gen=dict(), permute_control=True),
}
DISPLAY = {
    "absent_return_adequate": "Absent return, adequate sample",
    "absent_return_reduced": "Absent return, reduced sample (6 subjects)",
    "effect_within_zone": "Effect within the equivalence zone",
    "effect_within_zone_larger": "Effect within the zone, larger sample (40x400)",
    "effect_near_boundary": "Effect near the boundary",
    "effect_above_delta": "Effect clearly above delta_min",
    "return_present": "Return present",
    "measurement_error": "High but quantified measurement error",
    "unreliable_recording": "Unreliable recording of B",
    "misspecified_covariates": "Misspecified covariates",
    "latent_confounding": "Latent confounding",
    "failed_positive_control": "Failed positive control",
}
SCENARIO_KEYS = list(SCENARIOS.keys())


# ---------------------------------------------------------------------------
# per-subject bias-corrected effects + reliability
# ---------------------------------------------------------------------------

def subject_effects(A1, B, A2, Xc, Xk, rng, lam=1.0):
    n = A1.shape[0]
    tr, te = split_half(n, rng)
    strata_te = strata_labels(Xc, Xk)[te]

    dR2_1, _ = segment1_r2(A1, B, Xc, Xk, tr, te, lam)
    null1 = segment1_null(A1, B, Xc, Xk, tr, te, rng, n_perm=N_BIAS_PERM, lam=lam)
    e1 = dR2_1 - null1.mean()

    dR2_2, _ = segment2_r2(A1, B, A2, Xc, Xk, tr, te, lam)
    null2 = segment2_null(A1, B, A2, Xc, Xk, tr, te, rng, n_perm=N_BIAS_PERM, lam=lam)
    e2 = dR2_2 - null2.mean()

    q1, t1 = link1_query_target(A1, B, Xc, Xk, tr, te, lam)
    a1, _ = match_accuracy(q1, t1, strata_te)
    null_a1 = link_permutation_null(q1, t1, strata_te, rng, n_perm=N_BIAS_PERM)
    theta1 = a1 - null_a1.mean()

    q2, t2 = link2_query_target(A1, B, A2, Xc, Xk, tr, te, lam)
    a2, _ = match_accuracy(q2, t2, strata_te)
    null_a2 = link_permutation_null(q2, t2, strata_te, rng, n_perm=N_BIAS_PERM)
    theta2 = a2 - null_a2.mean()
    echain = min(theta1, theta2)

    # reliability gate ingredient: out-of-sample R2(B~X+A1)
    Zfull = design(Xc, Xk, A1)
    r2_b_given_xa1 = r2_oos(B[tr], B[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], B[tr], lam)))

    return e1, e2, echain, r2_b_given_xa1


# ---------------------------------------------------------------------------
# positive-control gate: independent synthetic calibration channel
# ---------------------------------------------------------------------------

def positive_control_recovery(study_seed, permuted, delta_min, n_trials=200, d=2, lam=1.0):
    estimates = np.empty(6)
    for k in range(6):
        rng = np.random.default_rng(study_seed + 3000 + k)
        Xc, Xk = make_covariates(n_trials, rng)
        dP, dQ = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
        P = dP + eps(n_trials, d, 1.0, rng)
        Q = dQ + P @ coupling(d, 0.5, rng) + eps(n_trials, d, 1.0, rng)
        if permuted:
            perm = rng.permutation(n_trials)
            Q = Q[perm]
        tr, te = split_half(n_trials, rng)
        Zx, Zfull = design(Xc, Xk), design(Xc, Xk, P)
        r2_base = r2_oos(Q[tr], Q[te], ridge_predict(Zx[te], ridge_fit(Zx[tr], Q[tr], lam)))
        r2_full = r2_oos(Q[tr], Q[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], Q[tr], lam)))
        estimates[k] = r2_full - r2_base
    recovery = float(estimates.mean())
    return recovery, recovery > 0.5 * delta_min


# ---------------------------------------------------------------------------
# one study -> one of the four outcomes
# ---------------------------------------------------------------------------

def run_study(i, n_subjects, n_trials, gen_kwargs, permute_control,
              delta_min=DELTA_MIN, delta_epi=DELTA_EPI):
    study_seed = 100000 + 37 * i
    e1s, e2s, echains, reliab = [], [], [], []
    for s in range(n_subjects):
        rng_gen = np.random.default_rng(100000 + 37 * i + 1000 * s)
        A1, B, A2, Xc, Xk = gen_scenario(n_trials, rng_gen, **gen_kwargs)
        rng_eff = np.random.default_rng(study_seed + 50000 + s)
        e1, e2, echain, r2_rel = subject_effects(A1, B, A2, Xc, Xk, rng_eff)
        e1s.append(e1); e2s.append(e2); echains.append(echain); reliab.append(r2_rel)

    reliability_pass = np.mean(reliab) > RELIABILITY_THRESHOLD
    recovery, control_pass = positive_control_recovery(study_seed, permute_control, delta_min, n_trials)
    # Estimability (>=5 training observations per free parameter) is satisfied
    # by construction at every trial count used in this package (Table S3).
    estimability_pass = True

    if not (reliability_pass and control_pass and estimability_pass):
        return "not_evaluable", (e1s, e2s, echains)

    rng_boot = np.random.default_rng(study_seed + 7)
    lo1, hi1 = bootstrap_ci(e1s, N_BOOT, BOOT_PCT, rng_boot)
    lo2, hi2 = bootstrap_ci(e2s, N_BOOT, BOOT_PCT, rng_boot)
    loc, hic = bootstrap_ci(echains, N_BOOT, BOOT_PCT, rng_boot)

    d1 = component_decision(lo1, hi1, delta_min)
    d2 = component_decision(lo2, hi2, delta_min)
    dc = component_decision(loc, hic, delta_epi)
    decisions = (d1, d2, dc)

    if "contradicted" in decisions:
        return "contradicted", (e1s, e2s, echains)
    if all(d == "survived" for d in decisions):
        return "survived", (e1s, e2s, echains)
    return "inconclusive", (e1s, e2s, echains)


def main():
    arch_slice = sys.argv[1] if len(sys.argv) > 1 else "0:12"
    n_studies = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    start, end = (int(x) for x in arch_slice.split(":"))
    keys = SCENARIO_KEYS[start:end]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdr = f"{'scenario':45s} {'contra':>7s} {'survived':>9s} {'inconcl':>8s} {'not eval':>9s}"
    print(hdr)
    print("-" * len(hdr))

    for key in keys:
        cfg = SCENARIOS[key]
        t0 = time.time()
        outcomes = []
        for i in range(n_studies):
            outcome, _ = run_study(i, cfg["subjects"], cfg["trials"], cfg["gen"],
                                    cfg.get("permute_control", False))
            outcomes.append(outcome)

        counts = {k: outcomes.count(k) for k in ("contradicted", "survived", "inconclusive", "not_evaluable")}
        pct = {k: 100.0 * v / n_studies for k, v in counts.items()}
        print(f"{DISPLAY[key]:45s} {pct['contradicted']:7.1f} {pct['survived']:9.1f} "
              f"{pct['inconclusive']:8.1f} {pct['not_evaluable']:9.1f}  ({time.time()-t0:.0f}s)")

        with open(OUT_DIR / f"out_{key}.json", "w") as f:
            json.dump({
                "scenario": key, "display_name": DISPLAY[key], "n_studies": n_studies,
                "subjects": cfg["subjects"], "trials": cfg["trials"],
                "contradicted_pct": pct["contradicted"], "survived_pct": pct["survived"],
                "inconclusive_pct": pct["inconclusive"], "not_evaluable_pct": pct["not_evaluable"],
            }, f, indent=2)

    print(f"\nJSON summaries written to {OUT_DIR}/out_*.json")


if __name__ == "__main__":
    main()
