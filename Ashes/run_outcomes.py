# -*- coding: utf-8 -*-
"""
Four-outcome classification over the conjunctive return-compatible criterion.

Each component receives its own equivalence decision from a bootstrap interval over
subjects. Component decisions are then aggregated by the rule of section 3.2:

    any validity gate fails            -> not evaluable
    any component contradicted         -> conjunction contradicted
    all components survived            -> conjunction survived
    otherwise                          -> conjunction inconclusive

Components entering the aggregation:
    C1  dR2_{A1->B}                 equivalence margin DELTA_R2
    C2  dR2_{B->A2 | A1, X}         equivalence margin DELTA_R2
    C3  chain episodic accuracy      equivalence margin DELTA_EPI

Both margins are illustrative of the generative model, not empirically calibrated.

Usage:  python3 run_outcomes.py <i0>:<i1> [R]
"""
import numpy as np, json, sys, time
from sim_return import _design, _fit, _r2, strata, DIM, LAM, wilson
from chain_return import _fl_parts, _match_acc, _split

DELTA_R2, DELTA_EPI = 0.01, 0.02
T_TRIAL, NADJ, BBOOT, CI = 200, 20, 800, 90
REL_MIN, PC_MIN, MIN_TPP = 0.10, 0.5 * 0.01, 5.0


# ------------------------------------------------------------------ per-subject effects

def subject_effects(A1, B, A2, Xc, Xk, seed=0, nadj=NADJ):
    """Bias-corrected point estimates of the three components for one subject."""
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    tr, te = _split(n, rng)
    st_te = strata(Xc, Xk)[te]

    def r2B(*blocks):
        Ztr = _design(Xc[tr], Xk[tr], *[b[tr] for b in blocks])
        Zte = _design(Xc[te], Xk[te], *[b[te] for b in blocks])
        return _r2(B[tr], B[te], _fit(Ztr, B[tr], Zte))

    def r2A2(*blocks):
        Ztr = _design(Xc[tr], Xk[tr], *[b[tr] for b in blocks])
        Zte = _design(Xc[te], Xk[te], *[b[te] for b in blocks])
        return _r2(A2[tr], A2[te], _fit(Ztr, A2[tr], Zte))

    # C1: A1 -> B, null A1 _||_ B | X
    base1 = r2B()
    obs1 = r2B(A1) - base1
    f1, r1 = _fl_parts(A1, _design(Xc, Xk), tr)
    null1 = np.array([r2B(f1 + r1[rng.permutation(n)]) - base1 for _ in range(nadj)])

    # C2: B -> A2 given A1, X
    base2 = r2A2(A1)
    obs2 = r2A2(A1, B) - base2
    f2, r2_ = _fl_parts(B, _design(Xc, Xk, A1), tr)
    null2 = np.array([r2A2(A1, f2 + r2_[rng.permutation(n)]) - base2 for _ in range(nadj)])

    # C3: chain episodic, both links; effect is the smaller of the two accuracies
    def link(Y, nuis, add):
        Zb_tr = _design(Xc[tr], Xk[tr], *[b[tr] for b in nuis])
        Zb_te = _design(Xc[te], Xk[te], *[b[te] for b in nuis])
        Zf_tr = _design(Xc[tr], Xk[tr], *[b[tr] for b in nuis], add[tr])
        Zf_te = _design(Xc[te], Xk[te], *[b[te] for b in nuis], add[te])
        bs = _fit(Zb_tr, Y[tr], Zb_te)
        fl = _fit(Zf_tr, Y[tr], Zf_te)
        a, ch = _match_acc(fl - bs, Y[te] - bs, st_te)
        return a - ch

    e1 = link(B, [], A1)
    e2 = link(A2, [A1], B)
    return (obs1 - null1.mean(), obs2 - null2.mean(), min(e1, e2))


def reliability(B, A1, Xc, Xk, seed=0):
    rng = np.random.default_rng(seed)
    n = B.shape[0]
    tr, te = _split(n, rng)
    Ztr = _design(Xc[tr], Xk[tr], A1[tr])
    Zte = _design(Xc[te], Xk[te], A1[te])
    return max(0.0, _r2(B[tr], B[te], _fit(Ztr, B[tr], Zte)))


# ---------------------------------------------------------------------- scenarios

SCN = ["absent_power", "absent_low_power", "within_equiv", "within_equiv_highpower",
       "near_boundary", "clearly_above", "return_present", "high_meas_error",
       "unreliable_signal", "misspecified_cov", "latent_confounding",
       "failed_positive_control"]

CONFIG = {  # (n_subjects, trials per subject)
    "absent_power": (20, 200), "absent_low_power": (6, 200),
    "within_equiv": (20, 200), "within_equiv_highpower": (40, 400),
    "near_boundary": (20, 200), "clearly_above": (20, 200),
    "return_present": (20, 200), "high_meas_error": (20, 200),
    "unreliable_signal": (20, 200), "misspecified_cov": (20, 200),
    "latent_confounding": (20, 200), "failed_positive_control": (20, 200),
}


def gen_subject(scn, seed):
    S, n = CONFIG[scn]
    r = np.random.default_rng(seed)
    Xc = r.integers(0, 4, n)
    Xk = r.standard_normal(n)
    dr = lambda: r.standard_normal((4, DIM))[Xc] + np.outer(Xk, r.standard_normal(DIM))
    dA1, dB, dA2 = dr(), dr(), dr()
    e = lambda s=1.0: s * r.standard_normal((n, DIM))
    W = lambda s: r.standard_normal((DIM, DIM)) * s

    A1 = dA1 + e()
    B = dB + A1 @ W(0.7) + e()
    A2 = dA2 + e()                                   # absent-return baseline

    if scn in ("absent_power", "absent_low_power", "failed_positive_control"):
        pass
    elif scn in ("within_equiv", "within_equiv_highpower"):
        A2 = dA2 + B @ W(0.04) + e()
    elif scn == "near_boundary":
        A2 = dA2 + B @ W(0.085) + e()
    elif scn == "clearly_above":
        A2 = dA2 + B @ W(0.30) + e()
    elif scn == "return_present":
        A2 = dA2 + B @ W(0.60) + e()
    elif scn == "high_meas_error":
        A2 = dA2 + B @ W(0.60) + e()
        B = B + 2.2 * r.standard_normal((n, DIM))
        A2 = A2 + 2.2 * r.standard_normal((n, DIM))
    elif scn == "unreliable_signal":
        A2 = dA2 + B @ W(0.60) + e()
        B = 6.0 * r.standard_normal((n, DIM))        # B block destroyed
    elif scn == "misspecified_cov":
        X2 = r.standard_normal(n)
        B = B + np.outer(X2, r.standard_normal(DIM))
        A2 = dA2 + np.outer(X2, r.standard_normal(DIM)) + e()
    elif scn == "latent_confounding":
        U = r.standard_normal((n, 3))
        L = lambda: r.standard_normal((3, DIM)) * 0.9
        A1 = dA1 + U @ L() + e()
        B = dB + A1 @ W(0.7) + U @ L() + e()
        A2 = dA2 + U @ L() + e()
    else:
        raise ValueError(scn)
    return A1, B, A2, Xc, Xk


def positive_control(seed, n, broken=False):
    r = np.random.default_rng(seed)
    Xc = r.integers(0, 4, n)
    Xk = r.standard_normal(n)
    dr = lambda: r.standard_normal((4, DIM))[Xc] + np.outer(Xk, r.standard_normal(DIM))
    A1 = dr() + r.standard_normal((n, DIM))
    B = dr() + A1 @ (r.standard_normal((DIM, DIM)) * 0.7) + r.standard_normal((n, DIM))
    A2 = dr() + B @ (r.standard_normal((DIM, DIM)) * 0.60) + r.standard_normal((n, DIM))
    if broken:
        B = B[r.permutation(n)]
    return subject_effects(A1, B, A2, Xc, Xk, seed=seed + 5, nadj=10)[1]


# ----------------------------------------------------------------- decision logic

def component_decision(vals, delta, rng):
    bs = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(BBOOT)])
    lo, hi = np.percentile(bs, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    if lo > delta:
        return "survived"
    if lo > -delta and hi < delta:
        return "contradicted"
    if hi < -delta:
        # Adverse direction: the interval lies entirely below -delta. A reliably
        # negative effect is incompatible with a component that asserts a positive
        # predictive gain, so this is a contradiction of the required condition,
        # not an inconclusive result. Folded into "contradicted" for reporting.
        return "contradicted"
    return "inconclusive"


def classify(scn, seed):
    S, n = CONFIG[scn]
    subj = [gen_subject(scn, seed + 1000 * s) for s in range(S)]

    rel = np.mean([reliability(sb[1], sb[0], sb[3], sb[4], seed + k)
                   for k, sb in enumerate(subj)])
    if rel < REL_MIN:
        return "not_evaluable", None
    if (n // 2) / (1 + 4 + 1 + 2 * DIM) < MIN_TPP:
        return "not_evaluable", None
    broken = (scn == "failed_positive_control")
    pc = np.mean([positive_control(seed + 3000 + k, n, broken=broken) for k in range(6)])
    if pc <= PC_MIN:
        return "not_evaluable", None

    eff = np.array([subject_effects(*sb, seed=seed + 50_000 + i) for i, sb in enumerate(subj)])
    rng = np.random.default_rng(seed + 7)
    d1 = component_decision(eff[:, 0], DELTA_R2, rng)
    d2 = component_decision(eff[:, 1], DELTA_R2, rng)
    d3 = component_decision(eff[:, 2], DELTA_EPI, rng)
    comps = [d1, d2, d3]

    if "contradicted" in comps:
        return "contradicted", comps
    if all(c == "survived" for c in comps):
        return "survived", comps
    return "inconclusive", comps


OUT = ["contradicted", "survived", "inconclusive", "not_evaluable"]

if __name__ == "__main__":
    i0, i1 = [int(v) for v in sys.argv[1].split(":")]
    R = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    t0 = time.time()
    res = {}
    for scn in SCN[i0:i1]:
        c = {o: 0 for o in OUT}
        for i in range(R):
            o, _ = classify(scn, 100_000 + 37 * i)
            c[o] += 1
        res[scn] = {o: c[o] for o in OUT}
        print(f"{scn:26s} " + " ".join(f"{o[:5]}={100*c[o]/R:5.1f}" for o in OUT), flush=True)
        json.dump(res, open(f"out_{i0}-{i1}.json", "w"))
    print(f"[{time.time()-t0:.0f}s]", flush=True)
