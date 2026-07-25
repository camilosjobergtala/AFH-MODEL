# -*- coding: utf-8 -*-
"""
Segment estimands, conditional nulls, chain-level episodic matching, and the
twelfth (token-disruption) architecture for the recurrent-return simulation
(§3.4-3.5 and S8-S10 of the article and its Supplementary Methods). Imported,
not executed (Table S6).

Three pieces of machinery live here rather than in sim_return.py:

  - the two segment estimands, dR2_{A1->B} and dR2_{B->A2|A1,X} (§3.2), and
    their conditional-null residual-permutation tests (§3.5): a nuisance
    block is regressed out on the training partition only, its residuals
    are permuted and added back to the training-fitted values to form a
    surrogate, and the whole segment estimand is recomputed on that
    surrogate. The nuisance block for the second segment must include A1
    (§3.5); omitting it would test a null no architecture of interest
    satisfies.

  - chain-level episode specificity (§3.4): two linked matching problems on
    held-out trials, matched by independent nearest-neighbour assignment
    within strata (stimulus category x arousal quartile, strata smaller
    than three discarded), with theta_link = accuracy - chance and
    theta_chain = min(theta_link_1, theta_link_2) -- the weakest link, since
    a mean or product would let a strong link mask a broken one.

  - the token-disruption architecture, which routes both segments through a
    coarse quintile-binned lookup so that both predictive gains are
    genuinely positive while episode identity is destroyed at each link
    (§3.8/S8's design test of whether the chain component is redundant).

The component decision rule of S10 (bootstrap interval vs. margin, including
the adverse-direction case) is also defined here, since run_chain.py,
run_outcomes.py, run_sweep.py, and run_thresh.py all need it.
"""
import numpy as np

from sim_return import (
    D, N_CAT, design, ridge_fit, ridge_predict, r2_oos, strata_labels,
    quantile_bin, make_covariates, dr, eps, category_pattern,
)

# ---------------------------------------------------------------------------
# train/test split
# ---------------------------------------------------------------------------

def split_half(n, rng):
    """50/50 train/test split by random permutation (§3.7)."""
    idx = rng.permutation(n)
    half = n // 2
    return idx[:half], idx[half:]

# ---------------------------------------------------------------------------
# segment estimands
# ---------------------------------------------------------------------------

def segment1_r2(A1, B, Xc, Xk, tr, te, lam=1.0):
    """dR2_{A1->B} = R2(B~X+A1) - R2(B~X), out-of-sample (§3.2)."""
    Zx = design(Xc, Xk)
    Zfull = design(Xc, Xk, A1)
    r2_base = r2_oos(B[tr], B[te], ridge_predict(Zx[te], ridge_fit(Zx[tr], B[tr], lam)))
    r2_full = r2_oos(B[tr], B[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], B[tr], lam)))
    return r2_full - r2_base, r2_base

def segment2_r2(A1, B, A2, Xc, Xk, tr, te, lam=1.0):
    """dR2_{B->A2|A1,X} = R2(A2~X+A1+B) - R2(A2~X+A1), out-of-sample (§3.2)."""
    Zbase = design(Xc, Xk, A1)
    Zfull = design(Xc, Xk, A1, B)
    r2_base = r2_oos(A2[tr], A2[te], ridge_predict(Zbase[te], ridge_fit(Zbase[tr], A2[tr], lam)))
    r2_full = r2_oos(A2[tr], A2[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], A2[tr], lam)))
    return r2_full - r2_base, r2_base

def _residual_surrogate(target_var, nuisance_Z, tr, rng):
    """Fit target_var ~ nuisance_Z on the training partition only; apply the
    training-fitted coefficients to every trial to form fitted values and
    residuals; permute the residuals and add them back to the fitted values
    (§3.5). Returns a surrogate of target_var for all trials."""
    coef = ridge_fit(nuisance_Z[tr], target_var[tr])
    fitted = ridge_predict(nuisance_Z, coef)
    resid = target_var - fitted
    perm = rng.permutation(resid.shape[0])
    return fitted + resid[perm]

def segment1_null(A1, B, Xc, Xk, tr, te, rng, n_perm=200, lam=1.0):
    """H0: A1 _||_ B | X (§3.5). Nuisance block is X; A1 is the permuted variable."""
    Zx = design(Xc, Xk)
    r2_base = r2_oos(B[tr], B[te], ridge_predict(Zx[te], ridge_fit(Zx[tr], B[tr], lam)))
    null = np.empty(n_perm)
    for b in range(n_perm):
        A1_perm = _residual_surrogate(A1, Zx, tr, rng)
        Zfull = design(Xc, Xk, A1_perm)
        r2_full = r2_oos(B[tr], B[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], B[tr], lam)))
        null[b] = r2_full - r2_base
    return null

def segment2_null(A1, B, A2, Xc, Xk, tr, te, rng, n_perm=200, lam=1.0):
    """H0: B _||_ A2 | A1, X (§3.5). Nuisance block is [X, A1] -- must include
    A1, or the permutation destroys the legitimate A1-B association and
    tests a null no architecture of interest satisfies."""
    Zbase = design(Xc, Xk, A1)
    r2_base = r2_oos(A2[tr], A2[te], ridge_predict(Zbase[te], ridge_fit(Zbase[tr], A2[tr], lam)))
    null = np.empty(n_perm)
    for b in range(n_perm):
        B_perm = _residual_surrogate(B, Zbase, tr, rng)
        Zfull = design(Xc, Xk, A1, B_perm)
        r2_full = r2_oos(A2[tr], A2[te], ridge_predict(Zfull[te], ridge_fit(Zfull[tr], A2[tr], lam)))
        null[b] = r2_full - r2_base
    return null

# ---------------------------------------------------------------------------
# chain-level episode specificity
# ---------------------------------------------------------------------------

def link1_query_target(A1, B, Xc, Xk, tr, te, lam=1.0):
    """Link 1 (§3.4): fit B~X and B~X+A1 on train; on held-out trials,
    query = Bhat_{X+A1} - Bhat_X, target = B - Bhat_X."""
    Zx, Zfull = design(Xc, Xk), design(Xc, Xk, A1)
    Bhat_x = ridge_predict(Zx[te], ridge_fit(Zx[tr], B[tr], lam))
    Bhat_full = ridge_predict(Zfull[te], ridge_fit(Zfull[tr], B[tr], lam))
    return Bhat_full - Bhat_x, B[te] - Bhat_x

def link2_query_target(A1, B, A2, Xc, Xk, tr, te, lam=1.0):
    """Link 2 (§3.4): fit A2~X+A1 and A2~X+A1+B on train; on held-out trials,
    query = A2hat_{X+A1+B} - A2hat_{X+A1}, target = A2 - A2hat_{X+A1}."""
    Zbase, Zfull = design(Xc, Xk, A1), design(Xc, Xk, A1, B)
    A2hat_base = ridge_predict(Zbase[te], ridge_fit(Zbase[tr], A2[tr], lam))
    A2hat_full = ridge_predict(Zfull[te], ridge_fit(Zfull[tr], A2[tr], lam))
    return A2hat_full - A2hat_base, A2[te] - A2hat_base

def match_accuracy(query, target, strata_te, min_stratum=3):
    """Independent nearest-neighbour assignment within each stratum (§3.4):
    each query selects its nearest target under squared Euclidean distance,
    independently of every other query -- not a one-to-one assignment.
    Returns (accuracy, chance); chance = |S| / n_te, computed rather than
    assumed, because each query in a stratum of size m selects its own
    target with probability 1/m and sum_s m_s * (1/m_s) = |S|."""
    correct, total, n_strata = 0, 0, 0
    for s in np.unique(strata_te):
        m = np.where(strata_te == s)[0]
        if m.size < min_stratum:
            continue
        d = ((query[m][:, None, :] - target[m][None, :, :]) ** 2).sum(-1)
        correct += np.sum(d.argmin(axis=1) == np.arange(m.size))
        total += m.size
        n_strata += 1
    if total == 0:
        return np.nan, np.nan
    return correct / total, n_strata / total

def link_permutation_null(query, target, strata_te, rng, n_perm=100, min_stratum=3):
    """Null for one link (§3.4): permute targets within strata, holding
    queries fixed; recompute matching accuracy each time."""
    null = np.empty(n_perm)
    for b in range(n_perm):
        tperm = target.copy()
        for s in np.unique(strata_te):
            m = np.where(strata_te == s)[0]
            if m.size < min_stratum:
                continue
            tperm[m] = target[m][rng.permutation(m.size)]
        acc, _ = match_accuracy(query, tperm, strata_te, min_stratum)
        null[b] = acc
    return null

def p_value(observed, null):
    """One-sided permutation p-value: fraction of the null at least as
    extreme as the observed statistic, plus-one smoothed."""
    return (1 + np.sum(null >= observed)) / (len(null) + 1)

def theta_chain(theta_link1, theta_link2):
    """theta_chain = min(theta_link_1, theta_link_2) (§3.4): the chain
    preserves episode identity only to the extent its weakest link does."""
    return min(theta_link1, theta_link2)

# ---------------------------------------------------------------------------
# component decision rule (S10), shared by run_outcomes/run_sweep/run_thresh
# ---------------------------------------------------------------------------

def component_decision(lo, hi, delta):
    """S10: compare a 90% bootstrap interval [lo, hi] over subjects to the
    margin delta.
        lo > delta                -> "survived"
        lo > -delta and hi < delta -> "contradicted" (negligible effect)
        hi < -delta                -> "contradicted" (adverse direction)
        otherwise                  -> "inconclusive"
    The adverse-direction row is explicit because an effect reliably below
    -delta is not negligible: it is large in the direction opposite to the
    one the component asserts, and is therefore incompatible with a
    required positive predictive gain."""
    if lo > delta:
        return "survived"
    if lo > -delta and hi < delta:
        return "contradicted"
    if hi < -delta:
        return "contradicted"
    return "inconclusive"

def bootstrap_ci(values, n_boot, pct, rng):
    """Percentile bootstrap interval over subjects (§3.7)."""
    values = np.asarray(values)
    n = values.shape[0]
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = values[idx].mean()
    lo_p, hi_p = (100 - pct) / 2, 100 - (100 - pct) / 2
    return np.percentile(boots, lo_p), np.percentile(boots, hi_p)

# ---------------------------------------------------------------------------
# token disruption: the twelfth architecture (Table S1, last row)
# ---------------------------------------------------------------------------

def gen_token_disruption(n, rng, d=D):
    """A1 drives B only through a five-level coarse summary of A1, and B
    drives A2 only through a five-level coarse summary of B: both predictive
    gains are genuinely positive, while the episode-specific token
    originating in A1 is destroyed at each link (§3.8)."""
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    c1 = quantile_bin(A1[:, 0], 5)
    G1 = category_pattern(5, d, 1.6, rng)
    B = dB + G1[c1] + eps(n, d, 1.0, rng)
    c2 = quantile_bin(B[:, 0], 5)
    G2 = category_pattern(5, d, 1.6, rng)
    A2 = dA2 + G2[c2] + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

ARCHITECTURES = None  # populated below to keep import order explicit

def _build_architecture_list():
    from sim_return import BASE_ARCHITECTURES
    return BASE_ARCHITECTURES + [("token_disruption", gen_token_disruption)]

ARCHITECTURES = _build_architecture_list()

if __name__ == "__main__":
    print(__doc__)
    print("Architectures, in Table 6 order:")
    for key, _ in ARCHITECTURES:
        print(" -", key)
