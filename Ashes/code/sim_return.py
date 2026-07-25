# -*- coding: utf-8 -*-
"""
Shared core module for the twelve-architecture recurrent-return simulation
(S2 of the Supplementary Methods to "An admissibility test for constitutive
claims about phenomenal presence: necessary conditions, asymmetric
falsification, and a conjunctive return-compatible criterion", C. A. Sjöberg
Tala). Imported, not executed (Table S6 of the Supplementary Methods).

Provides the machinery every other script in this package builds on:
  - the design matrix, closed-form ridge regression, and out-of-sample R^2
    used by every estimand in §3 of the article;
  - stimulus x arousal-quartile strata labels, used by the episodic matching
    test of chain_return.py;
  - a Wilson score interval, used to report positive rates over replicates;
  - generators for eleven of the twelve architectures of Table S1. The
    twelfth, token disruption, lives in chain_return.py (S8) because it
    shares the coarse quintile-binning helper the chain test itself relies
    on to be exercised.

Covariate, driver, noise, and coupling conventions follow S2 exactly:
    dr(Xc, Xk) = Gamma[Xc] + Xk @ gamma^T,   Gamma_ij, gamma_j ~ N(0,1)
    eps(s)     = s * Z,  Z ~ N(0, I_d) i.i.d. per trial
    W(s)       in R^{d x d}, entries ~ N(0, s^2)
Every occurrence of W(.), of a loading matrix L, or of a category pattern G
in Table S1 is an independent draw, even when the same scale is reused
elsewhere in the same equation; the helpers below draw a fresh matrix at
every call for that reason -- callers must not cache and reuse them.
"""
import numpy as np

D = 6       # population dimensionality (Table S4)
N_CAT = 4   # stimulus categories, uniform (Table S4)

# ---------------------------------------------------------------------------
# design matrix, ridge regression, out-of-sample R^2
# ---------------------------------------------------------------------------

def design(Xc, Xk, *extra):
    """Intercept + one-hot(stimulus) + arousal (+ any extra population blocks)."""
    n = Xc.shape[0]
    onehot = np.eye(N_CAT)[Xc]
    cols = [np.ones((n, 1)), onehot, Xk.reshape(-1, 1)]
    cols.extend(extra)
    return np.hstack(cols)

def ridge_fit(Ztr, Ytr, lam=1.0):
    """Closed-form multivariate ridge coefficients (Table S3: penalty fixed at 1.0)."""
    p = Ztr.shape[1]
    A = Ztr.T @ Ztr + lam * np.eye(p)
    return np.linalg.solve(A, Ztr.T @ Ytr)

def ridge_predict(Z, coef):
    return Z @ coef

def ridge_fit_predict(Ztr, Ytr, Zte, lam=1.0):
    return ridge_predict(Zte, ridge_fit(Ztr, Ytr, lam))

def r2_oos(Ytr, Yte, Yhat):
    """Out-of-sample R^2 = 1 - SSres/SStot, reference = training-set mean,
    summed jointly over all target dimensions so a single scalar summarises
    a multivariate fit (§3.7). Negative values are retained, not truncated."""
    mu = Ytr.mean(axis=0, keepdims=True)
    ss_res = ((Yte - Yhat) ** 2).sum()
    ss_tot = ((Yte - mu) ** 2).sum()
    return 1.0 - ss_res / ss_tot

# ---------------------------------------------------------------------------
# strata: stimulus category x arousal quartile
# ---------------------------------------------------------------------------

def quantile_bin(x, n_bins):
    """Bin a 1-D array into n_bins levels at its own empirical quantiles."""
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)).copy()
    qs[0], qs[-1] = -np.inf, np.inf
    return np.digitize(x, qs[1:-1])

def strata_labels(Xc, Xk, n_arousal_bins=4):
    """Strata = stimulus category x arousal quartile (16 cells, Table S3);
    §3.4 discards strata with fewer than three held-out trials at match time."""
    abin = quantile_bin(Xk, n_arousal_bins)
    return Xc * n_arousal_bins + abin

# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n (used throughout Table 6)."""
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = phat + z ** 2 / (2 * n)
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((centre - half) / denom, (centre + half) / denom)

# ---------------------------------------------------------------------------
# shared drivers, noise, coupling (S2)
# ---------------------------------------------------------------------------

def make_covariates(n, rng):
    Xc = rng.integers(0, N_CAT, n)
    Xk = rng.standard_normal(n)
    return Xc, Xk

def dr(Xc, Xk, d, rng):
    """Independently-drawn covariate driver: dr = Gamma[Xc] + Xk @ gamma^T."""
    Gamma = rng.standard_normal((N_CAT, d))
    gamma = rng.standard_normal(d)
    return Gamma[Xc] + np.outer(Xk, gamma)

def eps(n, d, scale, rng):
    """eps(s) = s * Z, Z ~ N(0, I_d) i.i.d. per trial."""
    return scale * rng.standard_normal((n, d))

def coupling(d, scale, rng):
    """W(s): d x d, entries ~ N(0, s^2). One fresh draw per call, as required."""
    return rng.standard_normal((d, d)) * scale

def loadings(q, d, sigma, rng):
    """L in R^{q x d}, entries ~ N(0, sigma^2). One fresh draw per call."""
    return rng.standard_normal((q, d)) * sigma

def category_pattern(K, d, sigma, rng):
    """G in R^{K x d}, entries ~ N(0, sigma^2). One fresh draw per call."""
    return rng.standard_normal((K, d)) * sigma

# ---------------------------------------------------------------------------
# eleven of the twelve architectures of Table S1
# (token disruption is defined in chain_return.py)
# ---------------------------------------------------------------------------
# Every generator returns (A1, B, A2, Xc, Xk, has_return) with A1, B, A2 of
# shape (n, D). "Each architecture draws its own dr for A1, B and A2" (S2):
# every generator below draws its own dA1, dB, dA2 independently, even where
# the equation for a population reduces to just that driver plus noise.

def gen_recurrent_return(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + A1 @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    A2 = dA2 + B @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, True

def gen_nonlinear_return(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + np.tanh(A1 @ coupling(d, 0.8, rng)) + eps(n, d, 1.0, rng)
    A2 = dA2 + np.tanh(B @ coupling(d, 0.8, rng)) + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, True

def gen_latent_confounder_weak_return(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    U = rng.standard_normal((n, 3))
    L1, L2, L3 = loadings(3, d, 0.9, rng), loadings(3, d, 0.9, rng), loadings(3, d, 0.9, rng)
    A1 = dA1 + U @ L1 + eps(n, d, 1.0, rng)
    B = dB + U @ L2 + A1 @ coupling(d, 0.15, rng) + eps(n, d, 1.0, rng)
    A2 = dA2 + U @ L3 + B @ coupling(d, 0.15, rng) + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, True

def gen_local_persistence(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + A1 @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    A2 = dA2 + A1 @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_feedforward_chain(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + A1 @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    _C = B @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)  # terminates outside A; unused downstream
    A2 = dA2 + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_observed_common_cause(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + eps(n, d, 1.0, rng)
    A2 = dA2 + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_parallel_pathways(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + eps(n, d, 1.0, rng)
    A2 = 1.3 * dr(Xc, Xk, d, rng) + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_latent_confounder(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    U = rng.standard_normal((n, 3))
    L1, L2, L3 = loadings(3, d, 0.9, rng), loadings(3, d, 0.9, rng), loadings(3, d, 0.9, rng)
    A1 = dA1 + U @ L1 + eps(n, d, 1.0, rng)
    B = dB + U @ L2 + eps(n, d, 1.0, rng)
    A2 = dA2 + U @ L3 + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_latent_feedforward_state(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    m = rng.standard_normal((n, 5))
    L1, L2, L3 = loadings(5, d, 0.9, rng), loadings(5, d, 0.9, rng), loadings(5, d, 0.9, rng)
    A1 = dA1 + m @ L1 + eps(n, d, 1.0, rng)
    B = dB + m @ L2 + eps(n, d, 1.0, rng)
    A2 = dA2 + m @ L3 + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_shared_low_rank_factor(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    c = rng.integers(0, 3, n)
    G = category_pattern(3, d, 1.6, rng)
    sigma = G[c]
    A1 = dA1 + sigma @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    B = dB + sigma @ coupling(d, 0.7, rng) + eps(n, d, 1.0, rng)
    A2 = dA2 + sigma + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

def gen_coarse_dependence(n, rng, d=D):
    Xc, Xk = make_covariates(n, rng)
    dA1, dB, dA2 = dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng), dr(Xc, Xk, d, rng)
    A1 = dA1 + eps(n, d, 1.0, rng)
    B = dB + eps(n, d, 1.0, rng)
    c = quantile_bin(A1[:, 0], 4)
    G = category_pattern(4, d, 1.5, rng)
    A2 = dA2 + G[c] + eps(n, d, 1.0, rng)
    return A1, B, A2, Xc, Xk, False

# Order matches Table 6 / Table S1, minus token disruption (chain_return.py).
BASE_ARCHITECTURES = [
    ("recurrent_return", gen_recurrent_return),
    ("nonlinear_return", gen_nonlinear_return),
    ("latent_confounder_weak_return", gen_latent_confounder_weak_return),
    ("local_persistence", gen_local_persistence),
    ("feedforward_chain", gen_feedforward_chain),
    ("observed_common_cause", gen_observed_common_cause),
    ("parallel_pathways", gen_parallel_pathways),
    ("latent_confounder", gen_latent_confounder),
    ("latent_feedforward_state", gen_latent_feedforward_state),
    ("shared_low_rank_factor", gen_shared_low_rank_factor),
    ("coarse_dependence", gen_coarse_dependence),
]

DISPLAY_NAME = {
    "recurrent_return": "Recurrent return",
    "nonlinear_return": "Nonlinear return",
    "latent_confounder_weak_return": "Latent confounder + weak return",
    "local_persistence": "Local persistence",
    "feedforward_chain": "Feedforward chain",
    "observed_common_cause": "Observed common cause",
    "parallel_pathways": "Parallel pathways",
    "latent_confounder": "Latent confounder",
    "latent_feedforward_state": "Latent feedforward state",
    "shared_low_rank_factor": "Shared low-rank factor",
    "coarse_dependence": "Coarse dependence",
    "token_disruption": "Token disruption",
}

if __name__ == "__main__":
    print(__doc__)
    print("This module defines shared machinery only; see run_chain.py, "
          "run_outcomes.py, run_sweep.py, run_thresh.py to reproduce results.")
