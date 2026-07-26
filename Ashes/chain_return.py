# -*- coding: utf-8 -*-
"""
Conjunctive observational return-compatible criterion on the path A1 -> B -> A2.

Components, all evaluated out of sample:

  N_A1_to_B          dR2_{A1->B}  = R2(B  ~ X + A1)      - R2(B  ~ X)
                     null: A1 independent of B given X

  N_B_to_A2_given    dR2_{B->A2}  = R2(A2 ~ X + A1 + B)  - R2(A2 ~ X + A1)
                     null: B independent of A2 given A1 and X

  N_chain_episode    episode identity preserved across A1_i -> B_i -> A2_i,
                     tested as two linked matching problems, both required

  N_same_source      A1 and A2 are the same preregistered population; a design
                     requirement, not estimable from the data, recorded as a flag

Conjunction: return-compatible only if all components are positive.

Both nulls are residual permutation schemes in the spirit of Freedman and Lane
(1983). The nuisance block is fitted on the training partition only; the fitted
values and residuals are then formed for all trials using those training-fitted
coefficients, so no test-partition information enters the conditional model. The
exchangeability assumption is that, under the null, the residuals of the permuted
variable with respect to the nuisance block are exchangeable across trials. This
is exact only if the conditional model is correctly specified; under
misspecification the null is approximate and the test can be anticonservative,
which is the behaviour the misspecified-covariate architecture is designed to
expose.
"""
import numpy as np
from sim_return import _design, _fit, _r2, strata, DIM, LAM, wilson


# ----------------------------------------------------------------------------- helpers

def _split(n, rng):
    idx = rng.permutation(n)
    return idx[:n // 2], idx[n // 2:]


def _fl_parts(Y, Z, tr):
    """Fit Y on nuisance design Z using the TRAINING rows only; return fitted values
    and residuals for all rows under those training-fitted coefficients."""
    A = Z[tr].T @ Z[tr] + LAM * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z[tr].T @ Y[tr])
    fitted = Z @ beta
    return fitted, Y - fitted


def _match_acc(query, target, st_te):
    """Nearest-neighbour matching of query_i to target_i within strata.
    Returns (accuracy, chance). Candidates are not reused; ties broken by argmin."""
    c = t = 0
    ch = 0.0
    for s in np.unique(st_te):
        m = np.where(st_te == s)[0]
        if m.size < 3:
            continue
        d = ((query[m][:, None, :] - target[m][None, :, :]) ** 2).sum(-1)
        c += np.sum(d.argmin(1) == np.arange(m.size))
        t += m.size
        ch += 1.0
    return (c / t, ch / t) if t else (np.nan, np.nan)


# ----------------------------------------------------------------- segment estimands

def estimand_A1_to_B(A1, B, Xc, Xk, nperm=200, seed=0):
    """dR2_{A1->B} with the conditional null A1 _||_ B | X.
    A1 is residualized on X; residuals are permuted and added back to fitted values."""
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    tr, te = _split(n, rng)
    Zx = _design(Xc, Xk)

    def r2_of(A1mat=None):
        blocks = [] if A1mat is None else [A1mat]
        Ztr = _design(Xc[tr], Xk[tr], *[b[tr] for b in blocks])
        Zte = _design(Xc[te], Xk[te], *[b[te] for b in blocks])
        return _r2(B[tr], B[te], _fit(Ztr, B[tr], Zte))

    r2x = r2_of(None)
    obs = r2_of(A1) - r2x

    fitted, resid = _fl_parts(A1, Zx, tr)
    null = np.empty(nperm)
    for b in range(nperm):
        null[b] = r2_of(fitted + resid[rng.permutation(n)]) - r2x
    p = (1 + np.sum(null >= obs)) / (nperm + 1)
    return obs - null.mean(), p


def estimand_B_to_A2(A1, B, A2, Xc, Xk, nperm=200, seed=0):
    """dR2_{B->A2} with the conditional null B _||_ A2 | A1, X.
    B is residualized on [X, A1], preserving the legitimate A1-B association."""
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    tr, te = _split(n, rng)
    Znuis = _design(Xc, Xk, A1)

    def r2_of(*blocks):
        Ztr = _design(Xc[tr], Xk[tr], *[b[tr] for b in blocks])
        Zte = _design(Xc[te], Xk[te], *[b[te] for b in blocks])
        return _r2(A2[tr], A2[te], _fit(Ztr, A2[tr], Zte))

    base = r2_of(A1)
    obs = r2_of(A1, B) - base

    fitted, resid = _fl_parts(B, Znuis, tr)
    null = np.empty(nperm)
    for b in range(nperm):
        null[b] = r2_of(A1, fitted + resid[rng.permutation(n)]) - base
    p = (1 + np.sum(null >= obs)) / (nperm + 1)
    return obs - null.mean(), p


# --------------------------------------------------------- chain episode specificity

def chain_episodic(A1, B, A2, Xc, Xk, nperm=200, seed=1):
    """Two linked matching problems, both required.

      link 1: does the incremental A1 contribution to B identify the correct B_i?
      link 2: does the incremental B contribution to A2 identify the correct A2_i?

    All fits are on the training partition; matching is on held-out trials within
    strata. Returns (acc1-chance, p1, acc2-chance, p2, passes_both).
    """
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    tr, te = _split(n, rng)
    st_te = strata(Xc, Xk)[te]

    def link(Y, nuis_blocks, add_block):
        Zb_tr = _design(Xc[tr], Xk[tr], *[b[tr] for b in nuis_blocks])
        Zb_te = _design(Xc[te], Xk[te], *[b[te] for b in nuis_blocks])
        Zf_tr = _design(Xc[tr], Xk[tr], *[b[tr] for b in nuis_blocks], add_block[tr])
        Zf_te = _design(Xc[te], Xk[te], *[b[te] for b in nuis_blocks], add_block[te])
        base = _fit(Zb_tr, Y[tr], Zb_te)
        full = _fit(Zf_tr, Y[tr], Zf_te)
        query, target = full - base, Y[te] - base
        a, ch = _match_acc(query, target, st_te)
        null = np.empty(nperm)
        for b in range(nperm):
            tp = target.copy()
            for s in np.unique(st_te):
                m = np.where(st_te == s)[0]
                tp[m] = target[m][rng.permutation(m.size)]
            null[b], _ = _match_acc(query, tp, st_te)
        return a - ch, (1 + np.sum(null >= a)) / (nperm + 1)

    d1, p1 = link(B, [], A1)              # A1 -> B
    d2, p2 = link(A2, [A1], B)            # B -> A2 given A1
    return d1, p1, d2, p2, (p1 < 0.05 and p2 < 0.05)


# --------------------------------------------------------------------- architectures

def gen_chain(arch, n=1200, seed=0):
    """Adds the token-disruption architecture to the original eleven.
    Returns (A1, B, A2, Xc, Xk, ret) where ret = episode-specific return through B."""
    if arch != "token_disruption":
        from sim_return import gen
        return gen(arch, n=n, seed=seed)

    r = np.random.default_rng(seed)
    Xc = r.integers(0, 4, n)
    Xk = r.standard_normal(n)
    dr = lambda: r.standard_normal((4, DIM))[Xc] + np.outer(Xk, r.standard_normal(DIM))
    dA1, dB, dA2 = dr(), dr(), dr()
    e = lambda s=1.0: s * r.standard_normal((n, DIM))

    A1 = dA1 + e()
    # A1 drives B only through a coarse five-level summary: predictive gain is real,
    # episode identity is destroyed at the first link
    q1 = np.quantile(A1[:, 0], [0.2, 0.4, 0.6, 0.8])
    cl1 = np.digitize(A1[:, 0], q1)
    B = dB + (r.standard_normal((5, DIM)) * 1.6)[cl1] + e()
    # B drives A2 only through a coarse five-level summary
    q2 = np.quantile(B[:, 0], [0.2, 0.4, 0.6, 0.8])
    cl2 = np.digitize(B[:, 0], q2)
    A2 = dA2 + (r.standard_normal((5, DIM)) * 1.6)[cl2] + e()
    return A1, B, A2, Xc, Xk, False


ARCHS_CHAIN = ["recurrent_return", "nonlinear_return", "confounder_plus_weak_return",
               "local_persistence", "feedforward_chain", "observed_common_cause",
               "parallel_pathways", "latent_confounder", "latent_ff_state",
               "shared_low_rank", "coarse_dependence", "token_disruption"]
