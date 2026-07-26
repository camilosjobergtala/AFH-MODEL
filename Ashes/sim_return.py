# -*- coding: utf-8 -*-
"""
Recurrent RETURN topology: A_early -> B -> A_late, where A_late is the SAME source
population as A_early measured in a later window, and B is an intermediate population
whose return path B -> A is the object of the necessary condition.

Estimands (out-of-sample, permutation-adjusted within strata of X):
  dR2_A = R2(A2 ~ X + A1) - R2(A2 ~ X)                      early-late dependence
  dR2_B = R2(A2 ~ X + A1 + B) - R2(A2 ~ X + A1)             incremental role of B on the path
Episodic specificity: matching of the incremental A1 contribution within strata.

Ground truth flag `ret` = TRUE only when A2 receives episode-specific return through B.

TRANSCRIPTION NOTE: reconstructed verbatim from S1_return_topology_core.pdf.
The only edits are repairs to line-wrap artifacts introduced by PDF text extraction
(e.g. "(Yte-\nmu)" -> "(Yte - mu)"). No logic, constant, or seed was altered.
"""
import numpy as np

DIM, LAM, NSTRAT = 6, 1.0, 4


def _design(Xc, Xk, *blocks):
    cols = [np.ones((Xc.shape[0], 1)), np.eye(4)[Xc], Xk.reshape(-1, 1)]
    cols += [b for b in blocks if b is not None]
    return np.hstack(cols)


def _fit(Ztr, Ytr, Zte):
    A = Ztr.T @ Ztr + LAM * np.eye(Ztr.shape[1])
    return Zte @ np.linalg.solve(A, Ztr.T @ Ytr)


def _r2(Ytr, Yte, Yhat):
    mu = Ytr.mean(0, keepdims=True)
    return 1 - ((Yte - Yhat) ** 2).sum() / ((Yte - mu) ** 2).sum()


def strata(Xc, Xk):
    q = np.quantile(Xk, np.linspace(0, 1, NSTRAT + 1))
    q[0], q[-1] = -np.inf, np.inf
    return Xc * NSTRAT + np.digitize(Xk, q[1:-1])


def _perm_within(M, st, rng):
    Mp = M.copy()
    for s in np.unique(st):
        idx = np.where(st == s)[0]
        Mp[idx] = M[rng.permutation(idx)]
    return Mp


def estimands(A1, B, A2, Xc, Xk, nperm=200, seed=0):
    """Returns (dR2_A, p_A, dR2_B, p_B) with within-stratum permutation nulls."""
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[:n // 2], idx[n // 2:]
    st = strata(Xc, Xk)
    Zx_tr, Zx_te = _design(Xc[tr], Xk[tr]), _design(Xc[te], Xk[te])
    r2x = _r2(A2[tr], A2[te], _fit(Zx_tr, A2[tr], Zx_te))

    def r2_with(*blocks):
        Ztr = _design(Xc[tr], Xk[tr], *[b[tr] for b in blocks])
        Zte = _design(Xc[te], Xk[te], *[b[te] for b in blocks])
        return _r2(A2[tr], A2[te], _fit(Ztr, A2[tr], Zte))

    r2_A = r2_with(A1)
    dA = r2_A - r2x
    dB = r2_with(A1, B) - r2_A

    groups = [np.where(st == s)[0] for s in np.unique(st)]
    nullA = np.empty(nperm)
    nullB = np.empty(nperm)
    for b in range(nperm):
        A1p = A1.copy()
        Bp = B.copy()
        for g in groups:
            A1p[g] = A1[rng.permutation(g)]
            Bp[g] = B[rng.permutation(g)]
        nullA[b] = r2_with(A1p) - r2x
        nullB[b] = r2_with(A1, Bp) - r2_A
    pA = (1 + np.sum(nullA >= dA)) / (nperm + 1)
    pB = (1 + np.sum(nullB >= dB)) / (nperm + 1)
    return dA - nullA.mean(), pA, dB - nullB.mean(), pB


def episodic(A1, A2, Xc, Xk, nperm=200, seed=1):
    """Trial-identifiability on the incremental A1 contribution; returns (acc-chance, p)."""
    rng = np.random.default_rng(seed)
    n = A1.shape[0]
    idx = rng.permutation(n)
    tr, te = idx[:n // 2], idx[n // 2:]
    st_te = strata(Xc, Xk)[te]
    Zx_tr, Zx_te = _design(Xc[tr], Xk[tr]), _design(Xc[te], Xk[te])
    Zf_tr, Zf_te = _design(Xc[tr], Xk[tr], A1[tr]), _design(Xc[te], Xk[te], A1[te])
    base = _fit(Zx_tr, A2[tr], Zx_te)
    full = _fit(Zf_tr, A2[tr], Zf_te)
    query, target = full - base, A2[te] - base

    def acc(tg):
        c = t = 0
        ch = 0.0
        for s in np.unique(st_te):
            m = np.where(st_te == s)[0]
            if m.size < 3:
                continue
            d = ((query[m][:, None, :] - tg[m][None, :, :]) ** 2).sum(-1)
            c += np.sum(d.argmin(1) == np.arange(m.size))
            t += m.size
            ch += 1.0
        return (c / t, ch / t) if t else (np.nan, np.nan)

    a, ch = acc(target)
    null = np.empty(nperm)
    for b in range(nperm):
        tp = target.copy()
        for s in np.unique(st_te):
            m = np.where(st_te == s)[0]
            tp[m] = target[m][rng.permutation(m.size)]
        null[b], _ = acc(tp)
    return a - ch, (1 + np.sum(null >= a)) / (nperm + 1)


def gen(arch, n=2000, seed=0):
    """Generates (A1, B, A2, Xc, Xk, ret) where ret = episode-specific return through B."""
    r = np.random.default_rng(seed)
    Xc = r.integers(0, 4, n)
    Xk = r.standard_normal(n)
    dr = lambda: r.standard_normal((4, DIM))[Xc] + np.outer(Xk, r.standard_normal(DIM))
    dA1, dB, dA2 = dr(), dr(), dr()
    e = lambda s=1.0: s * r.standard_normal((n, DIM))
    W = lambda s: r.standard_normal((DIM, DIM)) * s
    A1 = dA1 + e()

    if arch == "recurrent_return":                  # A1 -> B -> A2 (return to source)
        B = dB + A1 @ W(0.7) + e()
        A2 = dA2 + B @ W(0.7) + e()
        ret = True
    elif arch == "nonlinear_return":                # same, monotone nonlinearity
        B = dB + np.tanh(A1) @ W(0.8) + e()
        A2 = dA2 + np.tanh(B) @ W(0.8) + e()
        ret = True
    elif arch == "confounder_plus_weak_return":     # U everywhere + weak true return
        U = r.standard_normal((n, 3))
        A1 = dA1 + U @ (r.standard_normal((3, DIM)) * 0.9) + e()
        B = dB + U @ (r.standard_normal((3, DIM)) * 0.9) + A1 @ W(0.15) + e()
        A2 = dA2 + U @ (r.standard_normal((3, DIM)) * 0.9) + B @ W(0.15) + e()
        ret = True
    elif arch == "local_persistence":               # A1 -> A2 directly; B is a side branch
        B = dB + A1 @ W(0.7) + e()
        A2 = dA2 + A1 @ W(0.7) + e()
        ret = False
    elif arch == "feedforward_chain":               # A1 -> B -> C; A2 not on the path
        B = dB + A1 @ W(0.7) + e()
        A2 = dA2 + e()
        ret = False
    elif arch == "observed_common_cause":           # X drives all three
        B = dB + e()
        A2 = dA2 + e()
        ret = False
    elif arch == "parallel_pathways":               # different transforms of observed X
        B = dB + e()
        A2 = r.standard_normal((4, DIM))[Xc] * 1.3 + np.outer(Xk, r.standard_normal(DIM)) + e()
        ret = False
    elif arch == "latent_confounder":               # U drives all three, no path
        U = r.standard_normal((n, 3))
        L3 = lambda: r.standard_normal((3, DIM)) * 0.9
        A1 = dA1 + U @ L3() + e()
        B = dB + U @ L3() + e()
        A2 = dA2 + U @ L3() + e()
        ret = False
    elif arch == "latent_ff_state":                 # trial-specific hidden state
        m = r.standard_normal((n, 5))
        L5 = lambda: r.standard_normal((5, DIM)) * 0.9
        A1 = dA1 + m @ L5() + e()
        B = dB + m @ L5() + e()
        A2 = dA2 + m @ L5() + e()
        ret = False
    elif arch == "shared_low_rank":                 # coarse shared discrete factor
        cl = r.integers(0, 3, n)
        sh = (r.standard_normal((3, DIM)) * 1.6)[cl]
        A1 = dA1 + sh @ W(0.7) + e()
        B = dB + sh @ W(0.7) + e()
        A2 = dA2 + sh + e()
        ret = False
    elif arch == "coarse_dependence":               # A2 depends on a COARSE summary of A1 only
        cl = np.digitize(A1[:, 0], np.quantile(A1[:, 0], [0.25, 0.5, 0.75]))
        B = dB + e()
        A2 = dA2 + (r.standard_normal((4, DIM)) * 1.5)[cl] + e()
        ret = False
    else:
        raise ValueError(arch)

    return A1, B, A2, Xc, Xk, ret


ARCHS = ["recurrent_return", "nonlinear_return", "confounder_plus_weak_return", "local_persistence",
         "feedforward_chain", "observed_common_cause", "parallel_pathways", "latent_confounder",
         "latent_ff_state", "shared_low_rank", "coarse_dependence"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0., 0.)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))
