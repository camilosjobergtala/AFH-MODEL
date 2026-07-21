# -*- coding: utf-8 -*-
"""
Validación por simulación del diseño recurrencia-vs-causa-común (§6 de
"Poner a prueba afirmaciones constitutivas sobre la presencia fenomenológica:
falsación asimétrica y control de causas comunes", C. A. Sjöberg Tala).

CONTEXTO

§6 del paper propone un pipeline observacional de dos tests para distinguir
"la respuesta tardía L deriva recurrentemente de la respuesta temprana E del
mismo ensayo" de las arquitecturas alternativas que producen la misma
asociación E-L sin recurrencia genuina (causa común observada X, rutas
paralelas, o cualquier causa común NO observada U). Este script es el código
citado en §6.4: genera datos con VERDAD-BASE conocida bajo siete
arquitecturas y evalúa, en cuarenta réplicas por arquitectura, la TASA DE
POSITIVOS (fracción de datasets en que cada test rechaza el nulo p<0.05), el
tamaño de efecto medio, y la verdad-base (¿hay linaje E->L?). La salida de
este script reproduce, número a número, la Tabla 2 del paper.

QUÉ EVALÚA CADA TEST

  Test 1 (primario, §6.2): dependencia condicionada -> ganancia predictiva
  fuera de muestra de añadir E a un modelo que ya contiene X (identidad de
  estímulo + arousal), con surrogates que permutan E dentro de estratos de X.

  Test 2 (corroborativo, §6.2): identificabilidad por ensayo -> ¿el patrón
  temprano permite identificar el tardío correspondiente entre ensayos
  emparejados por estrato? Se reportan ambas versiones: NAIVE (sobre la
  predicción cruda de L) y RESIDUALIZADA (sobre el aporte incremental de E,
  tras descontar X de predicción y objetivo) -- el paper exige usar la
  residualizada (§6.2), y §6.4 muestra por qué: la naive da 60-65% de falsos
  positivos incluso cuando el confusor es puramente el X observado.

LAS SIETE ARQUITECTURAS (verdad-base)

  recurrencia            E->L genuino (linaje real).
  causa_comun_obs        S observado (X) causa E y L por separado.
  rutas_paralelas        Transformación de X distinta a la anterior; sin E->L.
  confusor_latente        U NO observado causa E y L; sin E->L.
  persistencia_local     Estado z compartido entre E y L (misma "población").
  feedforward_estado     Estado latente m compartido, sin bucle recurrente.
  compartido_bajo_rango  Factor discreto grueso (cluster) compartido.

Las primeras dos (recurrencia, causa_comun_obs/rutas_paralelas) validan que
el diseño hace lo que promete cuando el confusor SÍ está en X. Las últimas
cuatro son el resultado central de §6.4: el diseño no tiene defensa alguna
contra un confusor que no está en X, sin importar su forma -- ambos tests se
disparan al 100% pese a que ninguna de esas cuatro arquitecturas tiene
linaje E->L real. Ver `barrido_confusor_latente.py` para la extensión que
muestra que esa falla es GRADUAL (no de umbral) y ASIMÉTRICA entre Test 1 y
Test 2 residualizado.

CORRER: python simulacion_recurrencia_vs_causa_comun.py
TIEMPO: ~4-5 minutos (7 arquitecturas x 40 réplicas x 3 tests, cada uno con
        150 permutaciones).
"""
import numpy as np

rng_global = np.random.default_rng(0)

# ---------- utilidades ----------
def ridge_fit_predict(Ztr, Ytr, Zte, lam=1.0):
    """Ridge multivariada por forma cerrada. Devuelve predicción en test."""
    p = Ztr.shape[1]
    A = Ztr.T @ Ztr + lam * np.eye(p)
    B = np.linalg.solve(A, Ztr.T @ Ytr)
    return Zte @ B

def r2_oos(Ytr, Yte, Yhat):
    """R^2 fuera de muestra agregando todas las dimensiones (baseline = media de train)."""
    mu = Ytr.mean(axis=0, keepdims=True)
    ss_res = ((Yte - Yhat) ** 2).sum()
    ss_tot = ((Yte - mu) ** 2).sum()
    return 1.0 - ss_res / ss_tot

def design(Xcat, Xcont, extra=None):
    """Matriz de diseño: intercepto + one-hot(estimulo) + arousal (+ extra)."""
    n = Xcat.shape[0]
    onehot = np.eye(int(Xcat.max()) + 1)[Xcat]
    cols = [np.ones((n, 1)), onehot, Xcont.reshape(-1, 1)]
    if extra is not None:
        cols.append(extra)
    return np.hstack(cols)

def strata_labels(Xcat, Xcont, n_arousal_bins=4):
    """Estratos = estimulo x cuantil de arousal (covariables observadas discretizadas)."""
    qs = np.quantile(Xcont, np.linspace(0, 1, n_arousal_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    abin = np.digitize(Xcont, qs[1:-1])
    return Xcat * n_arousal_bins + abin

# ---------- Test 1: dependencia condicionada ----------
def test1(E, L, Xcat, Xcont, strata, lam=1.0, B=150, seed=0):
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2 :]
    Zbase = design(Xcat, Xcont)
    Zfull = design(Xcat, Xcont, extra=E)
    r2_base = r2_oos(L[tr], L[te], ridge_fit_predict(Zbase[tr], L[tr], Zbase[te], lam))
    r2_full = r2_oos(L[tr], L[te], ridge_fit_predict(Zfull[tr], L[tr], Zfull[te], lam))
    dR2 = r2_full - r2_base
    # nulo: permutar E DENTRO de estrato (preserva la relacion con el driver comun observado)
    null = np.empty(B)
    for b in range(B):
        Ep = E.copy()
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            Ep[m] = E[rng.permutation(m)]
        Zf = design(Xcat, Xcont, extra=Ep)
        r2f = r2_oos(L[tr], L[te], ridge_fit_predict(Zf[tr], L[tr], Zf[te], lam))
        null[b] = r2f - r2_base
    p = (1 + np.sum(null >= dR2)) / (B + 1)
    return dR2, p

# ---------- Test 2: identificabilidad por ensayo ----------
def _match_acc(Lhat_te, L_te, strata_te):
    """Top-1 accuracy emparejando cada L_hat con el L mas cercano dentro de su estrato."""
    correct = 0; total = 0; chance = 0.0
    for s in np.unique(strata_te):
        m = np.where(strata_te == s)[0]
        if m.size < 3:
            continue
        d = ((Lhat_te[m][:, None, :] - L_te[m][None, :, :]) ** 2).sum(-1)
        correct += np.sum(d.argmin(axis=1) == np.arange(m.size))
        total += m.size
        chance += 1.0  # 1/size por ensayo * size ensayos = 1 por estrato
    return (correct / total, chance / total) if total else (np.nan, np.nan)

def test2(E, L, Xcat, Xcont, strata, lam=1.0, B=150, seed=1, residualize=True):
    """Si residualize=True, empareja la contribucion INCREMENTAL de E (aislando X)."""
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2 :]
    st_te = strata[te]
    if residualize:
        Zb_tr = design(Xcat[tr], Xcont[tr]); Zb_te = design(Xcat[te], Xcont[te])
        Zf_tr = design(Xcat[tr], Xcont[tr], extra=E[tr]); Zf_te = design(Xcat[te], Xcont[te], extra=E[te])
        base = ridge_fit_predict(Zb_tr, L[tr], Zb_te, lam)
        full = ridge_fit_predict(Zf_tr, L[tr], Zf_te, lam)
        target = L[te] - base          # L sin la parte de X observado
        query = full - base            # aporte incremental de E a la prediccion
    else:
        Zf_tr = design(Xcat[tr], Xcont[tr], extra=E[tr]); Zf_te = design(Xcat[te], Xcont[te], extra=E[te])
        query = ridge_fit_predict(Zf_tr, L[tr], Zf_te, lam)
        target = L[te]
    acc, chance = _match_acc(query, target, st_te)
    null = np.empty(B)
    for b in range(B):
        tperm = target.copy()
        for s in np.unique(st_te):
            m = np.where(st_te == s)[0]
            tperm[m] = target[m][rng.permutation(m.size)]
        null[b], _ = _match_acc(query, tperm, st_te)
    p = (1 + np.sum(null >= acc)) / (B + 1)
    return acc - chance, p

# ---------- generadores (verdad-base) ----------
def base_drivers(n, dE, dL, rng):
    Xcat = rng.integers(0, 4, n)
    Xcont = rng.standard_normal(n)  # arousal
    MsE = rng.standard_normal((4, dE)); maE = rng.standard_normal(dE)
    MsL = rng.standard_normal((4, dL)); maL = rng.standard_normal(dL)
    drE = MsE[Xcat] + np.outer(Xcont, maE)
    drL = MsL[Xcat] + np.outer(Xcont, maL)
    return Xcat, Xcont, drE, drL

def gen(arch, n=2000, dE=6, dL=6, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    Xcat, Xcont, drE, drL = base_drivers(n, dE, dL, rng)
    eE = noise * rng.standard_normal((n, dE)); eL = noise * rng.standard_normal((n, dL))
    if arch == "recurrencia":                       # E -> L  (linaje verdadero)
        E = drE + eE
        A = rng.standard_normal((dE, dL)) * 0.6
        L = drL + E @ A + eL
        truth = True
    elif arch == "causa_comun_obs":                 # S -> E, S -> L (S observado); sin E->L
        E = drE + eE; L = drL + eL; truth = False
    elif arch == "rutas_paralelas":                 # transform. distinta de S observado; sin E->L
        MsL2 = rng.standard_normal((4, dL)) * 1.3
        L = MsL2[Xcat] + np.outer(Xcont, rng.standard_normal(dL)) + eL
        E = drE + eE; truth = False
    elif arch == "confusor_latente":                # U -> E, U -> L (U NO observado)
        dU = 3; U = rng.standard_normal((n, dU))
        Bue = rng.standard_normal((dU, dE)) * 0.9; Bul = rng.standard_normal((dU, dL)) * 0.9
        E = drE + U @ Bue + eE; L = drL + U @ Bul + eL; truth = False
    elif arch == "persistencia_local":              # misma poblacion temprano/tarde (estado z compartido)
        z = drL + 0.9 * rng.standard_normal((n, dL))
        Pz = rng.standard_normal((dL, dE)) * 0.7
        E = z @ Pz + eE; L = 0.8 * z + eL; truth = False  # no hay retorno por otra poblacion
    elif arch == "feedforward_estado":              # estado LATENTE m compartido, sin bucle recurrente
        k = 5; m = rng.standard_normal((n, k))      # estado oculto especifico del ensayo (no observado)
        Wl = rng.standard_normal((k, dL)) * 0.9
        We = rng.standard_normal((k, dE)) * 0.9
        E = drE + m @ We + eE; L = drL + m @ Wl + eL; truth = False
    elif arch == "compartido_bajo_rango":           # factor discreto grueso compartido (para Test1 vs Test2)
        clusters = rng.integers(0, 3, n)            # 3 clusters, no alineados con estimulo
        Cd = rng.standard_normal((3, dL)) * 1.6
        shared = Cd[clusters]
        PdE = rng.standard_normal((dL, dE)) * 0.7
        E = drE + shared @ PdE + eE
        L = drL + shared + eL; truth = False
    else:
        raise ValueError(arch)
    return E, L, Xcat, Xcont, truth

# ---------- correr ----------
ARCHS = ["recurrencia", "causa_comun_obs", "rutas_paralelas", "confusor_latente",
         "persistencia_local", "feedforward_estado", "compartido_bajo_rango"]
R = 40  # datasets por arquitectura
hdr = (f"{'arquitectura':22s} {'linaje':6s} | {'T1 ΔR²':>7s} {'T1+':>5s} | "
       f"{'T2n Δacc':>8s} {'T2n+':>5s} | {'T2r Δacc':>8s} {'T2r+':>5s}")
print(hdr); print("-" * len(hdr))
summary = {}
for arch in ARCHS:
    dR2s, p1s, d2n, p2n, d2r, p2r, truth = [], [], [], [], [], [], None
    for r in range(R):
        E, L, Xc, Xk, truth = gen(arch, seed=1000 + r)
        st = strata_labels(Xc, Xk)
        a, pa = test1(E, L, Xc, Xk, st, seed=7000 + r)
        bn, pbn = test2(E, L, Xc, Xk, st, seed=9000 + r, residualize=False)
        br, pbr = test2(E, L, Xc, Xk, st, seed=9500 + r, residualize=True)
        dR2s.append(a); p1s.append(pa); d2n.append(bn); p2n.append(pbn); d2r.append(br); p2r.append(pbr)
    f = lambda ps: 100 * np.mean(np.array(ps) < 0.05)
    summary[arch] = (truth, np.mean(dR2s), f(p1s), np.mean(d2n), f(p2n), np.mean(d2r), f(p2r))
    print(f"{arch:22s} {str(truth):6s} | {np.mean(dR2s):7.3f} {f(p1s):4.0f}% | "
          f"{np.mean(d2n):8.3f} {f(p2n):4.0f}% | {np.mean(d2r):8.3f} {f(p2r):4.0f}%")

print("\nT1 = dependencia condicionada (ganancia predictiva OOS de E sobre X).")
print("T2n = identificabilidad NAIVE; T2r = identificabilidad RESIDUALIZADA (aporte incremental de E).")
print("'+' = tasa de positivos (p<0.05). linaje=True: existe recurrencia E->L genuina.")
print("Correcto = positivo si linaje=True, negativo si linaje=False. Positivo con False = FALSO POSITIVO.")
