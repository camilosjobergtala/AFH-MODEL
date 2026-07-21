# -*- coding: utf-8 -*-
"""
Barrido de intensidad del confusor latente -- extensión de §6.4 del paper
"Poner a prueba afirmaciones constitutivas sobre la presencia fenomenológica:
falsación asimétrica y control de causas comunes" (C. A. Sjöberg Tala).

CONTEXTO

La Tabla 2 del paper (reproducida por `simulacion_recurrencia_vs_causa_comun.py`)
reporta, para la arquitectura `confusor_latente`, 100% de falsos positivos en
Test 1 y en ambas versiones de Test 2 -- pero con UNA sola intensidad de
acoplamiento U->E, U->L (kappa=0.9) y un solo N (2000). Ese único punto no
dice si la falla es de umbral (cualquier confusor no medido, por débil que
sea, dispara 100%) o proporcional (el FPR crece gradualmente con la fuerza
del confusor, como ya se había encontrado para un confusor distinto --la
calidad del proxy de identidad de estímulo-- en el script hermano de este
mismo autor, `NeuroscienceOfConsciousness_revision/code/test_rastreo_de_fuente.py`,
Bloque 7).

Este script barre kappa_U de 0 (sin confusor, calibra el nivel nominal) a
0.9 (el valor de la Tabla 2), manteniendo N y todo lo demás fijo, para las
tres pruebas: Test 1, Test 2 naive y Test 2 residualizado.

RESULTADO (40 réplicas por punto, ejecutado 2026-07-21)

    kappa_U |  T1 dR2   T1+ | T2n dAcc  T2n+ | T2r dAcc  T2r+
       0.00 |  -0.002    5% |    0.009   62% |    0.000    5%
       0.15 |   0.001   82% |    0.009   75% |    0.001    5%
       0.30 |   0.028  100% |    0.015   90% |    0.004   30%
       0.45 |   0.087  100% |    0.025  100% |    0.014   88%
       0.60 |   0.168  100% |    0.047  100% |    0.031  100%
       0.75 |   0.257  100% |    0.073  100% |    0.057  100%
       0.90 |   0.344  100% |    0.107  100% |    0.087  100%

Tres lecturas:

  1. Calibración correcta en kappa=0: Test 1 y Test 2 residualizado dan ~5%
     (el nivel nominal esperado bajo H0 verdadera), confirmando que la falla
     de la Tabla 2 no es un artefacto de implementación -- el test funciona
     bien cuando de verdad no hay confusor.

  2. La falla es GRADUAL, no de umbral -- igual que el hallazgo de Bloque 7
     de `test_rastreo_de_fuente.py` para el proxy de estímulo: la fragilidad
     es proporcional a la fuerza del confusor, no un interruptor todo-o-nada.

  3. La falla es ASIMÉTRICA entre los dos tests: el Test 1 supera 50% de
     positivos ya en kappa~0.15, mientras el Test 2 residualizado se mantiene
     calibrado hasta kappa~0.30-0.45 y recién después se dispara. OJO con la
     lectura: NO es que "el test primario sea el más frágil". Bajo confusor
     latente ninguno de los dos falla como test -- E y L son GENUINAMENTE
     dependientes dado el X observado (serían independientes recién dado X y
     U), así que el Test 1 detecta correctamente esa dependencia; lo "falso"
     está solo en la sobre-lectura como linaje, no en el test (el punto de
     §6.4-6.5). La asimetría es la curva sensibilidad/especificidad de
     manual: el Test 1 es el detector MÁS SENSIBLE de dependencia condicional
     (dispara con acoplamiento más débil, venga de linaje o de confusor); el
     Test 2 residualizado es MÁS ESPECÍFICO (exige señal más fuerte para
     sostener identificabilidad por ensayo). Razón estructural: el Test 1
     acumula covarianza residual sobre TODA la muestra y las 6 dimensiones de
     L, así que cualquier fuga no nula de un confusor no medido se vuelve
     detectable con N suficiente; el Test 2 residualizado exige que el aporte
     incremental de E identifique, ENSAYO POR ENSAYO, cuál L residual le
     corresponde, y una fuga débil se pierde en el ruido idiosincrático antes
     de discriminar nada.

  Implicación (decisión del autor, ya resuelta): el Test 1 se MANTIENE como
  primario -- es el test de la condición necesaria del §4 (E ⊥̸ L | X), y el
  Test 2 mide algo más fuerte que esa condición necesaria, así que invertir
  rompería la lógica falsacionista. Lo que sí cambia es el caveat: "primario"
  no implica que un positivo del Test 1 sea más diagnóstico de linaje que uno
  del Test 2 residualizado (ambos son igualmente compatibles con causa común
  latente), ni que sea el más robusto. Ver `../notes/adenda_limitacion4.md`
  para el texto propuesto para §8 y la aclaración para §6.2.

CORRER: python barrido_confusor_latente.py
TIEMPO: ~5 minutos (7 valores de kappa x 40 réplicas x 3 tests).
"""
import numpy as np

# ---------- utilidades (idénticas a simulacion_recurrencia_vs_causa_comun.py) ----------
def ridge_fit_predict(Ztr, Ytr, Zte, lam=1.0):
    p = Ztr.shape[1]
    A = Ztr.T @ Ztr + lam * np.eye(p)
    B = np.linalg.solve(A, Ztr.T @ Ytr)
    return Zte @ B

def r2_oos(Ytr, Yte, Yhat):
    mu = Ytr.mean(axis=0, keepdims=True)
    ss_res = ((Yte - Yhat) ** 2).sum()
    ss_tot = ((Yte - mu) ** 2).sum()
    return 1.0 - ss_res / ss_tot

def design(Xcat, Xcont, extra=None):
    n = Xcat.shape[0]
    onehot = np.eye(int(Xcat.max()) + 1)[Xcat]
    cols = [np.ones((n, 1)), onehot, Xcont.reshape(-1, 1)]
    if extra is not None:
        cols.append(extra)
    return np.hstack(cols)

def strata_labels(Xcat, Xcont, n_arousal_bins=4):
    qs = np.quantile(Xcont, np.linspace(0, 1, n_arousal_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    abin = np.digitize(Xcont, qs[1:-1])
    return Xcat * n_arousal_bins + abin

def test1(E, L, Xcat, Xcont, strata, lam=1.0, B=150, seed=0):
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2 :]
    Zbase = design(Xcat, Xcont)
    Zfull = design(Xcat, Xcont, extra=E)
    r2_base = r2_oos(L[tr], L[te], ridge_fit_predict(Zbase[tr], L[tr], Zbase[te], lam))
    r2_full = r2_oos(L[tr], L[te], ridge_fit_predict(Zfull[tr], L[tr], Zfull[te], lam))
    dR2 = r2_full - r2_base
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

def _match_acc(Lhat_te, L_te, strata_te):
    correct = 0; total = 0; chance = 0.0
    for s in np.unique(strata_te):
        m = np.where(strata_te == s)[0]
        if m.size < 3:
            continue
        d = ((Lhat_te[m][:, None, :] - L_te[m][None, :, :]) ** 2).sum(-1)
        correct += np.sum(d.argmin(axis=1) == np.arange(m.size))
        total += m.size
        chance += 1.0
    return (correct / total, chance / total) if total else (np.nan, np.nan)

def test2(E, L, Xcat, Xcont, strata, lam=1.0, B=150, seed=1, residualize=True):
    rng = np.random.default_rng(seed)
    n = E.shape[0]
    idx = rng.permutation(n); tr, te = idx[: n // 2], idx[n // 2 :]
    st_te = strata[te]
    if residualize:
        Zb_tr = design(Xcat[tr], Xcont[tr]); Zb_te = design(Xcat[te], Xcont[te])
        Zf_tr = design(Xcat[tr], Xcont[tr], extra=E[tr]); Zf_te = design(Xcat[te], Xcont[te], extra=E[te])
        base = ridge_fit_predict(Zb_tr, L[tr], Zb_te, lam)
        full = ridge_fit_predict(Zf_tr, L[tr], Zf_te, lam)
        target = L[te] - base
        query = full - base
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

def base_drivers(n, dE, dL, rng):
    Xcat = rng.integers(0, 4, n)
    Xcont = rng.standard_normal(n)
    MsE = rng.standard_normal((4, dE)); maE = rng.standard_normal(dE)
    MsL = rng.standard_normal((4, dL)); maL = rng.standard_normal(dL)
    drE = MsE[Xcat] + np.outer(Xcont, maE)
    drL = MsL[Xcat] + np.outer(Xcont, maL)
    return Xcat, Xcont, drE, drL

def gen_confusor_latente(n=2000, dE=6, dL=6, noise=1.0, kappa=0.9, seed=0):
    """Igual que 'confusor_latente' de simulacion_recurrencia_vs_causa_comun.py,
    pero con la fuerza de acoplamiento U->E, U->L parametrizada por kappa
    (0.9 = valor original de la Tabla 2). kappa=0 => sin confusor (E _|_ L | X
    genuino, calibra el nivel nominal del test)."""
    rng = np.random.default_rng(seed)
    Xcat, Xcont, drE, drL = base_drivers(n, dE, dL, rng)
    eE = noise * rng.standard_normal((n, dE)); eL = noise * rng.standard_normal((n, dL))
    dU = 3; U = rng.standard_normal((n, dU))
    Bue = rng.standard_normal((dU, dE)) * kappa
    Bul = rng.standard_normal((dU, dL)) * kappa
    E = drE + U @ Bue + eE
    L = drL + U @ Bul + eL
    return E, L, Xcat, Xcont

# ---------- barrido ----------
KAPPAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
R = 40

hdr = (f"{'kappa_U':>8s} | {'T1 dR2':>7s} {'T1+':>5s} | "
       f"{'T2n dAcc':>8s} {'T2n+':>5s} | {'T2r dAcc':>8s} {'T2r+':>5s}")
print(hdr); print("-" * len(hdr))
rows = []
for kappa in KAPPAS:
    dR2s, p1s, d2n, p2n, d2r, p2r = [], [], [], [], [], []
    for r in range(R):
        E, L, Xc, Xk = gen_confusor_latente(kappa=kappa, seed=2000 + r)
        st = strata_labels(Xc, Xk)
        a, pa = test1(E, L, Xc, Xk, st, seed=7000 + r)
        bn, pbn = test2(E, L, Xc, Xk, st, seed=9000 + r, residualize=False)
        br, pbr = test2(E, L, Xc, Xk, st, seed=9500 + r, residualize=True)
        dR2s.append(a); p1s.append(pa); d2n.append(bn); p2n.append(pbn); d2r.append(br); p2r.append(pbr)
    f = lambda ps: 100 * np.mean(np.array(ps) < 0.05)
    row = (kappa, np.mean(dR2s), f(p1s), np.mean(d2n), f(p2n), np.mean(d2r), f(p2r))
    rows.append(row)
    print(f"{kappa:8.2f} | {np.mean(dR2s):7.3f} {f(p1s):4.0f}% | "
          f"{np.mean(d2n):8.3f} {f(p2n):4.0f}% | {np.mean(d2r):8.3f} {f(p2r):4.0f}%")

print("\nkappa_U = fuerza del acoplamiento U->E, U->L (kappa=0: sin confusor, calibra alpha nominal).")
print("'+' = tasa de positivos (p<0.05) sobre 40 replicas. Todas las filas tienen linaje=False (E no causa L).")
print("Cualquier '+' > ~5% por encima del nivel nominal es un falso positivo.")

# ---------- interpretacion ----------
kappas = [r[0] for r in rows]
t1 = [r[2] for r in rows]
t2r = [r[6] for r in rows]
idx_t1 = next((i for i, v in enumerate(t1) if v > 50), None)
idx_t2r = next((i for i, v in enumerate(t2r) if v > 50), None)
print("\n" + "=" * 72)
print("  LECTURA")
print("=" * 72)
if idx_t1 is not None and idx_t2r is not None:
    print(f"  Test 1 supera 50% de positivos ya en kappa_U={kappas[idx_t1]:.2f};")
    print(f"  Test 2 residualizado recien lo hace en kappa_U={kappas[idx_t2r]:.2f}.")
print("  La falla es GRADUAL, no binaria (igual que en Bloque 7 de test_rastreo_de_fuente.py")
print("  para el proxy de estimulo), y ASIMETRICA entre los dos tests. OJO con la lectura:")
print("  NO es que 'el test primario sea el mas fragil'. Bajo confusor latente ninguno falla")
print("  como test -- E y L son GENUINAMENTE dependientes dado el X observado, y el positivo")
print("  es enganoso solo como evidencia de LINAJE (el punto de Sec. 6.4-6.5), no como test")
print("  de dependencia condicional. La asimetria es sensibilidad/especificidad: el Test 1 es")
print("  el detector MAS SENSIBLE de dependencia condicional (dispara con acoplamiento mas")
print("  debil, venga de linaje o de confusor); el Test 2 residualizado es MAS ESPECIFICO")
print("  (exige senal mas fuerte para sostener identificabilidad por ensayo). Razon")
print("  estructural: el Test 1 acumula covarianza residual sobre TODA la muestra; el Test 2")
print("  residualizado exige identificar ENSAYO POR ENSAYO, y una fuga debil se pierde en el")
print("  ruido idiosincratico antes de discriminar nada.")
print("  Decision del autor: el Test 1 se MANTIENE primario (es el test de la condicion")
print("  necesaria del Sec. 4); lo que se afina es el caveat -- 'primario' no implica que su")
print("  positivo sea mas diagnostico de linaje que el del Test 2r, ni que sea mas robusto.")
print("  Ver ../notes/adenda_limitacion4.md para el texto propuesto para Sec. 8 y Sec. 6.2.")
