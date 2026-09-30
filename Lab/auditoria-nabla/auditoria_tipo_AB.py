"""
Auditoria de la prueba Tipo A / Tipo B de AFH_modelo_N_nodos.py

Pregunta: la prueba F(Tipo B)=1088,9 vs F(Tipo A)=1,2 citada en Sec. 3.6 del manuscrito
¿mide "cierre reflexivo" (el retorno es funcion del propio pasado del hub) o mide otra cosa?

Usa el simulador ORIGINAL sin modificar (simular_AFH_N, granger_en_tau_escalar,
construir_matrices, modo_dominante) de NeuroscienceOfConsciousness_revision/code/.
Lo unico que se agrega es un "espia" que captura el proceso exogeno E que simular_AFH_N
genera internamente en Tipo A, para poder testear el retorno que efectivamente llega al hub.

Parametros identicos al bloque [3] de main() original: beta=1.5, m=4, n=5, tau_f=20 ms,
tau_s=400 ms, 10 repeticiones (semillas de simulacion 0..9). Se repite con tres semillas
de matrices: 2 (la que usa el original) y 1, 3.

  T1  original: C(t-tau_s) -> U(t), AR p=5 (2,5 ms de pasado del hub)
  T2  definicion formal literal (modelo .md Sec. 4, manuscrito Sec. 3.6):
      r(t-tau_s) -> U(t), con r = el retorno que entra al hub (B@C en Tipo B, B@E en Tipo A)
  T3  T1 condicionando ademas en el pasado del hub (las m componentes de U) a la latencia
      del lazo (tau_s+tau_f-20 .. tau_s+tau_f+60 ms)
  T4  indice de reflexividad: R2 fuera de muestra del retorno r(t) predicho desde el
      pasado del hub U(t-tau_f-j), j=0..100 ms -- direccion "fuente -> retorno"

Correr:  python auditoria_tipo_AB.py   (~15 s por semilla; escribe resultados_tipo_AB.txt)
"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "NeuroscienceOfConsciousness_revision" / "code"))
import AFH_modelo_N_nodos as M  # noqa: E402

_capt = {}
_orig = M._ruido_coloreado_vec


def _espia(*a, **k):
    E = _orig(*a, **k)
    _capt["E"] = E
    return E


M._ruido_coloreado_vec = _espia

DT = 0.5
m, n, beta, tau_f, tau_s = 4, 5, 1.5, 20.0, 400.0
k_f, k_s = int(tau_f / DT), int(tau_s / DT)
DESC = int(1000 / DT)  # mismo descarte de transitorio que granger_en_tau_escalar
P_AR = 5


def f_test(y, X_r, x_add):
    """F de Granger anidado con un regresor agregado: y ~ X_r  vs  y ~ X_r + x_add."""
    def rss(X):
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.sum((y - X @ c) ** 2)
    rr, rn = rss(X_r), rss(np.column_stack([X_r, x_add]))
    df2 = len(y) - X_r.shape[1] - 1
    F = (rr - rn) / (rn / df2)
    return F, 1 - stats.f.cdf(F, 1, df2)


def lags(x, idx, ks):
    return np.column_stack([x[idx - k] for k in ks])


def r2_oos(y, X):
    h = len(y) // 2
    c, *_ = np.linalg.lstsq(X[:h], y[:h], rcond=None)
    res = y[h:] - X[h:] @ c
    return 1 - np.sum(res ** 2) / np.sum((y[h:] - y[:h].mean()) ** 2)


def auditar(semilla_matrices):
    A, B, Cc = M.construir_matrices(m, n, semilla=semilla_matrices)
    v = M.modo_dominante(B, Cc)
    ev = np.linalg.eigvals(B @ Cc)
    mu = ev[np.argmax(np.abs(ev))]
    val = {k: {"A": [], "B": []} for k in ["T1", "T2", "T3", "T4"]}
    sig = {k: {"A": [], "B": []} for k in ["T1", "T2", "T3"]}

    for modo in ["B", "A"]:
        for rep in range(10):
            _capt.clear()
            U, C, _, _ = M.simular_AFH_N(m=m, n=n, tau_f=tau_f, tau_s=tau_s, beta=beta,
                                         modo=modo, A=A, B=B, Ccoup=Cc, semilla=rep)
            x = U @ v
            cbar = C.mean(axis=1)
            r = ((C if modo == "B" else _capt["E"]) @ B.T) @ v  # retorno real, modo dominante

            F1, p1, _ = M.granger_en_tau_escalar(x, cbar, dt=DT, tau=tau_s)

            L_loop = list(range(k_s + k_f - 40, k_s + k_f + 121, 4))
            idx = np.arange(DESC + max(L_loop) + 1, len(x))
            y = x[idx]
            X_r = np.column_stack([np.ones(len(idx)), lags(x, idx, range(1, P_AR + 1))])
            F2, p2 = f_test(y, X_r, r[idx - k_s])
            X_r3 = np.column_stack([X_r] + [lags(U[:, j], idx, L_loop) for j in range(m)])
            F3, p3 = f_test(y, X_r3, cbar[idx - k_s])

            idx4 = np.arange(DESC + k_f + 200, len(r))
            X4 = np.column_stack([np.ones(len(idx4))] +
                                 [lags(U[:, j], idx4, range(k_f, k_f + 200, 4)) for j in range(m)])
            R2 = r2_oos(r[idx4], X4)

            for k, F, p in [("T1", F1, p1), ("T2", F2, p2), ("T3", F3, p3)]:
                val[k][modo].append(F)
                sig[k][modo].append(p < 0.05)
            val["T4"][modo].append(R2)
    return mu, val, sig


DESC_T = {
    "T1": "original: C(t-400ms) -> U(t), AR p=5          (F medio)",
    "T2": "formal:   retorno real r(t-400ms) -> U(t)     (F medio)",
    "T3": "T1 + pasado de U a la latencia del lazo       (F medio)",
    "T4": "reflexividad: R2 de r(t) desde pasado de U    (R2 oos)",
}

if __name__ == "__main__":
    lineas = []
    for sem in [2, 1, 3]:
        mu, val, sig = auditar(sem)
        tipo = "real negativo" if mu.real < 0 and abs(mu.imag) < 1e-9 else (
            "real positivo" if abs(mu.imag) < 1e-9 else "complejo")
        lineas.append(f"semilla de matrices = {sem}   autovalor dominante del lazo L=B@C: "
                      f"{mu.real:+.3f}{mu.imag:+.3f}i ({tipo})")
        lineas.append(f"{'':60s} {'Tipo B':>10s} {'Tipo A':>10s}   p<0,05 (B / A)")
        for k in ["T1", "T2", "T3", "T4"]:
            b, a = np.mean(val[k]["B"]), np.mean(val[k]["A"])
            extra = f"   {np.mean(sig[k]['B']):.1f} / {np.mean(sig[k]['A']):.1f}" if k in sig else ""
            lineas.append(f"{k} {DESC_T[k]:57s} {b:10.2f} {a:10.2f}{extra}")
        lineas.append("")
    salida = "\n".join(lineas)
    print(salida)
    (HERE / "resultados_tipo_AB.txt").write_text(salida + "\n", encoding="utf-8")
