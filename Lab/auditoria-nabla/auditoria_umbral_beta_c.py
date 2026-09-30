"""
Auditoria del umbral H* = beta_c = 1 (manuscrito Sec. 3.6; modelo .md Sec. 5 y 8).

  [1] Origen del error de 1,5-5,5 % reportado: si fuera discretizacion (Euler), deberia
      bajar al reducir dt; si es sesgo del estimador por tiempo finito (el crecimiento cerca
      del umbral es lento: lambda ~ (beta-1)/(tau_u+tau_c+beta*D), D = tau_f+tau_s), deberia
      bajar al alargar T. Usa estimar_beta_c_directo() original, sin modificar.
  [2] La derivacion beta_c = 1 (bifurcacion estatica, pitchfork) supone que el autovalor
      dominante del lazo L = B@C es real y positivo. Se cuenta con que frecuencia lo es
      bajo construir_matrices() original (m=4, n=5, semillas 0..999).
  [3] Umbral exacto cuando el autovalor dominante es mu = e^{i theta} (|mu| = 1): raiz de
      (1+i w tau_u)(1+i w tau_c) = beta mu e^{-i w D} sobre el eje imaginario.

Correr:  python auditoria_umbral_beta_c.py   (~1,5 min con 4 nucleos; escribe resultados_umbral_beta_c.txt)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "NeuroscienceOfConsciousness_revision" / "code"))
import AFH_modelo_N_nodos as M  # noqa: E402

TAU_U = TAU_C = 10.0


def _estimar(args):
    tf, ts, T, dt = args
    r = M.estimar_beta_c_directo(m=4, n=5, tau_f=tf, tau_s=ts, dt=dt, T=T, semilla=1)
    return args, r["beta_c_estimado"], r["error_relativo"] * 100


def beta_c_exacto(theta, D):
    """Menor beta en que un modo con autovalor e^{i theta} cruza el eje imaginario."""
    mejor = np.inf
    ws = np.linspace(1e-9, 2.0, 200001)
    for k in range(-3, 4):
        g = lambda w: np.arctan(w * TAU_U) + np.arctan(w * TAU_C) + w * D - (theta + 2 * np.pi * k)
        v = g(ws)
        for i in np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
            w = brentq(g, ws[i], ws[i + 1])
            mejor = min(mejor, np.sqrt((1 + (w * TAU_U) ** 2) * (1 + (w * TAU_C) ** 2)))
    return mejor


if __name__ == "__main__":
    out = ["[1] Error de beta_c_hat (semilla=1, autovalor dominante +1, umbral verdadero = 1 exacto)"]
    casos = [(tf, ts, T, dt) for (tf, ts) in [(8, 200), (100, 600)]
             for (T, dt) in [(20000, 0.5), (20000, 0.25), (80000, 0.5)]]
    with Pool(4) as p:
        for (tf, ts, T, dt), b, e in p.map(_estimar, casos):
            out.append(f"    tau_f={tf:>3} tau_s={ts:>3}  T={T:>6} ms  dt={dt:<4}  "
                       f"beta_c_hat={b:.4f}  error={e:5.2f} %")

    cuenta = {"real positivo": 0, "real negativo": 0, "complejo": 0}
    for s in range(1000):
        _, B, C = M.construir_matrices(4, 5, semilla=s)
        ev = eig(B @ C, right=False)
        d = ev[np.argmax(np.abs(ev))]
        k = "complejo" if abs(d.imag) > 1e-9 else ("real positivo" if d.real > 0 else "real negativo")
        cuenta[k] += 1
    out.append("")
    out.append(f"[2] Autovalor dominante de L=B@C en 1000 semillas (m=4, n=5): {cuenta}")

    out.append("")
    out.append(f"[3] Umbral exacto por tipo de autovalor dominante (tau_u=tau_c={TAU_U:.0f} ms)")
    for D in [208, 430, 700]:
        out.append(f"    D=tau_f+tau_s={D:>3} ms   mu=+1: beta_c=1 (estatico)   "
                   f"mu=-1: beta_c={beta_c_exacto(np.pi, D):.4f} (oscilatorio)   "
                   f"mu=e^(i pi/2): beta_c={beta_c_exacto(np.pi / 2, D):.4f} (oscilatorio)")

    salida = "\n".join(out)
    print(salida)
    (HERE / "resultados_umbral_beta_c.txt").write_text(salida + "\n", encoding="utf-8")
