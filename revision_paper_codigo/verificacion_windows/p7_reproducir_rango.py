"""Punto 7: intento de reproducir el "4.4–5.0%" del Study 2b pasando por el motor completo.

Paper (Results, Study 2b): "Verified end to end through the engine, the realized
honest-scenario rate is 4.4–5.0% across the sampled designs".

Condiciones (fijadas de antemano):
  - Motor completo: cada estudio simulado pasa por stds_wrapper.run_stds(), que llama a
    StatisticalTestDataSnooping.perform_snooping_test() del core. No hay ruta vectorizada.
  - N = 60.000 estudios por celda (como la derivación de la Tabla 5). Semilla 20250805.
  - Escenario honesto: folds y holdout ~ Normal(0.70, 0.05), delta = 0.
    Muestras con la misma forma que honest_max_z() de 04_stds_screening_cutoffs.py:
    folds (N, M, K) y luego holdout (N, M), por celda.
  - Marcado: risk_level != "LOW", es decir, max_z > corte del percentil 95.
  - Dos conjuntos, cada uno con su propio generador iniciado con la semilla:
      (a) las 12 celdas de la Tabla 5: K en {3,5,10,20} x |M| en {1,3,5}
      (b) los 4 diseños del paso [2] de 04: (3,1), (5,3), (10,1), (20,5)

Controles positivos (se ejecutan antes; si alguno falla, el script se detiene):
  C1. max_z del motor = max_z calculado aparte con numpy, caso por caso (2000 casos).
      Comprueba que el motor recibe los datos que creemos.
  C2. El criterio risk_level != "LOW" coincide con max_z > corte p95 de EMPIRICAL_SCREEN
      en esos mismos 2000 casos. Comprueba que el criterio es el que dice el paper.
  C3. Escenario con holdout inflado (+4 SD de fold): el motor debe marcar > 90% de los
      casos. Comprueba que el marcado realmente se dispara cuando debe.

Uso (desde la raíz del clon en 93339b4):
    python p7_reproducir_rango.py            # N = 60.000 por celda (~10–20 min)
    python p7_reproducir_rango.py 2000       # prueba rápida con N menor (no comparable)
"""
import sys, hashlib
sys.dont_write_bytecode = True          # no deja __pycache__ en el clon
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path

ROOT = Path.cwd()
ENGINE = ROOT / "ScientificReports_revision" / "engine"
CORE = ENGINE / "eclipse_core.py"
ESPERADO = "9d4320d225375fb5ed326b1cc282779125595ef500ff81bda81d7dd528b059b3"

if not CORE.exists():
    sys.exit(f"No encuentro {CORE}. Ejecuta el script desde la raíz del clon (la carpeta que contiene ScientificReports_revision).")
_raw = CORE.read_bytes()
_h_raw = hashlib.sha256(_raw).hexdigest()
_h_lf = hashlib.sha256(_raw.replace(b"\r\n", b"\n")).hexdigest()
print("SHA-256 eclipse_core.py (bytes tal cual)        :", _h_raw)
print("SHA-256 eclipse_core.py (normalizado a LF)      :", _h_lf)
print("SHA-256 esperado (commit 93339b4)               :", ESPERADO)
if _h_lf != ESPERADO:
    sys.exit("** El archivo NO coincide con la versión del paper. Detengo el script. **")
print("Coincide con la versión del paper.\n")
sys.path.insert(0, str(ENGINE))


import time, logging, warnings
import numpy as np
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
from stds_wrapper import StudyMetrics, run_stds
import eclipse_core as ec

SEMILLA, MU, SIGMA = 20250805, 0.70, 0.05
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
CONJUNTOS = {
    "(a) Tabla 5, 12 celdas": [(k, m) for k in (3, 5, 10, 20) for m in (1, 3, 5)],
    "(b) paso [2] de 04, 4 diseños": [(3, 1), (5, 3), (10, 1), (20, 5)],
}

def motor(folds_i, hold_i):
    m = folds_i.shape[0]
    return run_stds(StudyMetrics(
        cv_fold_values={f"m{j}": folds_i[j].tolist() for j in range(m)},
        holdout_values={f"m{j}": float(hold_i[j]) for j in range(m)}))

print("=== Controles positivos ===")
rng_c = np.random.default_rng(12345)          # generador aparte: no altera la simulación
peor, desacuerdos = 0.0, 0
for _ in range(2000):
    k = int(rng_c.choice([3, 5, 10, 20])); m = int(rng_c.choice([1, 3, 5]))
    f = rng_c.normal(MU, SIGMA, size=(m, k)); h = rng_c.normal(MU, SIGMA, size=m)
    r = motor(f, h)
    vec = float(((h - f.mean(1)) / f.std(1)).max())
    peor = max(peor, abs(r["max_z_score"] - vec))
    cut95 = ec.StatisticalTestDataSnooping.EMPIRICAL_SCREEN[(k, m)][0]
    desacuerdos += (r["risk_level"] != "LOW") != (vec > cut95)
c1, c2 = peor < 1e-9, desacuerdos == 0
print(f"  C1 max_z motor vs numpy, máx. diferencia: {peor:.2e}  {'OK' if c1 else 'FALLA'}")
print(f"  C2 risk_level != LOW  vs  max_z > p95: {desacuerdos} desacuerdos en 2000  {'OK' if c2 else 'FALLA'}")
marc = 0
for _ in range(1000):
    f = rng_c.normal(MU, SIGMA, size=(1, 5)); h = f.mean(1) + 4 * f.std(1) + rng_c.normal(0, 1e-3, 1)
    marc += motor(f, h)["risk_level"] != "LOW"
c3 = marc / 1000 > 0.90
print(f"  C3 holdout inflado (+4 SD), K=5, M=1: {100*marc/1000:.1f}% marcado  {'OK' if c3 else 'FALLA'}")
if not (c1 and c2 and c3):
    sys.exit("** Un control falló: la simulación no se ejecuta porque su resultado no valdría. **")

print(f"\n=== Simulación honesta por el motor (N = {N} por celda, semilla {SEMILLA}) ===")
if N != 60000:
    print("  AVISO: N distinto de 60.000; el resultado no es comparable con la condición fijada.")
t0 = time.time()
for nombre, celdas in CONJUNTOS.items():
    rng = np.random.default_rng(SEMILLA)
    tasas = []
    print(f"\n{nombre}")
    for k, m in celdas:
        folds = rng.normal(MU, SIGMA, size=(N, m, k))
        hold = rng.normal(MU, SIGMA, size=(N, m))
        marcados = validos = 0
        for i in range(N):
            r = motor(folds[i], hold[i])
            if r.get("status") != "success":
                continue
            validos += 1
            marcados += r["risk_level"] != "LOW"
        p = marcados / validos
        tasas.append(100 * p)
        print(f"  K={k:>2} |M|={m}: {100*p:5.2f}%  (EE MC ±{100*np.sqrt(p*(1-p)/validos):.2f}; {marcados}/{validos})", flush=True)
    print(f"  Rango: {min(tasas):.2f}% – {max(tasas):.2f}%")
print(f"\nTiempo: {time.time() - t0:.0f} s")
print("Paper: 4.4–5.0%. Este script no saca conclusiones; compara tú los rangos.")
