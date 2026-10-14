"""Punto 8: rango "5.1–54.4%" bajo el corte fijo (Study 2b).

Paper (Results, Study 2b): "the realized honest-scenario rate is 4.4–5.0% across the
sampled designs, against 5.1–54.4% under the fixed cutoff". El README dice lo mismo.

Método: se leen los archivos depositados en outputs/ y se imprime el rango en cada uno.
  - outputs/study2b/stds_screening_cutoffs.csv : columna honest_rate_at_fixed_2 (Study 2b)
  - outputs/study2/study2_fpr_by_K_M.csv       : columna fpr (Study 2, Tabla 2)
  - Control positivo: los cortes p95/p99 del CSV de 2b deben coincidir con
    EMPIRICAL_SCREEN del motor (Tabla 5). Si no coinciden, se está leyendo mal el archivo.

Uso (desde la raíz del clon en 93339b4):  python p8_rango_corte_fijo.py
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


import csv, logging
logging.disable(logging.CRITICAL)
import eclipse_core as ec

OUT = ROOT / "ScientificReports_revision" / "outputs"
f2b = OUT / "study2b" / "stds_screening_cutoffs.csv"
f2 = OUT / "study2" / "study2_fpr_by_K_M.csv"
filas2b = list(csv.DictReader(open(f2b, encoding="utf-8")))
filas2 = list(csv.DictReader(open(f2, encoding="utf-8")))

print("=== Control positivo: CSV de 2b vs EMPIRICAL_SCREEN del motor ===")
dif = []
for r in filas2b:
    k, m = int(r["k_folds"]), int(r["n_metrics"])
    motor = ec.StatisticalTestDataSnooping.EMPIRICAL_SCREEN[(k, m)]
    archivo = (float(r["cutoff_p95"]), float(r["cutoff_p99"]))
    if tuple(round(x, 2) for x in motor) != archivo:
        dif.append((k, m, archivo, motor))
control_ok = not dif and len(filas2b) == 16
print(f"  Filas leídas: {len(filas2b)} (esperado 16). Diferencias: {dif or 'ninguna'}")
print("  Control:", "OK" if control_ok else "FALLA -> lo de abajo NO vale")

print(f"\n=== Study 2b: {f2b.relative_to(ROOT)} (honest_rate_at_fixed_2) ===")
for r in filas2b:
    print(f"  K={int(r['k_folds']):>2} |M|={r['n_metrics']}: {100*float(r['honest_rate_at_fixed_2']):5.2f}%")
v2b = [100 * float(r["honest_rate_at_fixed_2"]) for r in filas2b]
sub = [100 * float(r["honest_rate_at_fixed_2"]) for r in filas2b if r["n_metrics"] in ("1", "3", "5")]
print(f"  Rango (16 diseños)                         : {min(v2b):.2f}% – {max(v2b):.2f}%")
print(f"  Rango (12 diseños |M| en 1,3,5, como Fig. 4B): {min(sub):.2f}% – {max(sub):.2f}%")

print(f"\n=== Study 2: {f2.relative_to(ROOT)} (fpr, Tabla 2) ===")
for r in filas2:
    print(f"  K={int(r['K_folds']):>2} |M|={r['n_metrics']}: {100*float(r['fpr']):5.2f}%")
v2 = [100 * float(r["fpr"]) for r in filas2]
print(f"  Rango (9 diseños): {min(v2):.2f}% – {max(v2):.2f}%")

print("\n=== Resultado ===")
if control_ok:
    print("Paper: 5.1–54.4%.")
    print(f"Mínimo en Study 2b: {min(v2b):.2f}%   Máximo en Study 2b: {max(v2b):.2f}%")
    print(f"Mínimo en Study 2 : {min(v2):.2f}%   Máximo en Study 2 : {max(v2):.2f}%")
