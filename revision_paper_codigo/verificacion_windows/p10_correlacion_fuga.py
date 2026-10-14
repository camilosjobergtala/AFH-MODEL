"""Punto 10: "|r| ≤ 0.08" para el componente de fuga (Study 1).

Paper (Results, Study 1): "the leakage-risk component was essentially uncorrelated with
the others (|r| ≤ 0.08)."

Método: se lee la matriz de correlaciones depositada y se imprime la fila del componente
de fuga. Además se recalcula la matriz desde el corpus depositado (Spearman).
  - outputs/study1/component_correlation_matrix.csv
  - outputs/study1/study1_corpus.csv
  - outputs/study1/study1_summary.json
  - Control positivo: el máximo y la media de |r| fuera de la diagonal deben coincidir
    con study1_summary.json (0.40 / 0.14 en el paper), y la matriz recalculada desde el
    corpus debe coincidir con la depositada. Si no, se está leyendo mal.

Uso (desde la raíz del clon en 93339b4):  python p10_correlacion_fuga.py
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


import json
import pandas as pd

S1 = ROOT / "ScientificReports_revision" / "outputs" / "study1"
mat = pd.read_csv(S1 / "component_correlation_matrix.csv", index_col=0)
resumen = json.load(open(S1 / "study1_summary.json", encoding="utf-8"))
corpus = pd.read_csv(S1 / "study1_corpus.csv")

comps = list(mat.columns)
offdiag = [abs(mat.loc[a, b]) for i, a in enumerate(comps) for b in comps[i + 1:]]
recalc = corpus[comps].corr(method="spearman")

print("=== Control positivo ===")
mx, mn = max(offdiag), sum(offdiag) / len(offdiag)
nr = resumen["non_redundancy"]
ok1 = abs(mx - nr["max_abs_offdiag_spearman"]) < 1e-9 and abs(mn - nr["mean_abs_offdiag_spearman"]) < 1e-9
ok2 = float((recalc - mat).abs().max().max()) < 1e-9
print(f"  max |r| (matriz) = {mx:.4f}, summary = {nr['max_abs_offdiag_spearman']:.4f}; "
      f"media |r| (matriz) = {mn:.4f}, summary = {nr['mean_abs_offdiag_spearman']:.4f}  {'OK' if ok1 else 'FALLA'}")
print(f"  Matriz recalculada desde el corpus = matriz depositada: {'OK' if ok2 else 'FALLA'}")
control_ok = ok1 and ok2
print("  Control:", "OK" if control_ok else "FALLA -> lo de abajo NO vale")

print("\n=== Fila del componente de fuga (leakage_score) ===")
for c in comps:
    if c != "leakage_score":
        r = mat.loc["leakage_score", c]
        print(f"  leakage_score vs {c:<20}: r = {r:+.4f}   |r| = {abs(r):.4f}   (2 decimales: {r:+.2f})")
mx_leak = max(abs(mat.loc["leakage_score", c]) for c in comps if c != "leakage_score")

print("\n=== Resultado ===")
if control_ok:
    print(f"Paper: |r| ≤ 0.08.   Máximo |r| del componente de fuga en los datos depositados: {mx_leak:.4f}")
