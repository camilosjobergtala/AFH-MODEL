"""Punto 9: techo del EIS (0.975) frente al valor con el holdout en la media (≈0.957).

Paper (Methods, §2.6.4): "the maximum attainable EIS is therefore 0.975, obtained by
exhaustive evaluation over all combinations of the protocol indicators with a balanced
split and a holdout at or below the development mean."

Método: se calcula el EIS con el motor real (eis_wrapper -> EclipseIntegrityScore),
con todos los indicadores al máximo, split 50/50 y N = 200 (sin penalización por N).
  - Control positivo: S_leak en z = -1, 0, +1, +2 debe dar 0.98 / 0.88 / 0.50 / 0.12,
    los valores que el propio paper publica (§2.6.4). Si no, el cálculo no es el del paper.
  - Hallazgo: EIS con el holdout exactamente en la media, y con el holdout muy por debajo.
  - Además: el máximo con el split 70/30 por defecto.

Uso (desde la raíz del clon en 93339b4):  python p9_techo_eis.py
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


import logging, warnings
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
from eis_wrapper import ProtocolSpec, score_protocol

MEDIA, SD = 0.70, 0.05      # media y sd de los folds que usa ProtocolSpec por defecto

def eis(holdout, n_dev=100, n_hold=100):
    r = score_protocol(ProtocolSpec(n_development=n_dev, n_holdout=n_hold,
                                    holdout_metrics={"f1_score": holdout}))
    return r["eis"], r["components"]

print("=== Control positivo: S_leak frente a los valores publicados ===")
esperado = {-1: 0.98, 0: 0.88, 1: 0.50, 2: 0.12}
control_ok = True
for z, e in esperado.items():
    _, c = eis(MEDIA + z * SD)
    ok = round(c["leakage_score"], 2) == e
    control_ok &= ok
    print(f"  z = {z:+d}: S_leak = {c['leakage_score']:.4f}  (paper {e:.2f})  {'OK' if ok else 'FALLA'}")
print("  Control:", "OK" if control_ok else "FALLA -> lo de abajo NO vale")

print("\n=== Hallazgo: split 50/50, N = 200, indicadores al máximo ===")
for etiqueta, h in [("holdout EN la media (z = 0)", MEDIA),
                    ("holdout 1 SD bajo la media (z = -1)", MEDIA - SD),
                    ("holdout muy por debajo (f1 = 0.0, z = -14)", 0.0)]:
    e, c = eis(h)
    print(f"  {etiqueta:<44}: EIS = {e:.4f}   S_pre = {c['preregistration_score']:.2f}  "
          f"S_split = {c['split_strength']:.4f}  S_leak = {c['leakage_score']:.4f}")

print("\n=== Split 70/30 por defecto, N = 1000, holdout muy por debajo ===")
e, c = eis(0.0, 700, 300)
print(f"  EIS = {e:.4f}   S_split = {c['split_strength']:.4f}")

print("\n=== Resultado ===")
if control_ok:
    e_media, _ = eis(MEDIA)
    e_inf, _ = eis(0.0)
    print(f"Con el holdout en la media el máximo es {e_media:.4f}; 0.975 se alcanza solo con el holdout muy por debajo ({e_inf:.4f}).")
