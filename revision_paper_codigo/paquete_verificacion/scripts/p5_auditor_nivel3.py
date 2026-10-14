"""Punto 5: el ejemplo de 'Information leakage' de la Tabla 1 no lo detecta el nivel 3.

Paper, Tabla 1: "Information leakage | Critical | Data-flow | y=test_data.mean()".
Methods, Level 3: "Constructs a variable dependency graph to identify information
leakage via derived variables, cross-contamination between development and holdout,
and indirect access through aliasing."

Método: se audita cada fragmento con el CodeAuditor real. El nivel 3 del código emite
detection_method == 'semantic' (categoría indirect_holdout_access).
  - Control positivo 1 (nivel 3): una variable derivada del holdout que entra en fit()
    SÍ debe dar un hallazgo 'semantic'.
  - Control positivo 2 (nivel 1): print(X_test[0]) SÍ debe dar 'holdout_access'.
  - Caso: el ejemplo literal de la Tabla 1.

Uso (desde la raíz del clon en 93339b4):  python p5_auditor_nivel3.py
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

import tempfile, logging, warnings
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
import eclipse_core as ec

class _Cfg: audit_pass_threshold = 70.0; project_name = "p5"
class _Fw: config = _Cfg()
auditor = ec.CodeAuditor(_Fw())
tmp = Path(tempfile.mkdtemp(prefix="p5_"))

def auditar(nombre, codigo):
    p = tmp / "caso.py"
    p.write_text(codigo, encoding="utf-8")
    r = auditor.audit_analysis_code(code_paths=[str(p)])
    hallazgos = sorted({(v.category, v.detection_method) for v in r.violations})
    print(f"  {nombre}")
    print("    código:", codigo.strip().replace(chr(10), " | "))
    print("    hallazgos (categoría, método):", hallazgos or "ninguno")
    return hallazgos

print("=== Controles positivos ===")
c1 = auditar("Nivel 3: variable derivada del holdout en fit()",
             "tmp = test_data\nfeat = tmp\nmodel.fit(feat)\n")
c2 = auditar("Nivel 1: acceso directo", "print(X_test[0])\n")
ok1 = any(m == "semantic" for _, m in c1)
ok2 = any(c == "holdout_access" for c, _ in c2)
print("  Control nivel 3:", "OK" if ok1 else "FALLA")
print("  Control nivel 1:", "OK" if ok2 else "FALLA")
control_ok = ok1 and ok2
if not control_ok:
    print("  ** Al menos un control falló: el resultado de abajo NO vale. **")

print("\n=== Caso: ejemplo literal de la Tabla 1 ===")
h = auditar("Tabla 1, Information leakage", "y = test_data.mean()\n")

print("\n=== Resultado ===")
if control_ok:
    print("Ejemplo de la Tabla 1 -> hallazgo de nivel 3 ('semantic'):", any(m == "semantic" for _, m in h))
    print("Ejemplo de la Tabla 1 -> métodos que sí lo marcan       :", sorted({m for _, m in h}) or "ninguno")
