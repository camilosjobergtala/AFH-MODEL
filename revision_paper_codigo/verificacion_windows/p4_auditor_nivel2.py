"""Punto 4: el nivel 2 del auditor (control de flujo) no detecta bucles de optimización.

Paper (Methods, Multi-Level Analysis Architecture): "Level 2: Control-Flow Analysis.
Tracks execution paths to detect conditional holdout access (e.g., if accuracy>0.8:
evaluate(test_set)), loop-based optimization (e.g., for threshold in ...:
test_performance), and branch coverage anomalies."

Método: se audita cada fragmento con el CodeAuditor real y se listan los hallazgos con
su método de detección. El nivel 2 del código emite detection_method == 'controlflow'.
  - Control positivo: acceso condicional (if accuracy > 0.8: evaluate(model, X_test))
    SÍ debe dar un hallazgo 'controlflow'.
  - Casos: bucle de optimización de umbral, con y sin un identificador de holdout.
Además se listan los métodos del analizador, como evidencia (débil) de que no existe
un análisis de "branch coverage".

Uso (desde la raíz del clon en 93339b4):  python p4_auditor_nivel2.py
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

class _Cfg: audit_pass_threshold = 70.0; project_name = "p4"
class _Fw: config = _Cfg()
auditor = ec.CodeAuditor(_Fw())
tmp = Path(tempfile.mkdtemp(prefix="p4_"))

def auditar(nombre, codigo):
    p = tmp / "caso.py"
    p.write_text(codigo, encoding="utf-8")
    r = auditor.audit_analysis_code(code_paths=[str(p)])
    hallazgos = sorted({(v.category, v.detection_method) for v in r.violations})
    print(f"  {nombre}")
    print("    código:", codigo.strip().replace(chr(10), " | "))
    print("    hallazgos (categoría, método):", hallazgos or "ninguno")
    return {m for _, m in hallazgos}

print("=== Control positivo (debe dar 'controlflow') ===")
m = auditar("Acceso condicional", "if accuracy > 0.8:\n    evaluate(model, X_test)\n")
control_ok = "controlflow" in m
print("  Control:", "OK" if control_ok else "FALLA -> los resultados de abajo NO valen")

print("\n=== Casos del paper: bucle de optimización de umbral ===")
m1 = auditar("Bucle de umbral, datos neutros",
             "for threshold in grid:\n    perf = evaluate(model, X_eval, threshold)\n")
m2 = auditar("Bucle de umbral sobre holdout",
             "for threshold in grid:\n    test_performance = evaluate(model, X_test, threshold)\n")

print("\n=== Métodos de StaticCodeAnalyzer (búsqueda de un análisis de cobertura) ===")
metodos = sorted(n for n in vars(ec.StaticCodeAnalyzer) if callable(getattr(ec.StaticCodeAnalyzer, n)))
print(" ", metodos)
print("  ¿Algún método con 'coverage' o 'branch' en el nombre?:",
      any(("coverage" in n.lower() or "branch" in n.lower()) for n in metodos))

print("\n=== Resultado ===")
if control_ok:
    print("Bucle de umbral (neutro)  -> hallazgo 'controlflow':", "controlflow" in m1)
    print("Bucle de umbral (holdout) -> hallazgo 'controlflow':", "controlflow" in m2,
          "(si aparece 'holdout_access' es por el nombre X_test, nivel 1, no por el bucle)")
