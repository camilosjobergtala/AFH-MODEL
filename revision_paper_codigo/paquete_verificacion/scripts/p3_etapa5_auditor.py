"""Punto 3: la Etapa 5 no ejecuta el auditor de código.

Paper (Methods, Stage 5): "After validation, ECLIPSE automatically generates four
outputs: ... and a Code Audit Report." La Fig. 1 pone el auditor dentro de la Etapa 5.

Método: se espía CodeAuditor.audit_analysis_code (se cuenta cada llamada) y se ejecuta
el flujo completo Etapa 1 -> 5 en una carpeta temporal.
  - Hallazgo: llamadas al auditor durante la Etapa 5.
  - Control positivo: llamar a audit_code() a mano SÍ debe incrementar el contador y
    escribir el reporte de auditoría. Si no, el espía no funciona y el hallazgo no vale.

Uso (desde la raíz del clon en 93339b4):  python p3_etapa5_auditor.py
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

import io, builtins, tempfile, logging, warnings, contextlib
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
import eclipse_core as ec

llamadas = []
_original = ec.CodeAuditor.audit_analysis_code
def _espia(self, *a, **k):
    llamadas.append(1)
    return _original(self, *a, **k)
ec.CodeAuditor.audit_analysis_code = _espia

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

out = Path(tempfile.mkdtemp(prefix="p3_"))
cfg = ec.EclipseConfig(project_name="p3", researcher="test", sacred_seed=20250805, output_dir=str(out))
fw = quiet(ec.EclipseFramework, cfg)

ids = [f"s{i:03d}" for i in range(100)]
quiet(fw.stage1_irreversible_split, ids)
quiet(fw.stage2_register_criteria, [ec.FalsificationCriteria("acc", 0.5, ">=", "criterio de prueba")])
dev = quiet(fw.stage3_development, list(range(70)),
            training_function=lambda idx, **k: None,
            validation_function=lambda m, idx, **k: {"acc": 0.60 + (int(sum(idx)) % 7) / 100})
builtins.input = lambda *a: "I ACCEPT SINGLE-SHOT VALIDATION"   # simula la frase tecleada
val = quiet(fw.stage4_single_shot_validation, list(range(30)), None, lambda m, h, **k: {"acc": 0.62})

antes = len(llamadas)
final = quiet(fw.stage5_final_assessment, dev, val)      # parámetros por defecto
durante_etapa5 = len(llamadas) - antes

print("=== Hallazgo ===")
print("Llamadas al auditor durante la Etapa 5     :", durante_etapa5)
print("Claves de la salida de la Etapa 5          :", sorted(final.keys()))
print("Claves dentro de 'integrity_metrics'       :", sorted(final.get("integrity_metrics", {}).keys()))
print("Archivos escritos tras la Etapa 5          :", sorted(p.name for p in out.iterdir()))
hay_audit = any("AUDIT" in p.name.upper() for p in out.iterdir())
print("¿Hay algún reporte de auditoría?           :", hay_audit)

print("\n=== Control positivo: audit_code() llamado a mano ===")
script = out / "control.py"
script.write_text("print(X_test[0])\n", encoding="utf-8")
antes = len(llamadas)
res = quiet(fw.audit_code, code_paths=[str(script)])
print("Llamadas al auditor registradas por el espía:", len(llamadas) - antes, "(esperado: 1)")
print("Violaciones detectadas en el control        :", sorted({v.category for v in res.violations}))
print("Archivos con 'AUDIT' tras el control        :", sorted(p.name for p in out.iterdir() if "AUDIT" in p.name.upper()))
control_ok = (len(llamadas) - antes == 1) and any("AUDIT" in p.name.upper() for p in out.iterdir())
print("Control positivo:", "OK" if control_ok else "FALLA -> el hallazgo NO vale")

print("\n=== Resultado ===")
if control_ok:
    print("La Etapa 5 llamó al auditor", durante_etapa5, "veces." +
          (" No ejecuta el auditor." if durante_etapa5 == 0 else " SÍ lo ejecuta."))
