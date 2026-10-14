"""Verificaciones adicionales paper vs. código (commit 93339b4).

Uso (desde la raíz del repositorio, con requirements_frozen.txt instalado):
    python revision_paper_codigo/verificar_otros.py

Comprueba:
  A. Que la Etapa 5 no ejecuta el auditor de código (el paper dice que genera cuatro salidas).
     Evidencia débil (búsqueda en el código fuente), suficiente para una afirmación negativa.
  B. Que el auditor no detecta dos patrones que el paper atribuye a sus niveles 2 y 3,
     con dos controles que SÍ deben detectarse. Si un control falla, el bloque B no vale.
  D. Si algún script del depósito o del historial de git produce el rango "4.4–5.0%" que el
     paper reporta para el STDS. No simula nada: solo revisa el historial
     (ver reproducir_rango_stds.py para el intento de reproducción).
"""
import sys, io, json, inspect, logging, tempfile, contextlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "ScientificReports_revision"
sys.path.insert(0, str(ROOT / "engine"))
logging.disable(logging.CRITICAL)
import eclipse_core as ec
from auditor_wrapper import audit_script
from stds_wrapper import StudyMetrics, run_stds

print("A. ¿La Etapa 5 llama al auditor de código?")
src5 = inspect.getsource(ec.EclipseFramework.stage5_final_assessment)
src_im = inspect.getsource(ec.EclipseFramework.compute_integrity_metrics)
print("   stage5 menciona audit/CodeAuditor:", any(s in src5 for s in ("audit", "CodeAuditor")))
print("   compute_integrity_metrics menciona audit/CodeAuditor:", any(s in src_im for s in ("audit", "CodeAuditor")))

print("\nB. Patrones que el paper atribuye a los niveles 2 y 3 del auditor (con controles)")
controles = {
    "CONTROL - acceso condicional (nivel 2, sí debe detectarse)":
        ("if accuracy > 0.8:\n    evaluate(model, X_test)\n", "conditional_holdout_access"),
    "CONTROL - acceso directo al holdout (sí debe detectarse)":
        ("print(X_test[0])\n", "holdout_access"),
}
casos = {
    "Tabla 1, 'Information leakage' (data-flow): y = test_data.mean()":
        "import numpy as np\ny = test_data.mean()\n",
    "Nivel 2, loop-based optimization: for threshold in ...: performance":
        "for threshold in grid:\n    perf = evaluate(model, X_eval, threshold)\n",
}
d = Path(tempfile.mkdtemp())
controles_ok = True
for i, (nombre, (code, esperado)) in enumerate(controles.items()):
    p = d / f"control_{i}.py"; p.write_text(code)
    cats = audit_script(str(p))["categories"]
    ok = esperado in cats
    controles_ok &= ok
    print(f"   {nombre}\n      -> categorías reportadas: {cats}  (esperado {esperado!r}: {'OK' if ok else 'FALLA'})")
if not controles_ok:
    print("   ** Al menos un control falló: los resultados de B no son válidos. **")
for i, (nombre, code) in enumerate(casos.items()):
    p = d / f"caso_{i}.py"; p.write_text(code)
    print(f"   {nombre}\n      -> categorías reportadas: {audit_script(str(p))['categories']}")

print("\nD. ¿Qué script produce el '4.4–5.0%' del paper? (historial de git, sin simular)")
import subprocess
REPO = ROOT.parent
def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()
print("   Commits que introducen la cadena '4.4%' o '4.4–5.0':")
print("     " + (git("log", "--all", "--format=%h %cs %s", "-G", "4\\.4%|4\\.4.?[–-].?5\\.0", "--",
                   "ScientificReports_revision") or "(ninguno)").replace("\n", "\n     "))
print("   Versiones de code/04_stds_screening_cutoffs.py en el historial:")
print("     " + git("log", "--all", "--format=%h %cs %s", "--",
                    "ScientificReports_revision/code/04_stds_screening_cutoffs.py").replace("\n", "\n     "))
hits = set()
for c in git("rev-list", "--all").split():
    out = git("grep", "-l", "-E", "risk_level|screen_cutoff", c, "--", "ScientificReports_revision/code")
    hits.update(out.split())
print("   Scripts de code/ que, en algún commit, usan risk_level o screen_cutoff "
      "(tasa de marcado por el motor):", sorted(hits) or "ninguno")
