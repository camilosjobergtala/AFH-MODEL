"""Verificaciones adicionales paper vs. código (commit 93339b4).

Uso (desde la raíz del repositorio, con requirements_frozen.txt instalado):
    python revision_paper_codigo/verificar_otros.py

Comprueba:
  A. Que la Etapa 5 no ejecuta el auditor de código (el paper dice que genera cuatro salidas).
  B. Que el auditor no detecta dos patrones que el paper atribuye a sus niveles 2 y 3.
  C. Que las barras "empirical cutoff" de la Fig. 4B son una constante, no un valor medido.
  D. Tasa de marcado del STDS bajo el escenario honesto, de punta a punta por el motor
     (para contrastar con "4.4–5.0%" del paper). N=2000 por celda: error MC ≈ ±1 pp.
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

print("\nB. Patrones que el paper atribuye a los niveles 2 y 3 del auditor")
casos = {
    "Tabla 1, 'Information leakage' (data-flow): y = test_data.mean()":
        "import numpy as np\ny = test_data.mean()\n",
    "Nivel 2, loop-based optimization: for threshold in ...: performance":
        "for threshold in grid:\n    perf = evaluate(model, X_eval, threshold)\n",
}
d = Path(tempfile.mkdtemp())
for nombre, code in casos.items():
    p = d / "c.py"; p.write_text(code)
    print(f"   {nombre}\n      -> categorías reportadas: {audit_script(str(p))['categories']}")

print("\nC. Fig. 4B: ¿de dónde salen las barras 'empirical cutoff'?")
fig = (ROOT / "code" / "05_figures.py").read_text().splitlines()
for i, line in enumerate(fig, 1):
    if "np.full(len(sub), 5.0)" in line:
        print(f"   05_figures.py:{i}: {line.strip()}")

print("\nD. STDS de punta a punta por el motor, escenario honesto (N=2000 por celda)")
rng = np.random.default_rng(20250805); rates = []
for k in (3, 5, 10, 20):
    for m in (1, 2, 3, 5):
        f = 0
        for _ in range(2000):
            folds = {f"m{j}": list(rng.normal(0.7, 0.05, k)) for j in range(m)}
            hold = {f"m{j}": float(rng.normal(0.7, 0.05)) for j in range(m)}
            f += run_stds(StudyMetrics(folds, hold))["risk_level"] != "LOW"
        rates.append(f / 20); print(f"   K={k:>2} M={m}: {f/20:.2f}%")
print(f"   rango: {min(rates):.2f}%–{max(rates):.2f}%")
