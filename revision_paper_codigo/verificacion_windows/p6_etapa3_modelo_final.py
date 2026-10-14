"""Punto 6: la Etapa 3 no entrena un modelo final sobre todo D_dev.

Paper (Methods, Stage 3, k-fold cross-validation): "A final model is then trained on
all of D_dev."

Método: se espía la función de entrenamiento que recibe stage3_development y se
registra el tamaño del conjunto de entrenamiento de cada llamada.
  - Hallazgo: ¿alguna llamada entrena con los n_dev índices completos?
  - Control positivo: el espía registra exactamente K llamadas (una por fold), y una
    llamada hecha a mano con todos los índices SÍ queda registrada con tamaño n_dev.
  - También se listan las claves que devuelve la Etapa 3 (¿incluye un modelo?).

Uso (desde la raíz del clon en 93339b4):  python p6_etapa3_modelo_final.py
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

import io, tempfile, logging, warnings, contextlib
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
import eclipse_core as ec

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

K, N_DEV = 5, 70
tamanos = []
def entrenar(train_idx, **k):
    tamanos.append(len(train_idx))
    return {"modelo": len(train_idx)}

out = Path(tempfile.mkdtemp(prefix="p6_"))
cfg = ec.EclipseConfig(project_name="p6", researcher="test", sacred_seed=20250805,
                       output_dir=str(out), n_folds_cv=K)
fw = quiet(ec.EclipseFramework, cfg)
quiet(fw.stage1_irreversible_split, [f"s{i:03d}" for i in range(100)])
quiet(fw.stage2_register_criteria, [ec.FalsificationCriteria("acc", 0.5, ">=", "x")])
res = quiet(fw.stage3_development, list(range(N_DEV)), training_function=entrenar,
            validation_function=lambda m, idx, **k: {"acc": 0.6 + (int(sum(idx)) % 7) / 100})

print("=== Hallazgo ===")
print("Tamaños de entrenamiento en cada llamada:", tamanos, f"(n_dev = {N_DEV})")
print("¿Alguna llamada con todo D_dev?        :", N_DEV in tamanos)
print("Claves devueltas por la Etapa 3        :", sorted(res.keys()))
print("¿Devuelve algún modelo?                :", any("model" in k.lower() for k in res.keys()))

print("\n=== Control positivo ===")
n_llamadas = len(tamanos)
entrenar(list(range(N_DEV)))
control_ok = (n_llamadas == K) and (tamanos[-1] == N_DEV)
print(f"Llamadas registradas durante la Etapa 3: {n_llamadas} (esperado: {K})")
print(f"Llamada manual con todo D_dev registrada con tamaño: {tamanos[-1]} (esperado: {N_DEV})")
print("Control positivo:", "OK" if control_ok else "FALLA -> el hallazgo NO vale")

print("\n=== Resultado ===")
if control_ok:
    print("La Etapa 3 entrenó sobre todo D_dev:", N_DEV in tamanos[:n_llamadas])
