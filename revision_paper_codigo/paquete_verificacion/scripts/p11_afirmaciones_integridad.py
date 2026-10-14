"""Tarea 2 (barrido del texto): comprobaciones de las afirmaciones sobre hash,
inmutabilidad, sellado del holdout y precedencia temporal que no cubren P1–P4.

Cada bloque modifica un artefacto en una carpeta temporal y llama a verify_integrity()
del core, o ejecuta una etapa de forma que el paper dice que no es posible.
Control positivo común: editar el umbral de un criterio SIN recalcular el hash SÍ debe
detectarse. Si no se detecta, los demás resultados no valen.

  T1  Criterios editados y hash recalculado (sin cambiar la fecha)  -> ¿se detecta?
  T2  Fecha de registro de criterios cambiada (registration_date)    -> ¿se detecta?
  T3  Re-registro de criterios con force=True                        -> ¿queda rastro?
  T4  Re-split con force=True                                         -> ¿queda rastro?
  T5  ¿La Etapa 1 entrega los IDs del holdout al usuario?             (sellado)
  T6  ¿La Etapa 3 acepta datos que incluyen el holdout?               ("enforces")
  T7  Resultados finales editados tras la Etapa 5                     -> ¿se detecta?
  T8  ¿Algún código compara la fecha de los criterios con la de la validación?

Uso (desde la raíz del clon en 93339b4):  python p11_afirmaciones_integridad.py
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


import io, json, hashlib, builtins, tempfile, logging, warnings, contextlib
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
import eclipse_core as ec

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

IDS = [f"s{i:03d}" for i in range(100)]
CRIT = [ec.FalsificationCriteria("acc", 0.70, ">=", "criterio original")]

def nuevo(nombre):
    cfg = ec.EclipseConfig(project_name=nombre, researcher="test", sacred_seed=20250805,
                           output_dir=tempfile.mkdtemp(prefix=nombre + "_"))
    fw = quiet(ec.EclipseFramework, cfg)
    dev, hold = quiet(fw.stage1_irreversible_split, IDS)
    quiet(fw.stage2_register_criteria, CRIT)
    return fw, dev, hold

def valido(fw):
    v = quiet(fw.verify_integrity)
    return v["all_valid"], {k: x["valid"] for k, x in v["files"].items()}

def editar(path, cambio):
    d = json.load(open(path)); cambio(d); json.dump(d, open(path, "w"), indent=2)

print("=== Control positivo: editar un umbral SIN recalcular el hash ===")
fw, _, _ = nuevo("c")
editar(fw.criteria_file, lambda d: d["criteria"][0].__setitem__("threshold", 0.50))
ok, det = valido(fw)
control_ok = not ok
print("  verify_integrity:", det, "->", "DETECTADO (OK)" if control_ok else "NO detectado (FALLA)")
if not control_ok:
    sys.exit("** El control falló: el resto no vale. **")

print("\nT1 Criterios editados (umbral 0.70 -> 0.50) y hash recalculado con la misma receta del core")
fw, _, _ = nuevo("t1")
def t1(d):
    d["criteria"][0]["threshold"] = 0.50
    d["criteria_hash"] = hashlib.sha256(json.dumps(d["criteria"], sort_keys=True).encode()).hexdigest()
editar(fw.criteria_file, t1)
print("  verify_integrity:", valido(fw)[1])

print("\nT2 Fecha de registro de criterios cambiada a 2020-01-01")
fw, _, _ = nuevo("t2")
editar(fw.criteria_file, lambda d: d.__setitem__("registration_date", "2020-01-01T00:00:00"))
print("  verify_integrity:", valido(fw)[1])

print("\nT3 Re-registro de criterios con force=True (umbral nuevo 0.50)")
fw, _, _ = nuevo("t3")
antes = json.load(open(fw.criteria_file))
quiet(fw.stage2_register_criteria, [ec.FalsificationCriteria("acc", 0.50, ">=", "criterio nuevo")], force=True)
despues = json.load(open(fw.criteria_file))
print("  umbral antes/después:", antes["criteria"][0]["threshold"], "->", despues["criteria"][0]["threshold"])
print("  verify_integrity:", valido(fw)[1])
print("  archivos en la carpeta (¿queda copia del registro anterior?):", sorted(p.name for p in fw.output_dir.iterdir()))

print("\nT4 Re-split con force=True y otra semilla")
fw, dev0, hold0 = nuevo("t4")
fw.config.sacred_seed = 1
dev1, hold1 = quiet(fw.stage1_irreversible_split, IDS, force=True)
print("  holdout cambiado:", sorted(hold0) != sorted(hold1), "| IDs del holdout que cambiaron:", len(set(hold0) ^ set(hold1)) // 2)
print("  verify_integrity:", valido(fw)[1])
print("  archivos en la carpeta (¿queda copia del split anterior?):", sorted(p.name for p in fw.output_dir.iterdir()))

print("\nT5 ¿La Etapa 1 devuelve al usuario la lista de IDs del holdout?")
fw, dev, hold = nuevo("t5")
print(f"  stage1 devolvió {len(hold)} IDs de holdout al código que la llamó; ejemplo: {hold[:3]}")

print("\nT6 Etapa 3 con datos que incluyen el holdout (los 100 IDs, no solo los 70 de desarrollo)")
fw, dev, hold = nuevo("t6")
try:
    r = quiet(fw.stage3_development, IDS, training_function=lambda idx, **k: None,
              validation_function=lambda m, idx, **k: {"acc": 0.6 + (int(sum(idx)) % 7) / 100})
    print(f"  La Etapa 3 se ejecutó sin error con {len(IDS)} unidades (incluye {len(hold)} del holdout).")
except Exception as e:
    print("  La Etapa 3 rechazó los datos:", e)

print("\nT7 Resultados finales editados tras la Etapa 5")
fw, dev_ids, _ = nuevo("t7")
dev = quiet(fw.stage3_development, list(range(70)), training_function=lambda idx, **k: None,
            validation_function=lambda m, idx, **k: {"acc": 0.6 + (int(sum(idx)) % 7) / 100})
builtins.input = lambda *a: "I ACCEPT SINGLE-SHOT VALIDATION"
val = quiet(fw.stage4_single_shot_validation, list(range(30)), None, lambda m, h, **k: {"acc": 0.62})
quiet(fw.stage5_final_assessment, dev, val, generate_reports=False)
editar(fw.results_file, lambda d: (d["validation_summary"]["metrics"].__setitem__("acc", 0.99),
                                    d.__setitem__("verdict", "VALIDATED")))
ok, det = valido(fw)
print("  Archivos que verify_integrity comprueba:", sorted(det), "| all_valid:", ok)

print("\nT8 ¿Compara algún código la fecha de registro de criterios con la de la validación?")
src = CORE.read_text(encoding="utf-8")
lineas = [i for i, l in enumerate(src.splitlines(), 1) if "registration_date" in l]
print("  Líneas de eclipse_core.py que mencionan registration_date:", lineas)
for i in lineas:
    print(f"    {i}: {src.splitlines()[i-1].strip()}")
