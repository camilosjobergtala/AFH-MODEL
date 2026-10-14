"""Verificación independiente de las discrepancias P1 y P4 en eclipse_core.py (commit 93339b4).

Uso:
    pip install -r ScientificReports_revision/requirements_frozen.txt
    python verificar_p1_p4.py RUTA/A/ScientificReports_revision/engine

Si no se da la ruta, se busca ./ScientificReports_revision/engine desde el directorio actual.
"""
import sys, json, tempfile, io, contextlib, builtins, logging, hashlib
from pathlib import Path
ENGINE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ScientificReports_revision/engine")
if not (ENGINE / "eclipse_core.py").exists():
    sys.exit(f"No encuentro eclipse_core.py en {ENGINE.resolve()}")
print("eclipse_core.py SHA-256:", hashlib.sha256((ENGINE / "eclipse_core.py").read_bytes()).hexdigest())
print("(versión del paper, commit 93339b4: 9d4320d225375fb5ed326b1cc282779125595ef500ff81bda81d7dd528b059b3)\n")
sys.path.insert(0, str(ENGINE))
logging.disable(logging.CRITICAL)
from eclipse_core import EclipseConfig, EclipseFramework, FalsificationCriteria

def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)

def nuevo_framework(nombre):
    cfg = EclipseConfig(project_name=nombre, researcher="test", sacred_seed=20250805,
                        output_dir=tempfile.mkdtemp())
    return quiet(EclipseFramework, cfg)

ids = [f"s{i:03d}" for i in range(100)]

fw0 = nuevo_framework("inspect")
quiet(fw0.stage1_irreversible_split, ids)
d0 = json.load(open(fw0.split_file))
print("Claves del split_file:", sorted(d0.keys()))
print("  development_ratio =", d0.get("development_ratio"), "| split_date =", d0.get("split_date"))
print("  integrity_verification =", d0.get("integrity_verification"), "\n")

print("=== P1: ¿el hash del split detecta cambios en la asignación dev/holdout? ===")
def prueba(descripcion, modificar):
    fw = nuevo_framework("p1")
    quiet(fw.stage1_irreversible_split, ids)
    d = json.load(open(fw.split_file))
    antes = json.dumps(d, sort_keys=True)
    claves_antes = set(d)
    modificar(d)
    if json.dumps(d, sort_keys=True) == antes:
        print(f"  {descripcion:<62} -> SIN EFECTO (no cambió nada)")
        return
    nuevas = set(d) - claves_antes
    if nuevas:
        print(f"  {descripcion:<62} -> INVÁLIDA (creó claves nuevas: {sorted(nuevas)})")
        return
    json.dump(d, open(fw.split_file, "w"))
    res = quiet(fw.verify_integrity)
    ok = res["all_valid"]
    print(f"  {descripcion:<62} -> verify_integrity: {'VÁLIDO' if ok else 'DETECTADO'}")
    print(f"      dict completo: {json.dumps({k: v for k, v in res.items() if k != 'timestamp'}, ensure_ascii=False)}")

def swap1(d):
    d["development_ids"][0], d["holdout_ids"][0] = d["holdout_ids"][0], d["development_ids"][0]
def todo_al_reves(d):
    todos = sorted(d["development_ids"] + d["holdout_ids"])
    d["holdout_ids"], d["development_ids"] = todos[:30], todos[30:]   # holdout totalmente distinto
def ratio(d):
    d["development_ratio"], d["holdout_ratio"] = 0.5, 0.5
def fecha(d):
    d["split_date"] = "1999-01-01T00:00:00"
def semilla(d):
    d["sacred_seed"] = 1
def quitar_id(d):
    d["holdout_ids"].pop()

prueba("Intercambiar 1 ID entre dev y holdout", swap1)
prueba("Reemplazar el holdout por otros 30 IDs (mismo conjunto total)", todo_al_reves)
prueba("Cambiar el ratio guardado (0.7/0.3 -> 0.5/0.5)", ratio)
prueba("Cambiar la fecha del split", fecha)
prueba("Control: cambiar la semilla", semilla)
prueba("Control: eliminar un ID", quitar_id)

print("\n=== P4: ¿la Etapa 4 (validación única) queda bloqueada tras usarse? ===")
fw = nuevo_framework("p4")
quiet(fw.stage1_irreversible_split, ids)
quiet(fw.stage2_register_criteria, [FalsificationCriteria("acc", 0.5, ">=", "x")])
builtins.input = lambda *a: "I ACCEPT SINGLE-SHOT VALIDATION"   # simula que el usuario teclea la frase
ejecuciones = 0
for intento in range(1, 4):
    r = quiet(fw.stage4_single_shot_validation, list(range(30)), None, lambda m, h: {"acc": 0.6})
    ejecuciones += r is not None
    print(f"  Intento {intento} de Etapa 4: {'se EJECUTÓ sobre el holdout' if r else 'rechazado'}")
print(f"  ¿Existe el archivo de resultados tras la Etapa 4?: {fw.results_file.exists()}")
import os, stat
hp = getattr(fw, "holdout_file", None)
print(f"  ¿El framework tiene atributo holdout_file?: {hp is not None}")
if hp and Path(hp).exists():
    print(f"  Permisos del archivo de holdout tras la Etapa 4: {stat.filemode(os.stat(hp).st_mode)}")
print("  Archivos que el framework escribió en output_dir, con sus permisos tras la Etapa 4:")
for f in sorted(Path(fw.output_dir).iterdir()):
    print(f"    {stat.filemode(os.stat(f).st_mode)}  {f.name}")
sp = json.load(open(fw.split_file))
print(f"  (el split_file contiene la lista de holdout_ids: {'holdout_ids' in sp}, n={len(sp['holdout_ids'])})")
dev = {"n_folds": 5, "n_successful": 5, "aggregated_metrics": {}}
quiet(fw.stage5_final_assessment, dev, r, generate_reports=False, compute_integrity=False)
try:
    quiet(fw.stage4_single_shot_validation, list(range(30)), None, lambda m, h: {"acc": 0.6})
    print("  Intento tras Etapa 5: se EJECUTÓ")
except RuntimeError:
    print("  Intento tras Etapa 5: rechazado (el bloqueo existe solo después de la Etapa 5)")
try:
    quiet(fw.stage4_single_shot_validation, list(range(30)), None, lambda m, h: {"acc": 0.6}, force=True)
    print("  Intento tras Etapa 5 con force=True: se EJECUTÓ")
except RuntimeError:
    print("  Intento tras Etapa 5 con force=True: rechazado")
