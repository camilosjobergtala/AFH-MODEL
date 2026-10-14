"""Complemento de los puntos 1 y 2 del correo.

  O1  ¿El split depende del orden en que se entregan los IDs?
      Misma semilla y mismo conjunto de IDs, en orden directo e inverso.
      Control positivo: el mismo orden dos veces SÍ debe dar exactamente el mismo holdout.
      También se compara el split_hash de ambos casos.
  O2  ¿La Etapa 3 avisa de algo al recibir datos que incluyen el holdout?
      Se capturan warnings, logs y salida de pantalla.
      Control positivo: llamar a la Etapa 3 SIN criterios registrados SÍ debe emitir la
      advertencia del core "Criteria not registered" (prueba que la captura funciona).

Uso (desde la raíz del clon en 93339b4):  python p12_orden_y_avisos.py
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

import io, json, tempfile, warnings, contextlib, logging
import eclipse_core as ec

def fw(nombre):
    cfg = ec.EclipseConfig(project_name=nombre, researcher="t", sacred_seed=20250805,
                           output_dir=tempfile.mkdtemp(prefix=nombre + "_"))
    with contextlib.redirect_stdout(io.StringIO()):
        return ec.EclipseFramework(cfg)

def split(f, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        _, hold = f.stage1_irreversible_split(ids)
    return hold, json.load(open(f.split_file))["integrity_verification"]["split_hash"]

def etapa3(f, ids):
    salida, logs = io.StringIO(), io.StringIO()
    h = logging.StreamHandler(logs); logging.getLogger().addHandler(h)
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(salida):
        warnings.simplefilter("always")
        f.stage3_development(ids, training_function=lambda idx, **k: None,
                             validation_function=lambda m, idx, **k: {"acc": 0.6 + (int(sum(idx)) % 7) / 100})
    logging.getLogger().removeHandler(h)
    return [str(x.message) for x in w], logs.getvalue().strip(), salida.getvalue()

IDS = [f"s{i:03d}" for i in range(100)]

print("=== O1: dependencia del orden de entrada ===")
h_a, hash_a = split(fw("o1a"), IDS)
h_b, hash_b = split(fw("o1b"), IDS)
control1 = sorted(h_a) == sorted(h_b) and hash_a == hash_b
print("  Control (mismo orden dos veces): holdout idéntico y mismo hash:", "OK" if control1 else "FALLA")
h_c, hash_c = split(fw("o1c"), list(reversed(IDS)))
print("  Orden inverso -> holdout igual al del orden directo:", sorted(h_a) == sorted(h_c))
print("  IDs del holdout que difieren:", len(set(h_a) ^ set(h_c)) // 2, "de", len(h_a))
print("  split_hash igual en ambos órdenes:", hash_a == hash_c)
f_tmp = fw("o1f"); split(f_tmp, IDS)
print("  Claves del archivo del split (¿guarda el orden de entrada?):", sorted(json.load(open(f_tmp.split_file)).keys()))

print("\n=== O2: avisos de la Etapa 3 ===")
f = fw("o2c")
with contextlib.redirect_stdout(io.StringIO()):
    f.stage1_irreversible_split(IDS)
w, logs, _ = etapa3(f, IDS[:70])
control2 = any("Criteria not registered" in x for x in w)
print("  Control (Etapa 3 sin criterios): warnings capturados:", w, "->", "OK" if control2 else "FALLA")
f = fw("o2")
with contextlib.redirect_stdout(io.StringIO()):
    f.stage1_irreversible_split(IDS)
    f.stage2_register_criteria([ec.FalsificationCriteria("acc", 0.7, ">=", "x")])
w, logs, salida = etapa3(f, IDS)
print("  Etapa 3 con los 100 IDs (incluye 30 del holdout):")
print("    warnings:", w or "ninguno")
print("    logs    :", logs or "ninguno")
print("    líneas en pantalla que mencionan holdout/warn:",
      [l for l in salida.splitlines() if any(t in l.lower() for t in ("holdout", "warn", "⚠"))] or "ninguna")

print("\n=== Resultado ===")
if control1 and control2:
    print("O1: el holdout depende del orden de entrada y el hash no lo refleja:", sorted(h_a) != sorted(h_c) and hash_a == hash_c)
    print("O2: la Etapa 3 no emite ningún aviso con datos del holdout:", not w and not logs)
else:
    print("** Un control falló: el resultado no vale. **")
