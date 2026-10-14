"""Punto 7 (versión paralela): mismo cálculo que p7_reproducir_rango.py, repartido en varios
procesos para que termine antes.

Qué NO cambia respecto de p7:
  - Cada estudio simulado pasa por el motor completo (stds_wrapper.run_stds ->
    StatisticalTestDataSnooping.perform_snooping_test). No hay ruta vectorizada.
  - N = 60.000 por celda, semilla 20250805, escenario honesto Normal(0.70, 0.05), delta = 0.
  - Marcado: risk_level != "LOW".
  - Mismos datos simulados, número por número. El proceso principal recorre las celdas en el
    mismo orden que p7 y guarda el estado exacto del generador al inicio de cada una; cada
    proceso hijo restaura ese estado y genera las mismas matrices. Los conteos deben ser
    idénticos a los de p7.
  - Mismos controles positivos C1–C3, ejecutados antes de la simulación.

Controles adicionales propios de la versión paralela:
  C4. Cada celda debe tener sus N casos válidos; si no, el script se detiene y no da número.
  C5. Cada proceso usa su propio archivo temporal (arranque 'spawn'), así que no se pisan.

Qué SÍ cambia: varias celdas se calculan a la vez (una por proceso), y cada proceso informa su
avance cada 10.000 casos.

Uso (desde la raíz del clon en 93339b4):
    python p7b_reproducir_rango_paralelo.py              # N = 60.000, procesos = núcleos - 1
    python p7b_reproducir_rango_paralelo.py 60000 6      # N y número de procesos explícitos
    python p7b_reproducir_rango_paralelo.py 2000         # prueba rápida (no comparable)
"""
import sys, hashlib, os, time
sys.dont_write_bytecode = True          # no deja __pycache__ en el clon
from pathlib import Path

ROOT = Path.cwd()
ENGINE = ROOT / "ScientificReports_revision" / "engine"
CORE = ENGINE / "eclipse_core.py"
ESPERADO = "9d4320d225375fb5ed326b1cc282779125595ef500ff81bda81d7dd528b059b3"
SEMILLA, MU, SIGMA = 20250805, 0.70, 0.05
CONJUNTOS = {
    "(a) Tabla 5, 12 celdas": [(k, m) for k in (3, 5, 10, 20) for m in (1, 3, 5)],
    "(b) paso [2] de 04, 4 diseños": [(3, 1), (5, 3), (10, 1), (20, 5)],
}


def _preparar_proceso(engine_dir):
    """Se ejecuta en cada proceso hijo: silencia logs y hace importable el motor."""
    import logging, warnings
    sys.dont_write_bytecode = True
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)


def _motor(run_stds, StudyMetrics, folds_i, hold_i):
    m = folds_i.shape[0]
    return run_stds(StudyMetrics(
        cv_fold_values={f"m{j}": folds_i[j].tolist() for j in range(m)},
        holdout_values={f"m{j}": float(hold_i[j]) for j in range(m)}))


def evaluar_celda(tarea):
    """Proceso hijo: restaura el estado del generador, genera la celda y la pasa por el motor."""
    nombre, k, m, n, estado = tarea
    import numpy as np
    import stds_wrapper
    from stds_wrapper import StudyMetrics, run_stds
    archivo_temporal = str(stds_wrapper._RESULTS_PATH)     # propio de este proceso (arranque 'spawn')
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = estado
    folds = rng.normal(MU, SIGMA, size=(n, m, k))
    hold = rng.normal(MU, SIGMA, size=(n, m))
    marcados = validos = 0
    t0 = time.time()
    for i in range(n):
        r = _motor(run_stds, StudyMetrics, folds[i], hold[i])
        if r.get("status") == "success":
            validos += 1
            marcados += r["risk_level"] != "LOW"
        if (i + 1) % 10000 == 0 and i + 1 < n:
            print(f"    … {nombre[:3]} K={k:>2} |M|={m}: {i+1}/{n} casos ({time.time()-t0:.0f} s)", flush=True)
    return nombre, k, m, marcados, validos, archivo_temporal


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if not CORE.exists():
        sys.exit(f"No encuentro {CORE}. Ejecuta el script desde la raíz del clon (la carpeta que contiene ScientificReports_revision).")
    raw = CORE.read_bytes()
    h_lf = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    print("SHA-256 eclipse_core.py (bytes tal cual)        :", hashlib.sha256(raw).hexdigest())
    print("SHA-256 eclipse_core.py (normalizado a LF)      :", h_lf)
    print("SHA-256 esperado (commit 93339b4)               :", ESPERADO)
    if h_lf != ESPERADO:
        sys.exit("** El archivo NO coincide con la versión del paper. Detengo el script. **")
    print("Coincide con la versión del paper.\n")

    _preparar_proceso(str(ENGINE))
    import numpy as np
    import multiprocessing as mp
    from stds_wrapper import StudyMetrics, run_stds
    import eclipse_core as ec

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    procesos = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 1)

    print("=== Controles positivos (idénticos a p7) ===")
    rng_c = np.random.default_rng(12345)
    peor, desacuerdos = 0.0, 0
    for _ in range(2000):
        k = int(rng_c.choice([3, 5, 10, 20])); m = int(rng_c.choice([1, 3, 5]))
        f = rng_c.normal(MU, SIGMA, size=(m, k)); h = rng_c.normal(MU, SIGMA, size=m)
        r = _motor(run_stds, StudyMetrics, f, h)
        vec = float(((h - f.mean(1)) / f.std(1)).max())
        peor = max(peor, abs(r["max_z_score"] - vec))
        cut95 = ec.StatisticalTestDataSnooping.EMPIRICAL_SCREEN[(k, m)][0]
        desacuerdos += (r["risk_level"] != "LOW") != (vec > cut95)
    c1, c2 = peor < 1e-9, desacuerdos == 0
    print(f"  C1 max_z motor vs numpy, máx. diferencia: {peor:.2e}  {'OK' if c1 else 'FALLA'}")
    print(f"  C2 risk_level != LOW  vs  max_z > p95: {desacuerdos} desacuerdos en 2000  {'OK' if c2 else 'FALLA'}")
    marc = 0
    for _ in range(1000):
        f = rng_c.normal(MU, SIGMA, size=(1, 5)); h = f.mean(1) + 4 * f.std(1) + rng_c.normal(0, 1e-3, 1)
        marc += _motor(run_stds, StudyMetrics, f, h)["risk_level"] != "LOW"
    c3 = marc / 1000 > 0.90
    print(f"  C3 holdout inflado (+4 SD), K=5, M=1: {100*marc/1000:.1f}% marcado  {'OK' if c3 else 'FALLA'}")
    if not (c1 and c2 and c3):
        sys.exit("** Un control falló: la simulación no se ejecuta porque su resultado no valdría. **")

    # Recorre las celdas en el mismo orden que p7, guarda el estado del generador al inicio
    # de cada una y avanza el generador generando (sin evaluar) las mismas matrices.
    tareas = []
    for nombre, celdas in CONJUNTOS.items():
        rng = np.random.default_rng(SEMILLA)
        for k, m in celdas:
            tareas.append((nombre, k, m, N, rng.bit_generator.state))
            rng.normal(MU, SIGMA, size=(N, m, k))
            rng.normal(MU, SIGMA, size=(N, m))

    print(f"\n=== Simulación honesta por el motor (N = {N} por celda, semilla {SEMILLA}, {procesos} procesos) ===")
    if N != 60000:
        print("  AVISO: N distinto de 60.000; el resultado no es comparable con la condición fijada.")
    t0 = time.time()
    resultados, temporales = {}, set()
    # 'spawn' en todas las plataformas: cada proceso hijo importa el motor desde cero y crea su
    # PROPIO archivo temporal (con 'fork' compartirían el del proceso principal y se pisarían).
    ctx = mp.get_context("spawn")
    with ctx.Pool(procesos, initializer=_preparar_proceso, initargs=(str(ENGINE),)) as pool:
        for nombre, k, m, marcados, validos, tmpfile in pool.imap_unordered(evaluar_celda, tareas):
            resultados[(nombre, k, m)] = (marcados, validos)
            temporales.add(tmpfile)
            print(f"  terminada {nombre[:3]} K={k:>2} |M|={m}: {100*marcados/validos:5.2f}%  "
                  f"({marcados}/{validos})  [{len(resultados)}/{len(tareas)}, {time.time()-t0:.0f} s]", flush=True)
            if validos != N:
                pool.terminate()
                sys.exit(f"** Control C4 FALLA: solo {validos}/{N} casos válidos en esa celda. Resultado descartado. **")

    print(f"\n  C4 todos los casos válidos en todas las celdas: OK ({len(tareas)} celdas x {N})")
    print(f"  C5 archivos temporales distintos usados por los procesos: {len(temporales)} "
          f"(debe ser > 1 si hubo más de un proceso)  {'OK' if (procesos == 1 or len(temporales) > 1) else 'FALLA'}")

    for nombre, celdas in CONJUNTOS.items():
        print(f"\n{nombre}")
        tasas = []
        for k, m in celdas:
            marcados, validos = resultados[(nombre, k, m)]
            p = marcados / validos
            tasas.append(100 * p)
            print(f"  K={k:>2} |M|={m}: {100*p:5.2f}%  (EE MC ±{100*np.sqrt(p*(1-p)/validos):.2f}; {marcados}/{validos})")
        print(f"  Rango: {min(tasas):.2f}% – {max(tasas):.2f}%")
    print(f"\nTiempo: {time.time() - t0:.0f} s")
    print("Paper: 4.4–5.0%. Este script no saca conclusiones; compara tú los rangos.")


if __name__ == "__main__":      # obligatorio en Windows para multiprocessing
    main()
