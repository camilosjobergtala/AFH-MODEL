"""Intento de reproducir el rango "4.4–5.0%" que el paper reporta para el STDS (§3.2).

Frase del paper: "Verified end to end through the engine, the realized honest-scenario
rate is 4.4–5.0% across the sampled designs".

Condiciones (fijadas antes de correr):
  - Motor completo: cada estudio simulado pasa por stds_wrapper.run_stds(), que llama a
    StatisticalTestDataSnooping.perform_snooping_test() del core. No se usa la ruta vectorizada.
  - N = 60.000 estudios por celda (igual que la derivación de la Tabla 5).
  - Semilla maestra 20250805. Cada conjunto de diseños usa su propio generador,
    np.random.default_rng(20250805), así que (a) y (b) son reproducibles por separado.
  - Escenario honesto: folds y holdout ~ Normal(0.70, 0.05), delta = 0, sigma_fold = 0.05.
    Las muestras se extraen con la misma forma que honest_max_z() de
    04_stds_screening_cutoffs.py: folds (N, M, K) y luego holdout (N, M), por celda.
  - Criterio de marcado: risk_level != "LOW" en la salida del motor, es decir,
    max_z > corte del percentil 95 de EMPIRICAL_SCREEN para (K, M).
  - Conjuntos de diseños:
      (a) las 12 celdas de la Tabla 5 del paper: K en {3,5,10,20} x |M| en {1,3,5}
      (b) los 4 diseños del paso [2] de 04_stds_screening_cutoffs.py:
          (3,1), (5,3), (10,1), (20,5)

Uso (desde la raíz del repositorio, con requirements_frozen.txt instalado):
    python revision_paper_codigo/reproducir_rango_stds.py

Salida: tabla en pantalla y revision_paper_codigo/reproducir_rango_stds_resultados.json
Este script no saca conclusiones; solo reporta tasas y rangos.
"""
import json
import sys
import time
import logging
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ScientificReports_revision" / "engine"))
logging.disable(logging.CRITICAL)
from stds_wrapper import StudyMetrics, run_stds  # noqa: E402

MASTER_SEED = 20250805
MU, SIGMA = 0.70, 0.05
N_SIM = 60000

DESIGNS = {
    "a_tabla5_12_celdas": [(k, m) for k in (3, 5, 10, 20) for m in (1, 3, 5)],
    "b_04_paso2_4_disenos": [(3, 1), (5, 3), (10, 1), (20, 5)],
}


def flag_rate_engine(rng, k, m, n):
    folds = rng.normal(MU, SIGMA, size=(n, m, k))
    hold = rng.normal(MU, SIGMA, size=(n, m))
    flagged = valid = 0
    for i in range(n):
        res = run_stds(StudyMetrics(
            cv_fold_values={f"m{j}": folds[i, j].tolist() for j in range(m)},
            holdout_values={f"m{j}": float(hold[i, j]) for j in range(m)}))
        if res.get("status") != "success":
            continue
        valid += 1
        flagged += res["risk_level"] != "LOW"
    return flagged, valid


def main():
    t0 = time.time()
    out = {"master_seed": MASTER_SEED, "n_sim": N_SIM, "mu": MU, "sigma_fold": SIGMA,
           "criterion": 'risk_level != "LOW" (max_z > p95 cutoff)', "sets": {}}
    for name, cells in DESIGNS.items():
        rng = np.random.default_rng(MASTER_SEED)
        rows = []
        print(f"\n[{name}]")
        for k, m in cells:
            f, v = flag_rate_engine(rng, k, m, N_SIM)
            rate = 100 * f / v
            se = 100 * np.sqrt((f / v) * (1 - f / v) / v)
            rows.append({"k_folds": k, "n_metrics": m, "flagged": f, "valid": v,
                         "rate_pct": round(rate, 3), "mc_se_pct": round(se, 3)})
            print(f"  K={k:>2} |M|={m}: {rate:5.2f}%  (±{se:.2f} EE MC, {f}/{v})", flush=True)
        rates = [r["rate_pct"] for r in rows]
        out["sets"][name] = {"cells": rows, "range_pct": [min(rates), max(rates)]}
        print(f"  rango: {min(rates):.2f}%–{max(rates):.2f}%")
    out["seconds"] = round(time.time() - t0, 1)
    (HERE / "reproducir_rango_stds_resultados.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTiempo: {out['seconds']} s")


if __name__ == "__main__":
    main()
