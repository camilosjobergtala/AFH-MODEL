"""
Evaluacion de las predicciones AFH (Seccion 6.1) sobre datos REALES de
Sleep-EDF ya computados por el autor -- a diferencia de
`simulate_afh_predictions.py`, que usa datos sinteticos.

Por que no se descarga Sleep-EDF de nuevo aqui: este entorno de ejecucion no
tiene ruta de red permitida hacia physionet.org (bloqueado por la politica
del proxy: CONNECT -> 403) y no hay PSG crudo cacheado localmente. Lo que
existe en este paquete de revision son los resultados YA calculados por el
autor sobre datos reales de Sleep-EDF Cassette (153 sujetos, columna
`subject_id`), copiados aqui desde el archivo exploratorio original del autor
(`Lab/AFH vs ECLIPSE/FOLD/...`, fuera de este paquete) para que la revision no
dependa de esa carpeta:

    outputs/nivel3_pac_optimization/raw_results_full.csv
    outputs/deltatfold_sleepedf/DeltaTFold_SleepEDF_RealData_*.json

Este script carga y re-analiza esos archivos (no re-procesa EDF crudo).
IMPORTANTE -- correccion respecto a una version anterior de este script: NI
`raw_results_full.csv` NI `DeltaTFold_SleepEDF_RealData_*.json` son tests
oficiales, pre-registrados, de "Prediccion 3" o "Prediccion 4" tal como estan
escritas en el paper. Son analisis exploratorios previos del autor (carpeta
Lab/ del repositorio, separada del manuscrito) que resultan tematicamente
analogos a esas predicciones, pero con sus propias hipotesis y metricas
propias -- no la operacionalizacion que el paper mismo reclama como
"pendiente" en su Seccion 6.3. Se reportan aqui con esa salvedad explicita:

  (a) `raw_results_full.csv` / `OPTIMIZATION_SUMMARY.json` (NIVEL 3 PAC): para
      153 sujetos reales de Sleep-EDF, testea si una combinacion de features
      (hjorth_complexity + dfa + petrosian_fd, umbralizada) cruza su umbral
      antes o despues de la transicion de etapa anotada en el hipnograma.
      Esto es analogo a la LOGICA de la Prediccion 3 (organizador cambia antes
      que la evidencia observable), no una instancia oficial de ella. Resultado
      real: precede en 55.6% de 1542 transiciones (vs 50% azar), p=6.5e-6 --
      significativo pero el propio `OPTIMIZATION_SUMMARY.json` marca
      `reaches_70: false`, es decir ni el autor lo considera un efecto fuerte.

  (b) `DeltaTFold_SleepEDF_RealData_*.json` (NIVEL 0 FOLD): NO mide covarianza
      H*/Nabla (eso seria Prediccion 4). Mide una hipotesis distinta y anterior,
      H1: Delta-t-Fold_vigilia > Delta-t-Fold_sueno, donde Delta-t-Fold es el
      group delay (retardo de fase, banda 2-9Hz) entre los canales EEG Fpz-Cz
      y Pz-Oz (`compute_delta_t_fold` en FOLD ECLIPSE.py). Con criterios
      pre-registrados propios (d>=0.30 y p<=0.05, ambos requeridos), el
      holdout dio d=0.219 (falla) y p=0.021 (pasa) -> 1/2 requeridos ->
      FALSIFICADA bajo esa pre-registracion especifica -- no bajo los terminos
      de la Prediccion 4 del paper. Se reporta como contexto relacionado, no
      como test de P4.

  (c) por que las Predicciones 1, 2, 5, 6 y 7 (y, en rigor, tambien 3 y 4 en su
      forma exacta del paper) NO son testables con este dataset: Sleep-EDF es
      EEG de superficie, pasivo, sin brazo de anestesia, sin manipulacion
      graduada experimental, sin reporte en primera persona, y sin ningun canal
      que resuelva actividad de los ILN (estructura subcortical profunda). Esto
      es estructural, no una limitacion de cobertura de sujetos.

No se encontro en el repositorio del autor el numero exacto que el paper cita
en su Seccion 6.3 (F1=0,031 vs criterio >=0,60; d de Cohen=2,44-2,79, N=153
para PAC delta-gamma) -- probablemente proviene de un analisis no archivado.
Nota de la revision: esa cifra especifica de Cohen's d ya no aparece en el
texto del manuscrito auditado para este paquete de revision (que solo dice
"tamano de efecto grande, consistente con hallazgos previos en la
literatura", sin numero fijo, y declara explicitamente que ese analisis no
quedo archivado formalmente) -- el hallazgo de esta nota ya esta reflejado
en el texto actual.

Uso:
    python evaluate_real_sleepedf_predictions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from simulate_afh_predictions import PredictionResult  # reuse the same report shape

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent  # .../NeuroscienceOfConsciousness_revision (self-contained review package)
NIVEL3_DIR = REPO_ROOT / "outputs" / "nivel3_pac_optimization"
DELTAFOLD_DIR = REPO_ROOT / "outputs" / "deltatfold_sleepedf"

BEST_COMBO = "hjorth_dfa_petrosian"  # combinacion ya seleccionada por el autor (mayor pct_hstar_first)


def real_prediction_3() -> PredictionResult:
    csv_path = NIVEL3_DIR / "raw_results_full.csv"
    df = pd.read_csv(csv_path)

    onset_col = f"onset_hstar_{BEST_COMBO}"
    first_col = f"hstar_first_{BEST_COMBO}"

    onset = df[onset_col].dropna()
    first_flag = df[first_col].dropna().astype(bool)

    t_stat, p_ttest = stats.ttest_1samp(onset, 0.0)
    n_precede = int(first_flag.sum())
    n_valid = int(len(first_flag))
    pct_precede = 100.0 * n_precede / n_valid
    binom_p = stats.binomtest(n_precede, n_valid, p=0.5, alternative="greater").pvalue

    # criterio "significativo pero debil": mejor que azar con p<0.05, sin llegar
    # al 70% que el propio autor definio como umbral practico en OPTIMIZATION_SUMMARY.json
    passed = bool(binom_p < 0.05 and pct_precede > 50.0)
    strong = bool(pct_precede >= 70.0)

    return PredictionResult(
        "Analogo-P3",
        "Ordenamiento temporal -- ANALOGO exploratorio, no test oficial de Prediccion 3 "
        "(Sleep-EDF real, transiciones de etapa vs onset de proxy H*)",
        "% transiciones con proxy H* precediendo a la transicion anotada",
        pct_precede,
        "> 50% con p < 0.05 (binomial) para 'confirmada debilmente'; >= 70% para 'fuerte' "
        "(umbral practico ya fijado por el propio autor en OPTIMIZATION_SUMMARY.json)",
        passed,
        {
            "n_transitions_validas": n_valid,
            "n_sujetos": int(df["subject_id"].nunique()),
            "media_onset_s": float(onset.mean()),
            "p_value_ttest_onset_vs_0": float(p_ttest),
            "p_value_binomial_vs_azar": float(binom_p),
            "combinacion_de_features": BEST_COMBO,
            "alcanza_umbral_fuerte_70pct": strong,
            "fuente": str(csv_path.relative_to(REPO_ROOT)),
            "salvedad": (
                "Analisis exploratorio previo del autor (Lab/, no parte del manuscrito), "
                "tematicamente analogo a la logica de Prediccion 3 pero no una "
                "operacionalizacion oficial de esa prediccion tal como esta escrita en el paper."
            ),
        },
    )


def real_deltatfold_context() -> dict:
    """Contexto de un analisis relacionado pero DISTINTO: no mide covarianza H*/Nabla
    (eso seria Prediccion 4). Mide group delay inter-canal bajo la hipotesis
    Delta-t-Fold_vigilia > Delta-t-Fold_sueno (ver FOLD ECLIPSE.py, compute_delta_t_fold)."""
    result_path = DELTAFOLD_DIR / "DeltaTFold_SleepEDF_RealData_FINAL_RESULT.json"
    criteria_path = DELTAFOLD_DIR / "DeltaTFold_SleepEDF_RealData_CRITERIA_BINDING.json"
    data = json.loads(result_path.read_text())
    criteria = json.loads(criteria_path.read_text())
    holdout = data["validation_summary"]["metrics"]
    return {
        "que_mide": (
            "Group delay (retardo de fase, banda 2-9Hz) entre canales EEG Fpz-Cz y Pz-Oz. "
            "NO es covarianza H*/Nabla -- no corresponde a la Prediccion 4 del paper."
        ),
        "fuente_resultado": str(result_path.relative_to(REPO_ROOT)),
        "fuente_criterios": str(criteria_path.relative_to(REPO_ROOT)),
        "criterios_requeridos": [c["description"] for c in criteria["criteria"] if c["is_required"]],
        "veredicto": f"FALSIFICADA bajo su propia pre-registracion "
                     f"({sum(1 for c in criteria['criteria'] if c['is_required'])} requeridos, "
                     f"solo 1 superado en holdout)",
        "cohens_d_holdout": holdout["Cohen's d (Wake vs Sleep)"],
        "umbral_d_requerido": 0.30,
        "p_value_holdout": holdout["p-value (one-tailed t-test)"],
        "n_holdout": data["validation_summary"]["n_holdout_samples"],
        "nota": (
            "No es un nuevo analisis: es el resultado ya archivado en el repositorio para esta "
            "hipotesis especifica (Delta-t-Fold). Se reporta como contexto relacionado, no como "
            "test de ninguna de las 7 predicciones enumeradas en la Seccion 6.1 del paper."
        ),
    }


NOT_TESTABLE = [
    ("P1", "Correlacion fenomenologica gradada",
     "Requiere manipulacion graduada (dosis/tACS/DBS) Y reporte en primera persona por ensayo. "
     "Sleep-EDF es registro pasivo sin ninguna manipulacion experimental ni reporte."),
    ("P2", "Disociacion cualitativa N3/anestesia",
     "Requiere un brazo de anestesia profunda para contrastar con N3. Sleep-EDF solo contiene "
     "sueno natural (W/N1/N2/N3/REM); no existe condicion de anestesia en este dataset."),
    ("P5", "Covarianza bajo manipulaciones de la retencion",
     "Requiere una tarea de reproduccion de duracion/intervalo lleno con reporte de vividez. "
     "Sleep-EDF no incluye ninguna tarea conductual ni reporte subjetivo."),
    ("P6", "Origen reflexivo de la senal lenta (decisiva)",
     "Requiere registro intracraneal con cobertura de los ILN (CM/Pf, CL) para rastrear la "
     "fuente de la senal lenta. Sleep-EDF es EEG de superficie (Fpz-Cz, Pz-Oz): no resuelve "
     "actividad talamica profunda bajo ninguna circunstancia, sin importar cuantos sujetos se sumen."),
    ("P7", "Diferencial de latencia en los ILN",
     "Misma razon que P6: exige registro intracraneal talamico con resolucion temporal fina; "
     "el EEG de cuero cabelludo no tiene acceso a esa fuente."),
]


def print_report() -> dict:
    print("=" * 100)
    print("EVALUACION SOBRE DATOS REALES DE SLEEP-EDF (ya computados en este repositorio)")
    print("=" * 100)

    p3 = real_prediction_3()
    print(f"[{p3.id}] {p3.name}")
    print(f"      {p3.metric_name} = {p3.metric_value:.2f}%")
    print(f"      criterio: {p3.threshold_desc}")
    for k, v in p3.detail.items():
        print(f"      - {k}: {v}")
    print(f"      -> {'CONFIRMADA (debilmente)' if p3.passed else 'FALLA'}"
          f"{'  [NO alcanza el umbral fuerte del 70%]' if p3.passed and not p3.detail['alcanza_umbral_fuerte_70pct'] else ''}")
    print("-" * 100)

    dtf = real_deltatfold_context()
    print("[Contexto-DeltaTFold] Analisis relacionado pero DISTINTO -- NO es un test de Prediccion 4")
    for k, v in dtf.items():
        print(f"      - {k}: {v}")
    print("-" * 100)

    print("PREDICCIONES DEL PAPER (Seccion 6.1) NO TESTABLES CON SLEEP-EDF, EN NINGUNA VERSION:")
    for pid, name, reason in NOT_TESTABLE:
        print(f"  [{pid}] {name}")
        print(f"        {reason}")
    print("=" * 100)
    print("Conclusion: ninguno de los dos analisis reales de arriba es un test oficial de las")
    print("Predicciones 3 o 4 tal como estan escritas en el paper -- son exploraciones previas del")
    print("autor (Lab/), tematicamente relacionadas pero con sus propias hipotesis y metricas. El")
    print("analogo de P3 se confirma solo debilmente (55.6% vs 50% azar, lejos del 70% que el propio")
    print("autor fijo como umbral practico); el analisis Delta-t-Fold (group delay, no H*/Nabla) fue")
    print("falsificado bajo su propia pre-registracion. Las 5 predicciones restantes del paper")
    print("(incluyendo las dos decisivas, P6 y P7) exigen registro intracraneal en los ILN que ningun")
    print("dataset de EEG de superficie puede proveer, sin importar cuantos sujetos se agreguen.")
    print("=" * 100)

    return {
        "analogo_P3": p3.to_dict(),
        "contexto_delta_t_fold_no_es_P4": dtf,
        "no_testable_con_sleepedf": [
            {"id": pid, "name": name, "razon": reason} for pid, name, reason in NOT_TESTABLE
        ],
    }


def main() -> None:
    report = print_report()
    outdir = HERE.parent / "outputs" / "predicciones_simulacion"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "afh_predictions_real_sleepedf.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nReporte JSON guardado en: {out_path}")


if __name__ == "__main__":
    main()
