"""
Simulacion de las 7 predicciones de AFH (Seccion 6.1 de "Presencia fenomenologica
y el talamo intralaminar", Sjoberg Tala, 2026).

IMPORTANTE - que es esto y que NO es:
    Esto es una simulacion pedagogica con datos sinteticos, no un analisis sobre
    registros ILN reales. El paper es explicito en que las Predicciones 6/7
    requieren registro intracraneal talamico (Seccion 6.3: "pendiente"). Lo que
    este script hace es instanciar el mecanismo propuesto (via rapida 8-30ms +
    retorno lento tipo copia eferente 200-600ms, convergencia refleja en H*/V)
    como un generador de datos, y luego aplicar exactamente las pruebas
    estadisticas que el texto especifica para cada prediccion. Sirve para:
      1) verificar que las predicciones son operacionalizables y falsables
         (se puede fallar el test si se generan datos bajo el regimen nulo), y
      2) tener un banco de pruebas antes de intentarlo con datos reales.

Regimenes (--regime):
    afh_true          : la senal lenta ES una transformacion causal retardada
                         de la senal rapida (convergencia refleja, Tipo B).
    null_heterogeneous: la senal lenta es una fuente independiente que solo
                         coincide con la rapida (convergencia heterogenea,
                         Tipo A) -- el escenario que el paper dice que
                         reclasificaria Nabla y devolveria el argumento del
                         zombi a su forma estandar (Seccion 4.3, 6.1 Pred. 6).

Uso:
    python simulate_afh_predictions.py --regime afh_true
    python simulate_afh_predictions.py --regime null_heterogeneous --plots
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal, stats

FS = 200.0  # Hz simulados (dt = 5 ms) -- resolucion suficiente para latencias 5-700 ms
TRUE_LAG_MS = 300.0  # punto medio del rango 150-550 ms predicho (Prediccion 7)


# ---------------------------------------------------------------------------
# Generador de senales: mecanismo AFH (via rapida + retorno lento)
# ---------------------------------------------------------------------------

def _bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    b, a = signal.butter(4, [low / (FS / 2), high / (FS / 2)], btype="band")
    return signal.filtfilt(b, a, x)


def _lowpass(x: np.ndarray, high: float) -> np.ndarray:
    b, a = signal.butter(4, high / (FS / 2), btype="low")
    return signal.filtfilt(b, a, x)


def make_fast_signal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Entrada reticular ascendente rapida (8-30 ms -> banda ~30-45 Hz proxy)."""
    raw = rng.standard_normal(n)
    return _bandpass(raw, 30.0, 45.0)


def signal_envelope(x: np.ndarray) -> np.ndarray:
    """Envolvente (tasa de disparo / potencia instantanea) de una senal oscilatoria.
    Lo que se propaga causalmente hacia el retorno lento es esta envolvente, no la
    fase cruda -- por eso las pruebas de acoplamiento rapido-lento (P1, P6, P7)
    comparan la envolvente de F contra S, en vez de F crudo contra S."""
    env = np.abs(signal.hilbert(x))
    return (env - env.mean()) / (env.std() + 1e-9)


def _delay_kernel(lag_ms: float, width_ms: float = 40.0) -> np.ndarray:
    """Nucleo gaussiano causal centrado en lag_ms: representa la ventana de
    integracion del retorno tipo copia eferente (200-600 ms), en lugar de un
    lowpass zero-phase que difumina el retardo sobre un rango demasiado ancho."""
    length = int(round((lag_ms + 4 * width_ms) * FS / 1000.0))
    t_ms = np.arange(length) * 1000.0 / FS
    kernel = np.exp(-0.5 * ((t_ms - lag_ms) / width_ms) ** 2)
    return kernel / kernel.sum()


def make_slow_return(fast: np.ndarray, rng: np.random.Generator, *,
                      reflexive: bool, lag_ms: float = TRUE_LAG_MS,
                      coupling: float = 1.0, noise_sd: float = 0.6) -> np.ndarray:
    """Retorno lento tipo copia eferente (200-600 ms).

    reflexive=True  -> convergencia refleja (Tipo B): S(t) deriva causalmente
                        del envolvente de F retardado ~lag_ms via una ventana
                        de integracion angosta, tal como propone el Pliegue
                        Autopsiquico (la senal lenta es la propia salida
                        elaborada del sistema, no una fuente externa).
    reflexive=False -> convergencia heterogenea (Tipo A): S es una fuente
                        independiente de banda comparable, solo acoplada por
                        azar -- el nulo que el paper dice que falsaria la
                        lectura constitutiva (Prediccion 6).
    """
    n = len(fast)
    if reflexive:
        envelope = np.abs(signal.hilbert(fast))
        envelope = (envelope - envelope.mean()) / (envelope.std() + 1e-9)
        kernel = _delay_kernel(lag_ms)
        driven = np.convolve(envelope, kernel, mode="full")[:n]
        driven = driven / (driven.std() + 1e-9)
        slow = coupling * driven + noise_sd * rng.standard_normal(n)
    else:
        indep = rng.standard_normal(n)
        slow = _lowpass(indep, 4.0)
        slow = slow / (slow.std() + 1e-9) + noise_sd * rng.standard_normal(n)
    return slow


def _lagmat(v: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(v)
    return np.column_stack([v[max_lag - l: n - l] for l in range(1, max_lag + 1)])


def granger_like_test(cause: np.ndarray, effect: np.ndarray, max_lag: int):
    """F-test lineal tipo Granger: ¿agregar retardos de `cause` reduce la
    varianza residual al predecir `effect` mas alla de su propio pasado?
    Aproximacion simple (OLS + F-test anidado), suficiente para el proposito
    ilustrativo de la Prediccion 6 (rastreo de fuente)."""
    y = effect[max_lag:]
    X_r = np.column_stack([np.ones(len(y)), _lagmat(effect, max_lag)])
    X_u = np.column_stack([X_r, _lagmat(cause, max_lag)])
    beta_r, *_ = np.linalg.lstsq(X_r, y, rcond=None)
    beta_u, *_ = np.linalg.lstsq(X_u, y, rcond=None)
    rss_r = float(np.sum((y - X_r @ beta_r) ** 2))
    rss_u = float(np.sum((y - X_u @ beta_u) ** 2))
    df1, df2 = max_lag, len(y) - X_u.shape[1]
    f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2 + 1e-12)
    p_value = float(1 - stats.f.cdf(f_stat, df1, df2))
    variance_explained = (rss_r - rss_u) / (rss_r + 1e-12)
    return f_stat, p_value, variance_explained


@dataclass
class PredictionResult:
    id: str
    name: str
    metric_name: str
    metric_value: float
    threshold_desc: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = bool(self.passed)
        return d


# ---------------------------------------------------------------------------
# Prediccion 1 -- Correlacion fenomenologica gradada
# ---------------------------------------------------------------------------

def predict_1(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n_trials = 200
    theta = rng.uniform(0, 1, n_trials)  # intensidad de manipulacion gradada (dosis/tACS/DBS)
    fold_strength = np.empty(n_trials)
    for i, th in enumerate(theta):
        fast = make_fast_signal(400, rng)
        slow = make_slow_return(fast, rng, reflexive=reflexive, coupling=th)
        lag = int(round(TRUE_LAG_MS * FS / 1000.0))
        env = signal_envelope(fast)
        fold_strength[i] = abs(np.corrcoef(env[:-lag], slow[lag:])[0, 1]) if reflexive else \
            abs(np.corrcoef(env, slow)[0, 1])
    # Reporte en primera persona: bajo AFH, R sigue gradualmente a Nabla; bajo un
    # enfoque de acceso puro, R solo responde a un umbral de desempeno (escalon).
    if reflexive:
        report = 3.0 * fold_strength + rng.normal(0, 0.5, n_trials)
    else:
        report = 3.0 * (theta > 0.5).astype(float) + rng.normal(0, 0.5, n_trials)
    r, p = stats.pearsonr(fold_strength, report)
    passed = bool(r > 0.3 and p < 0.05)
    return PredictionResult(
        "P1", "Correlacion fenomenologica gradada", "r(Nabla, reporte)", float(r),
        "r > 0.30, p < 0.05 (umbral ilustrativo; el texto pide covariacion gradada, no un valor fijo)",
        passed, {"p_value": p, "n_trials": n_trials},
    )


# ---------------------------------------------------------------------------
# Prediccion 2 -- Disociacion cualitativa N3/anestesia
# ---------------------------------------------------------------------------

def predict_2(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n = 60
    # N3: horizonte preservado (H alto) con volumen colapsado (V~0).
    # Anestesia: horizonte abolido (H bajo) con volumen tambien colapsado.
    # Si AFH es correcto, H discrimina aunque V sea igual en ambos grupos.
    if reflexive:
        h_n3 = rng.normal(0.75, 0.08, n)
        h_anes = rng.normal(0.20, 0.10, n)
    else:
        # bajo el nulo, el "horizonte" no es mas que ruido de V colapsado: N3 y
        # anestesia se vuelven indistinguibles (la condicion de falsacion (4)
        # del Sec. 6.4: "N3 resulta indistinguible de la anestesia").
        h_n3 = rng.normal(0.30, 0.20, n)
        h_anes = rng.normal(0.28, 0.20, n)
    d = (h_n3.mean() - h_anes.mean()) / np.sqrt((h_n3.var() + h_anes.var()) / 2)
    _, p = stats.ttest_ind(h_n3, h_anes)
    passed = bool(abs(d) > 0.8 and p < 0.05)
    return PredictionResult(
        "P2", "Disociacion cualitativa N3/anestesia", "Cohen's d (H_N3 vs H_anestesia)", float(d),
        "|d| > 0.80 (efecto grande) y p < 0.05 -- deben diferir pese a V~0 en ambos",
        passed, {"p_value": p, "n_per_group": n},
    )


# ---------------------------------------------------------------------------
# Prediccion 3 -- Ordenamiento temporal durante la emergencia
# ---------------------------------------------------------------------------

def predict_3(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n_trials = 80
    lead_ms = np.empty(n_trials)
    for i in range(n_trials):
        # rampas de recuperacion con ruido de trial; bajo AFH, H*/Nabla cruzan
        # su umbral antes que la senal conductual (ignicion). Bajo el nulo
        # (enfoques de ignicion estandar), los cambios son simultaneos.
        true_lead = 300.0 if reflexive else 0.0
        onset_h = rng.normal(5000 - true_lead, 150)
        onset_behavior = rng.normal(5000, 150)
        lead_ms[i] = onset_behavior - onset_h
    t_stat, p = stats.ttest_1samp(lead_ms, 0.0)
    passed = bool(lead_ms.mean() > 50 and p < 0.05)
    return PredictionResult(
        "P3", "Ordenamiento temporal en la emergencia", "media(onset_conducta - onset_H*) [ms]",
        float(lead_ms.mean()),
        "media > 50 ms y p < 0.05 -- H*/Nabla deben cambiar antes que la evidencia conductual",
        passed, {"p_value": p, "n_trials": n_trials},
    )


# ---------------------------------------------------------------------------
# Prediccion 4 -- Covarianza H*/Nabla dentro de estados con horizonte presente
# ---------------------------------------------------------------------------

def predict_4(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n_per_state = 40
    states = ["Wake", "N1", "N2", "N3", "REM"]
    h_all, fold_all = [], []
    for state in states:
        latent = rng.normal(0, 1, n_per_state)
        if reflexive:
            h = 0.5 + 0.3 * latent + rng.normal(0, 0.15, n_per_state)
            fold = 0.5 + 0.3 * latent + rng.normal(0, 0.15, n_per_state)
        else:
            h = rng.normal(0.5, 0.3, n_per_state)
            fold = rng.normal(0.5, 0.3, n_per_state)
        h_all.append(h)
        fold_all.append(fold)
    h_all, fold_all = np.concatenate(h_all), np.concatenate(fold_all)
    r, p = stats.pearsonr(h_all, fold_all)
    passed = bool(r > 0.50 and p < 0.05)
    return PredictionResult(
        "P4", "Covarianza H*/Nabla con horizonte presente", "r(H*, Nabla)", float(r),
        "r > 0.50, p < 0.05 (umbral explicito del texto, Prediccion 4)",
        passed, {"p_value": p, "n_total": len(h_all), "states": states},
    )


# ---------------------------------------------------------------------------
# Prediccion 5 -- Covarianza bajo manipulaciones de la retencion
# ---------------------------------------------------------------------------

def predict_5(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n_trials = 100
    retention_manip = rng.uniform(0, 1, n_trials)  # p.ej. duracion del intervalo lleno
    if reflexive:
        slow_band_power = 2.0 * retention_manip + rng.normal(0, 0.4, n_trials)
        fast_band_power = rng.normal(0, 0.4, n_trials)  # control: no deberia covariar
        vividness = 2.0 * retention_manip + rng.normal(0, 0.4, n_trials)
    else:
        # mapeo meramente analogico: ninguna via covaria sistematicamente
        slow_band_power = rng.normal(0, 0.4, n_trials)
        fast_band_power = rng.normal(0, 0.4, n_trials)
        vividness = rng.normal(0, 0.4, n_trials)
    r_slow, p_slow = stats.pearsonr(slow_band_power, vividness)
    r_fast, p_fast = stats.pearsonr(fast_band_power, vividness)
    passed = bool(r_slow > 0.3 and p_slow < 0.05 and abs(r_slow) > abs(r_fast) + 0.2)
    return PredictionResult(
        "P5", "Covarianza bajo manipulacion de retencion", "r_lenta - r_rapida (vs vividez)",
        float(r_slow - r_fast),
        "r_lenta > 0.30 (p<0.05) y sustancialmente mayor que r_rapida -- disociacion, no mero analogo",
        passed, {"r_slow": r_slow, "p_slow": p_slow, "r_fast": r_fast, "p_fast": p_fast},
    )


# ---------------------------------------------------------------------------
# Prediccion 6 -- Origen reflexivo de la senal lenta (LA PRUEBA DECISIVA)
# ---------------------------------------------------------------------------

def predict_6(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n = 4000
    fast = make_fast_signal(n, rng)
    slow = make_slow_return(fast, rng, reflexive=reflexive, coupling=1.0, noise_sd=0.6)
    env = signal_envelope(fast)
    max_lag = int(round(700 * FS / 1000.0))  # cubre hasta 700 ms de retardo
    f_stat, p_value, var_explained = granger_like_test(env, slow, max_lag=max_lag)
    passed = bool(p_value < 0.05 and var_explained > 0.05)
    tipo = "Tipo B (reflexivo, confirma Nabla)" if passed else "Tipo A (heterogeneo, falsaria Nabla)"
    return PredictionResult(
        "P6", "Origen reflexivo de la senal lenta (decisiva)", "fraccion de varianza explicada por F->S",
        float(var_explained),
        "p < 0.05 y varianza explicada > 5% -- rastreo de fuente hasta la entrada rapida precedente",
        passed, {"f_stat": f_stat, "p_value": p_value, "clasificacion": tipo, "max_lag_ms": 700},
    )


# ---------------------------------------------------------------------------
# Prediccion 7 -- Diferencial de latencia en los ILN
# ---------------------------------------------------------------------------

def predict_7(rng: np.random.Generator, reflexive: bool) -> PredictionResult:
    n = 4000
    fast = make_fast_signal(n, rng)
    slow = make_slow_return(fast, rng, reflexive=reflexive, coupling=1.0, noise_sd=0.6)
    env = signal_envelope(fast)
    max_lag_samples = int(round(700 * FS / 1000.0))
    lags = np.arange(0, max_lag_samples)
    corrs = np.array([
        np.corrcoef(env[:n - lag] if lag else env, slow[lag:] if lag else slow)[0, 1]
        for lag in lags
    ])
    peak_idx = int(np.argmax(np.abs(corrs)))
    peak_lag_ms = float(lags[peak_idx] * 1000.0 / FS)
    peak_r = float(corrs[peak_idx])
    passed = bool(150.0 <= peak_lag_ms <= 550.0 and abs(peak_r) > 0.2)
    return PredictionResult(
        "P7", "Diferencial de latencia rapido-lento en los ILN", "lag del pico de correlacion [ms]",
        peak_lag_ms,
        "150-550 ms con |r_pico| > 0.20 -- ventana predicha por el texto (Seccion 6.1)",
        passed, {"peak_r": peak_r},
    )


# ---------------------------------------------------------------------------
# Orquestacion, reporte y (opcional) figuras
# ---------------------------------------------------------------------------

PREDICTIONS = [predict_1, predict_2, predict_3, predict_4, predict_5, predict_6, predict_7]


def run_all(regime: str, seed: int) -> list[PredictionResult]:
    reflexive = regime == "afh_true"
    rng = np.random.default_rng(seed)
    return [fn(rng, reflexive) for fn in PREDICTIONS]


def print_report(results: list[PredictionResult], regime: str) -> None:
    print("=" * 100)
    print(f"SIMULACION AFH -- Predicciones Seccion 6.1  |  regimen: {regime}")
    print("=" * 100)
    for r in results:
        mark = "PASA" if r.passed else "FALLA"
        print(f"[{r.id}] {r.name}")
        print(f"      {r.metric_name} = {r.metric_value:.4f}  |  criterio: {r.threshold_desc}")
        print(f"      -> {mark}")
        print("-" * 100)
    n_pass = sum(r.passed for r in results)
    p6 = next(r for r in results if r.id == "P6")
    print(f"RESUMEN: {n_pass}/{len(results)} predicciones confirmadas.")
    if not p6.passed:
        print("Prediccion 6 (decisiva) NO se confirma -> bajo la propia logica del texto (Sec. 4.3/6.1),")
        print("Nabla se reclasifica como Tipo A (heterogeneo) y la lectura constitutiva se abandona")
        print("en favor de una interpretacion causal o correlacional.")
    elif n_pass >= 6:
        print("Consistente con la lectura constitutiva propuesta por AFH (dato sintetico, no un hallazgo real).")
    else:
        print("Evidencia mixta: revisar predicciones individuales antes de sacar conclusiones.")
    print("=" * 100)
    print("Recordatorio: estos son datos sinteticos generados para ilustrar el mecanismo del")
    print("texto y probar que las 7 predicciones son computables y falsables. No sustituyen")
    print("registro intracraneal en los ILN (ver Seccion 6.3 del paper).")


def save_plots(results: list[PredictionResult], regime: str, outdir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()
    for ax, r in zip(axes, results):
        color = "#2a9d8f" if r.passed else "#e76f51"
        ax.bar(["obtenido"], [r.metric_value], color=color)
        ax.set_title(f"{r.id}: {r.name}", fontsize=9)
        ax.set_ylabel(r.metric_name, fontsize=8)
        ax.text(0, r.metric_value, f"{r.metric_value:.3f}", ha="center",
                va="bottom" if r.metric_value >= 0 else "top", fontsize=8)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle(f"Simulacion AFH -- regimen: {regime}", fontsize=12)
    fig.tight_layout()
    outpath = outdir / f"afh_predictions_{regime}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=["afh_true", "null_heterogeneous"], default="afh_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent / "outputs"))
    parser.add_argument("--plots", action="store_true", help="Guardar figura resumen en PNG")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_all(args.regime, args.seed)
    print_report(results, args.regime)

    json_path = outdir / f"afh_predictions_{args.regime}.json"
    json_path.write_text(json.dumps(
        {"regime": args.regime, "seed": args.seed,
         "predictions": [r.to_dict() for r in results]}, indent=2, ensure_ascii=False,
    ))
    print(f"\nReporte JSON guardado en: {json_path}")

    if args.plots:
        png_path = save_plots(results, args.regime, outdir)
        print(f"Figura guardada en: {png_path}")


if __name__ == "__main__":
    main()
