# ECLIPSE v2.0 — Reproducibility materials

Code, data, and reproducibility materials for the *Scientific Reports* manuscript
**"ECLIPSE v2.0: A Reproducible Scaffold for Severe Confirmatory Testing through Preregistration, Single-Shot Validation, and Static-Analysis Screening"** (C. A. Sjöberg Tala).

## Layout

- **`code/`** — analysis scripts:
  - `00_environment_check.py` — verifies the environment and prints package versions.
  - `01_study1_eis_coherence.py` — Study 1: EIS internal coherence and weight stability.
  - `02_study2_stds_characterization.py` — Study 2: STDS operating characteristics under the *fixed* `max_z > 2` rule used in the first submission. This is the analysis that motivated the change described below; it is retained unchanged so the problem it documents remains inspectable.
  - `03_study3_code_auditor_benchmark.py` — Study 3: code-auditor construct check (including the adversarial near-miss probe).
  - `04_stds_screening_cutoffs.py` — Study 2b: derivation of the empirical STDS screening cutoffs now used by the engine.
  - `05_figures.py` — regenerates Figures 2, 3, 3b and 4 from the deposited outputs. Runs no simulation: every plotted value is read from `outputs/`, so the figures cannot drift from the reported numbers.
- **`engine/`** — the ECLIPSE implementation and instrument wrappers (`eclipse_core.py`, `eis_wrapper.py`, `stds_wrapper.py`, `auditor_wrapper.py`) and unit tests (`test_eis_wrapper.py`). The wrappers are the canonical entry points used by every study script, so no analysis reimplements the scoring logic.
- **`outputs/`** — generated results (`study1/`, `study2/`, `study2b/`, `study3/`): summary JSON files and per-analysis CSVs, plus `figures/` (PNG at 600 dpi and PDF).
- **`STUDY4_specification.md`** — Study 4: the pre-specified nested-model specification (for prospective execution) and the cited external-evidence table. Study 4 is **not** an executed retrospective regression — see manuscript §3.4.
- **`requirements_frozen.txt`** — pinned package versions.

## Reproduce

```bash
pip install -r requirements_frozen.txt
python code/00_environment_check.py
python code/01_study1_eis_coherence.py
python code/02_study2_stds_characterization.py
python code/03_study3_code_auditor_benchmark.py
python code/04_stds_screening_cutoffs.py
python code/05_figures.py
```

All studies are deterministic under the fixed seed recorded in each script; outputs are written under `outputs/` and correspond to the figures and tables reported in the manuscript.

## Changes in this revision (instrument version 3.1)

Two reviewer requests required changes to the instruments themselves, not only to the manuscript text.

**EIS leakage component (Reviewer 1, Major 1).** The four-band step function on the standardized development-to-holdout discrepancy was discontinuous, non-monotone (a holdout far *below* the development mean received a higher risk value than an ordinary gap), and could not reach 0 or 1. It is replaced by a monotone continuous function,

```
z    = (holdout − dev_mean) / dev_std
r(z) = logistic((z − LEAK_Z0) / LEAK_SCALE),   LEAK_Z0 = 1.0, LEAK_SCALE = 0.5
S_leak = 1 − mean_m r(z_m)
```

`LEAK_Z0` and `LEAK_SCALE` are normative screening choices, not calibrated thresholds, and are exposed as documented class constants.

Reviewer 1 also asked that a *zero* development-to-holdout gap lower the score. Simulation of the honest scenario shows that a near-zero standardized gap occurs in roughly 12% of honest validations and a near-exact match in roughly 14%, so treating it as strongly suspicious would penalize ordinary honest work. The new function lowers the score monotonically as the gap shrinks (S_leak = 0.98 at z = −1, 0.88 at z = 0, 0.50 at z = +1) without treating z = 0 as anomalous. This is a partial, reasoned departure from the request; see the point-by-point response.

**STDS screening rule (Reviewer 1, Major 2; Reviewer 2, point 1).** All nominal-significance machinery is removed from the engine: no `alpha`, no `z_critical = Φ⁻¹(1 − α/2)`, no `is_significant`, no p-value equivalences. The `alpha` argument is accepted and ignored with a warning so existing call sites do not break. The fixed cutoffs of 2 and 3 are replaced by `EMPIRICAL_SCREEN`: the 95th and 99th percentiles of `max_z` under the Study 2 honest scenario, tabulated by (number of folds, number of metrics) and derived by `04_stds_screening_cutoffs.py`. The honest-scenario flag rate is therefore 5% and 1% by construction, verified end-to-end through the engine at 4.4%–5.0%. Under the previous fixed rule it ranged from 5.1% to 54.4% depending on design.

**Two limits that survive this change**, stated in the engine docstring and in `04_stds_screening_cutoffs.py`:

1. The calibration is *self-consistent, not independent* — the cutoffs come from the same generative model the engine is then screened against.
2. The simulation draws folds independently; real cross-validation folds are overlapping fits on shared data. The honest-scenario rate for a real study is therefore expected to **exceed** 5%; the tabulated value is a lower bound.

The EIS report's leakage advisory was recalibrated for the same reason: under the new component the previous fixed cutoffs (0.5 / 0.7) would have fired on 23% of honest studies. Advisory thresholds are now honest-scenario percentiles by fold count (`LEAK_HONEST_Q`), firing at ~5% and ~1%.

Study 3 and the code auditor are unchanged; Reviewer 2's point (b) was already satisfied — a flag is documented throughout as "this pattern is present", not "a violation is confirmed", with the 0% near-miss specificity cited in the class docstring and in the generated report.

## Scope notes

- Studies 1–3 are **construct checks** under controlled synthetic conditions: they characterize each instrument's internal behavior, **not** external validity (manuscript §3, §4.4).
- Study 4 is a pre-specified prospective test plus cited published evidence; it is not an executed analysis (see `STUDY4_specification.md`).

License: Apache License 2.0 (see the repository-level `LICENSE`).
