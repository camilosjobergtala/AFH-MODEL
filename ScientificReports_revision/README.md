# ECLIPSE v2.0 — Reproducibility materials

Code, data, and reproducibility materials for the *Scientific Reports* manuscript
**"ECLIPSE v2.0: A Reproducible Scaffold for Severe Confirmatory Testing through Preregistration, Single-Shot Validation, and Static-Analysis Screening"** (C. A. Sjöberg Tala).

## Layout

- **`code/`** — analysis scripts:
  - `00_environment_check.py` — verifies the environment and prints package versions.
  - `01_study1_eis_coherence.py` — Study 1: EIS internal coherence and weight stability.
  - `02_study2_stds_characterization.py` — Study 2: STDS operating characteristics.
  - `03_study3_code_auditor_benchmark.py` — Study 3: code-auditor construct check (including the adversarial near-miss probe).
- **`engine/`** — the ECLIPSE implementation and instrument wrappers (`eclipse_core.py`, `eis_wrapper.py`, `stds_wrapper.py`, `auditor_wrapper.py`) and unit tests (`test_eis_wrapper.py`).
- **`outputs/`** — generated results for each study (`study1/`, `study2/`, `study3/`): summary JSON files and per-analysis CSVs.
- **`STUDY4_specification.md`** — Study 4: the pre-specified nested-model specification (for prospective execution) and the cited external-evidence table. Study 4 is **not** an executed retrospective regression — see manuscript §3.4.
- **`requirements_frozen.txt`** — pinned package versions.

## Reproduce

```bash
pip install -r requirements_frozen.txt
python code/00_environment_check.py
python code/01_study1_eis_coherence.py
python code/02_study2_stds_characterization.py
python code/03_study3_code_auditor_benchmark.py
```

Studies 1–3 are deterministic under the fixed seed recorded in each script; outputs are written under `outputs/` and correspond to the figures and tables reported in the manuscript.

## Scope notes

- Studies 1–3 are **construct checks** under controlled synthetic conditions: they characterize each instrument's internal behavior, **not** external validity (manuscript §3, §4.4).
- Study 4 is a pre-specified prospective test plus cited published evidence; it is not an executed analysis (see `STUDY4_specification.md`).

License: Apache License 2.0 (see the repository-level `LICENSE`).