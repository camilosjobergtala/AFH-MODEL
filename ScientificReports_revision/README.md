# ECLIPSE v2.0 — Scientific Reports Revision: Reconstruction Workspace

This directory contains the reproducible reconstruction of the validation analyses
(Studies 1–4) for the revised ECLIPSE v2.0 manuscript submitted to *Scientific Reports*.

## Discipline (non-negotiable)

Every number that appears in the manuscript must be traceable to:

1. a **script** in `code/`,
2. a documented **seed** (master seed declared in `00_environment_check.py`),
3. an **output file** in `outputs/studyN/`,
4. a **log entry** in `logs/` recording environment and run metadata.

No number enters the manuscript unless this chain exists. Suspended values from the
original submission (d=3.54, AUC=0.991, STDS=89%, F1=0.908, ρ=0.68) are **not** used
unless a new script reproduces them.

## Structure

```
ScientificReports_revision/
├── README.md                    # this file
├── requirements_frozen.txt      # pinned versions (written by 00_environment_check.py)
├── engine/                      # ECLIPSE engine — single source of truth
│   ├── eis.py                   #   EIS calculator
│   ├── stds.py                  #   STDS permutation test
│   └── auditor.py               #   code auditor
├── code/                        # numbered reconstruction scripts (import engine/)
│   ├── 00_environment_check.py
│   ├── 01_study1_eis_coherence.py        # COH: coherence + weight stability
│   ├── 02_study2_stds_characterization.py
│   ├── 03_study3_code_auditor_benchmark.py
│   ├── 04_study4_incremental_validity.py
│   └── 05_generate_figures.py
├── data/
│   ├── raw/                     # source data (OSC, external inputs)
│   └── processed/               # intermediates (synthetic generated, etc.)
├── outputs/                     # traceable results, one folder per study
│   ├── study1/  study2/  study3/  study4/
├── figures/                     # Figures 2–5 regenerated
├── manuscript/                  # revised draft + versions
├── response/                    # point-by-point + cover letter
└── logs/                        # run logs: seed, timestamp, package versions
```

## Key rule: scripts import the engine, they do not reimplement it

The numbered scripts in `code/` **import** EIS, STDS, and the auditor from `engine/`.
Each instrument is defined exactly once. This guarantees that every reported number
has a single, unambiguous source — the failure mode that produced the original
untraceable values does not recur.

## Execution order

1. `00_environment_check.py`  — run first; verifies environment, writes provenance baseline.
2. `01_study1_eis_coherence.py` — Study 1 (most controllable): EIS component non-redundancy + weight stability + labeled synthetic sanity check.
3. `02_study2_stds_characterization.py` — STDS power curves under controlled conditions + boundary-condition violations.
4. `03_study3_code_auditor_benchmark.py` — auditor on synthetic scripts (designed violation categories).
5. `04_study4_incremental_validity.py` — **run last, most delicate.** Nested Models A–D (primary) + E (exploratory sensitivity). Determines the paper's empirical contribution.
6. `05_generate_figures.py` — regenerate Figures 2–5 from study outputs.

## Status

- Manuscript prose: revised, internally consistent (COH framing for Study 1).
- Results: all placeholders pending reconstruction. **Not submittable until filled.**
- Study 1 reconceived as internal coherence + weight stability (feasible solo; no raters required).
- Study 4 carries the external empirical weight; its outcome is reported honestly regardless of direction.

## Provenance

Master seed: declared in `00_environment_check.py`. License: Apache-2.0.
Author: Camilo Alejandro Sjöberg Tala, M.D. — ORCID 0009-0009-6052-0212.
