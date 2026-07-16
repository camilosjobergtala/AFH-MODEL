# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is C. A. Sjöberg Tala's personal research repository — not a product, not an org repo. It is
**not** a software product either — there is no package to build, no app to serve, no CI, and no
dependency manager beyond pip. It contains two unrelated bodies of work:

- **`ScientificReports_revision/`** — the current, active work: code, data, and reproducibility
  materials for the *Scientific Reports* manuscript "ECLIPSE v2.0: A Reproducible Scaffold for
  Severe Confirmatory Testing through Preregistration, Single-Shot Validation, and Static-Analysis
  Screening." Treat this directory as the primary thing to reason carefully about. This is the
  directory linked to reviewers of that submission — be extra conservative about changes here.
- Everything else is the author's separate, personal line of research on the AFH* model
  (Autopsychic Fold Hypothesis), unrelated to the manuscript above beyond an incidental naming
  overlap ("ECLIPSE" is reused as a script name in both, by historical coincidence, not shared
  logic):
  - **`Lab/`** — the earlier, exploratory research archive (AFH* model vs. IIT, EEG/PSG
    sleep-stage analyses, feature-filtering experiments). A much looser, notebook-style collection
    of one-off scripts with Spanish/English mixed naming (`NIVEL 0/1/2/3/4`, `FOLD`,
    `ECLIPSE 3.7/4.1.py`). Treat it as reference/archive, not code to refactor toward consistency
    with `ScientificReports_revision/`.
  - **`NeuroscienceOfConsciousness_revision/`** — a self-contained, reviewer-facing package
    (manuscript, formal model, code, results) for a separate, not-yet-submitted manuscript, "La
    convergencia temporal talámica intralaminar como mecanismo candidato de la presencia
    fenomenológica", curated out of `Lab/`. Its own README documents its layout; it references
    `Lab/` by relative path in prose and in one script's path resolution
    (`Lab/afh-predicciones-simulacion/evaluate_real_sleepedf_predictions.py` assumes `Lab/` hangs
    directly off the repo root) — keep that in mind before moving either directory.

Root-level `NISA'S CODE` is a joke file (the author's cat walked on the keyboard) — leave it as is.

## Setup and running

```bash
pip install -r ScientificReports_revision/requirements_frozen.txt   # pinned versions used in the manuscript
# or the looser root-level requirements.txt (numpy, networkx, matplotlib, scikit-learn, transformers, torch)
```

There is no build step, linter, or test runner configured (no `pytest.ini`, `Makefile`, `package.json`,
or CI workflow in the repo). "Tests" that exist are standalone verification scripts, run directly:

```bash
cd ScientificReports_revision/engine
python test_eis_wrapper.py     # prints wrapper sanity checks to stdout; not pytest, no assertions
```

### Reproducing the manuscript's studies

Run in order from `ScientificReports_revision/`; each script must fully complete before the next
(later scripts assume earlier ones ran, and `00_...` writes `requirements_frozen.txt` itself):

```bash
python code/00_environment_check.py           # verifies packages, writes logs/environment_*.json
python code/01_study1_eis_coherence.py        # Study 1: EIS internal coherence & weight stability
python code/02_study2_stds_characterization.py  # Study 2: STDS operating characteristics
python code/03_study3_code_auditor_benchmark.py # Study 3: code-auditor construct check
```

All three study scripts are deterministic under `MASTER_SEED = 20250805` (declared independently
at the top of each script, not imported — see Conventions below). Outputs land under
`ScientificReports_revision/outputs/study{1,2,3}/` as JSON summaries + CSVs, matching the
manuscript's figures/tables. Study 4 (`STUDY4_specification.md`) is a pre-registered specification
for *prospective* execution, not a script — there is nothing to run for it.

## Architecture: `ScientificReports_revision/`

```
engine/   the ECLIPSE v3.0 implementation (canonical logic) + thin in-memory wrappers
code/     study scripts (00-03) that call the wrappers and write outputs/
outputs/  generated results per study (do not hand-edit; regenerate by re-running code/)
```

### The canonical-engine-plus-wrapper pattern (the key architectural fact of this codebase)

`engine/eclipse_core.py` (~3,300 lines) is the single source of truth for all ECLIPSE v3.0 logic:
`EclipseIntegrityScore` (EIS), `StatisticalTestDataSnooping` (STDS), and `CodeAuditor` /
`StaticCodeAnalyzer`. Each of these classes expects to read its inputs from three specific JSON
files on disk (`SPLIT_IMMUTABLE.json`, `CRITERIA_BINDING.json`, `FINAL_RESULT.json`) via a
`framework` object passed into their constructor — this mirrors how the original interactive
ECLIPSE CLI tool works.

Because the study scripts need to run the *real* engine hundreds of times over synthetic protocols
(not interactively), `engine/{eis,stds,auditor}_wrapper.py` each implement a **minimal in-memory
adapter**: a small `_FrameworkAdapter`/`_Config` class exposing just the attributes the engine
class touches, plus a function that materializes the expected JSON files into a temp dir and calls
straight into the real `eclipse_core.py` class. The wrapper docstrings are explicit about *why*:
reimplementing the scoring logic a second time would create a second source of truth and risk
silent divergence from the engine actually described in the manuscript. **When changing scoring
behavior, always edit `eclipse_core.py`, never re-derive logic inside a wrapper or a study script.**

- `eis_wrapper.py` — `ProtocolSpec` (dataclass) → `score_protocol()` → calls `EclipseIntegrityScore`.
  Creates/deletes a fresh temp dir per call (correctness-first; called from Study 1).
- `stds_wrapper.py` — `StudyMetrics` (dataclass) → `run_stds()` → calls `StatisticalTestDataSnooping`.
  Deliberately reuses **one** temp file created at import time (`atexit`-cleaned) instead of a
  temp-dir-per-call, because Study 2 calls it a very large number of times and per-call temp-dir
  creation/deletion was a measured performance problem (especially under Windows AV scanning) —
  see the module docstring before changing this.
- `auditor_wrapper.py` — `audit_script(path)` → calls `CodeAuditor.audit_analysis_code()` on a real
  `.py` file on disk directly (no JSON materialization needed for this one).

### `engine/clean_engine_*.py`

`clean_engine_claims.py`, `clean_engine_pvalue.py`, `clean_engine_residuals.py` are one-shot,
already-applied text-cleanup scripts (each makes a `.bak*` backup and does exact-text or regex
substitution) used historically to strip inaccurate claims (e.g. a mentioned-but-never-implemented
Kolmogorov-Smirnov test, stray "p-value" wording for what is actually a z-score) out of
`eclipse_core.py`'s docstrings/comments. They are historical/idempotent utility scripts, not part
of the study pipeline — no need to run them unless doing similar documentation-accuracy cleanup.

### Study scripts (`code/`)

Each study script is heavily front-loaded with a docstring explaining precisely what the study
does and does **not** claim to show — read that docstring before modifying a study, since the
scope caveats are load-bearing for the manuscript's argument (e.g. Study 1's monotonicity check is
explicitly labeled "circular by construction," Study 3's detection results are explicitly labeled
as confirming designed behavior, not real-world generalization). Preserve these caveats in any
edits; don't silently strengthen a claim past what the study design supports.

## Conventions

- **`MASTER_SEED = 20250805`** (the manuscript's original submission date) is declared
  independently at the top of `00_environment_check.py` and again in each study script — it is a
  repeated literal, not an import, by convention throughout this codebase. Keep it consistent if
  you ever need to change it.
- Study scripts resolve their own `engine/` import path at runtime (`HERE.parent / "engine"` on
  `sys.path`), so `engine/` and `code/` must stay siblings.
- `np.std` in the wrappers is population std (`ddof=0`) to match `eclipse_core.py`'s own
  aggregation — don't switch to sample std (`ddof=1`) without checking both sides agree.
- Every generated artifact (provenance logs, frozen requirements, `outputs/*`) is written by code
  in the repo, not hand-maintained — regenerate rather than hand-edit.
- License is Apache-2.0 repo-wide (see root `LICENSE`); `eclipse_core.py` previously carried an
  inconsistent AGPL/commercial header, which the `clean_engine_*.py` scripts fixed to match.
