# NEW — most recent / active exploratory scripts

Flat folder, no further subfolders. Filenames look like a flat numbered sequence but are actually
several separate version trees plus a few standalone utilities. Grouped here by lineage rather
than alphabetically.

## AFH-BETA PILOT line

`ada.py` (v1.2, ancestor) → `3.py` (v1.3.2) → `3.2.py` ("HOLDOUT VALIDATION v2.2.3 FINAL") →
`3.2.1.py` (sub-analysis extracting B1/B2/B3 metrics from the holdout results).

`V2.2.1.py` and `HOLD OUT VALIDATION V2.2.1.py` are **different scripts despite the similar
name** — internally versioned v1.3.3 and v2.1 respectively, different content. Not a duplicate
pair; don't merge or dedupe them by the name alone.

## Multivariate Regression Analysis line

`3.3.1_notes.md` (renamed from `3.3.1.PY` — that file was never Python; it's Markdown/LaTeX
verification notes that had a misleading `.PY` extension) → `3.3.2.py` (v2.1) → `3.3.3.py` (v2.2)
→ `3.3.4.py` (v2.2, same version label as `3.3.3.py` — the two differ only in LaTeX table
formatting and plot cosmetics). **`3.3.4.py` is the more current/polished of the pair**; both are
kept.

## AFH Convergence Experiment

`afh_experiment_complete.py` ("(PATCHED)" version) → `2.py` ("AFH CONVERGENCE EXPERIMENT v2.5.1
FINAL", **most current**). A predecessor/successor pair despite the generic `2.py` name.

## Framework and utilities

- `eclipse_v4.py` — this folder's own ECLIPSE v4.0 framework copy (distinct from `FOLD/`'s v3.0
  copies of `eclipse_v3.py`).
- `debug.py`, `finddata.py` — small one-off path-search utilities, not part of any version chain.
- `figures.py` — generates `fig_correlations.{pdf,png}` and `fig_distributions.{pdf,png}` (Stage 1
  registered-report figures). **Unrelated in content** to `FOLD/figures.py` despite the identical
  filename — a coincidental name collision, not a duplicate.
