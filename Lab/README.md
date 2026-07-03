# Lab — exploratory research archive

This folder contains the author's **earlier and separate** exploratory research on the AFH*
model, distinct from the *Scientific Reports* manuscript materials in
[`../ScientificReports_revision/`](../ScientificReports_revision/) (see the root
[`README.md`](../README.md)). Nothing here is part of that manuscript.

This is an archive, not a maintained codebase: scripts accumulate as numbered/versioned
snapshots (e.g. `1.py` → `4.py`, `v2` → `v3.2` → `v4`), and older versions are kept alongside
newer ones intentionally, as a record of how each analysis evolved. Superseded, near-duplicate,
and one-off debug scripts are **kept, not deleted** — see each subfolder's README for which
version of a given lineage is the current/final one.

Generated outputs (JSON/CSV/PNG/HTML reports) are committed alongside the source script that
produced them, in the same folder. This mirrors the convention used in
`ScientificReports_revision/outputs/`: outputs are part of the scientific record of a run, not
build artifacts to `.gitignore`.

## Map

| Folder | Subject |
|---|---|
| [`AFH vs ECLIPSE/`](AFH%20vs%20ECLIPSE/) | The main falsification-framework line: `FOLD/` (NIVEL 0–4 test ladder), `NEW/` (most recent/active scripts), `legacy/` (ECLIPSE 3.7 → 4.1, already documented in its own README). |
| [`HORIZON/`](HORIZON/) | Feature-filtering / classification experiments (Wake vs N3, catch22/HCTSA feature exploration) for a separate Registered Report line. |
| [`IIT vs ECLIPSE/`](IIT%20vs%20ECLIPSE/) | AFH*/ECLIPSE vs. Integrated Information Theory (IIT) falsification comparison, versions V2 → V3.2 → v4. |
| [`MODELING FOLD/`](MODELING%20FOLD/) | A single orphan script, see below. |

See `AFH vs ECLIPSE/README.md`, `HORIZON/README.md`, and `IIT vs ECLIPSE/README.md` for
per-folder version details.

## Loose files directly under `Lab/`

- **`MODELING FOLD/0.4.py`** — despite the filename, its own header identifies it as "MODELO AFH
  v0.5" (changes relative to v0.4). Implements Granger-causality-based nabla/autoreferentiality
  metrics for the AFH model. Not part of any versioned chain elsewhere in `Lab/` — a standalone
  snapshot.
- **`analisis.py`** — a small (666 B) one-off script comparing anesthesia vs. sleep classification
  deltas from an Excel file at a hardcoded local path (`G:\Mi unidad\...`). Not reproducible as-is
  without that file; kept as a record of the analysis, not a reusable tool.
