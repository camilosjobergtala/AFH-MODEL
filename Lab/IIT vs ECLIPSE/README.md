# IIT vs ECLIPSE — AFH*/ECLIPSE vs. Integrated Information Theory comparison

A linear version progression across three folders, confirmed from each script's own header:

- **`V2/IIT vs ECLIPSE v2.py`** — baseline comparison.
- **`V3.2/IIT vs ECLIPSE v3.2.py`** — adds an LLM code auditor, multi-Φ computation, thermal
  monitoring.
- **`v4/f.py`** — **most current/complete version**, a fusion of v3.0 methodology with the
  v3.2.0 application. Two things worth knowing if you're looking for it: the file internally
  self-identifies as "ECLIPSE v3.3.1" even though it lives in a folder named `v4`, and the
  filename `f.py` gives no hint that this is the authoritative script in the lineage.

Both `V2/` and `V3.2/` also contain their own generated run outputs (JSON/HTML/TXT reports:
`CRITERIA`, `SPLIT`, `RESULT`, `REPORT`, `TERMINAL.txt`) alongside the source, kept as run
records.

## Helper / debug scripts (not part of the version lineage)

`V3.2/test_api.py` (checks an Anthropic API key is set), `V3.2/audit_eclipse.py` and
`V3.2/verify_results.py` (both hardcode a local Windows path, `G:\Mi unidad\AFH\GITHUB\...`, and
`verify_results.py` loads a `.pkl` checkpoint that isn't present in this repo), and the top-level
`log.py` (trivial log-directory debug utility) are one-off development aids, not reproducible
standalone, and not required to run the main `v4/f.py` script.

## `IIT VS ECLIPSE.bmp`

A 6.2 MB uncompressed bitmap image at the top of this folder — a static reference image, kept
as-is.
