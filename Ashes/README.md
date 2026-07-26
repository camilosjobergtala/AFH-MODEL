# Ashes — analysis code

Simulation code for *An admissibility test for constitutive claims about phenomenal
presence: necessary conditions, asymmetric falsification, and a conjunctive
return-compatible criterion.*

**These are the scripts that produced every number in the article.** They are
deterministic: fixed seeds give bit-identical output on re-run. Any reimplementation
written from the manuscript description will produce similar but not identical values
and should not be substituted for these files.

## Requirements

Python 3.12, NumPy 2.4.4, Matplotlib 3.10.8 (figures only). `pip install -r requirements.txt`.
No other dependency; no network access required.

## Files

| File | Role |
|---|---|
| `sim_return.py` | core module: design matrices, ridge fit, out-of-sample R², strata, Wilson intervals, eleven base architectures |
| `chain_return.py` | segment estimands, conditional nulls, chain-level matching, token-disruption architecture |
| `run_chain.py` | Table 6, Figure 3 |
| `run_outcomes.py` | Table 7, Figure 4 |
| `run_sweep.py` | Table 8 |
| `run_thresh.py` | Table 9 |
| `make_fig1.py` … `make_fig4.py` | Figures 1–4 |

`sim_return.py` and `chain_return.py` are modules, imported by the others; they are not
executed directly.

## Reproducing the reported results

```bash
python3 run_chain.py 0:12 200                              # Table 6   ~24 min
python3 run_outcomes.py 0:12 200                           # Table 7   ~13 min
python3 run_sweep.py 250,500,1000,2000,4000 200            # Table 8   ~9 min
python3 run_thresh.py 0.005 150                            # Table 9   ~5 min
python3 run_thresh.py 0.01 150
python3 run_thresh.py 0.02 150
python3 make_fig1.py && python3 make_fig2.py && python3 make_fig3.py && python3 make_fig4.py
```

Approximately 62 minutes total on a single core.

Two cells depart from these counts, as stated in the article's Supplementary Methods §S9:
the 4000-trial cell of Table 8 used 80 replicates, and Table 9 used 150 studies per cell.

## Licence

Apache License 2.0. See `LICENSE` at the repository root.
