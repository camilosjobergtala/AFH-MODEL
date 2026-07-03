# HORIZON — feature-filtering / classification experiments

Exploratory work for a separate Registered Report line (Wake vs. N3 sleep-stage classification
and general feature discovery). No further versioning ladder here — each subfolder is a distinct
piece of work.

## FILTERS/

- **`results_binary/Filter1 binary wake n3.py`** — Wake vs. N3 binary classification using
  spectral features, tied to OSF preregistration `10.17605/OSF.IO/GSJNH`. Its outputs
  (`binary_fisher_scores.csv`, `binary_results.json`) sit in the same folder as the source.
- **`results_enhanced/filter1_spectral_baseline.py`** — an enhanced-features variant of the same
  Wake/N3 task (adds band ratios, Hjorth parameters, cross-channel coherence, sample entropy).
  Outputs (`feature_results.csv`, `multi_feature_results.json`) again co-located with the source.

Both source scripts intentionally keep their own generated outputs alongside them, not in a
separate `outputs/` folder.

## Exploratory/

- **`HCTSA like exploration.py`** — AFH Level-3b exploratory analysis of H* candidate metrics
  (Hjorth, DFA, entropy measures) vs. PAC, explicitly labeled non-confirmatory in its own header.
- **`catch22_analysis_v3.1.py`** and **`HCTSA 2.py`** — share the same title header ("CATCH22 +
  EEG FEATURES FOR CONSCIOUSNESS DISCRIMINATION") and near-identical purpose (981 vs. 975 lines).
  `HCTSA 2.py` appears to be an earlier/parallel iteration of the same catch22 analysis line
  (not a distinct HCTSA-specific tool despite the name) — both kept per the archive convention.
- **`hash.py`** — generates a SHA-256 train/dev split for the Registered Report. Hardcodes a local
  Windows path (`G:\Mi unidad\...`); not runnable as-is outside the original machine.
