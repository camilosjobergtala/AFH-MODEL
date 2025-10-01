# AFH* Model: Structural Consciousness Theory with ECLIPSE Validation

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15541550-blue)](https://doi.org/10.5281/zenodo.15541550)
[![Scientific Reports](https://img.shields.io/badge/Scientific_Reports-Under_Review-orange)](https://www.nature.com/srep/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--6052--0212-green)](https://orcid.org/0009-0009-6052-0212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research repository accompanying:**  
*Sjöberg Tala, C.A. (2025). ECLIPSE: A systematic falsification framework for consciousness science. Submitted to Scientific Reports, under review.*

---

## ⚠️ Current Project Status (October 2025)

**This repository documents ongoing analysis following systematic falsification results.**

Initial ECLIPSE validations (v3.7, v4.1) yielded negative results (F1=0.031-0.037), systematically falsifying the tested implementations. We are currently analyzing what these results mean for the theoretical framework and determining next steps.

**Current focus:** Understanding why the implementations failed and what this tells us about operationalizing consciousness theories.

---

## Abstract

The **AFH*** (Autopsychic Fold + H* Horizon) model proposes a structural framework for consciousness emergence, integrating:

- **Topological organization** (κ_topo)
- **Multiscale temporal coherence** (Σ_stability)
- **Causal information integration** (Φ_integration)

This repository includes the full implementation of **ECLIPSE**, a falsification pipeline developed to validate AFH* across population-scale EEG data. Unlike post-hoc rationalizations, ECLIPSE enforces pre-registered, irreversible, and binary-outcome criteria.

---

## Empirical Results

| Version | Scope | F1 Score | Precision | Recall | Status |
|---------|-------|----------|-----------|--------|---------|
| **v3.7** | Basic AFH* prototype | 0.031 | 0.032 | 0.030 | ❌ Falsified |
| **v4.1** | Refined topological operations | 0.037 | 0.025 | 0.069 | ❌ Falsified |

### Current Analysis

We are analyzing these negative results to understand:
- What the falsification tells us about the specific implementations tested
- Whether the theoretical framework requires modification
- What alternative approaches might be more promising
- How to interpret these findings within consciousness science methodology

This analysis is ongoing and will inform future research directions.

---

## ECLIPSE Validation Protocol

### Five-Stage Falsification Framework

1. **Irreversible Data Partitioning** – Cryptographic split ensures zero optimization leakage
2. **Pre-Registered Thresholds** – Targets: F1 ≥ 0.60, Precision ≥ 0.70, Recall ≥ 0.50
3. **Clean Development** – Separation of exploratory and confirmatory phases
4. **Single-Shot Validation** – One-time test on holdout; no iterative tuning
5. **Binary Assessment** – Pass/fail outcome removes interpretative ambiguity

### Dataset: Sleep-EDF Expanded (PhysioNet)

| Metric | Value |
|--------|-------|
| **Subjects** | 153 full-night EEG recordings |
| **Epochs** | 126,160 × 30-second segments |
| **States** | Wake/REM (conscious) vs N1/N2/N3 (unconscious) |

---

## Repository Structure

AFH-MODEL/
├── README.md                       → Main project overview
├── requirements.txt                → Python dependencies
├── LICENSE                         → MIT license
├── .gitignore                      → Version control exclusions
├── CITATION.cff                    → Citation metadata
│
├── 1. Docs/                        → Presentations & conceptual material
│   ├── figures/                    → Scientific illustrations
│   ├── 1.1 presentation.md         → Lab presentation script
│   ├── 1.2 glossary.md             → Terms and variables
│   ├── 1.3 ethical_manifesto.md    → Ethical commitments
│   └── 1.4 acknowledgments.md      → Scientific credits
│
├── 2. Theory/                      → Core theoretical files
│   ├── 2.1 model_architecture.md   → AFH* formalism
│   ├── 2.2 horizon_thresholds.md   → H* criticality
│   ├── 2.3 resonant_residue.md     → Symbolic divergence
│   └── 2.4 falsifiability_criteria.md → Empirical test design
│
├── 3. Experimentation/             → ECLIPSE pipeline
│   ├── 3.1 COMPUTATIONAL STUFF/    → Main implementations
│   │   ├── legacy/                 → Older versions
│   │   ├── results 3.7/            → Output v3.7
│   │   ├── results 4.1/            → Output v4.1
│   │   ├── ECLIPSE 3.7.py          → Code v3.7
│   │   └── ECLIPSE 4.1.py          → Code v4.1
│   └── README.txt                  → Pipeline description
│
└── 4. Annexes/
└── 4_open_questions.md         → Theoretical frontiers
---

## Quick Start

### Setup
```bash
pip install -r requirements.txt

Run Validation
bashcd "3.Experimentation/3.1 COMPUTATIONAL STUFF/"
python "ECLIPSE 4.1.py" --model afh --validation complete

Outputs Include:

Holdout predictions
Cryptographic SHA256 checksum
Automatic threshold comparison
Reproducibility seed log (Seed 2025)


Scientific Contributions
ECLIPSE Pipeline

Systematic falsification tool for consciousness science
Prevents statistical artifacts and overfitting through irreversible pre-registration
Theory-agnostic: applicable to IIT, GWT, QSP, and other frameworks
Validated contribution: Methodology demonstrated regardless of AFH* outcomes

Transparent Result Reporting

Complete documentation of negative results
Analysis of what falsification means for theory development
Establishes precedent for honest reporting in consciousness research


Citation
bibtex@article{sjoberg2025eclipse,
  title={ECLIPSE: A systematic falsification framework for consciousness science},
  author={Sjöberg Tala, Camilo Alejandro},
  journal={Scientific Reports},
  year={2025},
  note={Submitted – under review},
  doi={10.5281/zenodo.15541550}
}

Author
Dr. Camilo Alejandro Sjöberg Tala, M.D.
Independent Researcher – Viña del Mar, Chile
Email: cst@afhmodel.org
ORCID: https://orcid.org/0009-0009-6052-0212

Current Status
Post-falsification analysis ongoing. We are carefully examining what the systematic negative results tell us about consciousness theory operationalization and determining appropriate next steps based on this analysis.
Updates will be posted as the analysis progresses.

"Systematic falsification provides critical information for theory development. We are analyzing what these results mean for the framework."
"Cogito ergo sum, quomodo sum?" — AFH-R Heuristic

