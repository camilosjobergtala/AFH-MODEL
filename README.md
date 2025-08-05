# AFH* Model: Structural Consciousness Theory with ECLIPSE Validation

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15541550-blue)](https://doi.org/10.5281/zenodo.15541550)
[![Scientific Reports](https://img.shields.io/badge/Scientific_Reports-Under_Review-orange)](https://www.nature.com/srep/)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0009--6052--0212-green)](https://orcid.org/0009-0009-6052-0212)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Research repository accompanying:**  
*Sjöberg Tala, C.A. (2025). ECLIPSE: A systematic falsification framework for consciousness science. Submitted to Scientific Reports, under review.*

---

## Abstract

The **AFH*** (Autopsychic Fold + H* Horizon) model proposes a structural framework for consciousness emergence, integrating:

- **Topological organization** (κ_topo)
- **Multiscale temporal coherence** (Σ_stability)
- **Causal information integration** (Φ_integration)

This repository includes the full implementation of **ECLIPSE**, a falsification pipeline developed to validate AFH* across population-scale EEG data. Unlike post-hoc rationalizations, ECLIPSE enforces pre-registered, irreversible, and binary-outcome criteria.

---

## Theoretical Framework

### Mathematical Formulation

```
Consciousness ⇔ (κ_topo ≥ θ_κ) ∧ (Σ_stability ≥ θ_Σ) ∧ (Φ_integration ≥ θ_Φ)
```

**Where:**
- κ_topo: Ollivier-Ricci curvature of functional brain networks
- Σ_stability: Multiscale temporal coherence across frequency bands  
- Φ_integration: Transfer entropy-based directional causal integration

### H* Horizon Hypothesis

Consciousness arises when neural systems cross a critical threshold in structural space, forming an **Autopsychic Fold**—a self-referential configuration capable of sustaining subjective experience.

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

## Empirical Results

| Version | Scope | F1 Score | Precision | Recall | Status |
|---------|-------|----------|-----------|--------|---------|
| **v3.7** | Basic AFH* prototype | 0.031 | 0.032 | 0.030 | ❌ Falsified |
| **v4.1** | Refined topological operations | 0.037 | 0.025 | 0.069 | ❌ Falsified |

### Interpretation

Systematic falsification confirms empirical limits of partial implementations and validates the robustness of the ECLIPSE methodology.

---

## Repository Structure

```
AFH-MODEL/
├── README.md                       → Main project overview
├── requirements.txt                → Python dependencies
├── LICENSE                         → MIT license
├── .gitignore                      → Version control exclusions
├── CITATION.cff                    → Citation metadata
├── NISA'S CODE                     → Alternative implementations
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
```

---

## Quick Start

### Setup

```bash
pip install -r requirements.txt
```

### Run Validation

```bash
cd "3.Experimentation/3.1 COMPUTATIONAL STUFF/"
python "ECLIPSE 4.1.py" --model afh --validation complete
```

### Outputs Include:

- Holdout predictions
- Cryptographic SHA256 checksum
- Automatic threshold comparison
- Reproducibility seed log (Seed 2025)

---

## Scientific Contributions

### AFH Model*

- Structural theory of consciousness grounded in topology, causality, and temporal dynamics
- Explicit, falsifiable predictions across neurotypical populations

### ECLIPSE Pipeline

- First systematic falsification tool for consciousness science
- Avoids statistical artifacts and overfitting
- Theory-agnostic: applicable to IIT, GWT, QSP, and beyond

---

## Clinical & Technological Applications

| Domain | Application |
|--------|-------------|
| **Anesthesia** | Structural detection of loss-of-consciousness transitions |
| **Disorders of Consciousness** | Diagnosis beyond behavioral inference |
| **BCI** | Integration into neural decoding pipelines |
| **Pharmacology** | Mechanistic assessment of consciousness-modulating compounds |

---

## Citation

```bibtex
@article{sjoberg2025eclipse,
  title={ECLIPSE: A systematic falsification framework for consciousness science},
  author={Sjöberg Tala, Camilo Alejandro},
  journal={Scientific Reports},
  year={2025},
  note={Submitted – under review},
  doi={10.5281/zenodo.15541550}
}
```

---

## Author

**Dr. Camilo Alejandro Sjöberg Tala, M.D.**  
Independent Researcher – Viña del Mar, Chile  
Email: cst@afhmodel.org  
ORCID: https://orcid.org/0009-0009-6052-0212

---

## Future Directions

- Implementation of fold detection (ψ) and symbolic resonance (∇Φ)
- Multimodal datasets (fMRI, MEG) for structural convergence
- Experimental paradigms targeting causal topology transitions
- Cross-cultural validation and developmental neuroscience applications

---

**"The systematic falsification of AFH v3.7 and v4.1 represents methodological success, not theoretical failure. ECLIPSE ensures consciousness science enters its post-hoc-free era."**

**"Cogito ergo sum, quamodo sum?"** — **AFH-R Heuristic**
