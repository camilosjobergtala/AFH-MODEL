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
