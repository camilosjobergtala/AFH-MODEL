# 🧠 AFH* MODEL – Simulation of Consciousness Emergence

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15541550.svg)](https://doi.org/10.5281/zenodo.15541550)

Welcome to the official repository of the **AFH*** model (Autopsychic Fold + Horizon), a falsifiable and structurally grounded hypothesis for the emergence of consciousness. The model proposes that consciousness arises when a neural system crosses a **topological threshold** (`H*`) and exhibits **resonant symbolic divergence** (`∇Φ > 0`), forming an **Autopsychic Fold** (`ψ > 0`).

---

## 🧩 Core Concepts

| Variable        | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `κ_topo`        | Topological curvature (e.g., Ricci, clustering coefficients) |
| `Σ_stability`   | Dynamic signal stability and perturbation resilience         |
| `Φ_integration` | Causal integration (e.g., Transfer Entropy, Granger Causality) |
| `∇Φ`            | Symbolic divergence (resonant residue)                       |
| `ψ`             | Fold emergence indicator (`ψ > 0` = consciousness detected)  |

---

## 🧪 What Does the Simulation Do?

This repository includes an **empirical simulation** based on real EEG data from the **Sleep-EDF dataset**. It performs the following:

1. Loads real `.edf` files (EEG and hypnogram) from human sleep stages.
2. Extracts features from EEG channels across annotated sleep segments.
3. Computes the three variables defining the **Horizon Threshold** (`κ`, `Σ`, `Φ`).
4. Estimates the symbolic divergence `∇Φ` across internal-external windows.
5. Determines whether the system enters a state where `ψ > 0` (Autopsychic Fold detected).
6. Outputs comparison between **conscious** (e.g., wakefulness/REM) and **non-conscious** states (e.g., NREM stages).

---

## 🚀 How to Run

### 1. Clone the repository:

```bash
git clone https://github.com/camilosjobergtala/AFH-MODEL.git
cd AFH-MODEL

2. Install dependencies (Python ≥ 3.10 recommended):
pip install -r 3_simulation/3.2_requirements.txt

3. Run the main experiment:
python 3_simulation/3.1_src/ECLIPSE_FINAL.py

📄 Dataset
This simulation uses the Sleep-EDF Expanded dataset (2013), a public EEG resource including full-night recordings with expert sleep stage annotations.

Required files (place in /data folder):

SC4051E0-PSG.edf (EEG)

SC4042EC-Hypnogram.edf (Hypnogram)

📂 Repository Structure
AFH-MODEL/
│
├── 0_README.md                         # 🔰 Model overview
├── LICENSE                             # MIT License
├── .gitignore
│
├── 1_docs/
│   ├── 1.1_presentation.md
│   ├── 1.2_glossary.md
│   ├── 1.3_ethical_manifesto.md
│   ├── 1.4_acknowledgments.md
│   └── figures/
│       └── horizon_region.png
│
├── 2_theory/
│   ├── 2.1_model_architecture.md
│   ├── 2.2_horizon_thresholds.md
│   ├── 2.3_resonant_residue.md
│   └── 2.4_falsifiability_criteria.md
│
├── 3_simulation/
│   ├── 3.1_src/
│   │   └── ECLIPSE_FINAL.py
│   ├── 3.2_requirements.txt
│   └── data/
│       └── (Sleep-EDF files here)
│
├── 4_preprints/
│   └── [... relevant PDFs ...]
│
├── 5_annexes/
│   ├── 5.1_psi_fold_geometry.md
│   ├── 5.2_research_roadmap.md
│   ├── 5.3_open_questions.md

🧑‍🔬 Author
Camilo A. Sjöberg Tala
Medical Doctor & Independent Researcher
Viña del Mar, Chile – 2025
ORCID: 0009-0009-6052-0212

🧾 Citation
If you use this repository or build upon the AFH* model, please cite:

@misc{sjöberg2025afh,
  author = {Camilo A. Sjöberg Tala},
  title = {AFH* Model: A Structural and Falsifiable Proposal for the Emergence of Consciousness},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15541551},
  url = {https://doi.org/10.5281/zenodo.15541551}
}

> 🌀 “Cogito ergo sum, quamodo sum?”  
> — AFH*-R Heuristic
