# 🧬 Sleep-EDF Dataset – EEG Input for AFH* Simulation

This folder contains references to real EEG data used in the AFH* model simulation.

## 📦 Dataset Origin

The data comes from the **Sleep-EDF Expanded Dataset** (version 1.0.0), a publicly available polysomnography dataset hosted by [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).

## 📁 Required Files

These files must be manually downloaded and placed in this folder:

- `SC4051E0-PSG.edf`: EEG signal recording (polysomnography)  
- `SC4042EC-Hypnogram.edf`: Corresponding expert sleep stage annotations

These files are **not distributed** in this repository due to licensing restrictions.

---

## 🔽 How to Download

1. Visit: [Sleep-EDF on PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)
2. Log in with a free PhysioNet account.
3. Download the following files from the *Sleep Cassette Study*:
   - `SC4051E0-PSG.edf`
   - `SC4042EC-Hypnogram.edf`
4. Place them in this folder: `3_simulation/3.3_data/`

---

## 📌 Notes

- These files are used to simulate topological and symbolic variables of the AFH* model.
- The data must match the filenames exactly for the Python script to work.
- For more details on how these files are used, see [`ECLIPSE_FINAL.py`](../3.1_src/ECLIPSE_FINAL.py).

---

> **Ethical Notice**: These data were collected with informed consent and are publicly shared for research purposes. Please respect all licensing terms and privacy considerations when using them.

python requirements
_____

mne>=1.5.1
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.2
pandas>=2.0.0
sklearn>=1.3.0
torch>=2.0.0
transformers>=4.35.0
tqdm>=4.66.0
statsmodels>=0.14.0
