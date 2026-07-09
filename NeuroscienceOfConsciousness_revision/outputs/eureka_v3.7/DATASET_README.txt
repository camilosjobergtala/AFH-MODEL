# 🧬 AFH* Experimentation – ECLIPSE 4.1 Pipeline

This folder contains the **computational experiments** that operationalize the AFH* model on EEG data.

## 📦 Dataset Origin

The analysis uses the **Sleep-EDF Expanded Dataset (v1.0.0)**, a public polysomnography dataset hosted by [PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/).

## 📁 Required Files

Manually download these files and place them in this folder’s `/data` subfolder:

- `SC4051E0-PSG.edf` → EEG signal recording
- `SC4042EC-Hypnogram.edf` → Sleep stage annotations

> ⚠️ These files are **not included** in this repository due to licensing restrictions.

---

## 🚀 How to Run the Pipeline

1️⃣ **Open in Visual Studio Code** (or any IDE).  
Make sure you have **Python ≥ 3.10** installed.

2️⃣ **Install dependencies:**
```bash
pip install -r "3. Experimentation/3.1 COMPUTATIONAL STUFF/requirements.txt"
```

3️⃣ **Run the latest pipeline:**
```bash
python "3. Experimentation/3.1 COMPUTATIONAL STUFF/ECLIPSE 4.1.py"
```

> 💡 **Note:** `ECLIPSE 4.1.py` is the **definitive version** used for the v4.1 falsification analysis (F1 = 0.037). Older scripts are archived for reference.

---

## 📊 What You Get

After running the script (with the dataset in place), a folder will be created:
```
3. Experimentation/3.1 COMPUTATIONAL STUFF/results 4.1/
```
This contains:
- 📈 **plots/** → κ_topo, Σ_stability, Φ_integration, ∇Φ graphs
- 📊 **metrics.csv** → summary table (includes F1 score)
- 📝 **log.txt** → run log for reproducibility

---

## 🐍 Python Requirements
```
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
```

---

> **Ethical Notice**: Sleep-EDF data were collected with informed consent and are shared for research purposes. Please respect licensing terms and privacy considerations.
