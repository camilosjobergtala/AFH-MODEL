"""
01b_figure2_correlation_heatmap.py
═══════════════════════════════════════════════════════════════════════════════
Regenerates Figure 2 — the Study 1 dimensional non-redundancy heatmap: the
Spearman correlation matrix among the five EIS component scores across the
300-protocol synthetic corpus.

Reads outputs/study1/component_correlation_matrix.csv (written by
01_study1_eis_coherence.py) — run that script first if it hasn't been run,
or if the S_leak component formula has changed.

Writes outputs/study1/Figure2_component_correlation_heatmap.png
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if (HERE.parent / "engine").exists() else HERE
CORR_CSV = ROOT / "outputs" / "study1" / "component_correlation_matrix.csv"
OUT_PNG = ROOT / "outputs" / "study1" / "Figure2_component_correlation_heatmap.png"

LABELS = {
    'preregistration': 'Pre-\nregistration',
    'split_strength': 'Split\nstrength',
    'protocol_adherence': 'Protocol\nadherence',
    'leakage_score': 'Leakage\nscore ($S_{leak}$)',
    'transparency': 'Transparency',
}


def main():
    if not CORR_CSV.exists():
        sys.exit(f"Missing {CORR_CSV} — run code/01_study1_eis_coherence.py first.")

    corr = pd.read_csv(CORR_CSV, index_col=0)
    order = [c for c in LABELS if c in corr.columns]
    corr = corr.loc[order, order]
    labels = [LABELS[c] for c in order]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=9, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman correlation", fontsize=9)

    ax.set_title("Figure 2. EIS component correlation matrix\n"
                  "(n = 300 synthetic protocols; Spearman)", fontsize=11)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 written -> {OUT_PNG}")


if __name__ == "__main__":
    main()
