# -*- coding: utf-8 -*-
"""
Figure 4 (outcome heat map), §4.2: four-outcome classification of the
conjunction across the twelve scenarios of Table 7, one row per scenario
and one column per outcome, each cell labelled with its percentage. A
vertical rule separates decisions taken under a valid design (contradicted,
survived, inconclusive) from the not-evaluable column, following the
article's own figure description. Reads the out_*.json summaries written by
run_outcomes.py.

RUN: python3 make_fig4.py   (after run_outcomes.py has produced out_*.json)
TIME: <1 min.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_outcomes import SCENARIO_KEYS, DISPLAY

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
COLUMNS = [("contradicted_pct", "Contradicted"), ("survived_pct", "Survived"),
           ("inconclusive_pct", "Inconclusive"), ("not_evaluable_pct", "Not evaluable")]


def main():
    rows = []
    labels = []
    missing = []
    for key in SCENARIO_KEYS:
        path = OUT_DIR / f"out_{key}.json"
        if not path.exists():
            missing.append(key)
            continue
        with open(path) as f:
            d = json.load(f)
        rows.append([d[field] for field, _ in COLUMNS])
        labels.append(DISPLAY[key])
    if missing:
        print(f"Missing out_*.json for: {missing}. Run run_outcomes.py first.", file=sys.stderr)

    mat = np.array(rows)
    n_rows = mat.shape[0]

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * n_rows + 1.6))
    im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels([c[1] for c in COLUMNS], fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=9.5)

    for i in range(n_rows):
        for j in range(4):
            val = mat[i, j]
            color = "white" if val > 55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9.5, color=color)

    ax.axvline(2.5, color="#b02a2a", linewidth=2.2)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title("Four-outcome classification of the conjunction by scenario\n"
                  "(red rule separates valid-design decisions from validity failures)",
                  fontsize=10.5)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Figure4_outcome_heatmap.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/Figure4_outcome_heatmap.png and .pdf")


if __name__ == "__main__":
    main()
