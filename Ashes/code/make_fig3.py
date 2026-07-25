# -*- coding: utf-8 -*-
"""
Figure 3 (component rates), §4.1: grouped bar chart of positive rates for
the two segment estimands, the chain component, and the conjunction, across
the twelve architectures, with 95% Wilson intervals as error bars and a
dotted line at the nominal 5% level. Reads the chain_*.json summaries
written by run_chain.py.

RUN: python3 make_fig3.py   (after run_chain.py has produced chain_*.json)
TIME: <1 min.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chain_return import ARCHITECTURES
from sim_return import DISPLAY_NAME

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

SERIES = [
    ("seg1", "$\\Delta R^2_{A_1\\rightarrow B}$", "#2b6cb0", ""),
    ("seg2", "$\\Delta R^2_{B\\rightarrow A_2\\,|\\,A_1,X}$", "#63a1d6", "//"),
    ("chain", "chain", "#2f7a3d", "xx"),
    ("conjunction", "conjunction", "#b02a2a", ".."),
]


def main():
    keys = [k for k, _ in ARCHITECTURES]
    data = {}
    missing = []
    for key in keys:
        path = OUT_DIR / f"chain_{key}.json"
        if not path.exists():
            missing.append(key)
            continue
        with open(path) as f:
            data[key] = json.load(f)
    if missing:
        print(f"Missing chain_*.json for: {missing}. Run run_chain.py first.", file=sys.stderr)
    keys = [k for k in keys if k in data]

    n = len(keys)
    x = np.arange(n)
    width = 0.19

    fig, ax = plt.subplots(figsize=(14, 6.2))
    for i, (field, label, color, hatch) in enumerate(SERIES):
        rates = np.array([data[k][f"{field}_rate"] for k in keys])
        cis = np.array([data[k][f"{field}_ci"] for k in keys])
        lo_err = np.clip(rates - cis[:, 0], 0, None)
        hi_err = np.clip(cis[:, 1] - rates, 0, None)
        offset = (i - 1.5) * width
        ax.bar(x + offset, rates, width, label=label, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.6,
               yerr=[lo_err, hi_err], capsize=2, error_kw={"linewidth": 0.8})

    ax.axhline(5.0, linestyle=":", color="grey", linewidth=1.2, zorder=0)
    ax.set_ylabel("Positive rate (%)")
    ax.set_ylim(-3, 108)
    ax.set_xticks(x)
    labels = [DISPLAY_NAME[k] + ("\n(return)" if data[k]["has_return"] else "") for k in keys]
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(loc="lower left", ncol=4, fontsize=9, frameon=False)
    ax.set_title("Positive rates by architecture, with 95% Wilson intervals "
                  "(dotted line: nominal 5% level)", fontsize=11)
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Figure3_component_rates.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/Figure3_component_rates.png and .pdf")


if __name__ == "__main__":
    main()
