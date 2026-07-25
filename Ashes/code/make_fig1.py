# -*- coding: utf-8 -*-
"""
Figure 1 (decision flow), §2.8. Three-stage diagram: seven hierarchical
validity gates in fixed order, each with a dashed red arrow to a red bar
reading "any gate fails, not evaluable"; four component boxes (three
estimated from data, one -- same source population -- a preregistered
design requirement, not an estimated statistic), feeding a green
conjunction-level aggregation box; and the three outcomes reachable once
every gate has passed: contradicted, survived testing, inconclusive.
Not evaluable is reached directly from stage 1, not through the conjunction.

RUN: python3 make_fig1.py
TIME: <1 min.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

BLUE_FILL, BLUE_EDGE = "#dbe9f7", "#2b6cb0"
RED_FILL, RED_EDGE = "#fbe3e3", "#b02a2a"
GREEN_FILL, GREEN_EDGE = "#e3f3e6", "#2f7a3d"
GREY_FILL, GREY_EDGE = "#eeeeee", "#7a7a7a"


def box(ax, xy, w, h, text, fill, edge, fontsize=10.5, dashed=False, weight="normal"):
    x, y = xy
    style = "round,pad=0.02,rounding_size=0.03"
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=style, facecolor=fill, edgecolor=edge,
        linewidth=1.6, linestyle="--" if dashed else "-", zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3, wrap=True)
    return patch


def arrow(ax, p0, p1, color="black", style="-", lw=1.4, dashed=False):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, color=color,
        linewidth=lw, linestyle="--" if dashed else "-", zorder=1,
    )
    ax.add_patch(a)


def main():
    fig, ax = plt.subplots(figsize=(13.4, 9.5))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(6.7, 9.15, "1.  Seven hierarchical validity gates, applied in fixed order",
            ha="center", va="center", fontsize=13, weight="bold")

    gates = ["domain\nmembership", "presence\nevidence (φ)", "observation\nmodel",
             "sufficient\nreliability", "sufficient\nsensitivity",
             "positive control\npassed", "minimal estimability\n& design adequacy"]
    n_gates = len(gates)
    gw, gh, gap = 1.55, 1.15, 0.30
    total_w = n_gates * gw + (n_gates - 1) * gap
    x0 = (13.4 - total_w) / 2
    gate_centers = []
    for i, label in enumerate(gates):
        x = x0 + i * (gw + gap)
        box(ax, (x, 7.55), gw, gh, label, BLUE_FILL, BLUE_EDGE, fontsize=9.7)
        gate_centers.append(x + gw / 2)
        if i > 0:
            arrow(ax, (x - gap, 7.55 + gh / 2), (x, 7.55 + gh / 2))

    box(ax, (x0, 6.15), total_w, 0.68, "any gate fails   →   NOT EVALUABLE",
        RED_FILL, RED_EDGE, fontsize=13, weight="bold")
    for cx in gate_centers:
        arrow(ax, (cx, 7.55), (cx, 6.83), color=RED_EDGE, dashed=True, lw=1.3)

    ax.text(6.7, 5.55, "2.  Component tests, evaluated only if every gate passes",
            ha="center", va="center", fontsize=13, weight="bold")

    comp_w, comp_h, comp_gap = 2.95, 1.85, 0.28
    comp_x0 = (13.4 - (4 * comp_w + 3 * comp_gap)) / 2
    comp_y = 3.35
    comps = [
        ("Component 1\n$\\Delta R^2_{A_1\\rightarrow B}$\n\npredictive gain\nfrom source to\nintermediate", BLUE_FILL, BLUE_EDGE, False),
        ("Component 2\n$\\Delta R^2_{B\\rightarrow A_2\\,|\\,A_1,X}$\n\nincremental gain\nfrom intermediate\nto late source", BLUE_FILL, BLUE_EDGE, False),
        ("Component 3\n$\\theta_{chain}$\n\nchain-level\nepisode\nspecificity", BLUE_FILL, BLUE_EDGE, False),
        ("Component 4\nsame source\npopulation\n\ndesign requirement,\nnot an estimated\nstatistic", GREY_FILL, GREY_EDGE, True),
    ]
    comp_centers = []
    for i, (label, fill, edge, dashed) in enumerate(comps):
        x = comp_x0 + i * (comp_w + comp_gap)
        box(ax, (x, comp_y), comp_w, comp_h, label, fill, edge, fontsize=10.3, dashed=dashed)
        comp_centers.append((x + comp_w / 2, comp_y))
    ax.text(comp_centers[3][0], comp_y - 0.22, "preregistered, frozen before analysis",
            ha="center", va="top", fontsize=8.5, style="italic", color="#555555")

    conj_w, conj_h = total_w, 0.75
    conj_y = 2.05
    box(ax, (x0, conj_y), conj_w, conj_h,
        "3.  Conjunction-level aggregation   (all four components required)",
        GREEN_FILL, GREEN_EDGE, fontsize=12.5, weight="bold")
    conj_cx = x0 + conj_w / 2
    for i, (cx, cy) in enumerate(comp_centers):
        dashed = (i == 3)
        arrow(ax, (cx, cy), (conj_cx + (cx - conj_cx) * 0.15, conj_y + conj_h),
              color=GREY_EDGE if dashed else "#444444", dashed=dashed, lw=1.4)

    out_w, out_h, out_gap = 3.55, 1.55, 0.55
    out_x0 = (13.4 - (3 * out_w + 2 * out_gap)) / 2
    out_y = 0.15
    outcomes = [
        ("CONTRADICTED\n\nany component\ncontradicted", GREEN_EDGE),
        ("SURVIVED TESTING\n\nall components\nsurvived", GREEN_EDGE),
        ("INCONCLUSIVE\n\nnone contradicted,\nat least one\ninconclusive", GREEN_EDGE),
    ]
    for i, (label, edge) in enumerate(outcomes):
        x = out_x0 + i * (out_w + out_gap)
        box(ax, (x, out_y), out_w, out_h, label, GREEN_FILL, edge, fontsize=11)
        arrow(ax, (conj_cx, conj_y), (x + out_w / 2, out_y + out_h), color=GREEN_EDGE, lw=1.6)

    ax.text(6.7, -0.28,
            "Survival confirms neither constitution nor identity; the conjunction is "
            "observational and does not establish causal return.",
            ha="center", va="center", fontsize=10, style="italic")

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Figure1_decision_flow.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/Figure1_decision_flow.png and .pdf")


if __name__ == "__main__":
    main()
