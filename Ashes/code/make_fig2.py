# -*- coding: utf-8 -*-
"""
Figure 2 (architecture diagram), §3.8. Six small directed graphs showing the
architectures that can generate segment-wise predictive dependence between
A1, B and A2. Only panel (a) contains episode-specific return to the source
population; panel (f) transmits only a coarse summary along each link, so
both predictive gains are positive while episode identity is destroyed in
transit (the token-disruption architecture).

RUN: python3 make_fig2.py
TIME: <1 min.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

NODE_FILL, NODE_EDGE = "#dbe9f7", "#2b6cb0"
LATENT_FILL, LATENT_EDGE = "#e6e6e6", "#7a7a7a"
RETURN_COLOR = "#b02a2a"


def node(ax, xy, label, r=0.30, fill=NODE_FILL, edge=NODE_EDGE, dashed=False):
    c = Circle(xy, r, facecolor=fill, edgecolor=edge, linewidth=1.6,
               linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(c)
    ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=11, zorder=3)


def edge(ax, p0, p1, r0=0.30, r1=0.30, color="black", dashed=False, lw=1.6):
    import numpy as np
    p0, p1 = np.array(p0, float), np.array(p1, float)
    v = p1 - p0
    d = (v ** 2).sum() ** 0.5
    u = v / d
    a = FancyArrowPatch(
        tuple(p0 + u * r0), tuple(p1 - u * r1), arrowstyle="-|>",
        mutation_scale=13, color=color, linewidth=lw, zorder=1,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(a)


def panel(ax, letter, caption):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.35, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-1.2, 1.0, f"({letter})", ha="left", va="center", fontsize=12, weight="bold")
    ax.text(0, -1.25, caption, ha="center", va="center", fontsize=9.5)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 9.6),
                              gridspec_kw={"hspace": 0.6, "wspace": 0.25})

    # (a) recurrent return: A1 -> B, B -> A2, labelled "return to source A"
    ax = axes[0, 0]
    node(ax, (-0.9, 0.4), "$A_1$")
    node(ax, (0, -0.4), "$B$")
    node(ax, (0.9, 0.4), "$A_2$")
    edge(ax, (-0.9, 0.4), (0, -0.4), color=RETURN_COLOR)
    edge(ax, (0, -0.4), (0.9, 0.4), color=RETURN_COLOR)
    panel(ax, "a", "return to source $A$")

    # (b) local persistence / no return through B: A1 -> B and A1 -> A2 directly
    ax = axes[0, 1]
    node(ax, (-0.9, 0.4), "$A_1$")
    node(ax, (0, -0.4), "$B$")
    node(ax, (0.9, 0.4), "$A_2$")
    edge(ax, (-0.9, 0.4), (0, -0.4))
    edge(ax, (-0.9, 0.4), (0.9, 0.4))
    panel(ax, "b", "no return through $B$")

    # (c) feedforward chain: A1 -> B -> C, terminates elsewhere
    ax = axes[0, 2]
    node(ax, (-0.9, 0), "$A_1$")
    node(ax, (0, 0), "$B$")
    node(ax, (0.9, 0), "$C$")
    edge(ax, (-0.9, 0), (0, 0))
    edge(ax, (0, 0), (0.9, 0))
    panel(ax, "c", "terminates elsewhere")

    # (d) observed common cause: X -> A1, X -> A2
    ax = axes[1, 0]
    node(ax, (0, 0.75), "$X$")
    node(ax, (-0.7, -0.35), "$A_1$")
    node(ax, (0.7, -0.35), "$A_2$")
    node(ax, (0, -0.9), "$B$", r=0.26)
    edge(ax, (0, 0.75), (-0.7, -0.35))
    edge(ax, (0, 0.75), (0.7, -0.35))
    panel(ax, "d", "observed common cause")

    # (e) latent confounder: same as (d) with a grey unobserved node U
    ax = axes[1, 1]
    node(ax, (0, 0.75), "$U$", fill=LATENT_FILL, edge=LATENT_EDGE, dashed=True)
    node(ax, (-0.7, -0.35), "$A_1$")
    node(ax, (0.7, -0.35), "$A_2$")
    node(ax, (0, -0.9), "$B$", r=0.26)
    edge(ax, (0, 0.75), (-0.7, -0.35), color=LATENT_EDGE, dashed=True)
    edge(ax, (0, 0.75), (0.7, -0.35), color=LATENT_EDGE, dashed=True)
    panel(ax, "e", "latent confounder")

    # (f) token disruption: A1 -> B -> A2, both arrows dashed grey, "coarse links, identity lost"
    ax = axes[1, 2]
    node(ax, (-0.9, 0.4), "$A_1$")
    node(ax, (0, -0.4), "$B$")
    node(ax, (0.9, 0.4), "$A_2$")
    edge(ax, (-0.9, 0.4), (0, -0.4), color=LATENT_EDGE, dashed=True)
    edge(ax, (0, -0.4), (0.9, 0.4), color=LATENT_EDGE, dashed=True)
    panel(ax, "f", "coarse links, identity lost")

    fig.suptitle(
        "Six architectures that can generate segment-wise predictive dependence.\n"
        "Only (a) contains episode-specific return to the source population; panel (f) transmits only a\n"
        "coarse summary along each link, so both predictive gains are positive while episode identity is\n"
        "destroyed in transit.",
        fontsize=10.5, y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"Figure2_architectures.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/Figure2_architectures.png and .pdf")


if __name__ == "__main__":
    main()
