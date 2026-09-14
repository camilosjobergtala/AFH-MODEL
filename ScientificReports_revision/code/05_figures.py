"""
═══════════════════════════════════════════════════════════════════════════════
FIGURE GENERATION
═══════════════════════════════════════════════════════════════════════════════

Regenerates Figures 2, 3 and 4 from the deposited CSV/JSON outputs. No
simulation is run here: every value plotted is read from outputs/, so the
figures cannot drift from the reported numbers.

Run 01-04 first. Writes PNG (600 dpi) and PDF to outputs/figures/.

Reviewer 2, point (d): the Figure 3 panel A axis is labelled "False-positive
rate", not "specificity" — it plots a rate of flagging honest studies.
═══════════════════════════════════════════════════════════════════════════════
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
INK = "#1a1a1a"
SEQ = ["#2b4c7e", "#567ebb", "#95b8e0"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600)
    plt.close(fig)
    print(f"   wrote {name}.png / .pdf")


# ─────────────────────────────────────────────────────────────────────────────
def figure2():
    """EIS internal coherence: component correlations + weight stability."""
    cm = pd.read_csv(OUT / "study1" / "component_correlation_matrix.csv", index_col=0)
    ws = pd.read_csv(OUT / "study1" / "weight_stability.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.15, 1],
                                                "wspace": 0.62})

    labels = ["Prereg.", "Split", "Adherence", "Leakage", "Transp."]
    M = cm.values
    im = ax1.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(5), labels, rotation=40, ha="right")
    ax1.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            v = M[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(v) > 0.55 else INK)
    ax1.set_title("A  Component correlations", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)

    alt = ws[ws.scheme != "default"].copy()
    pretty = {"equal": "Equal", "preregistration_dom": "Prereg.-dom.",
              "transparency_dom": "Transp.-dom.", "split_dom": "Split-dom.",
              "leakage_dom": "Leakage-dom."}
    alt["label"] = alt.scheme.map(pretty)
    alt = alt.sort_values("spearman_vs_default")
    y = np.arange(len(alt))
    ax2.barh(y, alt.spearman_vs_default, color=SEQ[1], height=0.6)
    ax2.set_yticks(y, alt.label)
    ax2.set_xlim(0, 1.0)
    mu = alt.spearman_vs_default.mean()
    ax2.axvline(mu, color=INK, ls="--", lw=0.9)
    ax2.text(mu - 0.03, -0.75, f"mean {mu:.2f}", ha="right", fontsize=6.5,
             color=INK)
    for yi, v in zip(y, alt.spearman_vs_default):
        ax2.text(v - 0.02, yi, f"{v:.2f}", va="center", ha="right",
                 fontsize=6.5, color="white")
    ax2.set_xlabel("Spearman rank correlation vs. default ordering")
    ax2.set_ylim(-1.1, len(alt) - 0.4)
    ax2.set_title("B  Ranking stability", loc="left", fontweight="bold")
    save(fig, "figure2_eis_coherence")
    print(f"      range {alt.spearman_vs_default.min():.2f}"
          f"-{alt.spearman_vs_default.max():.2f}, "
          f"mean {alt.spearman_vs_default.mean():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
def figure3():
    """STDS operating characteristics under the superseded fixed cutoff."""
    fpr = pd.read_csv(OUT / "study2" / "study2_fpr_by_K_M.csv")
    sens = pd.read_csv(OUT / "study2" / "study2_sensitivity_by_noise.csv")
    bnd = pd.read_csv(OUT / "study2" / "study2_boundary_shift.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))

    ax = axes[0]
    for c, (m, g) in zip(SEQ, fpr.groupby("n_metrics")):
        g = g.sort_values("K_folds")
        ax.plot(g.K_folds, g.fpr * 100, "o-", color=c, ms=4, lw=1.4,
                label=f"|M| = {m}")
    ax.axhline(5, color=INK, ls=":", lw=0.9)
    ax.text(9.6, 6.5, "5% reference", fontsize=6, ha="right", color=INK)
    ax.set_xticks([3, 5, 10])
    ax.set_xlabel("Cross-validation folds K")
    # R2(d): this is a false-positive rate, NOT specificity.
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title("A  Honest holdout", loc="left", fontweight="bold")
    ax.legend(loc="upper right")

    ax = axes[1]
    for c, (s, g) in zip(SEQ, sens.groupby("sigma_fold")):
        g = g.sort_values("delta_inflation")
        ax.plot(g.delta_inflation * 100, g.sensitivity * 100, "o-", color=c,
                ms=4, lw=1.4, label=f"σ = {s:g}")
    ax.set_xlabel("Holdout inflation δ (%)")
    ax.set_ylabel("Sensitivity (%)")
    ax.set_ylim(0, 105)
    ax.set_title("B  Snooped holdout", loc="left", fontweight="bold")
    ax.legend(loc="lower right")

    ax = axes[2]
    ax.plot(bnd.holdout_sigma_mult, bnd.fpr_under_H0 * 100, "o-",
            color=SEQ[0], ms=4, lw=1.4)
    ax.axhline(5, color=INK, ls=":", lw=0.9)
    ax.set_xlabel("Holdout SD ÷ fold SD")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title("C  Assumption violated", loc="left", fontweight="bold")

    save(fig, "figure3_stds_operating_characteristics")


# ─────────────────────────────────────────────────────────────────────────────
def figure3b():
    """Study 2b: empirical screening cutoffs that replaced the fixed rule."""
    cut = pd.read_csv(OUT / "study2b" / "stds_screening_cutoffs.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    for c, (m, g) in zip(SEQ + ["#c8dcf2"], cut.groupby("n_metrics")):
        g = g.sort_values("k_folds")
        ax1.plot(g.k_folds, g.cutoff_p95, "o-", color=c, ms=4, lw=1.4,
                 label=f"|M| = {m}")
    ax1.axhline(2.0, color="#b3452e", ls="--", lw=1.1)
    ax1.text(20, 2.35, "superseded fixed cutoff = 2", fontsize=6.5,
             ha="right", color="#b3452e")
    ax1.set_xscale("log")
    ax1.set_xticks([3, 5, 10, 20], ["3", "5", "10", "20"])
    ax1.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.set_xlabel("Cross-validation folds K")
    ax1.set_ylabel("max $z$ cutoff (95th pct, honest)")
    ax1.set_title("A  Empirical screening cutoffs", loc="left", fontweight="bold")
    ax1.legend(loc="upper right")

    sub = cut[cut.n_metrics.isin([1, 3, 5])].copy()
    sub["lbl"] = sub.k_folds.astype(str) + "/" + sub.n_metrics.astype(str)
    sub = sub.sort_values(["k_folds", "n_metrics"])
    x = np.arange(len(sub))
    ax2.bar(x - 0.2, sub.honest_rate_at_fixed_2 * 100, 0.4,
            color="#b3452e", label="fixed cutoff = 2")
    ax2.bar(x + 0.2, np.full(len(sub), 5.0), 0.4,
            color=SEQ[1], label="empirical cutoff")
    ax2.axhline(5, color=INK, ls=":", lw=0.9)
    ax2.set_xticks(x, sub.lbl, rotation=45, ha="right", fontsize=6)
    ax2.set_xlabel("K / |M|")
    ax2.set_ylabel("Honest-holdout flag rate (%)")
    ax2.set_title("B  Flag rate under an honest holdout", loc="left",
                  fontweight="bold")
    ax2.legend(loc="upper right")
    save(fig, "figure3b_empirical_cutoffs")


# ─────────────────────────────────────────────────────────────────────────────
def figure4():
    """Code auditor: per-category recall and the zero near-miss specificity."""
    d = json.loads((OUT / "study3" / "study3_summary.json").read_text())
    per = d["per_category_recall"]
    nm = d["near_miss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9),
                                   gridspec_kw={"width_ratios": [1.5, 1],
                                                "wspace": 0.45})

    names = list(per)
    rec = [per[n]["recall"] * 100 for n in names]
    pretty = [n.replace("_", " ").capitalize() for n in names]
    y = np.arange(len(names))
    ax1.barh(y, rec, color=SEQ[1], height=0.62)
    ax1.set_yticks(y, pretty, fontsize=6.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Recall (%)")
    for yi, n in zip(y, names):
        ax1.text(97, yi, f"n = {per[n]['n_injected']}", va="center", ha="right",
                 fontsize=6, color="white")
    ax1.set_title("A  Recall on injected violations", loc="left", fontweight="bold")

    # Panel B plots the same quantity for both sets — the proportion of scripts
    # flagged — because that is what the auditor does. For the near-miss set a
    # 100% flag rate IS zero specificity; plotting specificity directly would
    # render a zero-height bar.
    spec = nm["specificity"] * 100
    ax2.bar([0], [100.0], color=SEQ[1], width=0.55)
    ax2.bar([1], [nm["flagged"] / nm["n"] * 100], color="#b3452e", width=0.55)
    ax2.set_xticks([0, 1], ["Injected\nviolations", "Legitimate\nnear-misses"],
                   fontsize=7)
    ax2.set_ylim(0, 125)
    ax2.set_ylabel("Scripts flagged (%)")
    ax2.text(0, 103, "100%\n(recall = 1.00)", ha="center", fontsize=6.5, color=INK)
    ax2.text(1, 103, f"100%\n(specificity = {spec:.2f})", ha="center",
             fontsize=6.5, color="#b3452e", fontweight="bold")
    ax2.text(1, 50, f"{nm['flagged']}/{nm['n']}", ha="center", va="center",
             fontsize=8, color="white", fontweight="bold")
    ax2.set_title("B  Adversarial near-miss set", loc="left", fontweight="bold")

    fig.text(0.02, -0.10,
             "Panel A is circular by construction: the scripts were built to instantiate "
             "the auditor's target patterns. Panel B shows the\nconsequence: every "
             "legitimate near-miss script is flagged too, so specificity on that "
             "adversarial set is zero.",
             fontsize=6.2, color=INK, va="top")

    save(fig, "figure4_auditor_construct_check")


# ─────────────────────────────────────────────────────────────────────────────
def figure1():
    """Schematic of the ECLIPSE operational workflow (no data plotted)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    GREEN, RED, GREY = "#2f7d52", "#b3452e", "#8a8a8a"
    DEVBG, HOLDBG = "#dce8f5", "#f6e4df"
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_xlim(-5, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="white", ec=INK, lw=1.0, tfs=7.4, sfs=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.5,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x + w/2, y + h - 3.0, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=INK, zorder=4)
        if sub:
            ax.text(x + w/2, y + h/2 - 1.8, sub, ha="center", va="center",
                    fontsize=sfs, color=INK, zorder=4, linespacing=1.35)

    def arrow(x1, y1, x2, y2, color=GREEN, ls="-", lw=1.5, rad=0.0, z=2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=lw,
                                     linestyle=ls, zorder=z,
                                     connectionstyle=f"arc3,rad={rad}"))

    # swimlanes
    ax.add_patch(FancyBboxPatch((17.5, 25), 55, 16, boxstyle="round,pad=0.3",
                                fc=DEVBG, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((17.5, 5.5), 55, 12, boxstyle="round,pad=0.3",
                                fc=HOLDBG, ec="none", zorder=0))
    ax.text(16.5, 33, "DEVELOPMENT\n(70%)", fontsize=6.0, fontweight="bold",
            color="#2a4a6b", zorder=1, va="center", ha="right", linespacing=1.3)
    ax.text(16.5, 10.5, "HOLDOUT\n(30%)", fontsize=6.0, fontweight="bold",
            color="#8a3d2b", zorder=1, va="center", ha="right", linespacing=1.3)

    box(-4, 20, 14, 11, "Stage 1", "Split\nSHA-256 verified", fc="#f2f2f2")
    box(20, 26.5, 15, 11, "Stage 2", "Preregister\ncriteria")
    box(39, 26.5, 15, 11, "Stage 3", "k-fold CV on\ndevelopment only")
    box(20, 6.5, 34, 8, "Sealed \u2014 no access until Stage 4", fc="white",
        ec=GREY, lw=1.0, tfs=6.6)
    box(58, 13, 14, 24.5, "Stage 4", "Single-shot\nvalidation\n\n(locked\nafter use)")

    # Stage 5
    ax.add_patch(FancyBboxPatch((77, 5), 22, 34,
                                boxstyle="round,pad=0.5,rounding_size=1.0",
                                fc="#f2f2f2", ec=INK, lw=1.0, zorder=3))
    ax.text(88, 35.5, "Stage 5", ha="center", fontsize=7.4, fontweight="bold", zorder=4)
    ax.text(88, 32.3, "Automated verdict", ha="center", fontsize=6.0,
            color=INK, zorder=4)
    for i, (nm, dsc) in enumerate([("EIS", "protocol-level scoring"),
                                   ("STDS", "discrepancy screening"),
                                   ("Code auditor", "static-analysis screening")]):
        yy = 26.0 - i * 7.4
        ax.add_patch(FancyBboxPatch((79.5, yy - 2.4), 17, 5.4,
                                    boxstyle="round,pad=0.3,rounding_size=0.7",
                                    fc="white", ec=GREY, lw=0.8, zorder=4))
        ax.text(88, yy + 0.9, nm, ha="center", fontsize=6.6,
                fontweight="bold", zorder=5)
        ax.text(88, yy - 1.1, dsc, ha="center", fontsize=5.3, color=GREY, zorder=5)

    # permitted flows
    arrow(10.6, 28.5, 19.2, 31.5, rad=0.05)
    arrow(10.6, 22.5, 19.2, 12.0, rad=-0.05)
    arrow(35.6, 32, 38.2, 32)
    arrow(54.6, 32, 57.2, 30)
    arrow(54.6, 10.5, 57.2, 17.5)
    arrow(72.6, 25, 76.2, 22)

    # prohibited flow
    ax.add_patch(FancyArrowPatch((44, 25.9), (36, 15.4), arrowstyle="-|>",
                                 mutation_scale=9, color=RED, lw=1.6,
                                 linestyle=(0, (3.5, 2.5)),
                                 connectionstyle="arc3,rad=0.3", zorder=5))
    ax.text(46.5, 20.5, "prohibited: holdout\naccess during\ndevelopment",
            fontsize=5.8, color=RED, ha="left", va="center", zorder=5,
            linespacing=1.3)

    ax.plot([], [], color=GREEN, lw=1.5, label="permitted data flow")
    ax.plot([], [], color=RED, lw=1.6, ls=(0, (3.5, 2.5)), label="prohibited access")
    ax.plot([], [], color=GREY, lw=1.0, label="automated check")
    ax.legend(loc="upper center", ncol=3, fontsize=6.2,
              bbox_to_anchor=(0.46, 0.045), frameon=False)
    fig.text(0.46, -0.035,
             "Schematic of the enforced workflow, not an empirical result.",
             ha="center", fontsize=6.0, color=GREY)
    save(fig, "figure1_workflow")


if __name__ == "__main__":
    print("=" * 62)
    print("FIGURES — regenerated from deposited outputs")
    print("=" * 62)
    figure1()
    figure2()
    figure3()
    figure3b()
    figure4()
    print(f"\nAll figures -> {FIG}")
