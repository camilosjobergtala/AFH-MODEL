"""
01c_sensitivity_g_star.py
═══════════════════════════════════════════════════════════════════════════════
STUDY 1 SENSITIVITY ANALYSIS — S_leak saturation threshold g_star

Re-runs the Study 1 corpus generation (same 300 synthetic protocols, same
master seed 20250805) under g_star in {0.01, 0.02, 0.05} and reports how the
two Study 1 headline results change:
  1. Dimensional non-redundancy (Spearman correlations among the five
     components, in particular leakage_score's correlation with the other four).
  2. Weight stability (Spearman rank-corr of each weighting scheme's EIS
     ordering vs the default-scheme ordering).

g_star only rescales S_leak's saturation point — it does not change which
protocols are flagged as "no expected gap" (s=0) vs "safe" (s=1) at the
extremes, but it changes how many protocols fall in the partially-saturated
middle band, so it can shift leakage_score's variance and its correlations.

PROVENANCE: reuses generate_corpus() and WEIGHT_SCHEMES from
01_study1_eis_coherence.py — not a reimplementation. All scores come from the
canonical engine via eis_wrapper.score_protocol(g_star=...).
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "engine"))

from eis_wrapper import components_of  # noqa: E402
from importlib import import_module

study1 = import_module("01_study1_eis_coherence")

MASTER_SEED = 20250805
N = 300
G_STAR_VALUES = [0.01, 0.02, 0.05]


def analyze_one(g_star: float) -> dict:
    df = study1.generate_corpus(N, MASTER_SEED, g_star=g_star)

    corr = df[study1.COMPONENTS].corr(method='spearman')
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    max_abs_corr = float(np.nanmax(np.abs(off_diag.to_numpy())))
    mean_abs_corr = float(np.nanmean(np.abs(off_diag.to_numpy())))
    leakage_corrs = {
        other: float(corr.loc['leakage_score', other])
        for other in study1.COMPONENTS if other != 'leakage_score'
    }

    ws, linearity_diff = study1.analysis_weight_stability(df)
    alt = ws[ws['scheme'] != 'default']
    min_rho = float(alt['spearman_vs_default'].min())
    mean_rho = float(alt['spearman_vs_default'].mean())
    leakage_dom_rho = float(ws.loc[ws['scheme'] == 'leakage_dom', 'spearman_vs_default'].iloc[0])

    return {
        'g_star': g_star,
        'leakage_score_mean': float(df['leakage_score'].mean()),
        'leakage_score_std': float(df['leakage_score'].std()),
        'leakage_score_at_0': float((df['leakage_score'] == 0.0).mean()),
        'leakage_score_at_1': float((df['leakage_score'] == 1.0).mean()),
        'non_redundancy': {
            'max_abs_offdiag_spearman': max_abs_corr,
            'mean_abs_offdiag_spearman': mean_abs_corr,
            'leakage_score_correlations': leakage_corrs,
        },
        'weight_stability': {
            'min_rank_corr_vs_default': min_rho,
            'mean_rank_corr_vs_default': mean_rho,
            'leakage_dom_rank_corr_vs_default': leakage_dom_rho,
            'linearity_check_max_diff': linearity_diff,
        },
        'weight_stability_table': ws.to_dict('records'),
    }


def main():
    root = HERE.parent if (HERE.parent / "engine").exists() else HERE
    out_dir = root / "outputs" / "study1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STUDY 1 SENSITIVITY — S_leak saturation threshold g_star")
    print("=" * 70)
    print(f"Master seed: {MASTER_SEED} | corpus size: {N} | g_star values: {G_STAR_VALUES}")

    results = {}
    for g_star in G_STAR_VALUES:
        print(f"\n--- g_star = {g_star} ---")
        r = analyze_one(g_star)
        results[str(g_star)] = r
        print(f"  leakage_score: mean={r['leakage_score_mean']:.4f} "
              f"std={r['leakage_score_std']:.4f} "
              f"frac@0={r['leakage_score_at_0']:.3f} frac@1={r['leakage_score_at_1']:.3f}")
        print(f"  leakage_score correlations with other components:")
        for other, rho in r['non_redundancy']['leakage_score_correlations'].items():
            print(f"    {other:20s} {rho:+.4f}")
        print(f"  max |off-diag Spearman| = {r['non_redundancy']['max_abs_offdiag_spearman']:.4f}")
        print(f"  weight stability: min rank-corr = {r['weight_stability']['min_rank_corr_vs_default']:.4f}, "
              f"mean rank-corr = {r['weight_stability']['mean_rank_corr_vs_default']:.4f}, "
              f"leakage_dom rank-corr = {r['weight_stability']['leakage_dom_rank_corr_vs_default']:.4f}")

    # --- summary table across g_star values ---
    summary_rows = []
    for g_star in G_STAR_VALUES:
        r = results[str(g_star)]
        summary_rows.append({
            'g_star': g_star,
            'leakage_score_mean': r['leakage_score_mean'],
            'leakage_score_std': r['leakage_score_std'],
            'frac_at_0': r['leakage_score_at_0'],
            'frac_at_1': r['leakage_score_at_1'],
            'max_abs_offdiag_spearman': r['non_redundancy']['max_abs_offdiag_spearman'],
            'mean_abs_offdiag_spearman': r['non_redundancy']['mean_abs_offdiag_spearman'],
            'min_rank_corr_vs_default': r['weight_stability']['min_rank_corr_vs_default'],
            'mean_rank_corr_vs_default': r['weight_stability']['mean_rank_corr_vs_default'],
            'leakage_dom_rank_corr_vs_default': r['weight_stability']['leakage_dom_rank_corr_vs_default'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "sensitivity_g_star_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS g_star VALUES")
    print("=" * 70)
    print(summary_df.round(4).to_string(index=False))

    with open(out_dir / "sensitivity_g_star_full.json", 'w', encoding='utf-8') as f:
        json.dump({'master_seed': MASTER_SEED, 'corpus_size': N,
                   'g_star_values': G_STAR_VALUES,
                   'results': results,
                   'timestamp': datetime.now().isoformat()}, f, indent=2)

    print(f"\nOutputs -> {out_dir / 'sensitivity_g_star_summary.csv'}")
    print(f"           {out_dir / 'sensitivity_g_star_full.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
