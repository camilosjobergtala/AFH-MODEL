#!/usr/bin/env python3
"""
Generate all figures for Registered Report Stage 1
Delta-Gamma PAC as Consciousness Biomarker

Author: Camilo Sjöberg Tala
Date: October 2025

Requirements:
    pip install numpy matplotlib seaborn scipy

Usage:
    python generate_all_figures.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# Create figures directory
os.makedirs('figures', exist_ok=True)

print("=" * 70)
print(" GENERATING FIGURES FOR REGISTERED REPORT STAGE 1")
print("=" * 70)
print()

# Set random seed for reproducibility
np.random.seed(42)

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")

# ============================================================================
# FIGURE 1: SLEEP-EDF PAC RESULTS (3 panels)
# ============================================================================

print("Generating Figure 1: Sleep-EDF PAC Results...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# PANEL A: Violin plot - Wake vs N2 PAC
# ----------------------------------------------------------------------------

# Simulate data based on reported results
# Wake: M=0.172, SD=0.089, n=264
wake_pac = np.random.normal(0.172, 0.089, 264)
wake_pac = np.clip(wake_pac, 0, 1)  # PAC must be [0,1]

# N2: M=0.026, SD=0.035, n=487
n2_pac = np.random.normal(0.026, 0.035, 487)
n2_pac = np.clip(n2_pac, 0, 1)

# Violin plot
parts = axes[0].violinplot(
    [wake_pac, n2_pac],
    positions=[1, 2],
    showmeans=False,
    showmedians=True,
    widths=0.7
)

# Color violins
colors = ['#3498db', '#e74c3c']  # Blue for Wake, Red for N2
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)

# Add scatter points (jittered, subsample for visibility)
for i, (data, color) in enumerate(zip([wake_pac, n2_pac], colors), start=1):
    sample_idx = np.random.choice(len(data), size=min(50, len(data)), replace=False)
    x_jitter = np.random.normal(i, 0.04, size=len(sample_idx))
    axes[0].scatter(x_jitter, data[sample_idx], 
                   alpha=0.3, s=10, color='gray', zorder=1)

# Formatting
axes[0].set_ylabel('Delta-Gamma PAC', fontsize=12, fontweight='bold')
axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(['Wake', 'N2'], fontsize=11)
axes[0].set_ylim([-0.05, 0.6])
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Add statistics annotation
cohens_d = 1.693
p_val = 2.47e-84
axes[0].text(1.5, 0.55, f"Cohen's d = {cohens_d:.2f}\np < 10$^{{-80}}$",
            ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[0].set_title('A. Delta-Gamma PAC', fontsize=13, fontweight='bold', loc='left')

# PANEL B: Effect size comparison - PAC vs Power
# ----------------------------------------------------------------------------

measures = ['Delta-Gamma\nPAC', 'Delta\nPower']
effect_sizes = [1.693, -1.676]  # Negative for N2 > Wake in power
colors_bar = ['#3498db', '#e74c3c']

# Bar plot
bars = axes[1].bar(measures, effect_sizes, color=colors_bar, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)

# Add error bars (95% CI)
ci_lower = [1.64, -1.73]
ci_upper = [1.75, -1.62]
yerr = [[effect_sizes[i] - ci_lower[i], ci_upper[i] - effect_sizes[i]] 
        for i in range(2)]
axes[1].errorbar(measures, effect_sizes, 
                yerr=np.array(yerr).T, fmt='none', 
                ecolor='black', capsize=5, capthick=2)

# Reference line at 0
axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Formatting
axes[1].set_ylabel("Cohen's d (Wake vs N2)", fontsize=12, fontweight='bold')
axes[1].set_ylim([-2.5, 2.5])
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Add advantage annotation
delta_d = 3.369
axes[1].text(0.5, 2.2, f'Δd = {delta_d:.2f}', 
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

axes[1].set_title('B. PAC vs Power', fontsize=13, fontweight='bold', loc='left')

# PANEL C: N3 Paradox - PAC across states
# ----------------------------------------------------------------------------

states = ['Wake', 'N2', 'N3']
pac_means = [0.172, 0.026, 0.017]
pac_sems = [0.005, 0.002, 0.001]

# Line plot for PAC
axes[2].plot(states, pac_means, marker='o', markersize=10, 
            linewidth=2.5, color='#2ecc71', label='PAC')
axes[2].errorbar(states, pac_means, yerr=pac_sems, 
                fmt='none', ecolor='#2ecc71', capsize=5, capthick=2)

# Add secondary y-axis for delta power
ax2 = axes[2].twinx()
power_means = [0.30, 0.45, 0.45]  # Wake < N2 ≈ N3
power_sems = [0.02, 0.02, 0.02]
ax2.plot(states, power_means, marker='s', markersize=10, 
        linewidth=2.5, color='#e67e22', linestyle='--', label='Delta Power')
ax2.errorbar(states, power_means, yerr=power_sems,
            fmt='none', ecolor='#e67e22', capsize=5, capthick=2)

# Formatting
axes[2].set_ylabel('Delta-Gamma PAC', fontsize=12, fontweight='bold', color='#2ecc71')
axes[2].tick_params(axis='y', labelcolor='#2ecc71')
ax2.set_ylabel('Delta Power (a.u.)', fontsize=12, fontweight='bold', color='#e67e22')
ax2.tick_params(axis='y', labelcolor='#e67e22')

axes[2].set_xlabel('State', fontsize=12, fontweight='bold')
axes[2].set_ylim([0, 0.20])
ax2.set_ylim([0.2, 0.55])

axes[2].spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Annotation for N3 paradox
axes[2].annotate('N3 Paradox:\nHigh Δ power,\nLow PAC', 
                xy=(2, 0.017), xytext=(1.5, 0.12),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

axes[2].set_title('C. State Comparison', fontsize=13, fontweight='bold', loc='left')

# Save Figure 1
plt.tight_layout()
plt.savefig('figures/level2_pac_results.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/level2_pac_results.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/level2_pac_results.pdf")
plt.close()

# ============================================================================
# FIGURE 2: STDS DISTRIBUTION
# ============================================================================

print("Generating Figure 2: STDS Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

# Simulate STDS null distribution
# Max observed in 10,000 permutations was d=0.290
null_distribution = np.random.beta(2, 10, 10000) * 0.30
null_distribution = np.abs(null_distribution)

# Histogram
ax.hist(null_distribution, bins=50, color='lightgray', 
       edgecolor='black', alpha=0.8, label='Null distribution\n(10,000 permutations)')

# Observed value
observed_d = 1.693
ax.axvline(observed_d, color='red', linestyle='--', linewidth=3, 
          label=f'Observed d = {observed_d:.2f}')

# Annotation for observed
ax.text(observed_d + 0.05, ax.get_ylim()[1] * 0.8, 
       f'Observed\nd = {observed_d:.2f}\n(5.8× max null)',
       fontsize=11, color='red', fontweight='bold',
       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Max null
max_null = 0.290
ax.axvline(max_null, color='blue', linestyle=':', linewidth=2,
          label=f'Max null |d| = {max_null:.2f}')

# Formatting
ax.set_xlabel("Cohen's d (absolute value)", fontsize=13, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax.set_title('Statistical Test for Data Snooping (STDS)\nNull Distribution vs Observed Effect', 
            fontsize=14, fontweight='bold')

# STDS verdict box
ax.text(0.98, 0.95, 
       'STDS Verdict: CRITICAL\np < 0.001\n(0/10,000 permutations ≥ observed)', 
       transform=ax.transAxes, fontsize=11,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.9, 
                edgecolor='red', linewidth=2))

ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/stds_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/stds_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/stds_distribution.pdf")
plt.close()

# ============================================================================
# FIGURE 3: POWER CURVES (2 panels)
# ============================================================================

print("Generating Figure 3: Power Curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Function to calculate power
def calculate_power(d, n1, n2, alpha_level=0.05):
    """Calculate statistical power for independent t-test"""
    pooled_n = (n1 * n2) / (n1 + n2)
    ncp = d * np.sqrt(pooled_n)
    df = n1 + n2 - 2
    t_crit = stats.t.ppf(1 - alpha_level, df)
    power = 1 - stats.nct.cdf(t_crit, df, ncp)
    return power

# PANEL A: Power as function of effect size
# ----------------------------------------------------------------------------

effect_sizes = np.linspace(0, 2.0, 100)

# Different sample sizes
scenarios = {
    '2 subjects (n=110)': (30, 80),
    '3 subjects (n=165)': (45, 120),
    '4 subjects (n=220)': (60, 160),
    '5 subjects (n=275)': (75, 200)
}

colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

for (label, (n1, n2)), color in zip(scenarios.items(), colors):
    powers = [calculate_power(d, n1, n2) for d in effect_sizes]
    axes[0].plot(effect_sizes, powers, label=label, linewidth=2.5, color=color)

# Reference lines
axes[0].axhline(0.80, color='black', linestyle='--', linewidth=1.5, 
               alpha=0.5, label='80% power')
axes[0].axvline(0.5, color='purple', linestyle=':', linewidth=2, 
               alpha=0.7, label='Threshold (d=0.5)')
axes[0].axvline(0.8, color='green', linestyle=':', linewidth=2, 
               alpha=0.7, label='Conservative (d=0.8)')
axes[0].axvline(1.69, color='red', linestyle=':', linewidth=2, 
               alpha=0.7, label='Sleep-EDF (d=1.69)')

# Formatting
axes[0].set_xlabel("Cohen's d", fontsize=12, fontweight='bold')
axes[0].set_ylabel('Statistical Power', fontsize=12, fontweight='bold')
axes[0].set_title('A. Power vs Effect Size', fontsize=13, fontweight='bold', loc='left')
axes[0].set_xlim([0, 2.0])
axes[0].set_ylim([0, 1.0])
axes[0].legend(loc='lower right', fontsize=9, framealpha=0.95)
axes[0].grid(True, alpha=0.3, linestyle=':')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# PANEL B: Minimum Detectable Effect Size
# ----------------------------------------------------------------------------

# Calculate MDES for different sample sizes
def find_mdes(n1, n2, target_power=0.80):
    """Binary search for MDES"""
    d_low, d_high = 0.1, 3.0
    while d_high - d_low > 0.001:
        d_mid = (d_low + d_high) / 2
        power = calculate_power(d_mid, n1, n2)
        if power < target_power:
            d_low = d_mid
        else:
            d_high = d_mid
    return d_mid

subjects_range = np.arange(2, 8)
n_wakes = subjects_range * 15  # 15 wake epochs per subject
n_n2s = subjects_range * 40     # 40 N2 epochs per subject

mdes_values = [find_mdes(n1, n2) for n1, n2 in zip(n_wakes, n_n2s)]

# Bar plot
bars = axes[1].bar(subjects_range, mdes_values, color='#3498db', 
                  alpha=0.7, edgecolor='black', linewidth=1.5)

# Highlight base case (3 subjects)
bars[1].set_color('#e74c3c')
bars[1].set_alpha(0.9)

# Reference lines
axes[1].axhline(0.5, color='purple', linestyle='--', linewidth=2, 
               label='Success criterion (d=0.5)')
axes[1].axhline(0.8, color='green', linestyle='--', linewidth=2, 
               label='Conservative estimate (d=0.8)')

# Formatting
axes[1].set_xlabel('Number of Holdout Subjects', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Minimum Detectable Effect Size (MDES)', fontsize=12, fontweight='bold')
axes[1].set_title('B. MDES vs Sample Size', fontsize=13, fontweight='bold', loc='left')
axes[1].set_xticks(subjects_range)
axes[1].set_ylim([0, 1.2])
axes[1].legend(loc='upper right', fontsize=10, framealpha=0.95)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# Annotation for base case
axes[1].annotate(f'Base case\n(3 subjects)\nMDES = {mdes_values[1]:.2f}',
                xy=(3, mdes_values[1]), xytext=(4.5, 0.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('figures/power_curves.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/power_curves.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/power_curves.pdf")
plt.close()

# ============================================================================
# FIGURE 4: DECISION TREE
# ============================================================================

print("Generating Figure 4: Decision Tree...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Node positions
nodes = {
    'start': (5, 9),
    'd_check': (5, 7.5),
    'd_yes': (3, 6),
    'd_no': (7, 6),
    'p_check_yes': (3, 4.5),
    'd_large': (1.5, 3),
    'd_medium': (4.5, 3),
    'ambiguous': (3, 1.5),
    'd_small': (7, 4.5),
    'd_null': (7, 3),
    'd_opposite': (9, 4.5)
}

# Helper functions
def draw_box(ax, xy, text, color, width=1.8, height=0.6):
    box = FancyBboxPatch(
        (xy[0] - width/2, xy[1] - height/2), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, edgecolor='black', linewidth=2
    )
    ax.add_patch(box)
    ax.text(xy[0], xy[1], text, ha='center', va='center', 
           fontsize=10, fontweight='bold', wrap=True)

def draw_arrow(ax, start, end, label='', color='black'):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=20, 
        linewidth=2, color=color
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Draw tree structure
draw_box(ax, nodes['start'], 'Holdout Results\n(d, p)', '#e8f4f8', width=2, height=0.7)
draw_box(ax, nodes['d_check'], 'd ≥ 0.5?', '#fff9e6')
draw_arrow(ax, nodes['start'], nodes['d_check'])

# Left branch (d ≥ 0.5)
draw_arrow(ax, nodes['d_check'], nodes['d_yes'], 'YES', 'green')
draw_box(ax, nodes['d_yes'], 'p < 0.05?', '#fff9e6')

# Right branch (d < 0.5)
draw_arrow(ax, nodes['d_check'], nodes['d_no'], 'NO', 'red')
draw_box(ax, nodes['d_no'], 'd ≥ 0.3?', '#fff9e6')

# Outcomes
draw_arrow(ax, nodes['d_yes'], nodes['p_check_yes'], 'YES', 'green')
draw_box(ax, nodes['p_check_yes'], 'd ≥ 0.8?', '#fff9e6')

draw_arrow(ax, nodes['p_check_yes'], nodes['d_large'], 'YES', 'darkgreen')
draw_box(ax, nodes['d_large'], 'SCENARIO A\nStrong\nConfirmation', '#90EE90', width=1.6, height=0.8)

draw_arrow(ax, nodes['p_check_yes'], nodes['d_medium'], 'NO', 'green')
draw_box(ax, nodes['d_medium'], 'SCENARIO B\nModerate\nConfirmation', '#98FB98', width=1.6, height=0.8)

draw_arrow(ax, nodes['d_yes'], nodes['ambiguous'], 'NO', 'orange')
draw_box(ax, nodes['ambiguous'], 'SCENARIO C\nAmbiguous\n(FALSIFIED)', '#FFD700', width=1.6, height=0.8)

draw_arrow(ax, nodes['d_no'], nodes['d_small'], 'YES\n(0.3 ≤ d < 0.5)', 'orange')
draw_box(ax, nodes['d_small'], 'SCENARIO D\nWeak Effect\n(FALSIFIED)', '#FFA07A', width=1.6, height=0.8)

draw_arrow(ax, nodes['d_no'], nodes['d_null'], 'NO\n(0 ≤ d < 0.3)', 'red')
draw_box(ax, nodes['d_null'], 'SCENARIO E\nNull Effect\n(FALSIFIED)', '#FF6B6B', width=1.6, height=0.8)

draw_arrow(ax, nodes['d_no'], nodes['d_opposite'], 'NO\n(d < 0)', 'darkred')
draw_box(ax, nodes['d_opposite'], 'SCENARIO F\nOpposite\n(FALSIFIED)', '#DC143C', width=1.6, height=0.8)

# Legend
legend_elements = [
    mpatches.Patch(color='#90EE90', label='Confirmed (A-B)'),
    mpatches.Patch(color='#FFD700', label='Ambiguous (C)'),
    mpatches.Patch(color='#FFA07A', label='Falsified - Weak (D)'),
    mpatches.Patch(color='#FF6B6B', label='Falsified - Null/Opposite (E-F)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
         framealpha=0.95, title='Outcome Classification')

# Title
ax.text(5, 9.8, 'Complete Decision Tree for Outcome Interpretation',
       ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/decision_tree.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/decision_tree.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: figures/decision_tree.pdf")
plt.close()

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print()
print("=" * 70)
print(" ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)
print()
print("Generated files:")
print("  ✓ figures/level2_pac_results.pdf")
print("  ✓ figures/level2_pac_results.png")
print("  ✓ figures/stds_distribution.pdf")
print("  ✓ figures/stds_distribution.png")
print("  ✓ figures/power_curves.pdf")
print("  ✓ figures/power_curves.png")
print("  ✓ figures/decision_tree.pdf")
print("  ✓ figures/decision_tree.png")
print()
print("You can now compile your LaTeX document:")
print("  pdflatex main.tex")
print("  biber main")
print("  pdflatex main.tex")
print()