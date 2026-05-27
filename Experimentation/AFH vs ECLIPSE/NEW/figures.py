#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generación de Figuras 3-4 para Paper
Datos reales de holdout validation
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS
# ═══════════════════════════════════════════════════════════════════════════

# Ruta correcta a los archivos CSV
DATA_DIR = Path(r"G:\Mi unidad\AFH\PILOT_BETA\HOLDOUT")
OUTPUT_DIR = Path(r"G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\AFH vs ECLIPSE\NEW")

# Verificar que existan los archivos
csv_base = DATA_DIR / "holdout_results_base.csv"

if not csv_base.exists():
    print(f"ERROR: No se encuentra {csv_base}")
    print(f"\nArchivos disponibles en {DATA_DIR}:")
    for f in DATA_DIR.glob("*.csv"):
        print(f"  {f.name}")
    exit(1)

print(f"✓ Archivo encontrado: {csv_base}")

# ═══════════════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ═══════════════════════════════════════════════════════════════════════════

print("\nCargando datos...")
df = pd.read_csv(csv_base)

print(f"  Total observaciones: {len(df)}")
print(f"  Columnas: {list(df.columns)}")
print(f"\nPrimeras filas:")
print(df.head())

# Verificar que existan las columnas necesarias
required_cols = ['state', 'score_A', 'score_B']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n❌ ERROR: Faltan columnas: {missing_cols}")
    print(f"   Columnas disponibles: {list(df.columns)}")
    exit(1)

# Separar por estado
wake = df[df['state'] == 'wake'].copy()
n3 = df[df['state'] == 'n3'].copy()

print(f"\n✓ Wake observations: {len(wake)}")
print(f"✓ N3 observations: {len(n3)}")

# Calcular estadísticos
wake_sd_b = wake['score_B'].std()
n3_sd_b = n3['score_B'].std()

print(f"\nEstadísticos:")
print(f"  Wake SD(Score B): {wake_sd_b:.3f}")
print(f"  N3 SD(Score B): {n3_sd_b:.3f}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 3: SCATTER PLOT (STATE-DEPENDENT DISSOCIATION)
# ═══════════════════════════════════════════════════════════════════════════

print("\n[1/2] Generando Figura 3 (scatter plot)...")

fig, ax = plt.subplots(figsize=(8, 6))

# Wake
ax.scatter(wake['score_A'], wake['score_B'], 
           c='#1f77b4',  # Blue
           marker='o', 
           s=60,
           alpha=0.6, 
           edgecolors='black',
           linewidths=0.5,
           label='Wake (r=-0.390)',
           zorder=3)

# N3
ax.scatter(n3['score_A'], n3['score_B'], 
           c='#d62728',  # Red
           marker='^', 
           s=80,
           alpha=0.6, 
           edgecolors='black',
           linewidths=0.5,
           label='N3 (r=0.058)',
           zorder=3)

# Líneas de regresión (opcional, para visualizar mejor)
# Wake regression
wake_z = np.polyfit(wake['score_A'], wake['score_B'], 1)
wake_p = np.poly1d(wake_z)
x_wake = np.linspace(wake['score_A'].min(), wake['score_A'].max(), 100)
ax.plot(x_wake, wake_p(x_wake), 
        color='#1f77b4', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.7,
        zorder=2)

# N3 regression (nearly flat)
n3_z = np.polyfit(n3['score_A'], n3['score_B'], 1)
n3_p = np.poly1d(n3_z)
x_n3 = np.linspace(n3['score_A'].min(), n3['score_A'].max(), 100)
ax.plot(x_n3, n3_p(x_n3), 
        color='#d62728', 
        linestyle='--', 
        linewidth=2, 
        alpha=0.7,
        zorder=2)

# Formateo
ax.set_xlabel('Score A (Differentiation)', fontsize=12, fontweight='bold')
ax.set_ylabel('Score B (Temporal Organization)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(alpha=0.3, linestyle=':', zorder=1)
ax.set_axisbelow(True)

# Añadir líneas en cero para referencia
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)

plt.tight_layout()

# Guardar
output_fig3 = OUTPUT_DIR / "fig_correlations.pdf"
plt.savefig(output_fig3, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig_correlations.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Guardado: {output_fig3}")
print(f"  ✓ Guardado: {OUTPUT_DIR / 'fig_correlations.png'}")

plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# FIGURA 4: DISTRIBUCIONES (SCORE A Y B POR ESTADO)
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2/2] Generando Figura 4 (distribuciones)...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Estilo consistente
hist_kwargs = {
    'bins': 20,
    'edgecolor': 'black',
    'linewidth': 0.8,
    'alpha': 0.7
}

# Wake Score A
axes[0, 0].hist(wake['score_A'], color='#1f77b4', **hist_kwargs)
axes[0, 0].set_title('Wake - Score A', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Score A', fontsize=10)
axes[0, 0].set_ylabel('Count', fontsize=10)
axes[0, 0].grid(alpha=0.3, axis='y', linestyle=':')
axes[0, 0].set_axisbelow(True)

# Wake Score B
axes[0, 1].hist(wake['score_B'], color='#1f77b4', **hist_kwargs)
axes[0, 1].set_title(f'Wake - Score B (SD={wake_sd_b:.3f})', 
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Score B', fontsize=10)
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].grid(alpha=0.3, axis='y', linestyle=':')
axes[0, 1].set_axisbelow(True)

# N3 Score A
axes[1, 0].hist(n3['score_A'], color='#d62728', **hist_kwargs)
axes[1, 0].set_title('N3 - Score A', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Score A', fontsize=10)
axes[1, 0].set_ylabel('Count', fontsize=10)
axes[1, 0].grid(alpha=0.3, axis='y', linestyle=':')
axes[1, 0].set_axisbelow(True)

# N3 Score B (con threshold)
axes[1, 1].hist(n3['score_B'], color='#d62728', **hist_kwargs)
axes[1, 1].axvline(0.5, color='black', linestyle='--', linewidth=2, 
                   label='Threshold (SD=0.5)', zorder=3)
axes[1, 1].set_title(f'N3 - Score B (SD={n3_sd_b:.3f})', 
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Score B', fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=10)
axes[1, 1].legend(loc='best', fontsize=9)
axes[1, 1].grid(alpha=0.3, axis='y', linestyle=':')
axes[1, 1].set_axisbelow(True)

# Añadir anotación sobre SD observado vs threshold
if n3_sd_b < 0.5:
    axes[1, 1].text(0.95, 0.95, 
                    f'SD = {n3_sd_b:.3f} < 0.5\n(NOT MET)',
                    transform=axes[1, 1].transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Guardar
output_fig4 = OUTPUT_DIR / "fig_distributions.pdf"
plt.savefig(output_fig4, dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig_distributions.png", dpi=300, bbox_inches='tight')
print(f"  ✓ Guardado: {output_fig4}")
print(f"  ✓ Guardado: {OUTPUT_DIR / 'fig_distributions.png'}")

plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("FIGURAS GENERADAS EXITOSAMENTE")
print("="*80)
print(f"\nArchivos creados:")
print(f"  1. {output_fig3}")
print(f"  2. {output_fig3.with_suffix('.png')}")
print(f"  3. {output_fig4}")
print(f"  4. {output_fig4.with_suffix('.png')}")
print(f"\nEstadísticos clave:")
print(f"  Wake: N={len(wake)}, SD(B)={wake_sd_b:.3f}")
print(f"  N3: N={len(n3)}, SD(B)={n3_sd_b:.3f}")
print(f"  H1 criterion: SD(B|N3) ≥ 0.5 → {'NOT MET' if n3_sd_b < 0.5 else 'MET'}")
print("\n✓ Listo para insertar en LaTeX")
print("="*80)