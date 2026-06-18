import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pickle
from pathlib import Path

print("="*80)
print("VERIFICACIÓN DE RESULTADOS - ECLIPSE v3.2.0")
print("="*80)

# Cargar datos procesados
checkpoint_file = Path(r"G:\Mi unidad\AFH\GITHUB\AFH-MODEL\Experimentation\IIT vs ECLIPSE\V3.2\processing_v3_2_2ch_fast_natural.pkl")

with open(checkpoint_file, 'rb') as f:
    all_windows = pickle.load(f)

df = pd.DataFrame(all_windows)

print(f"\n📊 DATOS BÁSICOS:")
print(f"   Total ventanas: {len(df)}")
print(f"   Sujetos: {df['subject_id'].nunique()}")
print(f"   Conscientes: {(df['consciousness'] == 1).sum()}")
print(f"   Inconscientes: {(df['consciousness'] == 0).sum()}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Verificar etiquetas Sleep Stage
# ═══════════════════════════════════════════════════════════════════════
print(f"\n🔍 TEST 1: ETIQUETAS SLEEP STAGE")
print(f"{'='*80}")

print("\nSleep stages encontrados:")
for stage in sorted(df['sleep_stage'].unique()):
    count = (df['sleep_stage'] == stage).sum()
    consc = df[df['sleep_stage'] == stage]['consciousness'].values[0]
    print(f"   {stage}: {count} ventanas → Consciencia={consc}")

# ¿Hay REM?
has_rem = df['sleep_stage'].str.contains('R', na=False).any()
print(f"\n¿Contiene REM? {has_rem}")

if has_rem:
    rem_count = df['sleep_stage'].str.contains('R', na=False).sum()
    print(f"   Ventanas REM: {rem_count}")
    print(f"   % del total: {rem_count/len(df)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Distribución de Φ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n🔍 TEST 2: DISTRIBUCIÓN DE Φ")
print(f"{'='*80}")

phi_cols = [col for col in df.columns if col.startswith('phi_') and not col.endswith('_time')]

for phi_col in phi_cols:
    print(f"\n{phi_col.upper()}:")
    
    # Estadísticas básicas
    print(f"   Min: {df[phi_col].min():.6f}")
    print(f"   Max: {df[phi_col].max():.6f}")
    print(f"   Mean: {df[phi_col].mean():.6f}")
    print(f"   Std: {df[phi_col].std():.6f}")
    
    # Por grupo
    conscious = df[df['consciousness'] == 1][phi_col]
    unconscious = df[df['consciousness'] == 0][phi_col]
    
    print(f"   Vigilia (n={len(conscious)}): μ={conscious.mean():.6f}, σ={conscious.std():.6f}")
    print(f"   Sueño (n={len(unconscious)}): μ={unconscious.mean():.6f}, σ={unconscious.std():.6f}")
    print(f"   Diferencia: {conscious.mean() - unconscious.mean():.6f}")
    
    # Correlaciones
    pearson_r, pearson_p = pearsonr(df[phi_col], df['consciousness'])
    spearman_r, spearman_p = spearmanr(df[phi_col], df['consciousness'])
    
    print(f"   Pearson: r={pearson_r:.4f}, p={pearson_p:.2e}")
    print(f"   Spearman: ρ={spearman_r:.4f}, p={spearman_p:.2e}")
    
    # Test estadístico
    from scipy.stats import mannwhitneyu
    u_stat, u_p = mannwhitneyu(conscious, unconscious, alternative='two-sided')
    print(f"   Mann-Whitney U: p={u_p:.2e}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Análisis SIN REM
# ═══════════════════════════════════════════════════════════════════════
if has_rem:
    print(f"\n🔍 TEST 3: ANÁLISIS SIN REM")
    print(f"{'='*80}")
    
    df_no_rem = df[~df['sleep_stage'].str.contains('R', na=False)].copy()
    
    print(f"\nDatos sin REM:")
    print(f"   Total ventanas: {len(df_no_rem)}")
    print(f"   Conscientes: {(df_no_rem['consciousness'] == 1).sum()}")
    print(f"   Inconscientes: {(df_no_rem['consciousness'] == 0).sum()}")
    
    for phi_col in phi_cols:
        conscious = df_no_rem[df_no_rem['consciousness'] == 1][phi_col]
        unconscious = df_no_rem[df_no_rem['consciousness'] == 0][phi_col]
        
        spearman_r, spearman_p = spearmanr(df_no_rem[phi_col], df_no_rem['consciousness'])
        
        print(f"\n{phi_col.upper()} (sin REM):")
        print(f"   Vigilia: μ={conscious.mean():.6f}")
        print(f"   Sueño: μ={unconscious.mean():.6f}")
        print(f"   Spearman: ρ={spearman_r:.4f}, p={spearman_p:.2e}")

# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Verificar threshold problem
# ═══════════════════════════════════════════════════════════════════════
print(f"\n🔍 TEST 4: ANÁLISIS DE THRESHOLD")
print(f"{'='*80}")

phi_col = 'phi_multilevel_3'  # El que usaste

print(f"\nDistribución de {phi_col}:")
print(f"   Min: {df[phi_col].min():.6f}")
print(f"   25%: {df[phi_col].quantile(0.25):.6f}")
print(f"   50%: {df[phi_col].quantile(0.50):.6f}")
print(f"   75%: {df[phi_col].quantile(0.75):.6f}")
print(f"   Max: {df[phi_col].max():.6f}")

print(f"\n⚠️  Threshold usado: 0.2709")
print(f"   Ventanas con Φ ≥ 0.2709: {(df[phi_col] >= 0.2709).sum()}")
print(f"   % del total: {(df[phi_col] >= 0.2709).sum()/len(df)*100:.4f}%")

# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Samplear algunas ventanas específicas
# ═══════════════════════════════════════════════════════════════════════
print(f"\n🔍 TEST 5: MUESTRA DE VENTANAS INDIVIDUALES")
print(f"{'='*80}")

# 5 ventanas conscientes
print("\n5 Ventanas CONSCIENTES aleatorias:")
conscious_sample = df[df['consciousness'] == 1].sample(5, random_state=42)
for idx, row in conscious_sample.iterrows():
    print(f"   Sujeto {row['subject_id']}, Ventana {row['window_idx']}: "
          f"Stage={row['sleep_stage']}, Φ={row['phi_multilevel_3']:.6f}")

# 5 ventanas inconscientes
print("\n5 Ventanas INCONSCIENTES aleatorias:")
unconscious_sample = df[df['consciousness'] == 0].sample(5, random_state=42)
for idx, row in unconscious_sample.iterrows():
    print(f"   Sujeto {row['subject_id']}, Ventana {row['window_idx']}: "
          f"Stage={row['sleep_stage']}, Φ={row['phi_multilevel_3']:.6f}")

print("\n" + "="*80)
print("✅ VERIFICACIÓN COMPLETADA")
print("="*80)