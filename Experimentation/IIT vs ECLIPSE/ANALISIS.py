"""
Verificación rápida de resultados IIT - SIN re-procesamiento
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import matthews_corrcoef

print("=" * 80)
print("🔍 VERIFICACIÓN DE RESULTADOS IIT")
print("=" * 80)

# Cargar datos procesados desde checkpoint
checkpoint_file = Path("./eclipse_results_v2/processing_v2_8ch_combined.pkl")

if not checkpoint_file.exists():
    print(f"❌ No se encuentra: {checkpoint_file}")
    print("\n¿Archivos disponibles?")
    for f in Path("./eclipse_results_v2").glob("*.pkl"):
        print(f"   - {f.name}")
    exit(1)

print(f"\n📂 Cargando: {checkpoint_file}")
with open(checkpoint_file, 'rb') as f:
    all_windows = pickle.load(f)

df = pd.DataFrame(all_windows)
print(f"✅ {len(df)} ventanas cargadas")

print("\n" + "=" * 80)
print("TEST 1: Φ PROMEDIO POR ESTADO")
print("=" * 80)

phi_wake = df[df['consciousness'] == 1]['phi'].mean()
phi_sleep = df[df['consciousness'] == 0]['phi'].mean()
phi_wake_std = df[df['consciousness'] == 1]['phi'].std()
phi_sleep_std = df[df['consciousness'] == 0]['phi'].std()

print(f"\n📊 Estadísticas:")
print(f"   VIGILIA (consciente):")
print(f"      Media: {phi_wake:.4f} ± {phi_wake_std:.4f}")
print(f"      N = {(df['consciousness'] == 1).sum()}")
print(f"\n   SUEÑO (inconsciente):")
print(f"      Media: {phi_sleep:.4f} ± {phi_sleep_std:.4f}")
print(f"      N = {(df['consciousness'] == 0).sum()}")

ratio = phi_sleep / phi_wake
print(f"\n🔥 RATIO (sueño/vigilia): {ratio:.2f}x")

if ratio > 1.5:
    print(f"   ❌ Φ es {ratio:.1f}× MAYOR en sueño (OPUESTO a IIT)")
elif ratio > 1.1:
    print(f"   ⚠️  Φ es {ratio:.1f}× mayor en sueño (contradice IIT)")
else:
    print(f"   ✅ Φ es similar o mayor en vigilia")

print("\n" + "=" * 80)
print("TEST 2: DISTRIBUCIONES (Kolmogorov-Smirnov)")
print("=" * 80)

wake_phi = df[df['consciousness'] == 1]['phi'].values
sleep_phi = df[df['consciousness'] == 0]['phi'].values

ks_stat, ks_p = ks_2samp(wake_phi, sleep_phi)
print(f"\nKS Statistic: {ks_stat:.4f}")
print(f"P-value: {ks_p:.10f}")

if ks_p < 0.001:
    print(f"✅ Distribuciones son SIGNIFICATIVAMENTE diferentes (p < 0.001)")
else:
    print(f"⚠️  Distribuciones no son muy diferentes")

print("\n" + "=" * 80)
print("TEST 3: CLASIFICADOR INVERSO")
print("=" * 80)

# Clasificador normal: Φ alto → consciente (predicción IIT)
threshold_median = df['phi'].median()
pred_normal = (df['phi'] >= threshold_median).astype(int)
mcc_normal = matthews_corrcoef(df['consciousness'], pred_normal)

# Clasificador inverso: Φ BAJO → consciente (opuesto a IIT)
pred_inverse = (df['phi'] < threshold_median).astype(int)
mcc_inverse = matthews_corrcoef(df['consciousness'], pred_inverse)

print(f"\n🔵 Clasificador NORMAL (IIT: Φ alto → consciente):")
print(f"   MCC = {mcc_normal:.4f}")
acc_normal = (pred_normal == df['consciousness']).sum() / len(df)
print(f"   Accuracy = {acc_normal:.4f}")

print(f"\n🔴 Clasificador INVERSO (ANTI-IIT: Φ bajo → consciente):")
print(f"   MCC = {mcc_inverse:.4f}")
acc_inverse = (pred_inverse == df['consciousness']).sum() / len(df)
print(f"   Accuracy = {acc_inverse:.4f}")

print(f"\n🎯 DIFERENCIA: {abs(mcc_inverse - mcc_normal):.4f}")

if mcc_inverse > mcc_normal + 0.05:
    print(f"   ❌ Clasificador INVERSO es MEJOR → Φ tiene correlación NEGATIVA")
    print(f"   → IIT FALSIFICADA: Φ predice opuesto a consciencia")
elif abs(mcc_inverse - mcc_normal) < 0.05:
    print(f"   ⚠️  Ambos clasificadores similares → Φ no predice consciencia")
else:
    print(f"   ✅ Clasificador normal es mejor → Φ correlaciona con consciencia")

print("\n" + "=" * 80)
print("TEST 4: EJEMPLOS CONCRETOS")
print("=" * 80)

# Ventanas con Φ MÁS ALTO
top_phi = df.nlargest(10, 'phi')[['phi', 'consciousness', 'sleep_stage']]
print(f"\n🔝 TOP 10 ventanas con Φ MÁS ALTO:")
for idx, row in top_phi.iterrows():
    state = "VIGILIA ✓" if row['consciousness'] == 1 else "SUEÑO ✗"
    print(f"   Φ={row['phi']:.4f} | {state} | {row.get('sleep_stage', 'N/A')}")

conscious_count_top = (top_phi['consciousness'] == 1).sum()
print(f"\n   {conscious_count_top}/10 son VIGILIA")
if conscious_count_top < 3:
    print(f"   ❌ Φ alto ocurre principalmente en SUEÑO (contradice IIT)")

# Ventanas con Φ MÁS BAJO
bottom_phi = df.nsmallest(10, 'phi')[['phi', 'consciousness', 'sleep_stage']]
print(f"\n🔻 TOP 10 ventanas con Φ MÁS BAJO:")
for idx, row in bottom_phi.iterrows():
    state = "VIGILIA ✓" if row['consciousness'] == 1 else "SUEÑO ✗"
    print(f"   Φ={row['phi']:.4f} | {state} | {row.get('sleep_stage', 'N/A')}")

conscious_count_bottom = (bottom_phi['consciousness'] == 1).sum()
print(f"\n   {conscious_count_bottom}/10 son VIGILIA")
if conscious_count_bottom > 7:
    print(f"   ❌ Φ bajo ocurre principalmente en VIGILIA (contradice IIT)")

print("\n" + "=" * 80)
print("TEST 5: PERCENTILES")
print("=" * 80)

percentiles = [10, 25, 50, 75, 90]
print(f"\n{'Percentil':<12} {'Φ value':<12} {'% Vigilia':<12}")
print("-" * 36)

for p in percentiles:
    phi_threshold = np.percentile(df['phi'], p)
    subset = df[df['phi'] >= phi_threshold]
    pct_conscious = (subset['consciousness'] == 1).sum() / len(subset) * 100
    print(f"P{p:<10} {phi_threshold:<12.4f} {pct_conscious:<12.1f}%")

print("\n💡 Interpretación:")
print("   Si IIT es correcta: % Vigilia debería AUMENTAR con percentil")
print("   Si es incorrecta: % Vigilia DISMINUYE con percentil")

print("\n" + "=" * 80)
print("🎯 VEREDICTO FINAL")
print("=" * 80)

verdict_score = 0

# Criterio 1: Ratio
if ratio > 1.5:
    print("❌ Φ es 1.5x+ mayor en sueño")
    verdict_score += 2
elif ratio > 1.1:
    print("⚠️  Φ es algo mayor en sueño")
    verdict_score += 1

# Criterio 2: Clasificador inverso
if mcc_inverse > mcc_normal + 0.05:
    print("❌ Clasificador inverso es mejor")
    verdict_score += 2

# Criterio 3: Top Φ
if conscious_count_top < 3:
    print("❌ Alto Φ ocurre en sueño")
    verdict_score += 1

print(f"\n📊 Score: {verdict_score}/5")

if verdict_score >= 4:
    print("\n🔥 CONCLUSIÓN: IIT 3.0 FALSIFICADA")
    print("   Φ* tiene correlación NEGATIVA robusta con consciencia")
    print("   Resultado es publicable con limitaciones claras")
elif verdict_score >= 2:
    print("\n⚠️  CONCLUSIÓN: IIT 3.0 CUESTIONADA")
    print("   Φ* no predice bien consciencia")
elif verdict_score == 1:
    print("\n🤔 CONCLUSIÓN: RESULTADOS AMBIGUOS")
else:
    print("\n✅ CONCLUSIÓN: IIT 3.0 APOYADA")

print("\n" + "=" * 80)
print("✅ VERIFICACIÓN COMPLETA")
print("=" * 80)