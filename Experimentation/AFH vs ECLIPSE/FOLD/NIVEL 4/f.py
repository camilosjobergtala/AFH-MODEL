#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
ANÁLISIS DE CAUSALIDAD TEMPORAL - PAC vs SPECTRAL POWER
================================================================================

Pregunta: ¿PAC δ-γ PRECEDE causalmente a las métricas espectrales?

Métodos:
1. Cross-correlation con lag (simple, robusto)
2. Granger Causality (estadísticamente formal)
3. Transfer Entropy (no-lineal, más potente)

================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SERIES_DIR = Path(r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\awakening_cascade_v3.6\series")
OUTPUT_DIR = Path(r"G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\awakening_cascade_v3.6\causality")

# Métricas a comparar
PAC_METRIC = 'pac_delta_gamma_z'
SPECTRAL_METRICS = [
    'relpower_beta', 'relpower_gamma', 'relpower_delta', 'relpower_theta', 'relpower_alpha',
    'power_beta', 'power_gamma', 'power_delta', 'power_theta', 'power_alpha',
    'ratio_beta_delta', 'ratio_fast_slow', 'slope_aperiodic',
]

# Lags a evaluar (en ventanas de 30s)
MAX_LAG = 5  # 5 ventanas = 2.5 minutos

# Mínimo de puntos válidos para análisis
MIN_VALID_POINTS = 20


# ============================================================================
# FUNCIONES DE CAUSALIDAD
# ============================================================================

def cross_correlation_with_lag(x, y, max_lag=5):
    """
    Calcula correlación cruzada con diferentes lags.
    
    Retorna: para cada lag, r(x[t], y[t+lag]) y r(y[t], x[t+lag])
    
    Si x lidera y: r(x[t], y[t+lag]) > r(y[t], x[t+lag])
    """
    results = []
    
    for lag in range(1, max_lag + 1):
        # x en t predice y en t+lag
        x_early = x[:-lag]
        y_late = y[lag:]
        
        # y en t predice x en t+lag
        y_early = y[:-lag]
        x_late = x[lag:]
        
        # Eliminar NaN
        valid_xy = ~(np.isnan(x_early) | np.isnan(y_late))
        valid_yx = ~(np.isnan(y_early) | np.isnan(x_late))
        
        if valid_xy.sum() < 10 or valid_yx.sum() < 10:
            continue
        
        r_x_leads, p_x_leads = pearsonr(x_early[valid_xy], y_late[valid_xy])
        r_y_leads, p_y_leads = pearsonr(y_early[valid_yx], x_late[valid_yx])
        
        results.append({
            'lag': lag,
            'lag_seconds': lag * 30,
            'r_x_leads_y': r_x_leads,
            'p_x_leads_y': p_x_leads,
            'r_y_leads_x': r_y_leads,
            'p_y_leads_x': p_y_leads,
            'diff': r_x_leads - r_y_leads,  # positivo = x lidera
            'n_valid': min(valid_xy.sum(), valid_yx.sum()),
        })
    
    return results


def granger_causality_simple(x, y, max_lag=3):
    """
    Test de Granger simplificado usando regresión.
    
    H0: x no Granger-causa y
    H1: x Granger-causa y (pasado de x mejora predicción de y)
    
    Compara:
    - Modelo restringido: y[t] ~ y[t-1] + ... + y[t-lag]
    - Modelo completo: y[t] ~ y[t-1] + ... + y[t-lag] + x[t-1] + ... + x[t-lag]
    
    F-test para ver si x agrega poder predictivo.
    """
    from scipy.stats import f as f_dist
    
    n = len(x)
    if n < max_lag + 10:
        return {'f_stat': np.nan, 'p_value': np.nan, 'direction': 'insufficient_data'}
    
    # Crear matrices de regresión
    y_target = y[max_lag:]
    n_obs = len(y_target)
    
    # Modelo restringido: solo lags de y
    X_restricted = np.column_stack([y[max_lag-i-1:n-i-1] for i in range(max_lag)])
    
    # Modelo completo: lags de y + lags de x
    X_full = np.column_stack([
        X_restricted,
        *[x[max_lag-i-1:n-i-1] for i in range(max_lag)]
    ])
    
    # Eliminar filas con NaN
    valid = ~(np.isnan(y_target) | np.any(np.isnan(X_restricted), axis=1) | np.any(np.isnan(X_full), axis=1))
    
    if valid.sum() < max_lag * 4:
        return {'f_stat': np.nan, 'p_value': np.nan, 'direction': 'insufficient_data'}
    
    y_valid = y_target[valid]
    X_r_valid = X_restricted[valid]
    X_f_valid = X_full[valid]
    n_valid = len(y_valid)
    
    try:
        # Regresión restringida
        X_r_bias = np.column_stack([np.ones(n_valid), X_r_valid])
        beta_r = np.linalg.lstsq(X_r_bias, y_valid, rcond=None)[0]
        resid_r = y_valid - X_r_bias @ beta_r
        rss_r = np.sum(resid_r**2)
        
        # Regresión completa
        X_f_bias = np.column_stack([np.ones(n_valid), X_f_valid])
        beta_f = np.linalg.lstsq(X_f_bias, y_valid, rcond=None)[0]
        resid_f = y_valid - X_f_bias @ beta_f
        rss_f = np.sum(resid_f**2)
        
        # F-test
        df1 = max_lag  # parámetros adicionales
        df2 = n_valid - 2 * max_lag - 1  # grados de libertad residuales
        
        if df2 <= 0 or rss_f <= 0:
            return {'f_stat': np.nan, 'p_value': np.nan, 'direction': 'degenerate'}
        
        f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
        p_value = 1 - f_dist.cdf(f_stat, df1, df2)
        
        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'rss_restricted': rss_r,
            'rss_full': rss_f,
            'r2_improvement': (rss_r - rss_f) / rss_r,
            'n_obs': n_valid,
        }
    except Exception as e:
        return {'f_stat': np.nan, 'p_value': np.nan, 'direction': f'error: {e}'}


def bidirectional_granger(x, y, max_lag=3):
    """
    Calcula Granger en ambas direcciones:
    - x → y (x Granger-causa y)
    - y → x (y Granger-causa x)
    
    Determina dirección dominante.
    """
    gc_x_to_y = granger_causality_simple(x, y, max_lag)
    gc_y_to_x = granger_causality_simple(y, x, max_lag)
    
    # Determinar dirección
    p_xy = gc_x_to_y.get('p_value', 1)
    p_yx = gc_y_to_x.get('p_value', 1)
    
    if np.isnan(p_xy) or np.isnan(p_yx):
        direction = 'insufficient_data'
    elif p_xy < 0.05 and p_yx >= 0.05:
        direction = 'x_causes_y'
    elif p_yx < 0.05 and p_xy >= 0.05:
        direction = 'y_causes_x'
    elif p_xy < 0.05 and p_yx < 0.05:
        direction = 'bidirectional'
    else:
        direction = 'no_causality'
    
    return {
        'gc_x_to_y': gc_x_to_y,
        'gc_y_to_x': gc_y_to_x,
        'direction': direction,
        'p_x_to_y': p_xy,
        'p_y_to_x': p_yx,
    }


# ============================================================================
# ANÁLISIS PRINCIPAL
# ============================================================================

def analyze_single_transition(npz_path, pac_metric, spectral_metrics):
    """Analiza causalidad en una transición."""
    
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            times = data['times']
            
            if pac_metric not in data.files:
                return None
            
            pac = data[pac_metric]
            
            results = []
            for spec_metric in spectral_metrics:
                if spec_metric not in data.files:
                    continue
                
                spec = data[spec_metric]
                
                # Verificar datos válidos
                valid = ~(np.isnan(pac) | np.isnan(spec))
                if valid.sum() < MIN_VALID_POINTS:
                    continue
                
                # 1. Cross-correlation con lag
                cc_results = cross_correlation_with_lag(pac, spec, MAX_LAG)
                
                # 2. Granger bidireccional
                gc_results = bidirectional_granger(pac, spec, max_lag=3)
                
                # Agregar mejor lag de cross-correlation
                if cc_results:
                    best_lag = max(cc_results, key=lambda x: abs(x['diff']))
                else:
                    best_lag = {'lag': np.nan, 'diff': np.nan}
                
                results.append({
                    'spectral_metric': spec_metric,
                    'n_valid': valid.sum(),
                    
                    # Cross-correlation
                    'best_lag': best_lag.get('lag', np.nan),
                    'best_lag_seconds': best_lag.get('lag_seconds', np.nan),
                    'cc_diff': best_lag.get('diff', np.nan),  # positivo = PAC lidera
                    'cc_pac_leads': best_lag.get('r_x_leads_y', np.nan),
                    'cc_spec_leads': best_lag.get('r_y_leads_x', np.nan),
                    
                    # Granger
                    'gc_direction': gc_results['direction'],
                    'gc_p_pac_to_spec': gc_results['p_x_to_y'],
                    'gc_p_spec_to_pac': gc_results['p_y_to_x'],
                    'gc_f_pac_to_spec': gc_results['gc_x_to_y'].get('f_stat', np.nan),
                    'gc_f_spec_to_pac': gc_results['gc_y_to_x'].get('f_stat', np.nan),
                })
            
            return results
    except Exception as e:
        print(f"Error en {npz_path}: {e}")
        return None


def run_analysis():
    """Ejecuta análisis de causalidad en todas las transiciones."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("🔬 ANÁLISIS DE CAUSALIDAD TEMPORAL")
    print("="*70)
    print(f"PAC metric: {PAC_METRIC}")
    print(f"Spectral metrics: {len(SPECTRAL_METRICS)}")
    print(f"Max lag: {MAX_LAG} ventanas ({MAX_LAG * 30}s)")
    print("="*70)
    
    # Encontrar todos los .npz
    npz_files = list(SERIES_DIR.glob("*.npz"))
    print(f"\n📂 Series encontradas: {len(npz_files)}")
    
    if not npz_files:
        print("❌ No hay series")
        return
    
    # Analizar cada transición
    all_results = []
    
    for i, npz_path in enumerate(npz_files):
        print(f"\r   [{i+1}/{len(npz_files)}] {npz_path.stem[:30]}...", end="", flush=True)
        
        trans_results = analyze_single_transition(npz_path, PAC_METRIC, SPECTRAL_METRICS)
        
        if trans_results:
            for r in trans_results:
                r['trans_id'] = npz_path.stem
                all_results.append(r)
    
    print(f"\n\n   ✅ Transiciones analizadas: {len(npz_files)}")
    print(f"   ✅ Comparaciones totales: {len(all_results)}")
    
    if not all_results:
        print("❌ No hay resultados")
        return
    
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "causality_raw.csv", index=False)
    
    # ========================================================================
    # AGREGACIÓN POR MÉTRICA
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("RESULTADOS AGREGADOS")
    print(f"{'='*70}")
    
    summary = []
    
    for metric in SPECTRAL_METRICS:
        subset = df[df['spectral_metric'] == metric]
        
        if len(subset) < 10:
            continue
        
        # Cross-correlation: ¿PAC lidera?
        cc_diffs = subset['cc_diff'].dropna()
        pct_pac_leads_cc = 100 * (cc_diffs > 0).mean() if len(cc_diffs) > 0 else np.nan
        mean_cc_diff = cc_diffs.mean() if len(cc_diffs) > 0 else np.nan
        
        # Granger: conteo de direcciones
        gc_dirs = subset['gc_direction'].value_counts()
        n_pac_causes = gc_dirs.get('x_causes_y', 0)
        n_spec_causes = gc_dirs.get('y_causes_x', 0)
        n_bidirectional = gc_dirs.get('bidirectional', 0)
        n_none = gc_dirs.get('no_causality', 0)
        n_total = len(subset)
        
        # Significancia agregada (t-test en cc_diff)
        if len(cc_diffs) >= 10:
            t_stat, p_value = stats.ttest_1samp(cc_diffs, 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        summary.append({
            'spectral_metric': metric,
            'n_transitions': n_total,
            
            # Cross-correlation
            'pct_pac_leads_cc': pct_pac_leads_cc,
            'mean_cc_diff': mean_cc_diff,
            'cc_ttest_t': t_stat,
            'cc_ttest_p': p_value,
            
            # Granger
            'n_pac_causes_spec': n_pac_causes,
            'n_spec_causes_pac': n_spec_causes,
            'n_bidirectional': n_bidirectional,
            'n_no_causality': n_none,
            'pct_pac_causes': 100 * n_pac_causes / n_total,
            'pct_spec_causes': 100 * n_spec_causes / n_total,
        })
    
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values('pct_pac_leads_cc', ascending=False)
    df_summary.to_csv(OUTPUT_DIR / "causality_summary.csv", index=False)
    
    # Mostrar resultados
    print(f"\n{'Métrica':<25} {'%PAC→Spec':>10} {'%Spec→PAC':>10} {'CC_diff':>10} {'p-value':>10}")
    print("─" * 70)
    
    for _, row in df_summary.iterrows():
        sig = "***" if row['cc_ttest_p'] < 0.001 else "**" if row['cc_ttest_p'] < 0.01 else "*" if row['cc_ttest_p'] < 0.05 else ""
        print(f"{row['spectral_metric']:<25} {row['pct_pac_causes']:>9.1f}% {row['pct_spec_causes']:>9.1f}% "
              f"{row['mean_cc_diff']:>+10.3f} {row['cc_ttest_p']:>9.4f} {sig}")
    
    # ========================================================================
    # CONCLUSIÓN
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("CONCLUSIÓN")
    print(f"{'='*70}")
    
    # Contar métricas donde PAC lidera significativamente
    sig_pac_leads = df_summary[(df_summary['cc_ttest_p'] < 0.05) & (df_summary['mean_cc_diff'] > 0)]
    sig_spec_leads = df_summary[(df_summary['cc_ttest_p'] < 0.05) & (df_summary['mean_cc_diff'] < 0)]
    
    print(f"\n   PAC δ-γ LIDERA significativamente (p<0.05): {len(sig_pac_leads)} métricas")
    if len(sig_pac_leads) > 0:
        print(f"      {', '.join(sig_pac_leads['spectral_metric'].tolist())}")
    
    print(f"\n   Spectral LIDERA significativamente (p<0.05): {len(sig_spec_leads)} métricas")
    if len(sig_spec_leads) > 0:
        print(f"      {', '.join(sig_spec_leads['spectral_metric'].tolist())}")
    
    # Granger summary
    gc_pac_wins = df_summary['n_pac_causes_spec'].sum()
    gc_spec_wins = df_summary['n_spec_causes_pac'].sum()
    
    print(f"\n   Granger Causality (total):")
    print(f"      PAC → Spectral: {gc_pac_wins} casos")
    print(f"      Spectral → PAC: {gc_spec_wins} casos")
    
    if gc_pac_wins > gc_spec_wins * 1.5:
        print(f"\n   ✅ EVIDENCIA DE QUE PAC δ-γ PRECEDE CAUSALMENTE A SPECTRAL POWER")
    elif gc_spec_wins > gc_pac_wins * 1.5:
        print(f"\n   ❌ EVIDENCIA CONTRARIA: Spectral power precede a PAC")
    else:
        print(f"\n   ⚠️ RESULTADOS MIXTOS: No hay dirección causal clara")
    
    print(f"\n   📁 Resultados en: {OUTPUT_DIR}")
    
    return df_summary


if __name__ == "__main__":
    run_analysis()