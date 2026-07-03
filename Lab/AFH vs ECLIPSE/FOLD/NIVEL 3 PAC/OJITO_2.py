#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
NIVEL 3b-HCTSA: Precedencia H* -> PAC con Features HCTSA
Autopsychic Fold Hypothesis - Validacion Robusta con catch22 + tsfresh
================================================================================

OBJETIVO CIENTIFICO:
    Usar ~100+ features de series temporales (en vez de 3 proxies manuales)
    para caracterizar H* de forma robusta y data-driven.
    
METODOLOGIA:
    1. Extraer features catch22 (22 canonicas de HCTSA) por ventana temporal
    2. Extraer features tsfresh (complejidad, entropia, autocorrelacion)
    3. Agrupar features en categorias conceptuales:
       - COMPLEXITY: Features relacionadas con complejidad/organizacion (proxy H*)
       - PREDICTABILITY: Autocorrelacion, memoria temporal
       - DISTRIBUTION: Propiedades estadisticas de distribucion
    4. Analizar precedencia temporal de cada grupo vs PAC
    5. Identificar cuales features muestran patron H* -> PAC mas robusto

VENTAJA vs PROXIES MANUALES:
    - Data-driven: No asumimos que features capturan H*
    - Robusto: Multiples features independientes
    - Publicable: Metodologia HCTSA es gold-standard (Fulcher et al.)
    - Reproducible: Features estandarizadas

INSTALACION:
    pip install pycatch22 tsfresh antropy

Author: Camilo Sjoberg Tala
Date: 2025-12-12
================================================================================
"""

import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
from dataclasses import dataclass, asdict
from collections import Counter
import sys
import json
from datetime import datetime
import hashlib

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tensorpac para PAC
from tensorpac import Pac

# HCTSA features
try:
    import pycatch22
    HAS_CATCH22 = True
except ImportError:
    HAS_CATCH22 = False
    print("[WARNING] pycatch22 not installed. Run: pip install pycatch22")

# Entropia features
try:
    import antropy as ant
    HAS_ANTROPY = True
except ImportError:
    HAS_ANTROPY = False
    print("[WARNING] antropy not installed. Run: pip install antropy")

# tsfresh (opcional, mas lento)
try:
    from tsfresh.feature_extraction import feature_calculators as fc
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False
    print("[WARNING] tsfresh not installed. Run: pip install tsfresh")


#==============================================================================
# CATCH22 FEATURE CATEGORIES
#==============================================================================
"""
Agrupacion conceptual de las 22 features canonicas de HCTSA.
Basado en Lubba et al. 2019 y documentacion de catch22.
"""

CATCH22_FEATURES = {
    # COMPLEXITY / NONLINEARITY (candidatos para H*)
    'complexity': [
        'SB_BinaryStats_mean_longstretch1',      # Longest stretch of 1s in binary
        'SB_BinaryStats_diff_longstretch0',     # Binary stats
        'SB_TransitionMatrix_3ac_sumdiagcov',   # Transition matrix complexity
        'FC_LocalSimple_mean1_tauresrat',       # Local simple forecasting
        'FC_LocalSimple_mean3_stderr',          # Local prediction error
        'SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1',  # Fluctuation analysis
        'SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1',      # DFA scaling
    ],
    
    # PREDICTABILITY / MEMORY (estructura temporal)
    'predictability': [
        'CO_f1ecac',                            # First 1/e crossing of ACF
        'CO_FirstMin_ac',                       # First minimum of ACF
        'CO_HistogramAMI_even_2_5',            # Automutual information
        'IN_AutoMutualInfoStats_40_gaussian_fmmi',  # AMI stats
        'CO_trev_1_num',                        # Time reversibility
    ],
    
    # DISTRIBUTION (propiedades estadisticas)
    'distribution': [
        'DN_HistogramMode_5',                   # Mode of 5-bin histogram
        'DN_HistogramMode_10',                  # Mode of 10-bin histogram  
        'DN_OutlierInclude_p_001_mdrmd',       # Outlier timing
        'DN_OutlierInclude_n_001_mdrmd',       # Outlier timing (negative)
        'MD_hrv_classic_pnn40',                # HRV-like metric
    ],
    
    # PERIODICITY (oscilaciones)
    'periodicity': [
        'PD_PeriodicityWang_th0_01',           # Periodicity detection
        'SB_MotifThree_quantile_hh',           # Motif analysis
        'SP_Summaries_welch_rect_area_5_1',   # Spectral summary
        'SP_Summaries_welch_rect_centroid',   # Spectral centroid
    ]
}

# Features que esperamos correlacionen con H* (coordinacion organizacional)
H_STAR_CANDIDATE_FEATURES = (
    CATCH22_FEATURES['complexity'] + 
    CATCH22_FEATURES['predictability']
)


#==============================================================================
# CONFIGURATION
#==============================================================================

@dataclass
class HCTSAConfig:
    data_dir: Path = Path(r'G:\Mi unidad\NEUROCIENCIA\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\SLEEPEDF\sleep-cassette')
    output_dir: Path = Path('./nivel3b_hctsa_results')
    sampling_rate: float = 100.0
    lowcut: float = 0.5
    highcut: float = 45.0
    window_before: float = 300.0
    window_after: float = 300.0
    sliding_window: float = 30.0
    sliding_step: float = 5.0
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 45.0)
    onset_threshold_sd: float = 1.5
    wake_states: List[str] = None
    sleep_states: List[str] = None
    max_subjects: int = None  # None = all
    
    def __post_init__(self):
        self.wake_states = ['Sleep stage W']
        self.sleep_states = ['Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4']


#==============================================================================
# FEATURE EXTRACTORS
#==============================================================================

class HCTSAFeatureExtractor:
    """Extrae features HCTSA (catch22) + features adicionales de complejidad"""
    
    def __init__(self, config: HCTSAConfig):
        self.config = config
        self.fs = config.sampling_rate
        
        # PAC calculator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pac_calc = Pac(
                idpac=(1, 0, 0),
                f_pha=list(config.theta_band),
                f_amp=list(config.gamma_band),
                dcomplex='wavelet',
                width=7,
                verbose=False
            )
    
    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocesa senal EEG"""
        if len(data) < 10:
            return data
        sos = signal.butter(4, [self.config.lowcut, self.config.highcut],
                           btype='bandpass', fs=self.fs, output='sos')
        filt = signal.sosfiltfilt(sos, data)
        return (filt - np.mean(filt)) / (np.std(filt) + 1e-10)
    
    def extract_all_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extrae todas las features de una ventana de datos.
        
        Returns:
            Dict con nombre_feature -> valor
        """
        if len(data) < int(2 * self.fs):
            return {}
        
        clean = self.preprocess(data)
        features = {}
        
        # 1. PAC (nuestro proxy de Fold)
        features['PAC'] = self._compute_pac(clean)
        
        # 2. catch22 features (HCTSA canonicas)
        if HAS_CATCH22:
            try:
                c22 = pycatch22.catch22_all(clean.tolist(), catch24=True)
                for name, val in zip(c22['names'], c22['values']):
                    features[f'c22_{name}'] = val if not np.isnan(val) else None
            except Exception as e:
                logger.debug(f"catch22 error: {e}")
        
        # 3. Antropy features (entropias de complejidad)
        if HAS_ANTROPY:
            try:
                features['ant_perm_entropy'] = ant.perm_entropy(clean, normalize=True)
                features['ant_spectral_entropy'] = ant.spectral_entropy(clean, sf=self.fs, normalize=True)
                features['ant_svd_entropy'] = ant.svd_entropy(clean, normalize=True)
                features['ant_app_entropy'] = ant.app_entropy(clean)
                features['ant_sample_entropy'] = ant.sample_entropy(clean)
                features['ant_hjorth_complexity'] = ant.hjorth_params(clean)[1]
                features['ant_hjorth_mobility'] = ant.hjorth_params(clean)[0]
                features['ant_lziv_complexity'] = ant.lziv_complexity(clean > np.median(clean), normalize=True)
                features['ant_petrosian_fd'] = ant.petrosian_fd(clean)
                features['ant_katz_fd'] = ant.katz_fd(clean)
                features['ant_higuchi_fd'] = ant.higuchi_fd(clean)
                features['ant_detrended_fluctuation'] = ant.detrended_fluctuation(clean)
            except Exception as e:
                logger.debug(f"antropy error: {e}")
        
        # 4. tsfresh features basicas (si disponible)
        if HAS_TSFRESH:
            try:
                features['tsf_abs_energy'] = fc.abs_energy(clean)
                features['tsf_mean_abs_change'] = fc.mean_abs_change(clean)
                features['tsf_mean_second_derivative'] = fc.mean_second_derivative_central(clean)
                features['tsf_c3'] = fc.c3(clean, lag=3)
                features['tsf_cid_ce'] = fc.cid_ce(clean, normalize=True)
                features['tsf_binned_entropy'] = fc.binned_entropy(clean, max_bins=10)
                features['tsf_autocorr_lag1'] = fc.autocorrelation(clean, lag=1)
                features['tsf_autocorr_lag10'] = fc.autocorrelation(clean, lag=10)
            except Exception as e:
                logger.debug(f"tsfresh error: {e}")
        
        return features
    
    def _compute_pac(self, data: np.ndarray) -> float:
        """Calcula PAC theta-gamma"""
        if len(data) < int(2 * self.fs):
            return np.nan
        if np.ptp(data) > 8.0:
            return np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(self.pac_calc.filterfit(
                    self.fs, data[np.newaxis, :], data[np.newaxis, :]
                )[0, 0, 0])
        except:
            return np.nan


#==============================================================================
# TRANSITION DETECTOR
#==============================================================================

class TransitionDetector:
    def __init__(self, config: HCTSAConfig):
        self.config = config
    
    def find_transitions(self, annotations) -> List[Dict]:
        trans = []
        desc, onsets = annotations.description, annotations.onset
        for i in range(len(desc) - 1):
            if desc[i] in self.config.sleep_states and desc[i+1] in self.config.wake_states:
                trans.append({
                    'index': i, 'time': onsets[i+1],
                    'from': desc[i], 'to': desc[i+1]
                })
        return trans


#==============================================================================
# PRECEDENCE ANALYZER WITH HCTSA
#==============================================================================

class HCTSAPrecedenceAnalyzer:
    """Analiza precedencia temporal usando features HCTSA"""
    
    def __init__(self, config: HCTSAConfig):
        self.config = config
        self.feature_extractor = HCTSAFeatureExtractor(config)
        self.detector = TransitionDetector(config)
    
    def extract_window(self, raw, trans_time) -> Optional[np.ndarray]:
        fs = raw.info['sfreq']
        start = max(0, trans_time - self.config.window_before)
        end = min(raw.times[-1], trans_time + self.config.window_after)
        s_samp, e_samp = int(start * fs), int(end * fs)
        if e_samp - s_samp < int(60 * fs):
            return None
        for ch in ['Pz-Oz', 'EEG Pz-Oz', 'Fpz-Cz', 'EEG Fpz-Cz']:
            if ch in raw.ch_names:
                return raw.copy().pick_channels([ch])[:, s_samp:e_samp][0][0, :]
        return None
    
    def compute_sliding_features(self, data: np.ndarray, trans_in_win: float) -> pd.DataFrame:
        """Computa features HCTSA en ventanas deslizantes"""
        fs = self.config.sampling_rate
        win_s = int(self.config.sliding_window * fs)
        step_s = int(self.config.sliding_step * fs)
        
        results = []
        for i in range((len(data) - win_s) // step_s + 1):
            start = i * step_s
            end = start + win_s
            if end > len(data):
                break
            
            window = data[start:end]
            center = (start + win_s // 2) / fs
            rel_time = center - trans_in_win
            
            # Extract all features
            features = self.feature_extractor.extract_all_features(window)
            features['relative_time'] = rel_time
            features['window_idx'] = i
            
            results.append(features)
        
        return pd.DataFrame(results)
    
    def detect_onset(self, df: pd.DataFrame, feature_name: str) -> Optional[float]:
        """Detecta onset de ACTIVACION para una feature"""
        if feature_name not in df.columns:
            return None
        
        valid = df[['relative_time', feature_name]].dropna()
        if len(valid) < 10:
            return None
        
        baseline = valid.loc[valid['relative_time'] < -60, feature_name]
        if len(baseline) < 3:
            baseline = valid[feature_name].iloc[:len(valid)//3]
        
        mean_bl = baseline.mean()
        std_bl = baseline.std()
        
        if std_bl < 1e-10:
            return None
        
        thresh = mean_bl + self.config.onset_threshold_sd * std_bl
        
        for _, row in valid[valid['relative_time'] >= -60].iterrows():
            if row[feature_name] > thresh:
                return row['relative_time']
        
        return None
    
    def analyze_transition(self, raw, trans: Dict) -> Optional[Dict]:
        """Analiza una transicion con todas las features HCTSA"""
        data = self.extract_window(raw, trans['time'])
        if data is None:
            return None
        
        trans_in_win = min(self.config.window_before, trans['time'])
        df = self.compute_sliding_features(data, trans_in_win)
        
        if len(df) < 10:
            return None
        
        # Detectar onset de PAC
        onset_pac = self.detect_onset(df, 'PAC')
        
        result = {
            'transition_time': trans['time'],
            'from': trans['from'],
            'to': trans['to'],
            'onset_PAC': onset_pac,
            'feature_df': df  # Guardar para analisis detallado
        }
        
        # Detectar onset de TODAS las features
        feature_cols = [c for c in df.columns if c not in ['relative_time', 'window_idx']]
        
        for feat in feature_cols:
            if feat == 'PAC':
                continue
            onset = self.detect_onset(df, feat)
            result[f'onset_{feat}'] = onset
            
            if onset is not None and onset_pac is not None:
                result[f'delta_{feat}'] = onset - onset_pac  # Negativo = feature antes que PAC
        
        return result


#==============================================================================
# PROCESSOR
#==============================================================================

class HCTSAProcessor:
    def __init__(self, config: HCTSAConfig):
        self.config = config
        self.analyzer = HCTSAPrecedenceAnalyzer(config)
        self.detector = TransitionDetector(config)
    
    def find_files(self) -> List[Dict]:
        psg = sorted(self.config.data_dir.glob("*-PSG.edf"))
        hypno = sorted(self.config.data_dir.glob("*-Hypnogram.edf"))
        hmap = {h.stem.replace("-Hypnogram", "")[:-1]: h for h in hypno if len(h.stem) >= 6}
        files = []
        for p in psg:
            c = p.stem.replace("-PSG", "")
            if len(c) >= 7 and c.endswith('0') and c[:-1] in hmap:
                files.append({'psg': p, 'hypno': hmap[c[:-1]], 'subject_id': c})
        return files
    
    def process_subject(self, info: Dict) -> List[Dict]:
        try:
            raw = mne.io.read_raw_edf(info['psg'], preload=True, verbose=False)
            ann = mne.read_annotations(info['hypno'])
        except Exception as e:
            logger.error(f"Error {info['subject_id']}: {e}")
            return []
        
        trans = self.detector.find_transitions(ann)
        results = []
        for t in trans:
            r = self.analyzer.analyze_transition(raw, t)
            if r:
                r['subject_id'] = info['subject_id']
                # Remove large dataframe from results to save memory
                if 'feature_df' in r:
                    del r['feature_df']
                results.append(r)
        return results


#==============================================================================
# ANALYSIS
#==============================================================================

def analyze_feature_precedence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza cuales features muestran precedencia consistente sobre PAC.
    
    Returns:
        DataFrame con estadisticas por feature
    """
    delta_cols = [c for c in df.columns if c.startswith('delta_')]
    
    results = []
    for col in delta_cols:
        feat_name = col.replace('delta_', '')
        valid = df[col].dropna()
        
        if len(valid) < 10:
            continue
        
        n_total = len(valid)
        n_before_pac = (valid < 0).sum()  # Feature antes que PAC
        n_after_pac = (valid > 0).sum()   # Feature despues de PAC
        
        pct_before = 100 * n_before_pac / n_total
        mean_delta = valid.mean()
        std_delta = valid.std()
        
        # Test estadistico
        if n_before_pac + n_after_pac > 0:
            p_binom = scipy_stats.binomtest(
                n_before_pac, n_before_pac + n_after_pac, 0.5, 
                alternative='greater'
            ).pvalue
        else:
            p_binom = np.nan
        
        # Categorizar feature
        category = 'unknown'
        for cat, feats in CATCH22_FEATURES.items():
            if any(f in feat_name for f in feats):
                category = cat
                break
        if 'ant_' in feat_name:
            category = 'antropy_complexity'
        if 'tsf_' in feat_name:
            category = 'tsfresh'
        
        # Es candidato a H*?
        is_hstar_candidate = any(f in feat_name for f in H_STAR_CANDIDATE_FEATURES)
        
        results.append({
            'feature': feat_name,
            'category': category,
            'is_hstar_candidate': is_hstar_candidate,
            'n_valid': n_total,
            'n_before_pac': n_before_pac,
            'pct_before_pac': pct_before,
            'mean_delta_seconds': mean_delta,
            'std_delta_seconds': std_delta,
            'p_value': p_binom,
            'significant': p_binom < 0.05 if not np.isnan(p_binom) else False
        })
    
    return pd.DataFrame(results).sort_values('pct_before_pac', ascending=False)


def run_hctsa_analysis():
    """
    Ejecuta analisis NIVEL 3b con features HCTSA.
    """
    config = HCTSAConfig()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("NIVEL 3b-HCTSA: H* -> PAC PRECEDENCE WITH HCTSA FEATURES")
    print("=" * 100)
    
    print("\n[FEATURES AVAILABLE]")
    print(f"  catch22: {HAS_CATCH22} (22 canonical HCTSA features)")
    print(f"  antropy: {HAS_ANTROPY} (~12 complexity/entropy features)")
    print(f"  tsfresh: {HAS_TSFRESH} (~8 time series features)")
    
    if not HAS_CATCH22:
        print("\n[ERROR] pycatch22 required. Run: pip install pycatch22")
        return None, None
    
    # Process
    processor = HCTSAProcessor(config)
    files = processor.find_files()
    
    if config.max_subjects:
        files = files[:config.max_subjects]
    
    print(f"\n[DATA] {len(files)} subjects")
    
    all_results = []
    n_subj = 0
    
    for i, f in enumerate(files, 1):
        if i % 10 == 0 or i == 1:
            print(f"  [{i}/{len(files)}] {f['subject_id']}")
        r = processor.process_subject(f)
        if r:
            n_subj += 1
            all_results.extend(r)
    
    print(f"\n[RESULTS] {len(all_results)} transitions from {n_subj} subjects")
    
    if not all_results:
        print("[ERROR] No valid transitions")
        return None, None
    
    # Compile
    df = pd.DataFrame(all_results)
    df.to_csv(config.output_dir / 'raw_hctsa_results.csv', index=False)
    
    # Analyze feature precedence
    print("\n" + "=" * 100)
    print("FEATURE PRECEDENCE ANALYSIS")
    print("=" * 100)
    
    feat_stats = analyze_feature_precedence(df)
    feat_stats.to_csv(config.output_dir / 'feature_precedence_stats.csv', index=False)
    
    # Show top features that precede PAC (H* candidates)
    print("\n[TOP FEATURES PRECEDING PAC (H* CANDIDATES)]")
    print("-" * 80)
    
    top_features = feat_stats[
        (feat_stats['pct_before_pac'] > 50) & 
        (feat_stats['n_valid'] >= 20)
    ].head(20)
    
    for _, row in top_features.iterrows():
        sig = "*" if row['significant'] else " "
        hstar = "H*" if row['is_hstar_candidate'] else "  "
        print(f"  {sig}{hstar} {row['feature'][:40]:<40s} "
              f"Before PAC: {row['pct_before_pac']:5.1f}% "
              f"(n={row['n_valid']:3d}, delta={row['mean_delta_seconds']:+6.1f}s, p={row['p_value']:.4f})")
    
    # Analyze by category
    print("\n[PRECEDENCE BY FEATURE CATEGORY]")
    print("-" * 80)
    
    cat_stats = feat_stats.groupby('category').agg({
        'pct_before_pac': 'mean',
        'mean_delta_seconds': 'mean',
        'n_valid': 'sum',
        'significant': 'sum'
    }).round(2)
    
    print(cat_stats.to_string())
    
    # H* candidates summary
    print("\n[H* CANDIDATE FEATURES SUMMARY]")
    print("-" * 80)
    
    hstar_feats = feat_stats[feat_stats['is_hstar_candidate']]
    if len(hstar_feats) > 0:
        hstar_before = hstar_feats[hstar_feats['pct_before_pac'] > 50]
        print(f"  H* candidates total: {len(hstar_feats)}")
        print(f"  H* candidates preceding PAC (>50%): {len(hstar_before)}")
        print(f"  Mean precedence: {hstar_feats['pct_before_pac'].mean():.1f}%")
        print(f"  Mean delta: {hstar_feats['mean_delta_seconds'].mean():.1f}s")
    
    # ECLIPSE-style evaluation
    print("\n" + "=" * 100)
    print("AFH HYPOTHESIS EVALUATION")
    print("=" * 100)
    
    # Use complexity features as H* proxy
    complexity_feats = feat_stats[feat_stats['category'].isin(['complexity', 'antropy_complexity'])]
    
    if len(complexity_feats) > 0:
        mean_pct_before = complexity_feats['pct_before_pac'].mean()
        n_significant = (complexity_feats['significant']).sum()
        
        print(f"\n  Complexity features (H* proxies):")
        print(f"    Mean % preceding PAC: {mean_pct_before:.1f}%")
        print(f"    Significant (p<0.05): {n_significant}/{len(complexity_feats)}")
        
        # Verdict
        if mean_pct_before >= 60 and n_significant >= len(complexity_feats) // 2:
            verdict = "SUPPORTED"
            desc = "Complexity features (H* proxies) consistently precede PAC"
        elif mean_pct_before <= 40:
            verdict = "FALSIFIED"
            desc = "Complexity features do NOT precede PAC as predicted"
        else:
            verdict = "INCONCLUSIVE"
            desc = "Mixed results - some features precede PAC, others don't"
        
        print(f"\n  >>> VERDICT: {verdict} <<<")
        print(f"      {desc}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_subjects': n_subj,
        'n_transitions': len(all_results),
        'n_features_analyzed': len(feat_stats),
        'top_features_before_pac': top_features[['feature', 'pct_before_pac', 'p_value']].to_dict('records'),
        'category_summary': cat_stats.to_dict(),
        'verdict': verdict if 'verdict' in dir() else 'N/A'
    }
    
    with open(config.output_dir / 'HCTSA_SUMMARY.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n[OUTPUT] {config.output_dir}")
    
    return df, feat_stats


#==============================================================================
# MAIN
#==============================================================================

if __name__ == "__main__":
    try:
        df, stats = run_hctsa_analysis()
        
        if stats is not None:
            print("\n" + "=" * 100)
            print("INTERPRETACION CIENTIFICA")
            print("=" * 100)
            print("""
    Este analisis usa ~40+ features HCTSA para caracterizar H* de forma robusta.
    
    LOGICA:
    - Si AFH es correcta, features de COMPLEJIDAD (H*) deben activarse ANTES que PAC
    - Usamos catch22 (22 features canonicas de HCTSA) + antropy (entropias)
    - Analizamos cuales features preceden consistentemente a PAC
    
    VENTAJAS vs PROXIES MANUALES:
    - Data-driven: No asumimos a priori que features capturan H*
    - Robusto: Multiples features independientes dan misma conclusion
    - Publicable: Metodologia HCTSA es gold-standard en neurociencia
    - Conexion directa con grupo de Tsuchiya/Fulcher
    
    SIGUIENTE PASO:
    - Identificar features que mejor discriminan H* vs PAC
    - Usar estas features en analisis de prediccion
    - Preparar para paper con metodologia HCTSA formal
            """)
            print("=" * 100)
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)