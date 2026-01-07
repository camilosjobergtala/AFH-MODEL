#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AFH CONVERGENCE EXPERIMENT - 32GB RAM OPTIMIZED (PATCHED v2.5.1 FINAL)
-------------------------------------------------------------------------------
Objetivo:
- Estimar “convergencia” entre 4 vías de features EEG (spectral, pac, temporal,
  connectivity) usando CKNN-A (overlap de vecindarios kNN), más métricas auxiliares
  (dCor aproximada por muestreo de pares, dimensión intrínseca TwoNN, Information
  Imbalance basada en rank-fraction).

Optimización:
- CKNN-A sin matrices NxN (kNN overlap directo)
- dCor aproximada vía muestreo de pares (sin NxN)
- Carga incremental por sujeto (SLEEP-EDF) + concatenación por estado
- Submuestreo estratificado por sujeto para limitar memoria

FIXES METODOLÓGICOS v2.5 (y pequeños fixes v2.5.1):
1) Bootstrap de CKNN-A:
   - reemplaza “epoch-bootstrap con reemplazo” (inflador) por:
     subject-level bootstrap: re-muestrea sujetos (con reemplazo) y luego epochs
     dentro de sujeto (SIN reemplazo), hasta target_n.
2) Null model de CKNN-A:
   - reemplaza permutación global de epochs por:
     permutación within-subject de Y (preserva estructura por sujeto).
3) Information Imbalance:
   - devuelve rank fractions (0..1) y define II = 2*rank_fraction (0..2),
     sin hard-clip a 1.0 (evita saturación).
4) v2.5.1:
   - robustez extra (NaNs, tamaños pequeños), CLI ampliada, y guardado consistente.

Uso:
    python afh_experiment_complete.py --synthetic
    python afh_experiment_complete.py --real --sleep_edf_path "RUTA/AL/DATASET"
Opcional:
    python afh_experiment_complete.py --real --max_epochs 4000 --boot_target 3000

Requisitos:
- numpy, scipy, scikit-learn
- mne (solo si usa datos reales SLEEP-EDF)
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy import signal, stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: MNE not available. Real data loading will fail (install mne).")

warnings.filterwarnings("ignore")


# =============================================================================
# DEFAULT CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    "experiment": {
        "name": "AFH Convergence Experiment",
        "version": "2.5.1 - 32GB RAM Optimized + Subject-bootstrap + within-subject null (FINAL)",
        "author": "Dr. Sjöberg",
        "date": "2025-12-31",
        "random_seed": 42
    },
    "data": {
        "sampling_rate": 100,
        "epoch_duration": 30,
        "overlap": 0,
        "filter": {"lowcut": 0.5, "highcut": 40.0},
        "sleep_edf": {
            "path": "G:/Mi unidad/NEUROCIENCIA/AFH/EXPERIMENTO/FASE 2/SLEEP-EDF/SLEEPEDF/sleep-cassette/",
            "subjects": "all"
        }
    },
    "analysis": {
        "states": ["Wake", "N1", "N2", "N3", "REM"],
        "contrasts": [["Wake", "N3"], ["REM", "N3"], ["Wake", "N2"]],
        "stats": {
            "alpha": 0.05,
            "n_permutations": 200,
            "bootstrap_n": 150,
            "bootstrap_state_diff_n": 200,
            "max_epochs_per_state": 2000,         # límite duro por estado (post-carga)
            "bootstrap_target_epochs": 2000,      # target_n para subject-bootstrap (<= max_epochs recomendado)
            "min_epochs_per_subject": 5,          # sujetos con menos epochs se omiten en bootstrap/permutación
        }
    },
    "metrics": {
        "cknna": {"default_k": 25, "metric": "euclidean"},
        "dcor": {"n_pairs": 200000},
        "ii": {"k": 50}
    },
    "output": {
        "results_dir": "./results",
        "verbose": False
    }
}


# =============================================================================
# HELPERS
# =============================================================================

def _to_jsonable(obj: Any) -> Any:
    import numpy as _np
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (_np.floating, _np.integer)):
        return obj.item()
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def _safe_nanpercentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def _safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class PairResult:
    cknna: float
    dcor: float
    ii_x_to_y: float
    ii_y_to_x: float
    n: int

    def to_dict(self) -> dict:
        return {
            "cknna": float(self.cknna),
            "dcor": float(self.dcor),
            "ii_x_to_y": float(self.ii_x_to_y),
            "ii_y_to_x": float(self.ii_y_to_x),
            "n": int(self.n),
        }


class RepresentationalMetrics:
    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)

    @staticmethod
    def _check_correspondence(X: np.ndarray, Y: np.ndarray) -> None:
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"1:1 correspondence required. X={X.shape[0]}, Y={Y.shape[0]}")

    @staticmethod
    def _standardize(X: np.ndarray) -> np.ndarray:
        # StandardScaler maneja columnas constantes, pero puede producir NaN si todo es NaN.
        Xs = StandardScaler().fit_transform(X)
        Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
        return Xs

    # -------------------------------------------------------------------------
    # CKNN-A: neighborhood overlap (sin NxN)
    # -------------------------------------------------------------------------
    def cknna(self, X: np.ndarray, Y: np.ndarray, k: int = 25, normalize: bool = True,
              metric: str = "euclidean") -> float:
        self._check_correspondence(X, Y)
        n = X.shape[0]
        if n <= k + 1:
            return float("nan")

        if normalize:
            X = self._standardize(X)
            Y = self._standardize(Y)

        k_eff = int(min(k, n - 1))
        nnX = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1).fit(X)
        nnY = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1).fit(Y)

        idxX = nnX.kneighbors(X, return_distance=False)[:, 1:]  # drop self
        idxY = nnY.kneighbors(Y, return_distance=False)[:, 1:]  # drop self

        overlaps = np.empty(n, dtype=np.float64)
        for i in range(n):
            # k pequeño: set intersection es suficientemente rápido y claro
            overlaps[i] = len(set(map(int, idxX[i])).intersection(set(map(int, idxY[i])))) / k_eff

        val = float(np.mean(overlaps))
        if not np.isfinite(val):
            return float("nan")
        return val

    def cknna_matrix(self, pathways: Dict[str, np.ndarray], k: int = 25,
                     metric: str = "euclidean") -> Tuple[np.ndarray, List[str]]:
        labels = list(pathways.keys())
        n = len(labels)
        n_rows = {lab: pathways[lab].shape[0] for lab in labels}
        if len(set(n_rows.values())) != 1:
            raise ValueError(f"All pathways must have same N. Got: {n_rows}")

        M = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = labels[i], labels[j]
                s = self.cknna(pathways[a], pathways[b], k=k, normalize=True, metric=metric)
                M[i, j] = M[j, i] = s
        return M, labels

    @staticmethod
    def mean_offdiag(M: np.ndarray) -> float:
        idx = np.triu_indices_from(M, k=1)
        return float(np.mean(M[idx])) if len(idx[0]) else float("nan")

    # -------------------------------------------------------------------------
    # dCor aproximada: correlación entre distancias en pares muestreados
    # -------------------------------------------------------------------------
    def distance_correlation(self, X: np.ndarray, Y: np.ndarray, normalize: bool = True,
                             n_pairs: int = 200000, seed: int = 0) -> float:
        self._check_correspondence(X, Y)
        n = X.shape[0]
        if n < 10:
            return float("nan")

        if normalize:
            X = self._standardize(X)
            Y = self._standardize(Y)

        rng = np.random.default_rng(seed)
        m = int(min(n_pairs, n * (n - 1) // 2))
        if m <= 50:
            return float("nan")

        i = rng.integers(0, n, size=m, endpoint=False)
        j = rng.integers(0, n, size=m, endpoint=False)
        neq = i != j
        if np.sum(neq) <= 50:
            return float("nan")
        i = i[neq]
        j = j[neq]

        dx = np.linalg.norm(X[i] - X[j], axis=1)
        dy = np.linalg.norm(Y[i] - Y[j], axis=1)

        r = np.corrcoef(dx, dy)[0, 1]
        if not np.isfinite(r):
            return float("nan")
        return float(np.clip(abs(r), 0.0, 1.0))

    # -------------------------------------------------------------------------
    # Dimensión intrínseca TwoNN (aprox)
    # -------------------------------------------------------------------------
    def intrinsic_dimensionality(self, X: np.ndarray, method: str = "TwoNN") -> float:
        if X.ndim != 2 or X.shape[0] < 30:
            return float("nan")

        if method.lower() == "twonn":
            nbrs = NearestNeighbors(n_neighbors=3, n_jobs=-1).fit(X)
            dist, _ = nbrs.kneighbors(X)
            r1 = np.maximum(dist[:, 1], 1e-12)
            r2 = np.maximum(dist[:, 2], 1e-12)
            ratio = (r2 / r1)
            ratio = ratio[np.isfinite(ratio)]
            mu = np.sort(ratio)
            n = len(mu)
            if n < 30:
                return float("nan")
            F = np.arange(1, n + 1) / (n + 1)
            x = np.log(mu)
            y = -np.log(1 - F)
            slope, _ = np.polyfit(x, y, 1)
            return float(max(1.0, slope))
        return float("nan")

    # -------------------------------------------------------------------------
    # Information Imbalance v2.5: rank fractions sin clip (II = 2*r, rango 0..2)
    # -------------------------------------------------------------------------
    def information_imbalance(self, X: np.ndarray, Y: np.ndarray, k: int = 50,
                             normalize: bool = True, metric: str = "euclidean") -> Tuple[float, float]:
        self._check_correspondence(X, Y)
        n = X.shape[0]
        if n <= k + 1:
            return (float("nan"), float("nan"))

        if normalize:
            X = self._standardize(X)
            Y = self._standardize(Y)

        k_eff = int(min(k, n - 1))
        nnX = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1).fit(X)
        nnY = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=-1).fit(Y)

        idxX = nnX.kneighbors(X, return_distance=False)[:, 1:]
        idxY = nnY.kneighbors(Y, return_distance=False)[:, 1:]

        def avg_rank_fraction(idxA: np.ndarray, idxB: np.ndarray) -> float:
            ranks = []
            for i in range(n):
                pos = {int(j): r for r, j in enumerate(idxB[i])}
                for j in idxA[i]:
                    ranks.append(pos.get(int(j), k_eff) / k_eff)  # 0..1
            return float(np.mean(ranks)) if ranks else 1.0

        r_X_in_Y = avg_rank_fraction(idxX, idxY)
        r_Y_in_X = avg_rank_fraction(idxY, idxX)

        ii_xy = 2.0 * r_X_in_Y  # 0..2
        ii_yx = 2.0 * r_Y_in_X  # 0..2
        return float(ii_xy), float(ii_yx)

    def pair_metrics(self, X: np.ndarray, Y: np.ndarray, k: int = 25,
                     metric: str = "euclidean", dcor_pairs: int = 200000) -> PairResult:
        c = self.cknna(X, Y, k=k, normalize=True, metric=metric)
        d = self.distance_correlation(X, Y, normalize=True, n_pairs=dcor_pairs, seed=7)
        ii_xy, ii_yx = self.information_imbalance(X, Y, k=max(k, 25), normalize=True, metric=metric)
        return PairResult(cknna=c, dcor=d, ii_x_to_y=ii_xy, ii_y_to_x=ii_yx, n=X.shape[0])

    # -------------------------------------------------------------------------
    # Null model v2.5: permuta Y dentro de sujeto
    # -------------------------------------------------------------------------
    def cknna_null_within_subject(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        subject_ids: np.ndarray,
        k: int = 25,
        n_perm: int = 200,
        seed: int = 0,
        metric: str = "euclidean",
        min_epochs_per_subject: int = 5
    ) -> np.ndarray:
        self._check_correspondence(X, Y)
        subject_ids = np.asarray(subject_ids, dtype=object)
        if subject_ids.shape[0] != X.shape[0]:
            raise ValueError("subject_ids must match X/Y rows")

        rng = np.random.default_rng(seed)
        null_vals = np.empty(n_perm, dtype=np.float64)

        uniq = np.unique(subject_ids)
        subj_to_idx = {sid: np.where(subject_ids == sid)[0] for sid in uniq}

        # Filtra sujetos pequeños (evita permutación degenerada)
        subj_to_idx = {sid: idx for sid, idx in subj_to_idx.items() if idx.size >= int(min_epochs_per_subject)}
        if len(subj_to_idx) < 2:
            # fallback: permutación global (último recurso)
            for b in range(n_perm):
                perm = rng.permutation(Y.shape[0])
                null_vals[b] = self.cknna(X, Y[perm], k=k, normalize=True, metric=metric)
            return null_vals

        for b in range(n_perm):
            perm = np.arange(Y.shape[0])
            for sid, idx in subj_to_idx.items():
                if idx.size <= 1:
                    continue
                perm[idx] = rng.permutation(idx)
            null_vals[b] = self.cknna(X, Y[perm], k=k, normalize=True, metric=metric)

        return null_vals

    @staticmethod
    def p_value_greater(observed: float, null_vals: np.ndarray) -> float:
        null_vals = np.asarray(null_vals, dtype=float)
        null_vals = null_vals[np.isfinite(null_vals)]
        if null_vals.size == 0 or not np.isfinite(observed):
            return float("nan")
        return float((np.sum(null_vals >= observed) + 1) / (len(null_vals) + 1))

    # -------------------------------------------------------------------------
    # Subject-bootstrap v2.5: resample sujetos (con reemplazo) + epochs sin reemplazo
    # -------------------------------------------------------------------------
    def bootstrap_cknna_subject(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        subject_ids: np.ndarray,
        k: int = 25,
        n_boot: int = 150,
        seed: int = 0,
        metric: str = "euclidean",
        target_n: Optional[int] = None,
        min_epochs_per_subject: int = 5
    ) -> np.ndarray:
        self._check_correspondence(X, Y)
        subject_ids = np.asarray(subject_ids, dtype=object)
        if subject_ids.shape[0] != X.shape[0]:
            raise ValueError("subject_ids must match X/Y rows")

        rng = np.random.default_rng(seed)
        uniq = np.unique(subject_ids)

        subj_indices = {sid: np.where(subject_ids == sid)[0] for sid in uniq}
        good = [sid for sid, idx in subj_indices.items() if idx.size >= int(min_epochs_per_subject)]

        n = X.shape[0]
        tn_default = n if target_n is None else int(min(int(target_n), n))

        vals = np.empty(n_boot, dtype=np.float64)

        if len(good) < 2:
            # fallback: subsample sin reemplazo (evita duplicados)
            for b in range(n_boot):
                idx = rng.choice(n, size=tn_default, replace=False)
                vals[b] = self.cknna(X[idx], Y[idx], k=k, normalize=True, metric=metric)
            return vals

        good = np.array(good, dtype=object)

        for b in range(n_boot):
            chosen_subjects = rng.choice(good, size=len(good), replace=True)
            per_subj = max(1, int(math.ceil(tn_default / len(chosen_subjects))))

            chosen_rows = []
            for sid in chosen_subjects:
                idx = subj_indices[sid]
                take = min(per_subj, idx.size)
                chosen_rows.append(rng.choice(idx, size=take, replace=False))

            idx_all = np.concatenate(chosen_rows) if chosen_rows else np.array([], dtype=int)
            if idx_all.size > tn_default:
                idx_all = rng.choice(idx_all, size=tn_default, replace=False)

            if idx_all.size <= k + 1:
                vals[b] = float("nan")
            else:
                vals[b] = self.cknna(X[idx_all], Y[idx_all], k=k, normalize=True, metric=metric)

        return vals

    def state_convergence_summary(self, pathways: Dict[str, np.ndarray], k: int = 25,
                                  metric: str = "euclidean", dcor_pairs: int = 200000) -> Dict[str, object]:
        M, labels = self.cknna_matrix(pathways, k=k, metric=metric)
        out = {
            "labels": labels,
            "cknna_matrix": M,
            "cknna_mean_offdiag": self.mean_offdiag(M),
            "intrinsic_dim": {lab: self.intrinsic_dimensionality(pathways[lab], method="TwoNN") for lab in labels},
            "pairs": {},
            "n": int(pathways[labels[0]].shape[0]),
        }
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                out["pairs"][f"{a}__{b}"] = self.pair_metrics(
                    pathways[a], pathways[b], k=k, metric=metric, dcor_pairs=dcor_pairs
                ).to_dict()
        return out


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class EEGFeatureExtractor:
    def __init__(self, sfreq: float = 100.0, verbose: bool = False):
        self.sfreq = float(sfreq)
        self.verbose = bool(verbose)
        self._filter_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_filters()

    def _precompute_filters(self) -> None:
        nyq = 0.5 * self.sfreq
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 40.0),
        }
        for name, (low, high) in bands.items():
            low_n = low / nyq
            high_n = high / nyq
            b, a = signal.butter(4, [low_n, high_n], btype="bandpass")
            self._filter_cache[name] = (b, a)

    def _bandpass_filter(self, x: np.ndarray, low: float, high: float, order: int = 4) -> np.ndarray:
        nyq = 0.5 * self.sfreq
        low_n = low / nyq
        high_n = high / nyq
        b, a = signal.butter(order, [low_n, high_n], btype="bandpass")
        return signal.filtfilt(b, a, x).astype(np.float64)

    def _bandpass_filter_vectorized(self, data: np.ndarray, band_name: str) -> np.ndarray:
        if band_name not in self._filter_cache:
            raise ValueError(f"Band {band_name} not in cache")

        b, a = self._filter_cache[band_name]
        n_times = data.shape[1]
        min_len = 3 * (max(len(a), len(b)) - 1) + 1

        if n_times < min_len:
            try:
                return signal.filtfilt(b, a, data.astype(np.float64), axis=1, method="gust")
            except Exception:
                if self.verbose:
                    print(f"    WARNING: Segment too short for {band_name}")
                return data.astype(np.float64)

        return signal.filtfilt(b, a, data.astype(np.float64), axis=1)

    @staticmethod
    def _spectral_entropy(psd: np.ndarray) -> float:
        psd = np.maximum(psd, 1e-18)
        p = psd / (psd.sum() + 1e-18)
        return float(-(p * np.log(p + 1e-18)).sum())

    def pac_features(self, data: np.ndarray) -> np.ndarray:
        # Nota: computacionalmente costoso, mantenido por corrección.
        n_epochs, n_channels, _ = data.shape
        out = np.zeros((n_epochs, n_channels, 3), dtype=np.float64)
        phase_bins = np.linspace(-np.pi, np.pi, 19)
        u = np.ones(18, dtype=np.float64) / 18

        for e in range(n_epochs):
            for ch in range(n_channels):
                x = data[e, ch].astype(np.float64)
                x_delta = self._bandpass_filter(x, 0.5, 4.0)
                x_gamma = self._bandpass_filter(x, 30.0, 40.0)
                delta_phase = np.angle(signal.hilbert(x_delta))
                gamma_amp = np.abs(signal.hilbert(x_gamma))

                mean_amp = np.zeros(18, dtype=np.float64)
                for b in range(18):
                    mask = (delta_phase >= phase_bins[b]) & (delta_phase < phase_bins[b + 1])
                    mean_amp[b] = np.mean(gamma_amp[mask]) if np.any(mask) else 0.0

                mean_amp = np.maximum(mean_amp, 1e-18)
                p = mean_amp / (mean_amp.sum() + 1e-18)
                mi = float(np.sum(p * np.log((p + 1e-18) / u)) / np.log(18))

                out[e, ch, 0] = mi
                out[e, ch, 1] = float(np.mean(gamma_amp))
                out[e, ch, 2] = float(np.abs(np.mean(np.exp(1j * delta_phase))))

        return out.reshape(n_epochs, -1)

    def spectral_features(self, data: np.ndarray) -> np.ndarray:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 40.0)
        }
        n_epochs, n_channels, _ = data.shape
        out = np.zeros((n_epochs, n_channels, 8), dtype=np.float64)

        for e in range(n_epochs):
            for ch in range(n_channels):
                x = data[e, ch].astype(np.float64)
                freqs, psd = signal.welch(x, fs=self.sfreq, nperseg=min(1024, len(x)))
                psd = np.maximum(psd, 1e-18)

                def band_power(lo: float, hi: float) -> float:
                    m = (freqs >= lo) & (freqs < hi)
                    return float(np.trapz(psd[m], freqs[m])) if np.any(m) else 0.0

                bp = [band_power(*bands[k]) for k in ["delta", "theta", "alpha", "beta", "gamma"]]
                total = float(sum(bp) + 1e-18)
                rel = [b / total for b in bp]

                m_ent = (freqs >= 0.5) & (freqs <= 40.0)
                ent = self._spectral_entropy(psd[m_ent]) if np.any(m_ent) else float("nan")

                m_fit = (freqs >= 1.0) & (freqs <= 40.0)
                if np.any(m_fit):
                    xlog = np.log(np.maximum(freqs[m_fit], 1e-6))
                    ylog = np.log(psd[m_fit])
                    slope, _, r, _, _ = stats.linregress(xlog, ylog)
                else:
                    slope, r = float("nan"), float("nan")

                out[e, ch, 0:5] = np.array(rel, dtype=np.float64)
                out[e, ch, 5] = ent
                out[e, ch, 6] = slope
                out[e, ch, 7] = r

        return out.reshape(n_epochs, -1)

    def temporal_features(self, data: np.ndarray) -> np.ndarray:
        n_epochs, n_channels, _ = data.shape
        out = np.zeros((n_epochs, n_channels, 7), dtype=np.float64)

        for e in range(n_epochs):
            for ch in range(n_channels):
                x = data[e, ch].astype(np.float64)
                out[e, ch, 0] = np.mean(x)
                out[e, ch, 1] = np.std(x)
                out[e, ch, 2] = stats.skew(x)
                out[e, ch, 3] = stats.kurtosis(x)
                out[e, ch, 4] = np.sqrt(np.mean(x * x))
                out[e, ch, 5] = np.sum(np.abs(np.diff(x)))
                out[e, ch, 6] = np.mean((x[:-1] * x[1:]) < 0)

        return out.reshape(n_epochs, -1)

    def connectivity_features(self, data: np.ndarray) -> np.ndarray:
        bands_to_use = ["theta", "alpha", "beta"]
        n_epochs, n_channels, _ = data.shape
        pairs = np.triu_indices(n_channels, k=1)
        n_pairs = len(pairs[0])
        n_bands = len(bands_to_use)
        out = np.zeros((n_epochs, n_pairs * n_bands), dtype=np.float64)

        for e in range(n_epochs):
            band_offset = 0
            for band_name in bands_to_use:
                try:
                    filtered = self._bandpass_filter_vectorized(data[e], band_name)
                    analytic = signal.hilbert(filtered, axis=1)
                    z = np.exp(1j * np.angle(analytic))
                    zi = z[pairs[0]]
                    zj = z[pairs[1]]
                    plv_values = np.abs(np.mean(zi * np.conj(zj), axis=1))
                    out[e, band_offset * n_pairs: (band_offset + 1) * n_pairs] = plv_values
                except Exception as ex:
                    if self.verbose:
                        print(f"    WARNING: PLV failed for {band_name}: {ex}")
                    out[e, band_offset * n_pairs: (band_offset + 1) * n_pairs] = np.nan
                band_offset += 1

        return out

    def extract_all_features_separated(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "spectral": self.spectral_features(data),
            "pac": self.pac_features(data),
            "temporal": self.temporal_features(data),
            "connectivity": self.connectivity_features(data),
        }


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

class SyntheticEEGGenerator:
    def __init__(self, sfreq: float = 100.0, seed: int = 42):
        self.sfreq = float(sfreq)
        self.rng = np.random.default_rng(int(seed))

    def generate_synthetic_dataset(
        self,
        n_subjects: int = 3,
        n_epochs_per_state: int = 100,
        n_channels: int = 10,
        epoch_duration: float = 30.0
    ) -> dict:
        states = ["Wake", "N1", "N2", "N3", "REM"]
        n_samples = int(epoch_duration * self.sfreq)

        all_epochs, all_labels, all_subject_ids = [], [], []

        for subj_idx in range(n_subjects):
            subject_id = f"synthetic_S{subj_idx:03d}"
            for st in states:
                for _ in range(n_epochs_per_state):
                    all_epochs.append(self._generate_epoch_for_state(st, n_channels, n_samples))
                    all_labels.append(st)
                    all_subject_ids.append(subject_id)

        all_epochs = np.array(all_epochs, dtype=np.float64)
        all_labels = np.array(all_labels, dtype=object)
        all_subject_ids = np.array(all_subject_ids, dtype=object)

        idx = self.rng.permutation(len(all_epochs))
        return {
            "data": all_epochs[idx],
            "labels": all_labels[idx],
            "subject_ids": all_subject_ids[idx],
            "info": {
                "n_subjects": n_subjects,
                "n_epochs": int(len(all_epochs)),
                "n_channels": n_channels,
                "sfreq": self.sfreq,
                "synthetic": True
            }
        }

    def _generate_epoch_for_state(self, state: str, n_channels: int, n_samples: int) -> np.ndarray:
        t = np.arange(n_samples) / self.sfreq
        epoch = np.zeros((n_channels, n_samples), dtype=np.float64)

        for ch in range(n_channels):
            x = self.rng.standard_normal(n_samples) * 0.1
            if state == "Wake":
                x += np.sin(2 * np.pi * 10 * t) * 2.0 + np.sin(2 * np.pi * 20 * t) * 1.0
            elif state == "N1":
                x += np.sin(2 * np.pi * 6 * t) * 1.5 + np.sin(2 * np.pi * 10 * t) * 1.0
            elif state == "N2":
                env = np.zeros(n_samples)
                for _ in range(5):
                    start = self.rng.integers(0, max(1, n_samples - 500))
                    env[start: start + min(500, n_samples - start)] = 1.0
                x += env * np.sin(2 * np.pi * 13 * t) * 3.0
            elif state == "N3":
                x += np.sin(2 * np.pi * 1.5 * t) * 5.0 + np.sin(2 * np.pi * 2.5 * t) * 4.0
            elif state == "REM":
                x += np.sin(2 * np.pi * 6 * t) * 3.0 + np.sin(2 * np.pi * 25 * t) * 1.5
            epoch[ch, :] = x

        return epoch


# =============================================================================
# REAL DATA LOADER (SLEEP-EDF)
# =============================================================================

class EEGDataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.target_sfreq = float(config["data"]["sampling_rate"])
        self.epoch_duration = float(config["data"]["epoch_duration"])
        self.overlap = float(config["data"]["overlap"])
        self.verbose = bool(config.get("output", {}).get("verbose", False))

    def load_sleep_edf(self, data_path: str, subjects="all", feature_extractor: Optional[EEGFeatureExtractor] = None) -> dict:
        if not MNE_AVAILABLE:
            raise ImportError("MNE is required for loading real data. Install with: pip install mne")

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")

        psg_files = list(data_path.rglob("*PSG*.edf"))
        if subjects != "all":
            psg_files = [psg_files[i] for i in subjects if i < len(psg_files)]
        print(f"  Found {len(psg_files)} PSG files")

        hyp_files = list(data_path.rglob("*Hypnogram*.edf"))
        hyp_dict: Dict[str, Path] = {}
        for hyp in hyp_files:
            name = hyp.stem
            base = name.split("-")[0].split("_")[0]
            if len(base) >= 6:
                subject_base = base[:6]
                hyp_dict[subject_base] = hyp
                hyp_dict[base] = hyp
        print(f"  Found {len(hyp_files)} Hypnogram files")

        paired_files: List[Tuple[str, Path, Path]] = []
        for psg_file in sorted(psg_files):
            base = psg_file.stem.split("-")[0].split("_")[0]
            subject_id = base
            subject_base = base[:6] if len(base) >= 6 else base

            hyp_file = None
            for variant in [base, subject_base, base.replace("E", "C"), subject_base + "C"]:
                if variant in hyp_dict:
                    hyp_file = hyp_dict[variant]
                    break

            if hyp_file is None:
                candidates = list(psg_file.parent.glob(f"*{subject_base}*Hypnogram*.edf"))
                if candidates:
                    hyp_file = candidates[0]

            if hyp_file:
                paired_files.append((subject_id, psg_file, hyp_file))

        # Únicos por subject_id
        seen = set()
        unique_pairs = []
        for sid, psg, hyp in paired_files:
            if sid not in seen:
                seen.add(sid)
                unique_pairs.append((sid, psg, hyp))

        print(f"  Successfully paired: {len(unique_pairs)} unique subjects\n")

        if feature_extractor is None:
            raise ValueError("feature_extractor must be provided for incremental loading")

        states = ["Wake", "N1", "N2", "N3", "REM"]
        state_features: Dict[str, Dict[str, List[np.ndarray]]] = {
            state: {"spectral": [], "pac": [], "temporal": [], "connectivity": []} for state in states
        }
        state_subject_ids: Dict[str, List[str]] = {state: [] for state in states}

        loaded_subjects = 0
        failed_subjects: List[Tuple[str, str]] = []

        for subject_id, psg_file, hyp_file in unique_pairs:
            try:
                print(f"    Processing {psg_file.name}...", end="", flush=True)

                raw_psg = mne.io.read_raw_edf(str(psg_file), preload=True, verbose=False)
                raw_psg.pick_types(eeg=True, exclude="bads")

                if float(raw_psg.info["sfreq"]) != self.target_sfreq:
                    raw_psg.resample(self.target_sfreq)

                if "filter" in self.config["data"]:
                    raw_psg.filter(
                        l_freq=float(self.config["data"]["filter"]["lowcut"]),
                        h_freq=float(self.config["data"]["filter"]["highcut"]),
                        verbose=False
                    )

                labels_samples = self._load_hypnogram(hyp_file, raw_psg)
                epochs, epoch_labels = self._create_epochs(raw_psg, labels_samples)

                if len(epochs) == 0:
                    print(" ✗ no valid epochs")
                    failed_subjects.append((subject_id, "no_valid_epochs"))
                    continue

                subject_pathways = feature_extractor.extract_all_features_separated(epochs)

                for state in states:
                    mask = np.array(epoch_labels) == state
                    n_state = int(np.sum(mask))
                    if n_state > 0:
                        for pname in ["spectral", "pac", "temporal", "connectivity"]:
                            state_features[state][pname].append(subject_pathways[pname][mask])
                        state_subject_ids[state].extend([subject_id] * n_state)

                loaded_subjects += 1
                print(f" ✓ {len(epochs)} epochs", flush=True)

                del raw_psg, epochs, subject_pathways, labels_samples, epoch_labels
                import gc
                gc.collect()

            except Exception as e:
                msg = str(e)[:120]
                print(f" ✗ error: {msg}")
                failed_subjects.append((subject_id, msg))
                continue

        if loaded_subjects == 0:
            raise ValueError("No valid subjects loaded")

        print(f"\n  ✅ Successfully loaded {loaded_subjects} subjects")
        if failed_subjects:
            print(f"  ⚠️  Failed: {len(failed_subjects)} subjects")

        print("\n  Concatenating features by state...")
        final_data: Dict[str, dict] = {}

        for state in states:
            if not state_features[state]["spectral"]:
                continue

            state_data: Dict[str, np.ndarray] = {}
            for pname in ["spectral", "pac", "temporal", "connectivity"]:
                chunks = state_features[state][pname]
                if chunks:
                    state_data[pname] = np.concatenate(chunks, axis=0)

            if state_data:
                final_data[state] = {
                    "pathways": state_data,
                    "subject_ids": np.array(state_subject_ids[state], dtype=object),
                    "n_epochs": int(len(state_subject_ids[state]))
                }
                print(f"    {state}: {final_data[state]['n_epochs']} epochs")

        total_epochs = int(sum(d["n_epochs"] for d in final_data.values()))

        return {
            "data_by_state": final_data,
            "info": {
                "n_subjects": int(loaded_subjects),
                "n_epochs_total": total_epochs,
                "sfreq": float(self.target_sfreq),
                "states_available": list(final_data.keys())
            }
        }

    def _load_hypnogram(self, hypno_file: Path, raw_psg) -> np.ndarray:
        try:
            annotations = mne.read_annotations(str(hypno_file))
        except Exception as e:
            if self.verbose:
                print(f" [error reading annotations: {e}]", end="")
            return np.full(len(raw_psg.times), "Unknown", dtype=object)

        stage_mapping = {
            "Sleep stage W": "Wake",
            "Sleep stage 1": "N1",
            "Sleep stage 2": "N2",
            "Sleep stage 3": "N3",
            "Sleep stage 4": "N3",
            "Sleep stage R": "REM",
            "Sleep stage ?": "Unknown",
            "Movement time": "Unknown",
        }

        psg_sfreq = float(raw_psg.info["sfreq"])
        n_samples = int(len(raw_psg.times))
        psg_duration = n_samples / psg_sfreq

        labels = np.full(n_samples, "Unknown", dtype=object)
        if len(annotations) == 0:
            return labels

        onsets = np.asarray(annotations.onset, dtype=float)
        durations = np.asarray(annotations.duration, dtype=float)

        # Corrección simple si el hypnograma tiene offset extraño
        offset_correction = float(np.min(onsets)) if abs(float(np.min(onsets))) > 1.0 else 0.0

        for i in range(len(annotations)):
            onset = float(annotations.onset[i]) - offset_correction
            duration = float(annotations.duration[i])
            desc = str(annotations.description[i])

            if onset < 0 or onset >= psg_duration:
                continue

            onset_sample = int(onset * psg_sfreq)
            end_sample = min(onset_sample + int(duration * psg_sfreq), n_samples)
            if onset_sample >= n_samples:
                continue

            labels[onset_sample:end_sample] = stage_mapping.get(desc, "Unknown")

        return labels

    def _create_epochs(self, raw, labels: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        data = raw.get_data()
        n_channels, n_samples = data.shape
        epoch_samples = int(self.epoch_duration * self.target_sfreq)
        step = epoch_samples - int(self.overlap * self.target_sfreq)

        epochs_list: List[np.ndarray] = []
        epoch_labels: List[str] = []

        for start in range(0, n_samples - epoch_samples + 1, step):
            seg = labels[start: start + epoch_samples]
            uniq, cnt = np.unique(seg, return_counts=True)
            maj = str(uniq[np.argmax(cnt)])
            if maj != "Unknown":
                epochs_list.append(data[:, start: start + epoch_samples])
                epoch_labels.append(maj)

        if not epochs_list:
            return np.empty((0, n_channels, epoch_samples), dtype=np.float64), []

        return np.array(epochs_list, dtype=np.float64), epoch_labels


# =============================================================================
# EXPERIMENT
# =============================================================================

class AFHConvergenceExperiment:
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG

        seed = int(self.config["experiment"]["random_seed"])
        np.random.seed(seed)

        self.metrics = RepresentationalMetrics(verbose=bool(self.config["output"]["verbose"]))
        self.extractor = EEGFeatureExtractor(
            sfreq=float(self.config["data"]["sampling_rate"]),
            verbose=bool(self.config["output"]["verbose"])
        )

        self.results_dir = Path(self.config["output"]["results_dir"])
        (self.results_dir / "figures").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "reports").mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("AFH CONVERGENCE EXPERIMENT - 32GB RAM OPTIMIZED (PATCHED v2.5.1 FINAL)")
        print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # Submuestreo estratificado por sujeto
    # -------------------------------------------------------------------------
    def _subsample_state_data(
        self,
        pathways: Dict[str, np.ndarray],
        max_epochs: int,
        seed: int = 42,
        subject_ids: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, np.ndarray], int, Optional[np.ndarray]]:
        n_epochs = pathways[list(pathways.keys())[0]].shape[0]
        if n_epochs <= max_epochs:
            return pathways, n_epochs, subject_ids

        rng = np.random.default_rng(seed)

        if subject_ids is None:
            print(f" (subsampling {n_epochs} → {max_epochs})", end="", flush=True)
            indices = rng.choice(n_epochs, size=max_epochs, replace=False)
            subs_sid = None
        else:
            subject_ids = np.asarray(subject_ids, dtype=object)
            print(f" (stratified subsampling {n_epochs} → {max_epochs})", end="", flush=True)

            uniq = np.unique(subject_ids)
            n_subj = len(uniq)
            if n_subj <= 0:
                indices = rng.choice(n_epochs, size=max_epochs, replace=False)
            else:
                per = int(math.ceil(max_epochs / n_subj))
                chosen = []
                for sid in uniq:
                    idx = np.where(subject_ids == sid)[0]
                    if idx.size == 0:
                        continue
                    take = min(per, idx.size)
                    chosen.append(rng.choice(idx, size=take, replace=False))
                if chosen:
                    indices = np.concatenate(chosen)
                    if indices.size > max_epochs:
                        indices = rng.choice(indices, size=max_epochs, replace=False)
                else:
                    indices = rng.choice(n_epochs, size=max_epochs, replace=False)

            subs_sid = subject_ids[indices]

        subsampled = {name: pathways[name][indices] for name in pathways.keys()}
        return subsampled, int(len(indices)), subs_sid

    def run_full_pipeline(self, use_synthetic: bool = False, sleep_edf_path: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        print("STEP 1) Loading EEG data...")

        if use_synthetic:
            gen = SyntheticEEGGenerator(
                sfreq=float(self.config["data"]["sampling_rate"]),
                seed=int(self.config["experiment"]["random_seed"])
            )
            dataset = gen.generate_synthetic_dataset(n_subjects=3, n_epochs_per_state=120, n_channels=10)

            data_by_state = {}
            for state in self.config["analysis"]["states"]:
                mask = dataset["labels"] == state
                if int(np.sum(mask)) > 0:
                    epochs_state = dataset["data"][mask]
                    pathways = self.extractor.extract_all_features_separated(epochs_state)
                    data_by_state[state] = {
                        "pathways": pathways,
                        "subject_ids": dataset["subject_ids"][mask],
                        "n_epochs": int(np.sum(mask))
                    }

            data_dict = {"SYNTHETIC": {"data_by_state": data_by_state, "info": dataset["info"]}}

        else:
            loader = EEGDataLoader(self.config)
            path = sleep_edf_path or self.config["data"]["sleep_edf"]["path"]
            sleep_edf_data = loader.load_sleep_edf(
                path,
                subjects=self.config["data"]["sleep_edf"]["subjects"],
                feature_extractor=self.extractor
            )
            data_dict = {"SLEEP-EDF": sleep_edf_data}

        print("\nSTEP 2) Analyzing convergence by state...")
        results = self._analyze_all_states(data_dict)

        print("\nSTEP 3) Comparing states...")
        comparisons = self._compare_states(results, data_dict)

        print("\nSTEP 4) Saving results...")
        self._save_results(results, comparisons)

        print("\n" + "=" * 70)
        print("DONE. Results in:", str(self.results_dir))
        print("=" * 70 + "\n")

        return results, comparisons

    def _analyze_all_states(self, data_dict: Dict[str, dict]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        k = int(self.config["metrics"]["cknna"]["default_k"])
        metric = str(self.config["metrics"]["cknna"].get("metric", "euclidean"))
        dcor_pairs = int(self.config["metrics"]["dcor"].get("n_pairs", 200000))

        n_perm = int(self.config["analysis"]["stats"]["n_permutations"])
        n_boot = int(self.config["analysis"]["stats"]["bootstrap_n"])
        alpha = float(self.config["analysis"]["stats"]["alpha"])

        max_epochs = int(self.config["analysis"]["stats"]["max_epochs_per_state"])
        boot_target = int(self.config["analysis"]["stats"].get("bootstrap_target_epochs", max_epochs))
        min_eps_subj = int(self.config["analysis"]["stats"].get("min_epochs_per_subject", 5))

        for dataset_name, dataset in data_dict.items():
            print(f"\n  Dataset: {dataset_name}")
            results[dataset_name] = {}

            data_by_state = dataset["data_by_state"]

            for state in self.config["analysis"]["states"]:
                if state not in data_by_state:
                    print(f"    {state}: SKIP (no data)")
                    continue

                state_data = data_by_state[state]
                pathways = state_data["pathways"]
                n_state = int(state_data["n_epochs"])
                subject_ids = state_data.get("subject_ids", None)

                if n_state < 50:
                    print(f"    {state}: SKIP (only {n_state} epochs)")
                    continue

                print(f"    {state}: {n_state} epochs", end="", flush=True)

                pathways_s, n_sampled, sub_sid = self._subsample_state_data(
                    pathways, max_epochs, seed=42, subject_ids=subject_ids
                )
                print("...", flush=True)

                summary = self.metrics.state_convergence_summary(
                    pathways_s, k=k, metric=metric, dcor_pairs=dcor_pairs
                )

                tests = {}
                test_pairs = [
                    ("spectral", "pac"),
                    ("spectral", "temporal"),
                    ("pac", "temporal"),
                    ("temporal", "connectivity"),
                ]

                for a, b in test_pairs:
                    if a not in pathways_s or b not in pathways_s:
                        continue

                    X = pathways_s[a]
                    Y = pathways_s[b]
                    obs = self.metrics.cknna(X, Y, k=k, normalize=True, metric=metric)

                    if sub_sid is not None:
                        null = self.metrics.cknna_null_within_subject(
                            X, Y, subject_ids=sub_sid, k=k, n_perm=n_perm, seed=42, metric=metric,
                            min_epochs_per_subject=min_eps_subj
                        )
                        boot = self.metrics.bootstrap_cknna_subject(
                            X, Y, subject_ids=sub_sid, k=k, n_boot=n_boot, seed=42, metric=metric,
                            target_n=min(boot_target, X.shape[0]),
                            min_epochs_per_subject=min_eps_subj
                        )
                        boot_method = "subject_level"
                    else:
                        # fallback: permutación global + subsample sin reemplazo
                        rng = np.random.default_rng(42)
                        null = np.empty(n_perm, dtype=float)
                        for bb in range(n_perm):
                            perm = rng.permutation(Y.shape[0])
                            null[bb] = self.metrics.cknna(X, Y[perm], k=k, normalize=True, metric=metric)

                        boot = np.empty(n_boot, dtype=float)
                        tn = min(boot_target, X.shape[0])
                        for bb in range(n_boot):
                            idx = rng.choice(X.shape[0], size=tn, replace=False)
                            boot[bb] = self.metrics.cknna(X[idx], Y[idx], k=k, normalize=True, metric=metric)

                        boot_method = "subsample_no_replacement"

                    p = self.metrics.p_value_greater(obs, null)

                    tests[f"{a}__{b}"] = {
                        "observed": float(obs),
                        "p_value": float(p),
                        "significant": bool(np.isfinite(p) and (p < alpha)),
                        "null_mean": float(_safe_nanmean(null)),
                        "null_ci_lower": float(_safe_nanpercentile(null, 2.5)),
                        "null_ci_upper": float(_safe_nanpercentile(null, 97.5)),
                        "bootstrap_mean": float(_safe_nanmean(boot)),
                        "bootstrap_ci_lower": float(_safe_nanpercentile(boot, 2.5)),
                        "bootstrap_ci_upper": float(_safe_nanpercentile(boot, 97.5)),
                        "bootstrap_method": boot_method,
                        "n_used": int(X.shape[0]),
                    }

                results[dataset_name][state] = {
                    "state": state,
                    "n_epochs_total": int(n_state),
                    "n_epochs_analyzed": int(n_sampled),
                    "pathways": list(pathways_s.keys()),
                    "cknna_matrix": summary["cknna_matrix"].tolist(),
                    "cknna_mean_offdiag": float(summary["cknna_mean_offdiag"]),
                    "intrinsic_dim": {kk: (None if np.isnan(vv) else float(vv)) for kk, vv in summary["intrinsic_dim"].items()},
                    "pairs": _to_jsonable(summary["pairs"]),
                    "significance_tests": tests,
                    "n_unique_subjects_analyzed": int(len(np.unique(sub_sid))) if sub_sid is not None else None
                }

        return results

    def _compare_states(self, results: Dict[str, Any], data_dict: Dict[str, dict]) -> Dict[str, Any]:
        comparisons: Dict[str, Any] = {}

        k = int(self.config["metrics"]["cknna"]["default_k"])
        metric = str(self.config["metrics"]["cknna"].get("metric", "euclidean"))
        dcor_pairs = int(self.config["metrics"]["dcor"].get("n_pairs", 200000))

        n_boot = int(self.config["analysis"]["stats"]["bootstrap_state_diff_n"])
        max_epochs = int(self.config["analysis"]["stats"]["max_epochs_per_state"])

        boot_target = int(self.config["analysis"]["stats"].get("bootstrap_target_epochs", max_epochs))
        min_eps_subj = int(self.config["analysis"]["stats"].get("min_epochs_per_subject", 5))

        for dataset_name, states_data in results.items():
            print(f"\n  Contrasts in {dataset_name}:")
            data_by_state = data_dict[dataset_name]["data_by_state"]

            for A, B in self.config["analysis"]["contrasts"]:
                if A not in states_data or B not in states_data:
                    continue
                if A not in data_by_state or B not in data_by_state:
                    continue

                convA = float(states_data[A]["cknna_mean_offdiag"])
                convB = float(states_data[B]["cknna_mean_offdiag"])

                # baseline subsample (por reproducibilidad)
                pathways_A, _, sid_A = self._subsample_state_data(
                    data_by_state[A]["pathways"], max_epochs, seed=42, subject_ids=data_by_state[A].get("subject_ids", None)
                )
                pathways_B, _, sid_B = self._subsample_state_data(
                    data_by_state[B]["pathways"], max_epochs, seed=43, subject_ids=data_by_state[B].get("subject_ids", None)
                )

                boot_result = self._bootstrap_convergence_diff_subject(
                    pathways_A, sid_A, pathways_B, sid_B,
                    k=k, n_boot=n_boot, seed=7, metric=metric, dcor_pairs=dcor_pairs,
                    target_n=min(boot_target, max_epochs),
                    min_epochs_per_subject=min_eps_subj
                )

                key = f"{dataset_name}__{A}_vs_{B}"
                if boot_result.get("ok"):
                    comparisons[key] = {
                        "dataset": dataset_name,
                        "state_A": A,
                        "state_B": B,
                        "convergence_A": convA,
                        "convergence_B": convB,
                        "difference": convA - convB,
                        "A_greater": convA > convB,
                        **{kk: vv for kk, vv in boot_result.items() if kk != "ok"}
                    }
                    p = boot_result["p_one_sided_A_greater"]
                    print(f"    {A} vs {B}: Δ={convA-convB:.6f}, p={p:.4f}")

        return comparisons

    def _bootstrap_convergence_diff_subject(
        self,
        pathways_A: Dict[str, np.ndarray],
        subject_ids_A: Optional[np.ndarray],
        pathways_B: Dict[str, np.ndarray],
        subject_ids_B: Optional[np.ndarray],
        k: int,
        n_boot: int,
        seed: int = 0,
        metric: str = "euclidean",
        dcor_pairs: int = 200000,
        target_n: int = 2000,
        min_epochs_per_subject: int = 5
    ) -> Dict[str, Any]:
        pathway_names = list(pathways_A.keys())
        nA = pathways_A[pathway_names[0]].shape[0]
        nB = pathways_B[pathway_names[0]].shape[0]

        if nA < 30 or nB < 30:
            return {"ok": False, "reason": "Insufficient epochs"}

        rng = np.random.default_rng(seed)
        diffs = np.empty(n_boot, dtype=float)

        def draw_subject_boot(pathways: Dict[str, np.ndarray], sids: np.ndarray) -> Dict[str, np.ndarray]:
            sids = np.asarray(sids, dtype=object)
            uniq = np.unique(sids)
            subj_to_idx = {sid: np.where(sids == sid)[0] for sid in uniq}
            good = [sid for sid, idx in subj_to_idx.items() if idx.size >= int(min_epochs_per_subject)]
            if len(good) < 2:
                n = pathways[pathway_names[0]].shape[0]
                tn = min(int(target_n), n)
                idx = rng.choice(n, size=tn, replace=False)
                return {name: pathways[name][idx] for name in pathway_names}

            good = np.array(good, dtype=object)
            chosen_subjects = rng.choice(good, size=len(good), replace=True)
            per_subj = max(1, int(math.ceil(min(int(target_n), len(sids)) / len(chosen_subjects))))

            chosen_rows = []
            for sid in chosen_subjects:
                idx = subj_to_idx[sid]
                take = min(per_subj, idx.size)
                chosen_rows.append(rng.choice(idx, size=take, replace=False))

            idx_all = np.concatenate(chosen_rows) if chosen_rows else np.array([], dtype=int)
            tn = min(int(target_n), pathways[pathway_names[0]].shape[0], idx_all.size if idx_all.size else int(target_n))
            if idx_all.size > tn:
                idx_all = rng.choice(idx_all, size=tn, replace=False)

            return {name: pathways[name][idx_all] for name in pathway_names}

        use_subject = (subject_ids_A is not None) and (subject_ids_B is not None)

        for b in range(n_boot):
            if use_subject:
                boot_A = draw_subject_boot(pathways_A, subject_ids_A)
                boot_B = draw_subject_boot(pathways_B, subject_ids_B)
                boot_method = "subject_level"
            else:
                tnA = min(int(target_n), nA)
                tnB = min(int(target_n), nB)
                idxA = rng.choice(nA, size=tnA, replace=False)
                idxB = rng.choice(nB, size=tnB, replace=False)
                boot_A = {name: pathways_A[name][idxA] for name in pathway_names}
                boot_B = {name: pathways_B[name][idxB] for name in pathway_names}
                boot_method = "subsample_no_replacement"

            sumA = self.metrics.state_convergence_summary(boot_A, k=k, metric=metric, dcor_pairs=dcor_pairs)
            sumB = self.metrics.state_convergence_summary(boot_B, k=k, metric=metric, dcor_pairs=dcor_pairs)
            diffs[b] = float(sumA["cknna_mean_offdiag"] - sumB["cknna_mean_offdiag"])

        # p(A>B) = P(diff <= 0) cuando diff = A - B  (one-sided A_greater)
        p_one_sided = float(np.mean(diffs <= 0))

        return {
            "ok": True,
            "diff_mean": float(np.mean(diffs)),
            "diff_ci_lower": float(np.percentile(diffs, 2.5)),
            "diff_ci_upper": float(np.percentile(diffs, 97.5)),
            "p_one_sided_A_greater": p_one_sided,
            "bootstrap_method": boot_method
        }

    def _save_results(self, results: Dict, comparisons: Dict) -> None:
        output_file = self.results_dir / "metrics" / "experiment_results.json"
        payload = {
            "experiment": self.config["experiment"],
            "results_by_state": _to_jsonable(results),
            "state_comparisons": _to_jsonable(comparisons),
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  Results saved: {output_file}")


# =============================================================================
# CLI / ENTRYPOINT
# =============================================================================

def build_config_from_args(args: argparse.Namespace) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy simple

    if args.results_dir:
        cfg["output"]["results_dir"] = str(args.results_dir)

    if args.verbose:
        cfg["output"]["verbose"] = True

    if args.seed is not None:
        cfg["experiment"]["random_seed"] = int(args.seed)

    if args.max_epochs is not None:
        cfg["analysis"]["stats"]["max_epochs_per_state"] = int(args.max_epochs)

    if args.boot_target is not None:
        cfg["analysis"]["stats"]["bootstrap_target_epochs"] = int(args.boot_target)

    if args.n_perm is not None:
        cfg["analysis"]["stats"]["n_permutations"] = int(args.n_perm)

    if args.n_boot is not None:
        cfg["analysis"]["stats"]["bootstrap_n"] = int(args.n_boot)

    if args.n_boot_diff is not None:
        cfg["analysis"]["stats"]["bootstrap_state_diff_n"] = int(args.n_boot_diff)

    if args.min_eps_subj is not None:
        cfg["analysis"]["stats"]["min_epochs_per_subject"] = int(args.min_eps_subj)

    if args.k is not None:
        cfg["metrics"]["cknna"]["default_k"] = int(args.k)

    if args.dcor_pairs is not None:
        cfg["metrics"]["dcor"]["n_pairs"] = int(args.dcor_pairs)

    if args.sleep_edf_path:
        cfg["data"]["sleep_edf"]["path"] = str(args.sleep_edf_path)

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="AFH Convergence Experiment - 32GB RAM (PATCHED v2.5.1 FINAL)")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--synthetic", action="store_true", help="Use synthetic data (fast test)")
    mode.add_argument("--real", action="store_true", help="Use real SLEEP-EDF data")

    parser.add_argument("--sleep_edf_path", type=str, default=None, help="Override SLEEP-EDF path")
    parser.add_argument("--results_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")

    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=None, help="Max epochs per state (post-load)")
    parser.add_argument("--boot_target", type=int, default=None, help="Target epochs for subject-bootstrap")
    parser.add_argument("--min_eps_subj", type=int, default=None, help="Min epochs per subject (bootstrap/null)")

    parser.add_argument("--n_perm", type=int, default=None, help="Permutations for null")
    parser.add_argument("--n_boot", type=int, default=None, help="Bootstrap iterations (pair tests)")
    parser.add_argument("--n_boot_diff", type=int, default=None, help="Bootstrap iterations (state diffs)")

    parser.add_argument("--k", type=int, default=None, help="k for kNN overlap")
    parser.add_argument("--dcor_pairs", type=int, default=None, help="Pairs for dCor approximation")

    args = parser.parse_args()

    # default: si no especifica --real ni --synthetic, asumir real (como usted suele correr)
    use_synth = bool(args.synthetic)
    use_real = bool(args.real) or (not args.synthetic)

    cfg = build_config_from_args(args)
    exp = AFHConvergenceExperiment(config=cfg)

    if use_synth and use_real:
        # no debiera ocurrir por mutual_exclusive_group, pero se deja robustez
        use_real = False

    results, comparisons = exp.run_full_pipeline(
        use_synthetic=use_synth,
        sleep_edf_path=args.sleep_edf_path
    )

    print("\nSUMMARY:")
    for dataset_name, states_data in results.items():
        print(f"\n{dataset_name}:")
        for state, r in states_data.items():
            analyzed = r.get("n_epochs_analyzed", r.get("n_epochs", 0))
            total = r.get("n_epochs_total", r.get("n_epochs", 0))
            print(f"  {state}: convergence = {r['cknna_mean_offdiag']:.6f} (n={analyzed}/{total})")

    if comparisons:
        print("\nCONTRASTS:")
        for kkey, comp in comparisons.items():
            print(
                f"  {kkey}: Δ={comp['difference']:.6f} "
                f"CI=[{comp['diff_ci_lower']:.6f},{comp['diff_ci_upper']:.6f}] "
                f"p(one-sided A>B)={comp['p_one_sided_A_greater']:.4f} "
                f"({comp.get('bootstrap_method','?')})"
            )


if __name__ == "__main__":
    main()
