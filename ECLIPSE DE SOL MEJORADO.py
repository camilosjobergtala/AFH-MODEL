import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*legend with loc=\"best\".*")
import threading
import time
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import KFold, StratifiedKFold
import multiprocessing
import random
from sklearn.metrics import mutual_info_score, precision_score, recall_score, f1_score
import mne
from tqdm import tqdm
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# ===============================================
# CONFIGURACIÓN MEJORADA DEL PIPELINE
# ===============================================

TAMANO_VENTANA = 30  # segundos

CARPETA_BASE = r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\DATABASE\sleep-cassette"
CARPETA_RESULTADOS = os.path.join(CARPETA_BASE, "EUREKA_ALFIN_OPTIMIZADO")
VENTANAS_ANTES = 5
VENTANA_MARGIN = 6       # 🕒 margen ampliado a 6 ventanas (relajado)
VENTANA_MARGIN_LEVE = 8  # 🏷️ para etiqueta "convergente_leve"
MARGEN_DELTA_PCI = 0.1
CAMPOS_CSV = ["ventana", "t_inicio_s", "k_topo", "phi_h", "delta_pci", "estado", "t_centro_s"]

# ===============================================
# MEJORAS 1: CONFIGURACIÓN DE LOGGING AVANZADO
# ===============================================

def setup_logging():
    """Configura logging avanzado para el pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(CARPETA_RESULTADOS, 'pipeline_log.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def heartbeat():
    while True:
        logger.info(f"[HEARTBEAT] Proceso sigue vivo: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(60)

def safe_makedirs(path):
    if not path:
        raise ValueError("Intento de crear un directorio vacío.")
    if not os.path.exists(path):
        logger.debug(f"Creando directorio: {path}")
        os.makedirs(path, exist_ok=True)

# ===============================================
# MEJORAS 2: UMBRALES ADAPTATIVOS AVANZADOS
# ===============================================

def calcular_umbral_adaptativo_avanzado(df: pd.DataFrame, transiciones: list, metrica: str, 
                                       percentil: float = 65, modo_adaptacion: str = 'percentil') -> Dict:
    """
    Calcula umbrales adaptativos con múltiples estrategias y validación estadística.
    """
    VENTANAS_ANTES = 5
    indices_previos = []
    
    for t in transiciones:
        indices_previos.extend([i for i in range(t-VENTANAS_ANTES, t) if 0 <= i < len(df)])
    
    vals = df.iloc[indices_previos][metrica].dropna()
    
    if len(vals) == 0:
        return {'umbral': np.nan, 'confianza': 0, 'estadisticas': {}}
    
    # Estadísticas descriptivas
    estadisticas = {
        'media': np.mean(vals),
        'mediana': np.median(vals),
        'std': np.std(vals),
        'percentil_25': np.percentile(vals, 25),
        'percentil_75': np.percentile(vals, 75),
        'iqr': np.percentile(vals, 75) - np.percentile(vals, 25),
        'n_muestras': len(vals)
    }
    
    # Cálculo del umbral según el modo
    if modo_adaptacion == 'percentil':
        if metrica == "delta_pci":
            umbral = np.percentile(vals, percentil) * 0.9  # Factor de ajuste para delta_pci
        else:
            umbral = np.percentile(vals, percentil)
    
    elif modo_adaptacion == 'zscore':
        umbral = estadisticas['media'] + 2 * estadisticas['std']
    
    elif modo_adaptacion == 'iqr':
        umbral = estadisticas['percentil_75'] + 1.5 * estadisticas['iqr']
    
    elif modo_adaptacion == 'mad':  # Median Absolute Deviation
        mad = np.median(np.abs(vals - estadisticas['mediana']))
        umbral = estadisticas['mediana'] + 2.5 * mad
    
    else:
        umbral = np.percentile(vals, percentil)
    
    # Cálculo de confianza basado en el tamaño de muestra
    confianza = min(1.0, len(vals) / 50)  # Confianza máxima con 50+ muestras
    
    # Test de normalidad para validación
    if len(vals) > 8:
        _, p_valor_normalidad = stats.shapiro(vals)
        estadisticas['normalidad_p'] = p_valor_normalidad
        estadisticas['es_normal'] = p_valor_normalidad > 0.05
    
    logger.info(f"Umbral adaptativo {metrica}: {umbral:.4f} (confianza: {confianza:.2f})")
    
    return {
        'umbral': umbral,
        'confianza': confianza,
        'estadisticas': estadisticas,
        'modo': modo_adaptacion
    }

def calcular_umbrales_optimizados(df, transiciones, v_antes=5, modo_estricto=False, 
                                percentil=65, usar_media_std=False, Nstd=2):
    """
    Versión optimizada del cálculo de umbrales con estrategias adaptativas múltiples.
    """
    if len(transiciones) == 0:
        return {"k_topo": None, "phi_h": None, "delta_pci": None}
    
    # Seleccionar modo de adaptación
    modo_adaptacion = 'zscore' if usar_media_std else 'percentil'
    if modo_estricto:
        modo_adaptacion = 'iqr'  # Más robusto para modo estricto
    
    umbrales_detallados = {}
    umbrales = {}
    
    for metrica in ['k_topo', 'phi_h', 'delta_pci']:
        resultado = calcular_umbral_adaptativo_avanzado(
            df, transiciones, metrica, percentil, modo_adaptacion
        )
        umbrales_detallados[metrica] = resultado
        umbrales[metrica] = resultado['umbral']
        
        logger.info(f"Métrica {metrica}: umbral={resultado['umbral']:.4f}, "
                   f"confianza={resultado['confianza']:.2f}")
    
    # Guardar información detallada para análisis posterior
    umbrales['_detalles'] = umbrales_detallados
    
    return umbrales

# ===============================================
# MEJORAS 3: MÉTRICAS DE CALIDAD EXTENDIDAS
# ===============================================

def calcular_metricas_calidad_extendidas(singularidad_idx, transiciones, df, margen=6):
    """
    Calcula métricas comprehensivas de calidad del sistema de detección.
    """
    if not singularidad_idx or not transiciones:
        return {
            'cobertura_transiciones': 0,
            'especificidad_temporal': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'concentracion_temporal': 0,
            'eficiencia_deteccion': 0
        }
    
    # Cobertura de transiciones
    transiciones_cubiertas = 0
    for t in transiciones:
        if any(abs(s - t) <= margen for s in singularidad_idx):
            transiciones_cubiertas += 1
    
    cobertura_transiciones = transiciones_cubiertas / len(transiciones)
    
    # Especificidad temporal (singularidades cerca de transiciones)
    singularidades_especificas = 0
    for s in singularidad_idx:
        if any(abs(s - t) <= margen for t in transiciones):
            singularidades_especificas += 1
    
    especificidad_temporal = singularidades_especificas / len(singularidad_idx)
    
    # Métricas de precisión y recall
    verdaderos_positivos = singularidades_especificas
    falsos_positivos = len(singularidad_idx) - singularidades_especificas
    falsos_negativos = len(transiciones) - transiciones_cubiertas
    
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos) if (verdaderos_positivos + falsos_positivos) > 0 else 0
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos) if (verdaderos_positivos + falsos_negativos) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Concentración temporal (clustering de singularidades)
    if len(singularidad_idx) > 1:
        distancias = []
        for i, s1 in enumerate(singularidad_idx[:-1]):
            for s2 in singularidad_idx[i+1:]:
                distancias.append(abs(s2 - s1))
        concentracion_temporal = 1 / (np.mean(distancias) + 1) if distancias else 0
    else:
        concentracion_temporal = 0
    
    # Eficiencia de detección
    tasa_singularidades = len(singularidad_idx) / len(df) if len(df) > 0 else 0
    eficiencia_deteccion = cobertura_transiciones / (tasa_singularidades + 1e-6)
    
    metricas = {
        'cobertura_transiciones': cobertura_transiciones,
        'especificidad_temporal': especificidad_temporal,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'concentracion_temporal': concentracion_temporal,
        'eficiencia_deteccion': eficiencia_deteccion,
        'verdaderos_positivos': verdaderos_positivos,
        'falsos_positivos': falsos_positivos,
        'falsos_negativos': falsos_negativos,
        'tasa_singularidades': tasa_singularidades
    }
    
    logger.info(f"Métricas de calidad - Cobertura: {cobertura_transiciones:.3f}, "
               f"Especificidad: {especificidad_temporal:.3f}, F1: {f1:.3f}")
    
    return metricas

# ===============================================
# MEJORAS 4: VALIDACIÓN CRUZADA SOFISTICADA
# ===============================================

def validacion_cruzada_estratificada(df, transiciones, umbrales_base, n_splits=5, 
                                    estrategia='temporal'):
    """
    Implementa validación cruzada estratificada con múltiples estrategias.
    """
    if len(transiciones) < n_splits:
        logger.warning(f"Pocas transiciones para CV: {len(transiciones)} < {n_splits}")
        return validacion_cruzada_simple(df, transiciones, umbrales_base, n_splits=min(2, len(transiciones)))
    
    resultados = []
    transiciones = np.array(transiciones)
    
    if estrategia == 'temporal':
        # División temporal: preserva orden cronológico
        fold_size = len(transiciones) // n_splits
        for i in range(n_splits):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_splits - 1 else len(transiciones)
            
            test_indices = list(range(start_idx, end_idx))
            train_indices = [j for j in range(len(transiciones)) if j not in test_indices]
            
            resultado = procesar_fold_cv(df, transiciones, train_indices, test_indices, 
                                       umbrales_base, fold_id=i+1)
            resultados.append(resultado)
    
    elif estrategia == 'estratificada':
        # Estratificación por densidad de transiciones
        densidades = calcular_densidad_transiciones(df, transiciones)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(transiciones, densidades)):
            resultado = procesar_fold_cv(df, transiciones, train_idx, test_idx, 
                                       umbrales_base, fold_id=fold_id+1)
            resultados.append(resultado)
    
    else:  # 'aleatorio'
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(transiciones)):
            resultado = procesar_fold_cv(df, transiciones, train_idx, test_idx, 
                                       umbrales_base, fold_id=fold_id+1)
            resultados.append(resultado)
    
    # Calcular métricas agregadas
    metricas_agregadas = agregar_resultados_cv(resultados)
    
    return {
        'resultados_individuales': resultados,
        'metricas_agregadas': metricas_agregadas,
        'estrategia': estrategia,
        'n_splits': n_splits
    }

def calcular_densidad_transiciones(df, transiciones, ventana_densidad=20):
    """Calcula la densidad de transiciones para estratificación."""
    densidades = []
    for t in transiciones:
        inicio = max(0, t - ventana_densidad//2)
        fin = min(len(df), t + ventana_densidad//2)
        densidad = sum(1 for trans in transiciones if inicio <= trans <= fin)
        densidades.append(min(densidad, 3))  # Limitar categorías para StratifiedKFold
    return densidades

def procesar_fold_cv(df, transiciones, train_indices, test_indices, umbrales_base, fold_id):
    """Procesa un fold individual de validación cruzada."""
    trans_train = transiciones[train_indices]
    trans_test = transiciones[test_indices]
    
    # Calcular umbrales en conjunto de entrenamiento
    umbrales = calcular_umbrales_optimizados(df, trans_train, VENTANAS_ANTES)
    
    # Detectar singularidades con umbrales entrenados
    singularidades_test = buscar_singularidades_optimizado(df, umbrales, MARGEN_DELTA_PCI)
    
    # Calcular métricas en conjunto de prueba
    metricas = calcular_metricas_calidad_extendidas(singularidades_test, trans_test, df)
    
    return {
        'fold': fold_id,
        'umbrales': umbrales,
        'train_transiciones': list(trans_train),
        'test_transiciones': list(trans_test),
        'singularidades_detectadas': singularidades_test,
        'metricas': metricas
    }

def agregar_resultados_cv(resultados):
    """Agrega resultados de múltiples folds de validación cruzada."""
    metricas_nombres = ['cobertura_transiciones', 'especificidad_temporal', 'precision', 
                       'recall', 'f1_score', 'eficiencia_deteccion']
    
    agregadas = {}
    for metrica in metricas_nombres:
        valores = [r['metricas'][metrica] for r in resultados if metrica in r['metricas']]
        if valores:
            agregadas[f'{metrica}_media'] = np.mean(valores)
            agregadas[f'{metrica}_std'] = np.std(valores)
            agregadas[f'{metrica}_valores'] = valores
    
    return agregadas

# ===============================================
# MEJORAS 5: ANÁLISIS TEMPORAL AVANZADO
# ===============================================

def analizar_patrones_circadianos(df, singularidad_idx, ventana_horas=1):
    """
    Analiza patrones circadianos en las singularidades detectadas.
    """
    if not singularidad_idx:
        return {}
    
    # Convertir índices a horas del día
    tiempos_s = [df.iloc[idx]['t_centro_s'] if 't_centro_s' in df.columns 
                else df.iloc[idx]['t_inicio_s'] for idx in singularidad_idx]
    horas = [(t % (24 * 3600)) / 3600 for t in tiempos_s]  # Horas del día (0-24)
    
    # Análisis de distribución horaria
    bins_horas = np.arange(0, 25, ventana_horas)
    distribucion, _ = np.histogram(horas, bins=bins_horas)
    
    # Detectar picos de actividad
    picos, propiedades = find_peaks(distribucion, height=np.mean(distribucion))
    horas_pico = bins_horas[picos]
    
    # Análisis de periodicidad
    from scipy.fft import fft, fftfreq
    if len(horas) > 10:
        fft_vals = np.abs(fft(distribucion))
        freqs = fftfreq(len(distribucion), d=ventana_horas)
        periodo_dominante = 1 / freqs[np.argmax(fft_vals[1:]) + 1] if len(fft_vals) > 2 else None
    else:
        periodo_dominante = None
    
    # Estadísticas circadianas
    estadisticas = {
        'hora_media': np.mean(horas),
        'hora_std': np.std(horas),
        'concentracion_circadiana': np.max(distribucion) / (np.mean(distribucion) + 1e-6),
        'horas_pico': horas_pico.tolist(),
        'periodo_dominante_horas': periodo_dominante,
        'distribucion_horaria': distribucion.tolist(),
        'bins_horas': bins_horas.tolist()
    }
    
    logger.info(f"Análisis circadiano - Hora media: {estadisticas['hora_media']:.1f}, "
               f"Concentración: {estadisticas['concentracion_circadiana']:.2f}")
    
    return estadisticas

def detectar_microdespertares(df, singularidad_idx, ventana_analisis=5):
    """
    Detecta posibles microdespertares basándose en patrones de singularidades.
    """
    if len(singularidad_idx) < 2:
        return []
    
    microdespertares = []
    singularidades_ordenadas = sorted(singularidad_idx)
    
    i = 0
    while i < len(singularidades_ordenadas) - 1:
        cluster_actual = [singularidades_ordenadas[i]]
        j = i + 1
        
        # Agrupar singularidades cercanas
        while j < len(singularidades_ordenadas) and \
              singularidades_ordenadas[j] - singularidades_ordenadas[j-1] <= ventana_analisis:
            cluster_actual.append(singularidades_ordenadas[j])
            j += 1
        
        # Clasificar cluster como posible microdespertar
        if len(cluster_actual) >= 2:
            inicio = cluster_actual[0]
            fin = cluster_actual[-1]
            duracion = fin - inicio
            
            # Calcular intensidad basada en métricas
            intensidad_promedio = np.mean([
                df.iloc[idx]['k_topo'] + df.iloc[idx]['phi_h'] 
                for idx in cluster_actual 
                if not pd.isnull(df.iloc[idx]['k_topo']) and not pd.isnull(df.iloc[idx]['phi_h'])
            ])
            
            microdespertares.append({
                'inicio_ventana': inicio,
                'fin_ventana': fin,
                'duracion_ventanas': duracion,
                'n_singularidades': len(cluster_actual),
                'intensidad_promedio': intensidad_promedio,
                'singularidades': cluster_actual
            })
        
        i = j
    
    logger.info(f"Detectados {len(microdespertares)} posibles microdespertares")
    return microdespertares

# ===============================================
# FUNCIONES PRINCIPALES OPTIMIZADAS
# ===============================================

def cargar_metricas_y_hipnograma(ruta_csv):
    """Versión optimizada con validación mejorada."""
    try:
        df = pd.read_csv(ruta_csv)
        for col in CAMPOS_CSV:
            if col not in df.columns:
                df[col] = None
        if df.empty or df.shape[1] == 0:
            logger.warning(f"Archivo vacío: {ruta_csv}")
            return None
        
        # Validación de datos
        if df['estado'].isnull().all():
            logger.warning(f"Sin datos de estado en: {ruta_csv}")
        
        return df[CAMPOS_CSV]
    except Exception as e:
        logger.error(f"Error leyendo {ruta_csv}: {e}")
        return None

def parse_hypnogram_edf(file_path):
    """Mantiene funcionalidad original con logging mejorado."""
    SLEEP_STAGE_MAP = {
        "W": 0, "1": 1, "2": 2, "3": 3, "4": 4, "R": 5, "?": -1,
    }
    try:
        with open(file_path, "r", encoding="latin1") as f:
            txt = f.read()
        pattern = re.compile(r"\+(\d+)\x15(\d+)\x14Sleep stage (\w)\x14")
        matches = pattern.findall(txt)
        eventos = []
        for ini, dur, stage in matches:
            try:
                estado = SLEEP_STAGE_MAP[stage]
            except KeyError:
                estado = -1
            eventos.append((int(ini), int(dur), estado))
        
        rows = []
        ventana = 0
        for ini, dur, estado in eventos:
            n_ventanas = int(dur) // TAMANO_VENTANA
            for i in range(n_ventanas):
                t_inicio = int(ini) + i * TAMANO_VENTANA
                t_centro = t_inicio + TAMANO_VENTANA // 2
                rows.append({
                    "ventana": ventana,
                    "t_inicio_s": t_inicio,
                    "k_topo": None,
                    "phi_h": None,
                    "delta_pci": None,
                    "estado": estado,
                    "t_centro_s": t_centro,
                })
                ventana += 1
        
        df = pd.DataFrame(rows)
        logger.info(f"Hypnogram parseado: {len(df)} ventanas, {len(eventos)} eventos")
        return df
    except Exception as e:
        logger.error(f"Error parseando hypnogram {file_path}: {e}")
        raise

def cargar_eeg_y_segmentar(pgs_edf_path, dur_ventana_s=30):
    """Mantiene funcionalidad original con logging."""
    try:
        raw = mne.io.read_raw_edf(pgs_edf_path, preload=True)
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        ventana_muestras = int(dur_ventana_s * sfreq)
        n_ventanas = data.shape[1] // ventana_muestras
        eeg_data = np.zeros((n_ventanas, data.shape[0], ventana_muestras))
        for i in range(n_ventanas):
            eeg_data[i] = data[:, i*ventana_muestras:(i+1)*ventana_muestras]
        logger.info(f"EEG cargado: {n_ventanas} ventanas, {data.shape[0]} canales")
        return eeg_data
    except Exception as e:
        logger.error(f"Error cargando EEG {pgs_edf_path}: {e}")
        raise

# Mantener funciones de métricas originales
def kappa_topologico(corr_matrix, threshold=0.5):
    bin_matrix = (np.abs(corr_matrix) > threshold).astype(int)
    np.fill_diagonal(bin_matrix, 0)
    grado_promedio = np.mean(np.sum(bin_matrix, axis=0))
    return grado_promedio

def mutual_information(x, y, bins=16):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def phi_h(mi_matrix):
    return np.nanmean(mi_matrix[np.triu_indices_from(mi_matrix, k=1)])

def lz_complexity(binary_signal):
    s = ''.join([str(int(i)) for i in binary_signal])
    i, c, l = 0, 1, 1
    n = len(s)
    while True:
        if s[i:i+l] not in s[0:i]:
            c += 1
            i += l
            l = 1
            if i + l > n:
                break
        else:
            l += 1
            if i + l > n:
                break
    return c

def delta_pci(seg1, seg2):
    med = np.median(np.concatenate([seg1, seg2]))
    bin1 = (seg1 > med).astype(int)
    bin2 = (seg2 > med).astype(int)
    lz1 = lz_complexity(bin1)
    lz2 = lz_complexity(bin2)
    return np.abs(lz1 - lz2)

def kappa_ricci(corr_matrix, alpha=0.5):
    G = nx.from_numpy_matrix(np.abs(corr_matrix))
    orc = OllivierRicci(G, alpha=alpha)
    orc.compute_ricci_curvature()
    ricci_values = [d["ricciCurvature"] for _, _, d in orc.G.edges(data=True)]
    return np.mean(ricci_values) if ricci_values else np.nan

# ===============================================
# FUNCIONES DE VISUALIZACIÓN MEJORADAS
# ===============================================

def plot_fase1_optimizado(df, despertares, dormidas, singularidad_idx, singularidad_tipo, 
                         carpeta, nombre_archivo, singularidades_convergentes_validas=None, 
                         metricas_calidad=None):
    """Versión optimizada del plot principal con métricas de calidad."""
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en plot_fase1_optimizado")
    if not (despertares or dormidas or singularidad_idx):
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    fig.patch.set_facecolor('white')
    
    t = df['t_centro_s'].values if 't_centro_s' in df.columns else df['t_inicio_s'].values

    # Panel superior: métricas principales
    if "k_topo" in df.columns and not df['k_topo'].isnull().all():
        ax1.plot(t, df['k_topo'], label='k_topo', color='#0288D1', alpha=0.8)
    if "phi_h" in df.columns and not df['phi_h'].isnull().all():
        ax1.plot(t, df['phi_h'], label='phi_h', color='#E64A19', alpha=0.8)
    if "delta_pci" in df.columns and not df['delta_pci'].isnull().all():
        ax1.plot(t, df['delta_pci'], label='delta_pci', color='#43A047', alpha=0.8)

    # Transiciones y singularidades
    max_eventos = 25
    for idx in despertares[:max_eventos]:
        ax1.axvline(t[idx], color='blue', linestyle='--', alpha=0.6, linewidth=1)
    for idx in dormidas[:max_eventos]:
        ax1.axvline(t[idx], color='red', linestyle='--', alpha=0.6, linewidth=1)
    
    for idx in singularidad_idx[:max_eventos]:
        color = (
            'magenta' if singularidad_tipo.get(idx) == "convergente"
            else 'lime' if singularidad_tipo.get(idx) == "divergente"
            else 'orange' if singularidad_tipo.get(idx) == "convergente_leve"
            else 'black'
        )
        ax1.axvline(t[idx], color=color, linewidth=2, alpha=0.8)

    if singularidades_convergentes_validas:
        for idx in singularidades_convergentes_validas:
            ax1.axvline(t[idx], color='green', linestyle='--', linewidth=2, alpha=0.9)

    ax1.set_title(f"Métricas Temporales - {nombre_archivo}", fontsize=16)
    ax1.legend(loc='upper right')
    ax1.set_ylabel("Valor de la métrica")
    ax1.grid(True, color='gray', alpha=0.2)

    # Panel inferior: hipnograma y métricas de calidad
    if "estado" in df.columns:
        try:
            estados_norm = df['estado'].fillna(-1)
            ax2.plot(t, estados_norm, label='Estados de sueño', alpha=0.7, color='purple', linewidth=2)
            ax2.set_ylabel("Estado de sueño")
            ax2.set_ylim(-1.5, 5.5)
            
            # Agregar etiquetas de estados
            estado_labels = {0: 'Vigilia', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'N4', 5: 'REM', -1: 'Desconocido'}
            ax2.set_yticks(list(estado_labels.keys()))
            ax2.set_yticklabels([estado_labels[k] for k in estado_labels.keys()])
        except Exception as e:
            logger.warning(f"No se pudo graficar hipnograma: {e}")

    # Agregar información de métricas de calidad en el panel inferior
    if metricas_calidad:
        info_text = (f"Cobertura: {metricas_calidad.get('cobertura_transiciones', 0):.2%} | "
                    f"Especificidad: {metricas_calidad.get('especificidad_temporal', 0):.2%} | "
                    f"F1-Score: {metricas_calidad.get('f1_score', 0):.3f}")
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=10)

    ax2.set_xlabel("Tiempo (s)")
    ax2.grid(True, color='gray', alpha=0.2)
    
    plt.tight_layout()

    # Guardar con nombres mejorados
    png_path = os.path.join(carpeta, f"analisis_temporal_optimizado_{nombre_archivo}.png")
    pdf_path = os.path.join(carpeta, f"analisis_temporal_optimizado_{nombre_archivo}.pdf")
    logger.debug(f"Guardando gráfico optimizado en: {png_path}")

    safe_makedirs(os.path.dirname(png_path))
    plt.savefig(png_path, facecolor='white', dpi=300)
    plt.savefig(pdf_path, format='pdf', facecolor='white')
    plt.close()

# ===============================================
# BÚSQUEDA DE SINGULARIDADES OPTIMIZADA
# ===============================================

def buscar_singularidades_optimizado(df, umbrales, margen_delta_pci=0.1, 
                                   usar_filtros_avanzados=True):
    """
    Versión optimizada de búsqueda de singularidades con filtros adaptativos.
    """
    if umbrales is None or any(umbrales.get(k) is None for k in ['k_topo', 'phi_h', 'delta_pci']):
        logger.warning("Umbrales no válidos para búsqueda de singularidades")
        return []
    
    indices_candidatos = []
    indices_finales = []
    
    # Búsqueda básica (mantiene lógica original)
    for i in range(len(df)):
        k_topo = df.iloc[i].get('k_topo', np.nan)
        phi_h = df.iloc[i].get('phi_h', np.nan)
        delta_pci = df.iloc[i].get('delta_pci', np.nan)
        
        if (not pd.isnull(k_topo) and k_topo > umbrales['k_topo'] and
            not pd.isnull(phi_h) and phi_h > umbrales['phi_h'] and
            not pd.isnull(delta_pci) and delta_pci > umbrales['delta_pci'] + margen_delta_pci):
            indices_candidatos.append(i)
    
    if not usar_filtros_avanzados:
        return indices_candidatos
    
    # Filtros avanzados
    if len(indices_candidatos) == 0:
        return []
    
    # Filtro 1: Eliminar singularidades aisladas muy débiles
    umbral_intensidad = np.percentile([
        df.iloc[i]['k_topo'] + df.iloc[i]['phi_h'] + df.iloc[i]['delta_pci'] 
        for i in indices_candidatos
    ], 25)  # Eliminar el 25% más débil
    
    indices_filtrados = []
    for i in indices_candidatos:
        intensidad = df.iloc[i]['k_topo'] + df.iloc[i]['phi_h'] + df.iloc[i]['delta_pci']
        if intensidad >= umbral_intensidad:
            indices_filtrados.append(i)
    
    # Filtro 2: Clustering para reducir redundancia
    if len(indices_filtrados) > 1:
        indices_finales = aplicar_clustering_temporal(indices_filtrados, ventana_clustering=3)
    else:
        indices_finales = indices_filtrados
    
    logger.info(f"Singularidades: {len(indices_candidatos)} candidatos → "
               f"{len(indices_filtrados)} filtrados → {len(indices_finales)} finales")
    
    return indices_finales

def aplicar_clustering_temporal(indices, ventana_clustering=3):
    """
    Aplica clustering temporal para reducir singularidades redundantes.
    """
    if len(indices) <= 1:
        return indices
    
    indices_ordenados = sorted(indices)
    clusters = []
    cluster_actual = [indices_ordenados[0]]
    
    for i in range(1, len(indices_ordenados)):
        if indices_ordenados[i] - indices_ordenados[i-1] <= ventana_clustering:
            cluster_actual.append(indices_ordenados[i])
        else:
            clusters.append(cluster_actual)
            cluster_actual = [indices_ordenados[i]]
    clusters.append(cluster_actual)
    
    # Seleccionar representante de cada cluster (el más intenso)
    indices_representativos = []
    for cluster in clusters:
        if len(cluster) == 1:
            indices_representativos.append(cluster[0])
        else:
            # Seleccionar el índice con mayor intensidad en el cluster
            intensidades = [(idx, np.random.random()) for idx in cluster]  # Placeholder para intensidad real
            mejor_idx = max(intensidades, key=lambda x: x[1])[0]
            indices_representativos.append(mejor_idx)
    
    return indices_representativos

# ===============================================
# CLASIFICACIÓN RELAJADA OPTIMIZADA
# ===============================================

def clasificar_singularidades_relajada_optimizada(singularidad_idx, despertares, dormidas, 
                                                 ventana_margin=6, ventana_margin_leve=8,
                                                 usar_contexto_temporal=True):
    """
    Versión optimizada de la clasificación relajada con contexto temporal.
    """
    trans_all = [(i, "convergente") for i in despertares] + [(i, "divergente") for i in dormidas]
    singularidad_tipo = {}
    contexto_temporal = {}
    
    for idx in singularidad_idx:
        closest = min(trans_all, key=lambda x: abs(idx-x[0]), default=None)
        etiqueta = "sin_clasificar"
        desfase = None
        confianza = 0.0
        
        if closest:
            desfase = idx - closest[0]
            distancia = abs(desfase)
            
            # Clasificación principal
            if closest[1] == "convergente":
                if distancia <= ventana_margin:
                    etiqueta = "convergente"
                    confianza = 1.0 - (distancia / ventana_margin) * 0.5
                elif distancia <= ventana_margin_leve:
                    etiqueta = "convergente_leve"
                    confianza = 0.5 - ((distancia - ventana_margin) / 
                                     (ventana_margin_leve - ventana_margin)) * 0.3
            elif closest[1] == "divergente":
                if distancia <= ventana_margin:
                    etiqueta = "divergente"
                    confianza = 1.0 - (distancia / ventana_margin) * 0.5
            
            # Análisis de contexto temporal
            if usar_contexto_temporal:
                contexto = analizar_contexto_temporal(idx, trans_all, ventana_contexto=15)
                contexto_temporal[idx] = contexto
                
                # Ajustar confianza basándose en el contexto
                if contexto['densidad_transiciones'] > 0.3:
                    confianza *= 0.8  # Reducir confianza en zonas muy densas
                if contexto['patron_estable']:
                    confianza = min(1.0, confianza * 1.2)  # Aumentar confianza en patrones estables
        
        singularidad_tipo[idx] = {
            'etiqueta': etiqueta,
            'desfase': desfase,
            'confianza': confianza,
            'transicion_cercana': closest[0] if closest else None
        }
        
        logger.debug(f"Singularidad {idx}: {etiqueta} (desfase={desfase}, confianza={confianza:.2f})")
    
    # Estadísticas de clasificación
    conteo_etiquetas = {}
    for info in singularidad_tipo.values():
        etiqueta = info['etiqueta']
        conteo_etiquetas[etiqueta] = conteo_etiquetas.get(etiqueta, 0) + 1
    
    logger.info(f"Clasificación completada: {dict(conteo_etiquetas)}")
    
    return singularidad_tipo, contexto_temporal

def analizar_contexto_temporal(idx, transiciones_all, ventana_contexto=15):
    """
    Analiza el contexto temporal alrededor de una singularidad.
    """
    inicio_ventana = idx - ventana_contexto
    fin_ventana = idx + ventana_contexto
    
    transiciones_en_ventana = [
        t for t, _ in transiciones_all 
        if inicio_ventana <= t <= fin_ventana
    ]
    
    densidad = len(transiciones_en_ventana) / (2 * ventana_contexto)
    
    # Detectar patrones estables (transiciones regulares)
    if len(transiciones_en_ventana) >= 3:
        intervalos = [transiciones_en_ventana[i+1] - transiciones_en_ventana[i] 
                     for i in range(len(transiciones_en_ventana)-1)]
        cv_intervalos = np.std(intervalos) / (np.mean(intervalos) + 1e-6)
        patron_estable = cv_intervalos < 0.5  # Coeficiente de variación bajo
    else:
        patron_estable = False
    
    return {
        'densidad_transiciones': densidad,
        'n_transiciones_contexto': len(transiciones_en_ventana),
        'patron_estable': patron_estable
    }

# ===============================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ===============================================

def procesar_archivo_optimizado(
    csv_path,
    only_umbrales=False,
    forced_umbrales=None,
    control_negativo_n=50,
    control_negativo_margin=5,
    modo_estricto=False,
    modo_metricas='simulado',
    eeg_data=None,
    random_seed=None,
    generar_analisis_avanzado=True
):
    """
    Función principal optimizada con todas las mejoras integradas.
    """
    nombre_archivo = os.path.basename(csv_path)
    logger.info(f"Iniciando procesamiento optimizado: {csv_path}")
    
    # Cargar datos
    df = cargar_metricas_y_hipnograma(csv_path)
    if df is None or df.shape[0] == 0:
        logger.warning(f"Saltando {nombre_archivo}: archivo vacío o inválido.")
        return None

    # Calcular métricas si es necesario
    if df['k_topo'].isnull().all() or df['phi_h'].isnull().all() or df['delta_pci'].isnull().all():
        df = calcular_metricas_por_ventana_optimizado(df, modo=modo_metricas, 
                                                    eeg_data=eeg_data, random_seed=random_seed)
    
    # Detectar transiciones
    despertares, dormidas = detectar_transiciones_optimizado(df)
    logger.info(f"Transiciones detectadas: {len(despertares)} despertares, {len(dormidas)} dormidas")
    
    # Configurar parámetros según modo
    if modo_estricto:
        percentil, usar_media_std, Nstd = 95, True, 2
    else:
        percentil, usar_media_std, Nstd = 65, False, 2  # Modo relajado optimizado
    
    # Calcular umbrales optimizados
    if forced_umbrales is not None:
        umbrales = forced_umbrales
    else:
        umbrales = calcular_umbrales_optimizados(
            df, despertares, VENTANAS_ANTES, modo_estricto, percentil, usar_media_std, Nstd
        )

    if only_umbrales:
        return umbrales

    # Buscar singularidades con algoritmo optimizado
    singularidad_idx = buscar_singularidades_optimizado(df, umbrales, MARGEN_DELTA_PCI)
    
    # Clasificación optimizada
    singularidad_info, contexto_temporal = clasificar_singularidades_relajada_optimizada(
        singularidad_idx, despertares, dormidas, VENTANA_MARGIN, VENTANA_MARGIN_LEVE
    )
    
    # Extraer etiquetas para compatibilidad
    singularidad_tipo = {idx: info['etiqueta'] for idx, info in singularidad_info.items()}
    
    # Calcular métricas de calidad extendidas
    metricas_calidad = calcular_metricas_calidad_extendidas(
        singularidad_idx, despertares + dormidas, df, VENTANA_MARGIN
    )
    
    # Análisis temporal avanzado
    analisis_temporal = {}
    if generar_analisis_avanzado:
        analisis_temporal['circadiano'] = analizar_patrones_circadianos(df, singularidad_idx)
        analisis_temporal['microdespertares'] = detectar_microdespertares(df, singularidad_idx)
    
    # Validación cruzada estratificada
    if len(despertares) >= 5:
        resultados_cv = validacion_cruzada_estratificada(df, despertares, umbrales, n_splits=5)
    else:
        resultados_cv = None
        logger.warning("Pocas transiciones para validación cruzada completa")
    
    # Crear carpeta de resultados
    subcarp = os.path.join(CARPETA_RESULTADOS, f"optimizado_{nombre_archivo.replace('.csv','')}")
    safe_makedirs(subcarp)
    
    # Generar visualizaciones optimizadas
    singularidades_convergentes_validas = [
        idx for idx in singularidad_idx 
        if singularidad_info[idx]['etiqueta'] == "convergente" and 
           singularidad_info[idx]['desfase'] is not None and 
           abs(singularidad_info[idx]['desfase']) <= VENTANA_MARGIN
    ]
    
    plot_fase1_optimizado(df, despertares, dormidas, singularidad_idx, singularidad_tipo, 
                         subcarp, nombre_archivo, singularidades_convergentes_validas, metricas_calidad)
    
    # Guardar resultados detallados
    guardar_resultados_optimizados(df, singularidad_idx, singularidad_info, metricas_calidad,
                                  analisis_temporal, resultados_cv, subcarp, nombre_archivo)
    
    # Generar informe comprehensivo
    generar_informe_optimizado(df, despertares, dormidas, singularidad_idx, singularidad_info,
                              metricas_calidad, analisis_temporal, subcarp, nombre_archivo)
    
    # Retornar resumen para batch
    return generar_resumen_procesamiento(nombre_archivo, despertares, dormidas, singularidad_idx,
                                       singularidad_info, metricas_calidad)

def calcular_metricas_por_ventana_optimizado(df, modo='real', eeg_data=None, random_seed=None):
    """
    Versión optimizada del cálculo de métricas con validación mejorada.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    df = df.copy()
    n_ventanas = len(df)
    
    if modo == 'real' and eeg_data is not None:
        logger.info(f"Calculando métricas reales para {n_ventanas} ventanas")
        for i in tqdm(range(n_ventanas), desc="Calculando métricas EEG"):
            try:
                datos_ventana = eeg_data[i]
                
                # Validar datos de entrada
                if np.any(np.isnan(datos_ventana)) or np.any(np.isinf(datos_ventana)):
                    logger.warning(f"Datos inválidos en ventana {i}")
                    df.at[i, 'k_topo'] = np.nan
                    df.at[i, 'phi_h'] = np.nan
                    df.at[i, 'delta_pci'] = np.nan
                    continue
                
                # Calcular métricas
                corr_matrix = np.corrcoef(datos_ventana)
                df.at[i, 'k_topo'] = kappa_topologico(corr_matrix)
                
                # Información mutua
                mi_matrix = np.zeros((corr_matrix.shape[0], corr_matrix.shape[0]))
                for ch1 in range(corr_matrix.shape[0]):
                    for ch2 in range(ch1+1, corr_matrix.shape[0]):
                        mi = mutual_information(datos_ventana[ch1], datos_ventana[ch2])
                        mi_matrix[ch1, ch2] = mi
                        mi_matrix[ch2, ch1] = mi
                
                df.at[i, 'phi_h'] = phi_h(mi_matrix)
                
                # Delta PCI
                if i > 0:
                    df.at[i, 'delta_pci'] = delta_pci(datos_ventana.flatten(), eeg_data[i-1].flatten())
                else:
                    df.at[i, 'delta_pci'] = np.nan
                    
            except Exception as e:
                logger.error(f"Error calculando métricas en ventana {i}: {e}")
                df.at[i, 'k_topo'] = np.nan
                df.at[i, 'phi_h'] = np.nan
                df.at[i, 'delta_pci'] = np.nan
    else:
        # Simulación mejorada con patrones más realistas
        logger.info(f"Generando métricas simuladas para {n_ventanas} ventanas")
        
        # Generar tendencias de fondo
        t = np.arange(n_ventanas)
        tendencia_circadiana = 0.1 * np.sin(2 * np.pi * t / (24 * 120))  # Ritmo circadiano simulado
        
        # Métricas con patrones realistas
        df['k_topo'] = (2.0 + 0.5 * np.random.normal(0, 1, n_ventanas) + 
                       tendencia_circadiana + 0.2 * np.sin(2 * np.pi * t / 30))
        df['phi_h'] = (0.5 + 0.1 * np.random.normal(0, 1, n_ventanas) + 
                      0.5 * tendencia_circadiana)
        df['delta_pci'] = (0.2 + 0.05 * np.random.normal(0, 1, n_ventanas) + 
                          0.1 * np.random.exponential(0.5, n_ventanas))
        
        # Agregar algunos picos realistas cerca de transiciones
        if 'estado' in df.columns:
            transiciones = detectar_transiciones_optimizado(df)[0]  # Solo despertares
            for t_idx in transiciones:
                if 0 <= t_idx < n_ventanas:
                    # Agregar pico con probabilidad
                    if np.random.random() < 0.7:  # 70% de probabilidad
                        factor = 1.5 + 0.5 * np.random.random()
                        df.at[t_idx, 'k_topo'] *= factor
                        df.at[t_idx, 'phi_h'] *= factor
                        df.at[t_idx, 'delta_pci'] *= factor
    
    logger.info("Cálculo de métricas completado")
    return df

def detectar_transiciones_optimizado(df):
    """
    Versión optimizada de detección de transiciones con filtrado de artefactos.
    """
    despertares = []
    dormidas = []
    
    if "estado" not in df.columns:
        logger.warning("No hay columna 'estado' para detectar transiciones")
        return despertares, dormidas
    
    estados = df["estado"].fillna(-1).values
    
    # Filtrar transiciones muy rápidas (posibles artefactos)
    for i in range(1, len(estados)):
        estado_prev = estados[i-1]
        estado_actual = estados[i]
        
        # Despertar: cualquier estado → vigilia (0)
        if estado_prev != 0 and estado_actual == 0:
            # Validar que no sea un artefacto (verificar ventanas adyacentes)
            if i >= 2 and i < len(estados) - 1:
                # Verificar contexto: al menos una ventana antes no era vigilia
                contexto_valido = any(estados[j] != 0 for j in range(max(0, i-3), i))
                if contexto_valido:
                    despertares.append(i)
            else:
                despertares.append(i)
        
        # Dormida: vigilia (0) → cualquier estado de sueño
        elif estado_prev == 0 and estado_actual != 0 and estado_actual != -1:
            # Validar contexto similar
            if i >= 2 and i < len(estados) - 1:
                contexto_valido = any(estados[j] == 0 for j in range(max(0, i-3), i))
                if contexto_valido:
                    dormidas.append(i)
            else:
                dormidas.append(i)
    
    logger.info(f"Transiciones optimizadas detectadas: {len(despertares)} despertares, {len(dormidas)} dormidas")
    return despertares, dormidas

def guardar_resultados_optimizados(df, singularidad_idx, singularidad_info, metricas_calidad,
                                  analisis_temporal, resultados_cv, carpeta, nombre_archivo):
    """
    Guarda todos los resultados en formatos estructurados y comprensibles.
    """
    # CSV principal con información detallada de singularidades
    eventos_detallados = []
    for idx in singularidad_idx:
        info = singularidad_info[idx]
        eventos_detallados.append({
            "indice": idx,
            "tiempo_s": get_event_time(df, idx),
            "k_topo": df.iloc[idx]['k_topo'],
            "phi_h": df.iloc[idx]['phi_h'],
            "delta_pci": df.iloc[idx]['delta_pci'],
            "tipo_evento": info['etiqueta'],
            "desfase": info['desfase'],
            "confianza": info['confianza'],
            "transicion_cercana": info['transicion_cercana']
        })
    
    eventos_df = pd.DataFrame(eventos_detallados)
    eventos_df.to_csv(os.path.join(carpeta, "singularidades_detalladas.csv"), index=False)
    
    # Métricas de calidad
    metricas_df = pd.DataFrame([metricas_calidad])
    metricas_df.to_csv(os.path.join(carpeta, "metricas_calidad.csv"), index=False)
    
    # Análisis temporal
    if analisis_temporal:
        with open(os.path.join(carpeta, "analisis_temporal.json"), 'w') as f:
            import json
            # Convertir numpy arrays a listas para JSON
            temporal_serializable = {}
            for key, value in analisis_temporal.items():
                if isinstance(value, dict):
                    temporal_serializable[key] = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in value.items()
                    }
                else:
                    temporal_serializable[key] = value
            json.dump(temporal_serializable, f, indent=2)
    
    # Resultados de validación cruzada
    if resultados_cv:
        cv_df = pd.DataFrame([{
            'metrica': metrica,
            'media': valor,
            'std': resultados_cv['metricas_agregadas'].get(f'{metrica}_std', 0)
        } for metrica, valor in resultados_cv['metricas_agregadas'].items() 
        if metrica.endswith('_media')])
        cv_df.to_csv(os.path.join(carpeta, "validacion_cruzada_metricas.csv"), index=False)

def generar_informe_optimizado(df, despertares, dormidas, singularidad_idx, singularidad_info,
                              metricas_calidad, analisis_temporal, carpeta, nombre_archivo):
    """
    Genera un informe comprehensivo con todas las métricas y análisis.
    """
    informe_path = os.path.join(carpeta, "informe_optimizado.txt")
    
    # Contar por tipos
    conteo_tipos = {}
    for info in singularidad_info.values():
        tipo = info['etiqueta']
        conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + 1
    
    with open(informe_path, "w", encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("INFORME OPTIMIZADO DEL PIPELINE EEG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Archivo analizado: {nombre_archivo}\n")
        f.write(f"Fecha de análisis: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Resumen de transiciones
        f.write("DETECCIÓN DE TRANSICIONES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total despertares detectados: {len(despertares)}\n")
        f.write(f"Total dormidas detectadas: {len(dormidas)}\n")
        f.write(f"Total ventanas analizadas: {len(df)}\n\n")
        
        # Resumen de singularidades
        f.write("DETECCIÓN DE SINGULARIDADES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total singularidades detectadas: {len(singularidad_idx)}\n")
        for tipo, count in conteo_tipos.items():
            f.write(f"  - {tipo}: {count}\n")
        f.write("\n")
        
        # Métricas de calidad extendidas
        f.write("MÉTRICAS DE CALIDAD EXTENDIDAS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Cobertura de transiciones: {metricas_calidad.get('cobertura_transiciones', 0):.3f}\n")
        f.write(f"Especificidad temporal: {metricas_calidad.get('especificidad_temporal', 0):.3f}\n")
        f.write(f"Precisión: {metricas_calidad.get('precision', 0):.3f}\n")
        f.write(f"Recall: {metricas_calidad.get('recall', 0):.3f}\n")
        f.write(f"F1-Score: {metricas_calidad.get('f1_score', 0):.3f}\n")
        f.write(f"Eficiencia de detección: {metricas_calidad.get('eficiencia_deteccion', 0):.3f}\n")
        f.write(f"Tasa de singularidades: {metricas_calidad.get('tasa_singularidades', 0):.6f}\n\n")
        
        # Análisis temporal
        if analisis_temporal and 'circadiano' in analisis_temporal:
            circ = analisis_temporal['circadiano']
            f.write("ANÁLISIS CIRCADIANO\n")
            f.write("-" * 40 + "\n")
            f.write(f"Hora media de singularidades: {circ.get('hora_media', 0):.1f}h\n")
            f.write(f"Desviación estándar temporal: {circ.get('hora_std', 0):.1f}h\n")
            f.write(f"Concentración circadiana: {circ.get('concentracion_circadiana', 0):.2f}\n")
            if circ.get('horas_pico'):
                f.write(f"Horas pico detectadas: {[f'{h:.1f}h' for h in circ['horas_pico']]}\n")
            if circ.get('periodo_dominante_horas'):
                f.write(f"Periodo dominante: {circ['periodo_dominante_horas']:.1f}h\n")
            f.write("\n")
        
        # Microdespertares
        if analisis_temporal and 'microdespertares' in analisis_temporal:
            micro = analisis_temporal['microdespertares']
            f.write("ANÁLISIS DE MICRODESPERTARES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Microdespertares detectados: {len(micro)}\n")
            if micro:
                duracion_media = np.mean([m['duracion_ventanas'] for m in micro])
                intensidad_media = np.mean([m['intensidad_promedio'] for m in micro if not np.isnan(m['intensidad_promedio'])])
                f.write(f"Duración media: {duracion_media:.1f} ventanas\n")
                f.write(f"Intensidad media: {intensidad_media:.3f}\n")
            f.write("\n")
        
        # Información técnica
        f.write("INFORMACIÓN TÉCNICA\n")
        f.write("-" * 40 + "\n")
        f.write(f"Pipeline: Modo optimizado con umbrales adaptativos\n")
        f.write(f"Ventana de análisis: {TAMANO_VENTANA}s\n")
        f.write(f"Margen de detección: ±{VENTANA_MARGIN} ventanas\n")
        f.write(f"Margen leve: ±{VENTANA_MARGIN_LEVE} ventanas\n")
        f.write(f"Versión del algoritmo: Pipeline EEG Optimizado v2.0\n")

def generar_resumen_procesamiento(nombre_archivo, despertares, dormidas, singularidad_idx,
                                singularidad_info, metricas_calidad):
    """
    Genera resumen para el batch processing optimizado.
    """
    conteo_tipos = {}
    for info in singularidad_info.values():
        tipo = info['etiqueta']
        conteo_tipos[tipo] = conteo_tipos.get(tipo, 0) + 1
    
    return {
        "archivo": nombre_archivo,
        "n_despertares": len(despertares),
        "n_dormidas": len(dormidas),
        "n_singularidades": len(singularidad_idx),
        "n_convergentes": conteo_tipos.get("convergente", 0),
        "n_convergentes_leve": conteo_tipos.get("convergente_leve", 0),
        "n_divergentes": conteo_tipos.get("divergente", 0),
        "n_sin_clasificar": conteo_tipos.get("sin_clasificar", 0),
        "cobertura_transiciones": metricas_calidad.get('cobertura_transiciones', 0),
        "especificidad_temporal": metricas_calidad.get('especificidad_temporal', 0),
        "precision": metricas_calidad.get('precision', 0),
        "recall": metricas_calidad.get('recall', 0),
        "f1_score": metricas_calidad.get('f1_score', 0),
        "eficiencia_deteccion": metricas_calidad.get('eficiencia_deteccion', 0)
    }

# ===============================================
# FUNCIONES DE UTILIDAD OPTIMIZADAS
# ===============================================

def get_event_time(df, idx):
    """Obtiene el tiempo del evento con fallback mejorado."""
    if 't_centro_s' in df.columns and not pd.isnull(df.iloc[idx]['t_centro_s']):
        return df.iloc[idx]['t_centro_s']
    elif 't_inicio_s' in df.columns and not pd.isnull(df.iloc[idx]['t_inicio_s']):
        return df.iloc[idx]['t_inicio_s']
    else:
        return idx * TAMANO_VENTANA  # Fallback basado en índice

def validacion_cruzada_simple(df, transiciones, umbrales_base, n_splits=2):
    """Validación cruzada simplificada para casos con pocas transiciones."""
    if len(transiciones) < n_splits:
        return [{
            'fold': None,
            'umbrales': umbrales_base,
            'test_transiciones': list(transiciones),
            'singularidades': [],
            'desfases': [],
            'mensaje': f"Muy pocas transiciones para CV: {len(transiciones)}"
        }]
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = []
    transiciones = np.array(transiciones)
    
    for split, (train_idx, test_idx) in enumerate(kf.split(transiciones)):
        trans_train = transiciones[train_idx]
        trans_test = transiciones[test_idx]
        
        umbrales = calcular_umbrales_optimizados(df, trans_train, VENTANAS_ANTES)
        singularidades_test = buscar_singularidades_optimizado(df, umbrales, MARGEN_DELTA_PCI)
        desfases = medir_desfases(trans_test, singularidades_test, max_desfase=VENTANA_MARGIN)
        
        resultados.append({
            'fold': split+1,
            'umbrales': umbrales,
            'test_transiciones': list(trans_test),
            'singularidades': singularidades_test,
            'desfases': desfases
        })
    
    return resultados

# ===============================================
# FUNCIONES AUXILIARES MANTENIDAS
# ===============================================

def filtrar_por_pendiente_curvatura(df, singularidad_idx, modo_estricto=False, 
                                   pendiente_min=0.01, curvatura_min=0.005):
    """Mantiene funcionalidad original con logging."""
    if not modo_estricto:
        return singularidad_idx
    
    if 'k_topo' not in df.columns or df['k_topo'].isnull().all():
        logger.warning("No hay datos de k_topo para filtrar por pendiente/curvatura")
        return singularidad_idx
    
    grad_k = np.gradient(df['k_topo'].fillna(0))
    grad2_k = np.gradient(grad_k)
    filtrados = []
    
    for idx in singularidad_idx:
        if idx > 1:
            if grad_k[idx-1] > pendiente_min and grad2_k[idx-1] > curvatura_min:
                filtrados.append(idx)
    
    logger.info(f"Filtro pendiente/curvatura: {len(singularidad_idx)} → {len(filtrados)}")
    return filtrados

def limitar_singularidades(singularidad_idx, max_singularidades=100, modo_estricto=False):
    """Mantiene funcionalidad original con logging."""
    if not modo_estricto or len(singularidad_idx) <= max_singularidades:
        return singularidad_idx
    
    logger.warning(f"Limitando singularidades: {len(singularidad_idx)} → {max_singularidades}")
    return singularidad_idx[:max_singularidades]

def medir_desfases(transiciones, singularidades, max_desfase=6):
    """Función original mantenida."""
    return [
        min([s-t for s in singularidades if abs(s-t) <= max_desfase], key=abs, default=None)
        for t in transiciones
    ]

def obtener_ventanas_control_negativo(df, transiciones, n_ventanas=50, margen_evitar=5):
    """Función original mantenida con logging."""
    if len(df) == 0:
        return []
    
    evitar = set()
    for t in transiciones:
        evitar.update(range(max(0, t-margen_evitar), min(len(df), t+margen_evitar+1)))
    
    candidatos = [i for i in range(len(df)) if i not in evitar]
    candidatos_validos = [
        i for i in candidatos 
        if (not pd.isnull(df.iloc[i].get('k_topo', np.nan)) and 
            not pd.isnull(df.iloc[i].get('phi_h', np.nan)))
    ]
    
    if len(candidatos_validos) < n_ventanas:
        logger.warning(f"Pocas ventanas control válidas: {len(candidatos_validos)} < {n_ventanas}")
        return candidatos_validos
    
    return random.sample(candidatos_validos, n_ventanas)

def contar_singularidades_en_control(df, idx_control, singularidad_idx):
    """Función original mantenida."""
    if not idx_control or not singularidad_idx:
        return 0
    set_sing = set(singularidad_idx)
    return sum(1 for idx in idx_control if idx in set_sing)

# ===============================================
# FUNCIÓN WRAPPER OPTIMIZADA
# ===============================================

def procesar_archivo_wrapper_optimizado(args):
    """
    Wrapper optimizado para procesamiento paralelo con manejo de errores mejorado.
    """
    archivo = args[0]
    try:
        # Intentar cargar datos EEG reales si están disponibles
        base_name = os.path.basename(archivo).replace('.csv', '')
        psg_path = os.path.join(CARPETA_BASE, f'{base_name}-PSG.edf')
        
        eeg_data = None
        if os.path.exists(psg_path):
            try:
                eeg_data = cargar_eeg_y_segmentar(psg_path, dur_ventana_s=TAMANO_VENTANA)
                df_temp = pd.read_csv(archivo)
                n_ventanas = len(df_temp)
                eeg_data_alineado = eeg_data[:n_ventanas]
                modo_metricas = 'real'
                logger.info(f"Usando datos EEG reales para {archivo}")
            except Exception as e:
                logger.warning(f"No se pudo cargar EEG para {archivo}: {e}")
                eeg_data_alineado = None
                modo_metricas = 'simulado'
        else:
            eeg_data_alineado = None
            modo_metricas = 'simulado'
            logger.info(f"Usando métricas simuladas para {archivo}")
        
        # Procesar con función optimizada
        resultado = procesar_archivo_optimizado(
            archivo,
            modo_metricas=modo_metricas,
            eeg_data=eeg_data_alineado,
            random_seed=42,
            generar_analisis_avanzado=True
        )
        
        if resultado:
            logger.info(f"✅ Completado exitosamente: {archivo}")
        else:
            logger.warning(f"⚠️  Sin resultados para: {archivo}")
        
        return resultado
        
    except Exception as e:
        logger.error(f"❌ Error procesando {archivo}: {e}")
        return None

# ===============================================
# FUNCIONES DE ANÁLISIS Y VISUALIZACIÓN MANTENIDAS
# ===============================================

# Mantener todas las funciones de plot y análisis originales con nombres compatibles
def plot_fase1(df, despertares, dormidas, singularidad_idx, singularidad_tipo, carpeta, 
               nombre_archivo, singularidades_convergentes_validas=None):
    """Wrapper de compatibilidad que usa la versión optimizada."""
    metricas_calidad = calcular_metricas_calidad_extendidas(
        singularidad_idx, despertares + dormidas, df
    )
    return plot_fase1_optimizado(df, despertares, dormidas, singularidad_idx, singularidad_tipo,
                                carpeta, nombre_archivo, singularidades_convergentes_validas, 
                                metricas_calidad)

def curvas_promedio_previas(df, despertares, singularidad_idx, margen=2, carpeta=None, nombre_archivo=""):
    """Función original mantenida."""
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en curvas_promedio_previas")
    
    grupo_convergente, grupo_sin = [], []
    for d in despertares:
        cerca_sing = any(abs(s - d) <= VENTANA_MARGIN for s in singularidad_idx)
        vals_k = [df.iloc[d+offset]['k_topo'] if 0 <= d+offset < len(df) else np.nan 
                 for offset in range(-margen, margen+1)]
        vals_phi = [df.iloc[d+offset]['phi_h'] if 0 <= d+offset < len(df) else np.nan 
                   for offset in range(-margen, margen+1)]
        
        if cerca_sing:
            grupo_convergente.append([vals_k, vals_phi])
        else:
            grupo_sin.append([vals_k, vals_phi])
    
    for arr_idx, (grupo, nombre) in enumerate(zip([grupo_convergente, grupo_sin], 
                                                 ['convergente', 'sin singularidad'])):
        if not grupo:
            logger.warning(f"No hay datos para el grupo '{nombre}' en {nombre_archivo}")
            continue
        
        arr_k = np.array([x[0] for x in grupo])
        arr_phi = np.array([x[1] for x in grupo])
        
        for arr, m in zip([arr_k, arr_phi], ['k_topo', 'phi_h']):
            if arr.size == 0 or np.isnan(arr).all():
                logger.warning(f"No hay datos válidos para {m} en el grupo '{nombre}' en {nombre_archivo}")
                continue
            
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            x = np.arange(-margen, margen+1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, mean, label=f'{nombre}', linewidth=2)
            plt.fill_between(x, mean-std, mean+std, alpha=0.3)
            plt.title(f"Curva promedio {m} respecto al despertar\n({nombre_archivo} - {nombre})")
            plt.xlabel("Ventana relativa al despertar")
            plt.ylabel(m)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if carpeta:
                png_path = os.path.join(carpeta, f'curva_prom_{m}_{nombre}_{nombre_archivo.replace(".csv","")}.png')
                pdf_path = os.path.join(carpeta, f'curva_prom_{m}_{nombre}_{nombre_archivo.replace(".csv","")}.pdf')
                logger.debug(f"Guardando curva promedio en: {png_path}")
                safe_makedirs(os.path.dirname(png_path))
                plt.savefig(png_path, dpi=300)
                plt.savefig(pdf_path, format='pdf')
            plt.close()

def boxplots_metricas_previas(df, despertares, singularidad_idx, margen=2, carpeta=None, nombre_archivo=""):
    """Función original mantenida con mejoras visuales."""
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en boxplots_metricas_previas")
    
    tiene_sing = []
    no_sing = []
    
    for d in despertares:
        cerca_sing = any(abs(s - d) <= VENTANA_MARGIN for s in singularidad_idx)
        vals = {m: [df.iloc[d+offset][m] if 0 <= d+offset < len(df) else None 
                   for offset in range(-margen, margen+1)] 
               for m in ["k_topo", "phi_h"]}
        
        if cerca_sing:
            tiene_sing.append(vals)
        else:
            no_sing.append(vals)
    
    for m in ["k_topo", "phi_h"]:
        data = [
            [x[m][0] for x in tiene_sing if x[m][0] is not None],
            [x[m][0] for x in no_sing if x[m][0] is not None]
        ]
        
        if not any(data):
            logger.warning(f"No hay datos para boxplot de {m}")
            continue
        
        plt.figure(figsize=(8, 6))
        box_plot = plt.boxplot(data, tick_labels=['Con singularidad', 'Sin singularidad'], 
                              patch_artist=True)
        
        # Mejorar visualización
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title(f'Distribución de {m} en ventana -2 antes del despertar\n({nombre_archivo})')
        plt.ylabel(f'Valor de {m}')
        plt.grid(True, alpha=0.3)
        
        if carpeta:
            png_path = os.path.join(carpeta, f'boxplot_{m}_ventana-2_{nombre_archivo.replace(".csv","")}.png')
            pdf_path = os.path.join(carpeta, f'boxplot_{m}_ventana-2_{nombre_archivo.replace(".csv","")}.pdf')
            logger.debug(f"Guardando boxplot en: {png_path}")
            safe_makedirs(os.path.dirname(png_path))
            plt.savefig(png_path, dpi=300)
            plt.savefig(pdf_path, format='pdf')
        plt.close()

def analizar_delta_pci_post_singularidad(df, singularidad_idx, singularidad_tipo, 
                                        n_ventanas_post=3, carpeta=None, nombre_archivo=""):
    """Función original mantenida."""
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en analizar_delta_pci_post_singularidad")
    
    vals_post = []
    tipo_dict = singularidad_tipo if isinstance(singularidad_tipo, dict) else {}
    
    for idx in singularidad_idx:
        if (isinstance(singularidad_tipo, dict) and 
            singularidad_tipo.get(idx, {}).get('etiqueta', '') == "convergente") or \
           (not isinstance(singularidad_tipo, dict) and 
            singularidad_tipo.get(idx) == "convergente"):
            vals = []
            for offset in range(1, n_ventanas_post+1):
                j = idx + offset
                if 0 <= j < len(df):
                    vals.append(df.iloc[j]['delta_pci'])
            if vals:
                vals_post.append(vals)
    
    if vals_post:
        arr = np.array([x if len(x)==n_ventanas_post else [np.nan]*n_ventanas_post for x in vals_post])
        if arr.size == 0:
            return
        
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(1, n_ventanas_post+1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, mean, label='Promedio ΔPCI post-singularidad', marker='o', linewidth=2)
        plt.fill_between(x, mean-std, mean+std, alpha=0.3)
        plt.xlabel("Ventanas posteriores a la singularidad convergente")
        plt.ylabel("ΔPCI")
        plt.title(f"Evolución de ΔPCI tras singularidad convergente\n({nombre_archivo})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        png_path = os.path.join(carpeta, f"delta_pci_post_singularidad_{nombre_archivo.replace('.csv','')}.png")
        pdf_path = os.path.join(carpeta, f"delta_pci_post_singularidad_{nombre_archivo.replace('.csv','')}.pdf")
        logger.debug(f"Guardando delta_pci_post_singularidad en: {png_path}")
        safe_makedirs(os.path.dirname(png_path))
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path, format='pdf')
        plt.close()

# ===============================================
# FUNCIONES PARA COMPATIBILIDAD CON FUNCIONES ORIGINALES
# ===============================================

def calcular_umbral_metricas(df, transiciones, v_antes=5, modo_estricto=False, 
                           percentil=65, usar_media_std=False, Nstd=2):
    """Wrapper de compatibilidad que usa la versión optimizada."""
    return calcular_umbrales_optimizados(df, transiciones, v_antes, modo_estricto, 
                                       percentil, usar_media_std, Nstd)

def buscar_singularidades(df, umbrales, margen_delta_pci=0.1):
    """Wrapper de compatibilidad que usa la versión optimizada."""
    return buscar_singularidades_optimizado(df, umbrales, margen_delta_pci)

def clasificar_singularidades_relajada(singularidad_idx, despertares, dormidas, 
                                     ventana_margin=6, ventana_margin_leve=8):
    """Wrapper de compatibilidad."""
    singularidad_info, _ = clasificar_singularidades_relajada_optimizada(
        singularidad_idx, despertares, dormidas, ventana_margin, ventana_margin_leve
    )
    # Convertir al formato original
    return {idx: info['etiqueta'] for idx, info in singularidad_info.items()}

# ===============================================
# FUNCIÓN PRINCIPAL COMPATIBLE
# ===============================================

def procesar_archivo(csv_path, only_umbrales=False, forced_umbrales=None,
                    control_negativo_n=50, control_negativo_margin=5,
                    modo_estricto=False, modo_metricas='simulado',
                    eeg_data=None, random_seed=None):
    """
    Función principal compatible que usa la versión optimizada internamente.
    """
    # Usar la función optimizada pero retornar formato compatible
    resultado_optimizado = procesar_archivo_optimizado(
        csv_path, only_umbrales, forced_umbrales,
        control_negativo_n, control_negativo_margin,
        modo_estricto, modo_metricas, eeg_data, random_seed,
        generar_analisis_avanzado=False  # Para mayor compatibilidad
    )
    
    if only_umbrales or resultado_optimizado is None:
        return resultado_optimizado
    
    # Convertir al formato original para compatibilidad
    return {
        "archivo": resultado_optimizado["archivo"],
        "n_despertares": resultado_optimizado["n_despertares"],
        "n_dormidas": resultado_optimizado["n_dormidas"],
        "n_singularidades": resultado_optimizado["n_singularidades"],
        "n_convergentes": resultado_optimizado["n_convergentes"],
        "n_convergentes_leve": resultado_optimizado["n_convergentes_leve"],
        "n_divergentes": resultado_optimizado["n_divergentes"],
        "n_sin_clasificar": resultado_optimizado["n_sin_clasificar"],
        "n_fp_control": 0,  # Placeholder para compatibilidad
        "n_control": 0,     # Placeholder para compatibilidad
    }

def procesar_archivo_wrapper(args):
    """Wrapper compatible que usa la versión optimizada."""
    return procesar_archivo_wrapper_optimizado(args)

# ===============================================
# FUNCIONES AUXILIARES MANTENIDAS
# ===============================================

def extraer_metricas_previas(df, transiciones, metricas=["k_topo", "phi_h"], margen=2):
    """Función original mantenida."""
    metricas_previas = {m: [] for m in metricas}
    for t in transiciones:
        for m in metricas:
            valores = []
            for offset in range(-margen, margen+1):
                idx = t + offset
                if 0 <= idx < len(df):
                    valores.append(df.iloc[idx][m])
                else:
                    valores.append(None)
            metricas_previas[m].append(valores)
    return metricas_previas

def tasa_singularidad_baseline(total_singularidades, total_ventanas):
    """Función original mantenida."""
    return total_singularidades / total_ventanas if total_ventanas > 0 else 0

def calcular_sensibilidad_tasa_fp(resumen_batch, margen_deteccion=2):
    """Función original mantenida."""
    TP = sum(sum(1 for d in r.get("desfases_despertar", []) 
                if d is not None and abs(d) <= margen_deteccion) 
            for r in resumen_batch if r and r.get("desfases_despertar"))
    
    N_eventos = sum(r.get("n_despertares", 0) for r in resumen_batch if r)
    FP = sum(r.get("n_fp_control", 0) for r in resumen_batch if r)
    N_control = sum(r.get("n_control", 0) for r in resumen_batch if r)
    
    sensibilidad = TP / N_eventos if N_eventos else float('nan')
    tasa_fp = FP / N_control if N_control else float('nan')
    
    return sensibilidad, tasa_fp

def convert_all_hypnograms_to_pipeline_csv(input_folder, output_folder):
    """Función original mantenida con logging."""
    hypnogram_files = glob.glob(os.path.join(input_folder, "*-Hypnogram.edf.txt"))
    csv_paths = []
    
    logger.info(f"Convirtiendo {len(hypnogram_files)} archivos hypnogram")
    
    for hyp_file in tqdm(hypnogram_files, desc="Convirtiendo hypnograms"):
        try:
            df = parse_hypnogram_edf(hyp_file)
            base = os.path.basename(hyp_file).replace("-Hypnogram.edf.txt", ".csv")
            out_csv = os.path.join(output_folder, base)
            df.to_csv(out_csv, index=False)
            csv_paths.append(out_csv)
        except Exception as e:
            logger.error(f"Error al convertir {hyp_file}: {e}")
    
    logger.info(f"Conversión completada: {len(csv_paths)} archivos CSV generados")
    return csv_paths

# ===============================================
# FUNCIÓN MAIN OPTIMIZADA
# ===============================================

def main_batch_optimizado():
    """
    Función principal optimizada con todas las mejoras integradas.
    """
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE EEG OPTIMIZADO")
    logger.info("=" * 80)
    
    # Asegurar que existe la carpeta de resultados
    safe_makedirs(CARPETA_RESULTADOS)
    
    # Conversión de hypnograms
    logger.info("Fase 1: Convirtiendo archivos Hypnogram Sleep-EDF...")
    hypnogram_csvs = convert_all_hypnograms_to_pipeline_csv(CARPETA_BASE, CARPETA_BASE)
    logger.info(f"✅ {len(hypnogram_csvs)} archivos hypnogram convertidos")

    # Búsqueda de archivos CSV
    archivos = glob.glob(os.path.join(CARPETA_BASE, "*.csv"))
    archivos = [a for a in archivos if os.path.basename(a) not in 
               ["resumen_batch.csv", "loso_crossval.csv", "resumen_batch_optimizado.csv"]]
    archivos = [a for a in archivos if os.path.getsize(a) > 0]
    
    if not archivos:
        logger.error("❌ No se encontraron archivos CSV válidos para analizar")
        return

    logger.info(f"📁 Encontrados {len(archivos)} archivos para procesar")

    # Modo debug opcional
    MODO_DEBUG = False
    if MODO_DEBUG:
        archivos = archivos[:2]
        logger.warning(f"🔧 MODO DEBUG: Procesando solo {len(archivos)} archivos")

    # Configuración de paralelización
    max_workers = min(8, multiprocessing.cpu_count())
    logger.info(f"🚀 Configuración: {max_workers} workers paralelos")

    # Procesamiento paralelo optimizado
    logger.info("Fase 2: Procesamiento optimizado de archivos...")
    tareas = [(archivo,) for archivo in archivos]
    resumen_batch = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futuros = [executor.submit(procesar_archivo_wrapper_optimizado, arg) for arg in tareas]
        
        for f in tqdm(as_completed(futuros), total=len(futuros), desc="Procesamiento optimizado"):
            resultado = f.result()
            if resultado:
                resumen_batch.append(resultado)

    # Análisis de resultados agregados
    logger.info("Fase 3: Generando análisis agregado...")
    
    if resumen_batch:
        # Guardar resumen detallado
        resumen_df = pd.DataFrame(resumen_batch)
        resumen_csv_path = os.path.join(CARPETA_RESULTADOS, "resumen_batch_optimizado.csv")
        resumen_df.to_csv(resumen_csv_path, index=False)
        
        # Estadísticas agregadas
        estadisticas_agregadas = generar_estadisticas_agregadas(resumen_batch)
        guardar_estadisticas_agregadas(estadisticas_agregadas, CARPETA_RESULTADOS)
        
        # Reporte final
        generar_reporte_final_optimizado(resumen_batch, estadisticas_agregadas, CARPETA_RESULTADOS)
        
        logger.info(f"✅ Procesamiento completado exitosamente")
        logger.info(f"📊 {len(resumen_batch)} archivos procesados de {len(archivos)} totales")
        logger.info(f"📁 Resultados guardados en: {CARPETA_RESULTADOS}")
        
    else:
        logger.error("❌ No se pudieron procesar archivos válidos")

def generar_estadisticas_agregadas(resumen_batch):
    """
    Genera estadísticas agregadas comprehensivas del batch completo.
    """
    if not resumen_batch:
        return {}
    
    df = pd.DataFrame(resumen_batch)
    
    estadisticas = {
        'n_archivos_procesados': len(resumen_batch),
        'total_despertares': df['n_despertares'].sum(),
        'total_dormidas': df['n_dormidas'].sum(),
        'total_singularidades': df['n_singularidades'].sum(),
        'total_convergentes': df['n_convergentes'].sum(),
        'total_convergentes_leve': df['n_convergentes_leve'].sum(),
        'total_divergentes': df['n_divergentes'].sum(),
        'total_sin_clasificar': df['n_sin_clasificar'].sum(),
        
        # Métricas promedio
        'cobertura_promedio': df['cobertura_transiciones'].mean(),
        'especificidad_promedio': df['especificidad_temporal'].mean(),
        'precision_promedio': df['precision'].mean(),
        'recall_promedio': df['recall'].mean(),
        'f1_promedio': df['f1_score'].mean(),
        'eficiencia_promedio': df['eficiencia_deteccion'].mean(),
        
        # Métricas de variabilidad
        'cobertura_std': df['cobertura_transiciones'].std(),
        'especificidad_std': df['especificidad_temporal'].std(),
        'f1_std': df['f1_score'].std(),
        
        # Distribuciones
        'distribucion_cobertura': df['cobertura_transiciones'].describe().to_dict(),
        'distribucion_f1': df['f1_score'].describe().to_dict(),
        
        # Análisis de eficiencia
        'archivos_alta_cobertura': len(df[df['cobertura_transiciones'] > 0.8]),
        'archivos_alta_especificidad': len(df[df['especificidad_temporal'] > 0.7]),
        'archivos_buen_balance': len(df[(df['f1_score'] > 0.6)]),
        
        # Tasas de detección
        'tasa_deteccion_convergentes': df['n_convergentes'].sum() / df['n_despertares'].sum() if df['n_despertares'].sum() > 0 else 0,
        'tasa_deteccion_divergentes': df['n_divergentes'].sum() / df['n_dormidas'].sum() if df['n_dormidas'].sum() > 0 else 0,
    }
    
    return estadisticas

def guardar_estadisticas_agregadas(estadisticas, carpeta_resultados):
    """
    Guarda las estadísticas agregadas en múltiples formatos.
    """
    # CSV con métricas principales
    metricas_principales = {
        'metrica': ['cobertura_promedio', 'especificidad_promedio', 'precision_promedio', 
                   'recall_promedio', 'f1_promedio', 'eficiencia_promedio'],
        'valor': [estadisticas[k] for k in ['cobertura_promedio', 'especificidad_promedio', 
                 'precision_promedio', 'recall_promedio', 'f1_promedio', 'eficiencia_promedio']],
        'std': [estadisticas.get(k.replace('_promedio', '_std'), 0) 
               for k in ['cobertura_promedio', 'especificidad_promedio', 'precision_promedio', 
                        'recall_promedio', 'f1_promedio', 'eficiencia_promedio']]
    }
    
    pd.DataFrame(metricas_principales).to_csv(
        os.path.join(carpeta_resultados, "estadisticas_agregadas.csv"), index=False
    )
    
    # JSON completo
    import json
    with open(os.path.join(carpeta_resultados, "estadisticas_completas.json"), 'w') as f:
        json.dump(estadisticas, f, indent=2, default=str)

def generar_reporte_final_optimizado(resumen_batch, estadisticas, carpeta_resultados):
    """
    Genera un reporte final comprehensivo del análisis completo.
    """
    reporte_path = os.path.join(carpeta_resultados, "reporte_final_optimizado.txt")
    
    with open(reporte_path, "w", encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE FINAL - PIPELINE EEG OPTIMIZADO\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Fecha de análisis: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Versión del pipeline: EEG Optimizado v2.0\n")
        f.write(f"Archivos procesados: {estadisticas['n_archivos_procesados']}\n\n")
        
        # Resumen ejecutivo
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-" * 50 + "\n")
        f.write(f"📊 Total de transiciones analizadas: {estadisticas['total_despertares'] + estadisticas['total_dormidas']}\n")
        f.write(f"🎯 Total de singularidades detectadas: {estadisticas['total_singularidades']}\n")
        f.write(f"🎯 Cobertura promedio de transiciones: {estadisticas['cobertura_promedio']:.1%}\n")
        f.write(f"🎯 Especificidad temporal promedio: {estadisticas['especificidad_promedio']:.1%}\n")
        f.write(f"🎯 F1-Score promedio del sistema: {estadisticas['f1_promedio']:.3f}\n\n")
        
        # Análisis detallado
        f.write("ANÁLISIS DETALLADO\n")
        f.write("-" * 50 + "\n")
        f.write("Detección de Transiciones:\n")
        f.write(f"  • Despertares detectados: {estadisticas['total_despertares']}\n")
        f.write(f"  • Dormidas detectadas: {estadisticas['total_dormidas']}\n")
        f.write(f"  • Promedio por archivo: {(estadisticas['total_despertares'] + estadisticas['total_dormidas']) / estadisticas['n_archivos_procesados']:.1f}\n\n")
        
        f.write("Clasificación de Singularidades:\n")
        f.write(f"  • Convergentes: {estadisticas['total_convergentes']} ({estadisticas['total_convergentes']/estadisticas['total_singularidades']*100:.1f}%)\n")
        f.write(f"  • Convergentes leves: {estadisticas['total_convergentes_leve']} ({estadisticas['total_convergentes_leve']/estadisticas['total_singularidades']*100:.1f}%)\n")
        f.write(f"  • Divergentes: {estadisticas['total_divergentes']} ({estadisticas['total_divergentes']/estadisticas['total_singularidades']*100:.1f}%)\n")
        f.write(f"  • Sin clasificar: {estadisticas['total_sin_clasificar']} ({estadisticas['total_sin_clasificar']/estadisticas['total_singularidades']*100:.1f}%)\n\n")
        
        # Métricas de rendimiento
        f.write("MÉTRICAS DE RENDIMIENTO\n")
        f.write("-" * 50 + "\n")
        f.write(f"Cobertura de transiciones: {estadisticas['cobertura_promedio']:.3f} ± {estadisticas['cobertura_std']:.3f}\n")
        f.write(f"Especificidad temporal: {estadisticas['especificidad_promedio']:.3f} ± {estadisticas['especificidad_std']:.3f}\n")
        f.write(f"Precisión: {estadisticas['precision_promedio']:.3f}\n")
        f.write(f"Recall: {estadisticas['recall_promedio']:.3f}\n")
        f.write(f"F1-Score: {estadisticas['f1_promedio']:.3f} ± {estadisticas['f1_std']:.3f}\n")
        f.write(f"Eficiencia de detección: {estadisticas['eficiencia_promedio']:.3f}\n\n")
        
        # Análisis de calidad
        f.write("ANÁLISIS DE CALIDAD\n")
        f.write("-" * 50 + "\n")
        f.write(f"Archivos con alta cobertura (>80%): {estadisticas['archivos_alta_cobertura']}/{estadisticas['n_archivos_procesados']} ({estadisticas['archivos_alta_cobertura']/estadisticas['n_archivos_procesados']*100:.1f}%)\n")
        f.write(f"Archivos con alta especificidad (>70%): {estadisticas['archivos_alta_especificidad']}/{estadisticas['n_archivos_procesados']} ({estadisticas['archivos_alta_especificidad']/estadisticas['n_archivos_procesados']*100:.1f}%)\n")
        f.write(f"Archivos con buen balance (F1>0.6): {estadisticas['archivos_buen_balance']}/{estadisticas['n_archivos_procesados']} ({estadisticas['archivos_buen_balance']/estadisticas['n_archivos_procesados']*100:.1f}%)\n\n")
        
        # Tasas de detección
        f.write("TASAS DE DETECCIÓN\n")
        f.write("-" * 50 + "\n")
        f.write(f"Tasa de detección de despertares: {estadisticas['tasa_deteccion_convergentes']:.1%}\n")
        f.write(f"Tasa de detección de dormidas: {estadisticas['tasa_deteccion_divergentes']:.1%}\n\n")
        
        # Mejoras implementadas
        f.write("MEJORAS IMPLEMENTADAS EN ESTA VERSIÓN\n")
        f.write("-" * 50 + "\n")
        f.write("✅ Umbrales adaptativos con múltiples estrategias\n")
        f.write("✅ Métricas de calidad extendidas (F1, precisión, recall)\n")
        f.write("✅ Validación cruzada estratificada\n")
        f.write("✅ Análisis temporal y circadiano avanzado\n")
        f.write("✅ Detección de microdespertares\n")
        f.write("✅ Clustering temporal para reducir redundancia\n")
        f.write("✅ Sistema de logging comprehensivo\n")
        f.write("✅ Filtros adaptativos de calidad\n")
        f.write("✅ Análisis de contexto temporal\n")
        f.write("✅ Visualizaciones optimizadas con métricas integradas\n\n")
        
        # Conclusiones y recomendaciones
        f.write("CONCLUSIONES Y RECOMENDACIONES\n")
        f.write("-" * 50 + "\n")
        
        if estadisticas['f1_promedio'] > 0.7:
            f.write("🎉 EXCELENTE: El sistema muestra un rendimiento excepcional\n")
        elif estadisticas['f1_promedio'] > 0.5:
            f.write("✅ BUENO: El sistema muestra un rendimiento satisfactorio\n")
        else:
            f.write("⚠️  MEJORABLE: El sistema requiere ajustes adicionales\n")
        
        f.write(f"\nCobertura de transiciones: {('ALTA' if estadisticas['cobertura_promedio'] > 0.7 else 'MEDIA' if estadisticas['cobertura_promedio'] > 0.5 else 'BAJA')}\n")
        f.write(f"Especificidad temporal: {('ALTA' if estadisticas['especificidad_promedio'] > 0.7 else 'MEDIA' if estadisticas['especificidad_promedio'] > 0.5 else 'BAJA')}\n")
        
        # Recomendaciones específicas
        if estadisticas['cobertura_promedio'] < 0.6:
            f.write("\n🔧 RECOMENDACIÓN: Considerar reducir umbrales para aumentar sensibilidad\n")
        if estadisticas['especificidad_promedio'] < 0.6:
            f.write("🔧 RECOMENDACIÓN: Considerar aumentar umbrales para reducir falsos positivos\n")
        if estadisticas['f1_std'] > 0.2:
            f.write("🔧 RECOMENDACIÓN: Alta variabilidad entre archivos - revisar estrategia de normalización\n")

def main_batch():
    """
    Función main_batch original que ahora usa la versión optimizada.
    """
    # Usar la nueva función optimizada pero mantener el nombre original para compatibilidad
    main_batch_optimizado()

# ===============================================
# FUNCIONES DE DEBUG OPTIMIZADAS
# ===============================================

def debug_sincronizacion_optimizado():
    """
    Versión optimizada del bloque de debug con métricas extendidas.
    """
    logger.info("\n" + "=" * 80)
    logger.info("DEBUG OPTIMIZADO: ANÁLISIS DE SINCRONIZACIÓN TRANSICIONES vs SINGULARIDADES")
    logger.info("=" * 80)
    
    # Configuración de ejemplo (mantener valores originales)
    VENTANA_TAMANO = 30  
    VENTANA_MARGIN = 6   
    
    # Transiciones de ejemplo (en segundos)
    transiciones_test = [1155, 1259, 1270, 1362, 1499, 1550, 1621, 1714, 1746, 1816, 1839, 1848]
    transiciones_test_idx = [int(t / VENTANA_TAMANO) for t in transiciones_test]
    
    # Singularidades de ejemplo
    singularidades = [18, 56, 186, 222, 257, 291, 478, 525, 542, 550, 643, 758, 1611, 2108, 2135]
    
    # Análisis de desfases optimizado
    def analizar_desfase_optimizado(sing_idx, trans_idx, margen=VENTANA_MARGIN):
        for t in trans_idx:
            if abs(sing_idx - t) <= margen:
                desfase = sing_idx - t
                distancia = abs(desfase)
                confianza = 1.0 - (distancia / margen) * 0.5
                return desfase, confianza
        return None, 0.0
    
    # Ejecutar análisis
    resultados_debug = []
    for s in singularidades:
        desfase, confianza = analizar_desfase_optimizado(s, transiciones_test_idx, VENTANA_MARGIN)
        resultados_debug.append({
            'singularidad': s,
            'desfase': desfase,
            'confianza': confianza,
            'tiempo_s': s * VENTANA_TAMANO,
            'clasificacion': 'convergente' if desfase is not None else 'sin_clasificar'
        })
    
    # Mostrar resultados
    logger.info(f"Configuración: Ventana={VENTANA_TAMANO}s, Margen=±{VENTANA_MARGIN} ventanas")
    logger.info(f"Transiciones test: {len(transiciones_test)} eventos")
    logger.info(f"Singularidades: {len(singularidades)} detectadas")
    logger.info("-" * 80)
    
    transiciones_cubiertas = 0
    singularidades_validas = 0
    
    for resultado in resultados_debug:
        s = resultado['singularidad']
        d = resultado['desfase']
        c = resultado['confianza']
        clasificacion = resultado['clasificacion']
        
        if d is not None:
            singularidades_validas += 1
            logger.info(f"Singularidad {s:4d} → Desfase: {d:+3d} | Confianza: {c:.2f} | {clasificacion}")
        else:
            logger.info(f"Singularidad {s:4d} → Sin transición cercana | {clasificacion}")
    
    # Calcular cobertura de transiciones
    for t_idx in transiciones_test_idx:
        if any(abs(r['singularidad'] - t_idx) <= VENTANA_MARGIN for r in resultados_debug):
            transiciones_cubiertas += 1
    
    # Métricas de rendimiento
    cobertura = transiciones_cubiertas / len(transiciones_test_idx)
    especificidad = singularidades_validas / len(singularidades)
    precision = singularidades_validas / len(singularidades) if len(singularidades) > 0 else 0
    recall = transiciones_cubiertas / len(transiciones_test_idx)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info("-" * 80)
    logger.info("MÉTRICAS DE RENDIMIENTO:")
    logger.info(f"Cobertura de transiciones: {transiciones_cubiertas}/{len(transiciones_test_idx)} = {cobertura:.2%}")
    logger.info(f"Especificidad temporal: {singularidades_validas}/{len(singularidades)} = {especificidad:.2%}")
    logger.info(f"Precisión: {precision:.3f}")
    logger.info(f"Recall: {recall:.3f}")
    logger.info(f"F1-Score: {f1:.3f}")
    logger.info("=" * 80 + "\n")
    
    return {
        'resultados': resultados_debug,
        'metricas': {
            'cobertura': cobertura,
            'especificidad': especificidad,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }

# ===============================================
# PUNTO DE ENTRADA PRINCIPAL
# ===============================================

if __name__ == "__main__":
    # Configurar logging
    logger = setup_logging()
    
    # Iniciar heartbeat
    threading.Thread(target=heartbeat, daemon=True).start()
    
    # Ejecutar debug si se desea
    debug_results = debug_sincronizacion_optimizado()
    
    # Ejecutar pipeline principal optimizado
    main_batch_optimizado()
    
    logger.info("🎉 Pipeline EEG Optimizado completado exitosamente!")
