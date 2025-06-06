import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*legend with loc=\"best\".*")

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import KFold
import multiprocessing
import random

# Mapa de estados de sueño
SLEEP_STAGE_MAP = {
    "W": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "R": 5,
    "?": -1,
}

CARPETA_BASE = r"G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\DATABASE\sleep-cassette"
CARPETA_RESULTADOS = os.path.join(CARPETA_BASE, "resultados_pah")
VENTANAS_ANTES = 5
MARGEN_DELTA_PCI = 0.1
VENTANA_MARGIN = 2
CAMPOS_CSV = ["ventana", "t_inicio_s", "k_topo", "phi_h", "delta_pci", "estado", "t_centro_s"]

def safe_makedirs(path):
    if not path:
        raise ValueError("Intento de crear un directorio vacío.")
    if not os.path.exists(path):
        print(f"[DEBUG] Creando directorio: {path}")
        os.makedirs(path, exist_ok=True)

def parse_hypnogram_edf(file_path):
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
        n_ventanas = int(dur) // 30
        for i in range(n_ventanas):
            t_inicio = int(ini) + i * 30
            t_centro = t_inicio + 15
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
    return df

def convert_all_hypnograms_to_pipeline_csv(input_folder, output_folder):
    print("Iniciando conversión de hypnogramas...")
    os.makedirs(output_folder, exist_ok=True)
    generated_files = []
    ignored_files = []
    for fname in os.listdir(input_folder):
        print(f"Revisando: {fname}")
        if "Hypnogram" in fname and (fname.endswith(".edf") or fname.endswith(".txt") or fname.endswith(".csv")):
            file_path = os.path.join(input_folder, fname)
            try:
                df = parse_hypnogram_edf(file_path)
            except Exception as e:
                print(f"[ERROR] No se pudo procesar {file_path}: {e}")
                ignored_files.append(file_path)
                continue
            if df.empty or df.shape[1] == 0:
                print(f"[AVISO] {file_path} está vacío y será ignorado.")
                ignored_files.append(file_path)
                continue
            base_name = fname.replace('.edf','').replace('.txt','').replace('.csv','')
            while base_name.startswith("WINDOWS_"):
                base_name = base_name[len("WINDOWS_"):]
            outname = f"WINDOWS_{base_name}.csv"
            outpath = os.path.join(output_folder, outname)
            df.to_csv(outpath, index=False)
            print(f"[GENERADO] {outpath}")
            generated_files.append(outpath)
    print(f"\nTotal archivos generados: {len(generated_files)}")
    print(f"Total archivos ignorados por estar vacíos o con error: {len(ignored_files)}")
    if ignored_files:
        print("Archivos ignorados:")
        for f in ignored_files:
            print("   ", f)
    print("Terminé de recorrer la carpeta.")
    return generated_files

def calcular_metricas_por_ventana(df, modo='simulado', eeg_data=None, random_seed=None):
    np.random.seed(random_seed)
    n = len(df)
    if modo == 'simulado':
        df['k_topo'] = np.random.normal(5, 1, size=n)
        df['phi_h'] = np.random.normal(0.8, 0.2, size=n)
        df['delta_pci'] = np.random.normal(0, 0.08, size=n)
    elif modo == 'real':
        if eeg_data is None:
            raise ValueError("En modo 'real', eeg_data (señal cruda) debe ser provisto para calcular las métricas.")
        for i, row in df.iterrows():
            df.at[i, 'k_topo'] = np.random.normal(5, 1)
            df.at[i, 'phi_h'] = np.random.normal(0.8, 0.2)
            df.at[i, 'delta_pci'] = np.random.normal(0, 0.08)
    else:
        raise ValueError("El modo debe ser 'simulado' o 'real'.")
    for col in ['k_topo', 'phi_h', 'delta_pci']:
        if df[col].isnull().all():
            raise ValueError(f"Todas las filas de {col} son NaN luego de calcular las métricas.")
    return df

def plot_fase1(df, despertares, dormidas, singularidad_idx, singularidad_tipo, carpeta, nombre_archivo):
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en plot_fase1")
    if not (despertares or dormidas or singularidad_idx):
        return

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    t = df['t_centro_s'].values if 't_centro_s' in df.columns else df['t_inicio_s'].values

    if "k_topo" in df.columns and not df['k_topo'].isnull().all():
        ax.plot(t, df['k_topo'], label='k_topo', color='#0288D1')
    if "phi_h" in df.columns and not df['phi_h'].isnull().all():
        ax.plot(t, df['phi_h'], label='phi_h', color='#E64A19')
    if "delta_pci" in df.columns and not df['delta_pci'].isnull().all():
        ax.plot(t, df['delta_pci'], label='delta_pci', color='#43A047')

    max_eventos = 25
    despertares_plot = despertares[:max_eventos]
    dormidas_plot = dormidas[:max_eventos]
    singularidad_idx_plot = singularidad_idx[:max_eventos]

    for idx in despertares_plot:
        ax.axvline(get_event_time(df, idx), color='blue', linestyle='--', alpha=0.5, linewidth=1)
    for idx in dormidas_plot:
        ax.axvline(get_event_time(df, idx), color='red', linestyle='--', alpha=0.5, linewidth=1)
    for idx in singularidad_idx_plot:
        color = (
            'magenta' if singularidad_tipo[idx] == "convergente"
            else 'lime' if singularidad_tipo[idx] == "divergente"
            else 'black'
        )
        ax.axvline(get_event_time(df, idx), color=color, linewidth=2, alpha=0.8)

    ax.set_title(f"Evolución de métricas - {nombre_archivo}", fontsize=16)
    ax.legend()
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Valor de la métrica")
    ax.grid(True, color='gray', alpha=0.2)

    plt.tight_layout()
    png_path = os.path.join(carpeta, f"grafico_fase1_{nombre_archivo}.png")
    pdf_path = os.path.join(carpeta, f"grafico_fase1_{nombre_archivo}.pdf")
    print(f"[DEBUG] Guardando gráfico en: {png_path}")

    safe_makedirs(os.path.dirname(png_path))
    plt.savefig(png_path, facecolor='white')
    plt.savefig(pdf_path, format='pdf', facecolor='white')
    plt.close()

def curvas_promedio_previas(df, despertares, singularidad_idx, margen=2, carpeta=None, nombre_archivo=""):
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en curvas_promedio_previas")
    grupo_convergente, grupo_sin = [], []
    for d in despertares:
        cerca_sing = any(abs(s - d) <= VENTANA_MARGIN for s in singularidad_idx)
        vals_k = [df.iloc[d+offset]['k_topo'] if 0 <= d+offset < len(df) else np.nan for offset in range(-margen, margen+1)]
        vals_phi = [df.iloc[d+offset]['phi_h'] if 0 <= d+offset < len(df) else np.nan for offset in range(-margen, margen+1)]
        if cerca_sing:
            grupo_convergente.append([vals_k, vals_phi])
        else:
            grupo_sin.append([vals_k, vals_phi])
    for arr_idx, (grupo, nombre) in enumerate(zip([grupo_convergente, grupo_sin], ['convergente', 'sin singularidad'])):
        if not grupo:
            print(f"[AVISO] No hay datos para el grupo '{nombre}' en {nombre_archivo}")
            continue
        arr_k = np.array([x[0] for x in grupo])
        arr_phi = np.array([x[1] for x in grupo])
        for arr, m in zip([arr_k, arr_phi], ['k_topo', 'phi_h']):
            if arr.size == 0 or np.isnan(arr).all():
                print(f"[AVISO] No hay datos válidos para {m} en el grupo '{nombre}' en {nombre_archivo}")
                continue
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            x = np.arange(-margen, margen+1)
            plt.plot(x, mean, label=f'{nombre}')
            plt.fill_between(x, mean-std, mean+std, alpha=0.2)
            plt.title(f"Curva promedio {m} respecto al despertar ({nombre_archivo} - {nombre})")
            plt.xlabel("Ventana relativa al despertar")
            plt.ylabel(m)
            plt.legend()
            if carpeta:
                png_path = os.path.join(carpeta, f'curva_prom_{m}_{nombre}_{nombre_archivo.replace(".csv","")}.png')
                pdf_path = os.path.join(carpeta, f'curva_prom_{m}_{nombre}_{nombre_archivo.replace(".csv","")}.pdf')
                print(f"[DEBUG] Guardando curva promedio en: {png_path}")
                safe_makedirs(os.path.dirname(png_path))
                plt.savefig(png_path)
                plt.savefig(pdf_path, format='pdf')
            plt.close()

def cargar_metricas_y_hipnograma(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        for col in CAMPOS_CSV:
            if col not in df.columns:
                df[col] = None
        if df.empty or df.shape[1] == 0:
            return None
        return df[CAMPOS_CSV]
    except Exception as e:
        print(f"Error leyendo {ruta_csv}: {e}")
        return None

def detectar_transiciones(df):
    estados = df['estado'].values
    despertares, dormidas = [], []
    for i in range(1, len(estados)):
        if estados[i-1] in [1,2,3,4] and estados[i] == 0:
            despertares.append(i)
        elif estados[i-1] == 0 and estados[i] in [1,2,3,4]:
            dormidas.append(i)
    return despertares, dormidas

def calcular_umbral_metricas(df, transiciones, v_antes, modo_estricto=False, percentil=75, usar_media_std=False, Nstd=2):
    if len(transiciones) == 0 or df['k_topo'].isnull().all():
        return {"k_topo_75": 0, "phi_h_75": 0, "delta_pci_med": 0}
    valores_k, valores_phi, valores_delta = [], [], []
    for t in transiciones:
        ini = max(0, t-v_antes)
        fin = t
        if fin > ini:
            valores_k.extend(df.iloc[ini:fin]["k_topo"].values)
            valores_phi.extend(df.iloc[ini:fin]["phi_h"].values)
            valores_delta.extend(df.iloc[ini:fin]["delta_pci"].values)
    if modo_estricto:
        percentil = 95
        usar_media_std = True
        Nstd = 2
    if usar_media_std:
        umbral_k = np.nanmean(valores_k) + Nstd * np.nanstd(valores_k)
        umbral_phi = np.nanmean(valores_phi) + Nstd * np.nanstd(valores_phi)
    else:
        umbral_k = np.nanpercentile(valores_k, percentil)
        umbral_phi = np.nanpercentile(valores_phi, percentil)
    umbrales = {
        "k_topo_75": umbral_k,
        "phi_h_75": umbral_phi,
        "delta_pci_med": np.nanmedian(valores_delta) if valores_delta else 0
    }
    return umbrales

def buscar_singularidades(df, umbrales, margen_delta):
    if df.shape[0] == 0:
        return []
    cumple_k = np.full(df.shape[0], True) if df['k_topo'].isnull().all() else (df['k_topo'] >= umbrales["k_topo_75"])
    cumple_phi = np.full(df.shape[0], True) if df['phi_h'].isnull().all() else (df['phi_h'] >= umbrales["phi_h_75"])
    cumple_delta = np.full(df.shape[0], True) if df['delta_pci'].isnull().all() else (np.abs(df['delta_pci']) <= margen_delta)
    return list(df.index[cumple_k & cumple_phi & cumple_delta])

def filtrar_por_pendiente_curvatura(df, singularidad_idx, modo_estricto=False, pendiente_min=0.01, curvatura_min=0.005):
    if not modo_estricto:
        return singularidad_idx
    grad_k = np.gradient(df['k_topo'].fillna(0))
    grad2_k = np.gradient(grad_k)
    filtrados = []
    for idx in singularidad_idx:
        if idx > 1:
            if grad_k[idx-1] > pendiente_min and grad2_k[idx-1] > curvatura_min:
                filtrados.append(idx)
    return filtrados

def limitar_singularidades(singularidad_idx, max_singularidades=100, modo_estricto=False):
    if not modo_estricto:
        return singularidad_idx
    if len(singularidad_idx) > max_singularidades:
        return singularidad_idx[:max_singularidades]
    return singularidad_idx

def tasa_singularidad_baseline(total_singularidades, total_ventanas):
    return total_singularidades / total_ventanas if total_ventanas > 0 else 0

def clasificar_singularidades_unico(singularidad_idx, despertares, dormidas, ventana_margin=2):
    trans_all = [(i, "convergente") for i in despertares] + [(i, "divergente") for i in dormidas]
    trans_all = sorted(trans_all, key=lambda x: x[0])
    singularidad_tipo = {}
    for idx in singularidad_idx:
        closest = min(trans_all, key=lambda x: abs(idx-x[0]), default=None)
        if closest and abs(closest[0]-idx) <= ventana_margin:
            singularidad_tipo[idx] = closest[1]
        else:
            singularidad_tipo[idx] = "sin_clasificar"
    return singularidad_tipo

def medir_desfase_y_clasificar(singularidad_idx, transiciones, tipo, margen=2):
    resultados = []
    for idx in singularidad_idx:
        desfase = None
        tipo_cercano = "sin_clasificar"
        if transiciones:
            desfases_posibles = [t - idx for t in transiciones if abs(t - idx) <= margen]
            if desfases_posibles:
                desfase = min(desfases_posibles, key=abs)
                tipo_cercano = tipo
        resultados.append((idx, desfase, tipo_cercano))
    return resultados

def medir_desfases(transiciones, singularidades, max_desfase=2):
    return [
        min([s-t for s in singularidades if abs(s-t) <= max_desfase], key=abs, default=None)
        for t in transiciones
    ]

def extraer_metricas_previas(df, transiciones, metricas=["k_topo", "phi_h"], margen=2):
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

def validacion_cruzada_umbrales(df, transiciones, v_antes, margen_delta, n_splits=2):
    if len(transiciones) < n_splits:
        return [{
            'fold': None,
            'umbrales': None,
            'test_transiciones': list(transiciones),
            'singularidades': [],
            'desfases': [],
            'mensaje': f"No se puede hacer KFold: n_splits={n_splits} > n_transiciones={len(transiciones)}"
        }]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    resultados = []
    transiciones = np.array(transiciones)
    for split, (train_idx, test_idx) in enumerate(kf.split(transiciones)):
        trans_train = transiciones[train_idx]
        trans_test = transiciones[test_idx]
        umbrales = calcular_umbral_metricas(df, trans_train, v_antes)
        singularidades_test = buscar_singularidades(df, umbrales, margen_delta)
        desfases = medir_desfases(trans_test, singularidades_test, max_desfase=VENTANA_MARGIN)
        resultados.append({
            'fold': split+1,
            'umbrales': umbrales,
            'test_transiciones': list(trans_test),
            'singularidades': singularidades_test,
            'desfases': desfases
        })
    return resultados

def boxplots_metricas_previas(df, despertares, singularidad_idx, margen=2, carpeta=None, nombre_archivo=""):
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en boxplots_metricas_previas")
    tiene_sing = []
    no_sing = []
    for d in despertares:
        cerca_sing = any(abs(s - d) <= VENTANA_MARGIN for s in singularidad_idx)
        vals = {m: [df.iloc[d+offset][m] if 0 <= d+offset < len(df) else None for offset in range(-margen, margen+1)] for m in ["k_topo", "phi_h"]}
        if cerca_sing:
            tiene_sing.append(vals)
        else:
            no_sing.append(vals)
    for m in ["k_topo", "phi_h"]:
        data = [
            [x[m][0] for x in tiene_sing if x[m][0] is not None],
            [x[m][0] for x in no_sing if x[m][0] is not None]
        ]
        plt.boxplot(data, tick_labels=['convergente', 'sin singularidad'])
        plt.title(f'Boxplot {m} ventana -2 antes del despertar\n({nombre_archivo})')
        plt.ylabel(m)
        if carpeta:
            png_path = os.path.join(carpeta, f'boxplot_{m}_ventana-2_{nombre_archivo.replace(".csv","")}.png')
            pdf_path = os.path.join(carpeta, f'boxplot_{m}_ventana-2_{nombre_archivo.replace(".csv","")}.pdf')
            print(f"[DEBUG] Guardando boxplot en: {png_path}")
            safe_makedirs(os.path.dirname(png_path))
            plt.savefig(png_path)
            plt.savefig(pdf_path, format='pdf')
        plt.close()

def analizar_delta_pci_post_singularidad(df, singularidad_idx, singularidad_tipo, n_ventanas_post=3, carpeta=None, nombre_archivo=""):
    if not carpeta:
        raise ValueError("El argumento 'carpeta' está vacío en analizar_delta_pci_post_singularidad")
    vals_post = []
    for idx in singularidad_idx:
        if singularidad_tipo[idx] == "convergente":
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
        plt.plot(x, mean, label='Promedio ΔPCI post-singularidad')
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)
        plt.xlabel("Ventanas posteriores a la singularidad convergente")
        plt.ylabel("ΔPCI")
        plt.title(f"ΔPCI tras singularidad convergente ({nombre_archivo})")
        plt.legend()
        png_path = os.path.join(carpeta, f"delta_pci_post_singularidad_{nombre_archivo.replace('.csv','')}.png")
        pdf_path = os.path.join(carpeta, f"delta_pci_post_singularidad_{nombre_archivo.replace('.csv','')}.pdf")
        print(f"[DEBUG] Guardando delta_pci_post_singularidad en: {png_path}")
        safe_makedirs(os.path.dirname(png_path))
        plt.savefig(png_path)
        plt.savefig(pdf_path, format='pdf')
        plt.close()

def get_event_time(df, idx):
    if 't_centro_s' in df.columns:
        return df.iloc[idx]['t_centro_s']
    else:
        return df.iloc[idx]['t_inicio_s']

def obtener_ventanas_control_negativo(df, transiciones, n_ventanas=50, margen_evitar=5):
    if len(df) == 0:
        return []
    evitar = set()
    for t in transiciones:
        evitar.update(range(max(0, t-margen_evitar), min(len(df), t+margen_evitar+1)))
    candidatos = [i for i in range(len(df)) if i not in evitar]
    candidatos_validos = [i for i in candidatos if (not pd.isnull(df.iloc[i].get('k_topo', np.nan)) and not pd.isnull(df.iloc[i].get('phi_h', np.nan)))]
    if len(candidatos_validos) < n_ventanas:
        return candidatos_validos
    return random.sample(candidatos_validos, n_ventanas)

def contar_singularidades_en_control(df, idx_control, singularidad_idx):
    if not idx_control or not singularidad_idx:
        return 0
    set_sing = set(singularidad_idx)
    return sum(1 for idx in idx_control if idx in set_sing)

def calcular_sensibilidad_tasa_fp(resumen_batch, margen_deteccion=2):
    TP = sum(sum(1 for d in r["desfases_despertar"] if d is not None and abs(d) <= margen_deteccion) for r in resumen_batch if r and r.get("desfases_despertar"))
    N_eventos = sum(r["n_despertares"] for r in resumen_batch if r and r.get("n_despertares") is not None)
    FP = sum(r.get("n_fp_control", 0) for r in resumen_batch if r)
    N_control = sum(r.get("n_control", 0) for r in resumen_batch if r)
    sensibilidad = TP / N_eventos if N_eventos else float('nan')
    tasa_fp = FP / N_control if N_control else float('nan')
    return sensibilidad, tasa_fp

def procesar_archivo(
    csv_path,
    only_umbrales=False,
    forced_umbrales=None,
    control_negativo_n=50,
    control_negativo_margin=5,
    modo_estricto=False,
    modo_metricas='simulado',
    eeg_data=None,
    random_seed=None
):
    nombre_archivo = os.path.basename(csv_path)
    print(f"Procesando archivo: {csv_path}")
    df = cargar_metricas_y_hipnograma(csv_path)
    if df is None or df.shape[0] == 0:
        print(f"Saltando {nombre_archivo}: archivo vacío o inválido.")
        return None

    if df['k_topo'].isnull().all() or df['phi_h'].isnull().all() or df['delta_pci'].isnull().all():
        df = calcular_metricas_por_ventana(df, modo=modo_metricas, eeg_data=eeg_data, random_seed=random_seed)

    despertares, dormidas = detectar_transiciones(df)
    percentil = 75
    usar_media_std = False
    Nstd = 2
    max_singularidades = 100
    pendiente_min = 0.01
    curvatura_min = 0.005
    if modo_estricto:
        percentil = 95
        usar_media_std = True
        Nstd = 2
        max_singularidades = 100
        pendiente_min = 0.01
        curvatura_min = 0.005
    if forced_umbrales is not None:
        umbrales = forced_umbrales
    else:
        umbrales = calcular_umbral_metricas(df, despertares, VENTANAS_ANTES, modo_estricto, percentil, usar_media_std, Nstd)

    if only_umbrales:
        return umbrales

    singularidad_idx = buscar_singularidades(df, umbrales, MARGEN_DELTA_PCI)
    singularidad_idx = filtrar_por_pendiente_curvatura(df, singularidad_idx, modo_estricto, pendiente_min, curvatura_min)
    singularidad_idx = limitar_singularidades(singularidad_idx, max_singularidades, modo_estricto)

    singularidad_tipo = clasificar_singularidades_unico(singularidad_idx, despertares, dormidas, VENTANA_MARGIN)
    convergentes = [idx for idx in singularidad_idx if singularidad_tipo[idx] == "convergente"]
    divergentes = [idx for idx in singularidad_idx if singularidad_tipo[idx] == "divergente"]
    sin_clasificar = [idx for idx in singularidad_idx if singularidad_tipo[idx] == "sin_clasificar"]

    desfases_despertar = medir_desfases(despertares, singularidad_idx, max_desfase=VENTANA_MARGIN)
    desfases_dormida = medir_desfases(dormidas, singularidad_idx, max_desfase=VENTANA_MARGIN)

    subcarp = os.path.join(CARPETA_RESULTADOS, ("estricto_" if modo_estricto else "") + nombre_archivo.replace('.csv',''))
    safe_makedirs(subcarp)
    plot_fase1(df, despertares, dormidas, singularidad_idx, singularidad_tipo, subcarp, nombre_archivo)

    singularidades_conver = medir_desfase_y_clasificar(convergentes, despertares, "convergente", VENTANA_MARGIN)
    singularidades_diver = medir_desfase_y_clasificar(divergentes, dormidas, "divergente", VENTANA_MARGIN)
    singularidades_sin = [(idx, None, "sin_clasificar") for idx in sin_clasificar]
    todos_eventos = singularidades_conver + singularidades_diver + singularidades_sin

    eventos_df = pd.DataFrame([
        {
            "indice": idx,
            "tiempo_s": get_event_time(df, idx),
            "k_topo": df.iloc[idx]['k_topo'],
            "phi_h": df.iloc[idx]['phi_h'],
            "delta_pci": df.iloc[idx]['delta_pci'],
            "tipo": tipo,
            "desfase_a_transicion": desfase,
            "falso_positivo": desfase is None
        }
        for idx, desfase, tipo in todos_eventos
    ])
    eventos_df.to_csv(os.path.join(subcarp, "eventos_detectados.csv"), index=False)

    n_falsos_positivos = sum(1 for _, d, _ in todos_eventos if d is None)

    informe_path = os.path.join(subcarp, "informe.txt")
    with open(informe_path, "w") as f:
        f.write(f"Archivo analizado: {nombre_archivo}\n")
        f.write(f"Total despertares detectados: {len(despertares)}\n")
        f.write(f"Total dormidas detectadas: {len(dormidas)}\n")
        f.write(f"Total singularidades detectadas: {len(singularidad_idx)}\n")
        f.write(f"Singularidades convergentes: {len(convergentes)}\n")
        f.write(f"Singularidades divergentes: {len(divergentes)}\n")
        f.write(f"Singularidades sin clasificar: {len(sin_clasificar)}\n")
        f.write(f"FALSOS POSITIVOS (sin transición cercana): {n_falsos_positivos}\n")
        f.write(f"Tasa basal de singularidad: {tasa_singularidad_baseline(len(singularidad_idx), len(df)):.4f}\n")

    val_cruzada_result = validacion_cruzada_umbrales(df, despertares, VENTANAS_ANTES, MARGEN_DELTA_PCI, n_splits=2)
    with open(os.path.join(subcarp, "validacion_cruzada.txt"), "w") as fvc:
        for fold in val_cruzada_result:
            fvc.write(f"Fold {fold.get('fold')}\n")
            if "mensaje" in fold:
                fvc.write(f"AVISO: {fold['mensaje']}\n")
            else:
                fvc.write(f"Umbrales: {fold['umbrales']}\n"
                          f"Transiciones test: {fold['test_transiciones']}\nSingularidades detectadas en test: {fold['singularidades']}\nDesfases: {fold['desfases']}\n")
            fvc.write("---\n")

    boxplots_metricas_previas(df, despertares, singularidad_idx, margen=2, carpeta=subcarp, nombre_archivo=nombre_archivo)
    curvas_promedio_previas(df, despertares, singularidad_idx, margen=2, carpeta=subcarp, nombre_archivo=nombre_archivo)
    analizar_delta_pci_post_singularidad(df, singularidad_idx, singularidad_tipo, n_ventanas_post=3, carpeta=subcarp, nombre_archivo=nombre_archivo)

    metricas_previas_despertar = extraer_metricas_previas(df, despertares, metricas=["k_topo", "phi_h"], margen=2)
    metricas_previas_dormida = extraer_metricas_previas(df, dormidas, metricas=["k_topo", "phi_h"], margen=2)
    with open(os.path.join(subcarp, "metricas_previas.txt"), "w") as f:
        for m in metricas_previas_despertar:
            f.write(f"{m} antes/después de despertares: {metricas_previas_despertar[m]}\n")
        for m in metricas_previas_dormida:
            f.write(f"{m} antes/después de dormidas: {metricas_previas_dormida[m]}\n")

    with open(os.path.join(subcarp, "desfases_transiciones.txt"), "w") as f:
        f.write(f"Desfases (singularidad - transición) para despertares: {desfases_despertar}\n")
        f.write(f"Desfases (singularidad - transición) para dormidas: {desfases_dormida}\n")

    idx_control = obtener_ventanas_control_negativo(
        df, despertares+dormidas,
        n_ventanas=control_negativo_n, margen_evitar=control_negativo_margin
    )
    n_fp_control = contar_singularidades_en_control(df, idx_control, singularidad_idx)
    pd.DataFrame({"idx_control": idx_control}).to_csv(os.path.join(subcarp, "control_negativo_idx.csv"), index=False)

    return {
        "archivo": nombre_archivo,
        "n_despertares": len(despertares),
        "n_dormidas": len(dormidas),
        "n_singularidades": len(singularidad_idx),
        "n_convergentes": len(convergentes),
        "n_divergentes": len(divergentes),
        "n_sin_clasificar": len(sin_clasificar),
        "desfases_despertar": desfases_despertar,
        "desfases_dormida": desfases_dormida,
        "eventos_tipo": [singularidad_tipo[idx] for idx in singularidad_idx],
        "n_fp_control": n_fp_control,
        "n_control": len(idx_control),
    }

def get_optimal_workers(estimated_usage_per_process_gb=2):
    return min(12, multiprocessing.cpu_count())

def main_batch():
    print("Buscando y convirtiendo archivos Hypnogram Sleep-EDF a formato ventana...")
    hypnogram_csvs = convert_all_hypnograms_to_pipeline_csv(CARPETA_BASE, CARPETA_BASE)
    print(f"Total archivos Hypnogram convertidos: {len(hypnogram_csvs)}")

    archivos = glob.glob(os.path.join(CARPETA_BASE, "*.csv"))
    archivos = [a for a in archivos if os.path.basename(a) not in ["resumen_batch.csv", "loso_crossval.csv"]]
    archivos = [a for a in archivos if os.path.getsize(a) > 0]
    if not archivos:
        print("No se encontraron archivos para analizar. Asegúrate de tener archivos .csv en la carpeta base.")
        return

    safe_makedirs(CARPETA_RESULTADOS)

    num_workers = get_optimal_workers(estimated_usage_per_process_gb=2)
    print(f"Usando {num_workers} procesos paralelos.")

    resumen_batch = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(procesar_archivo, archivo, modo_metricas='simulado', random_seed=42): archivo for archivo in archivos}
        for future in as_completed(future_to_file):
            try:
                resultado = future.result()
                if resultado:
                    resumen_batch.append(resultado)
            except Exception as e:
                print(f"[ERROR] Archivo {future_to_file[future]}: {e}")
    resumen_df = pd.DataFrame(resumen_batch)
    resumen_csv_path = os.path.join(CARPETA_RESULTADOS, "resumen_batch.csv")
    resumen_df.to_csv(resumen_csv_path, index=False)
    print(f"\nResumen batch guardado en {resumen_csv_path}")

    resumen_batch_estricto = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(procesar_archivo, archivo, False, None, 50, 5, True, 'simulado', None, 42): archivo for archivo in archivos}
        for future in as_completed(future_to_file):
            try:
                resultado = future.result()
                if resultado:
                    resumen_batch_estricto.append(resultado)
            except Exception as e:
                print(f"[ERROR] Archivo (estricto) {future_to_file[future]}: {e}")
    resumen_df_estricto = pd.DataFrame(resumen_batch_estricto)
    resumen_csv_path_estricto = os.path.join(CARPETA_RESULTADOS, "resumen_batch_estricto.csv")
    resumen_df_estricto.to_csv(resumen_csv_path_estricto, index=False)
    print(f"\nResumen batch ESTRICTO guardado en {resumen_csv_path_estricto}")

if __name__ == "__main__":
    main_batch()