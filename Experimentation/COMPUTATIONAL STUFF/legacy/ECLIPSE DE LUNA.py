import os
import numpy as np
import pandas as pd
import mne
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import logging
import time
from datetime import datetime
import traceback

# =========== CONFIGURACIÓN ===========
nombre_exp = "ECLIPSE LUNAR"
carpeta_base = r'G:\Mi unidad\AFH\EXPERIMENTO\FASE 2\SLEEP-EDF\DATABASE\sleep-cassette'
resultados_dir = os.path.join(carpeta_base, nombre_exp)
os.makedirs(resultados_dir, exist_ok=True)
canals = None  # Puedes poner ['EEG Fpz-Cz'] si quieres solo un canal
VENTANA_S = 30
PASO_S = 5
N_BINS = 16
MAX_WORKERS = 8  # Puedes aumentar si tu PC lo soporta

UMBRAL_PERCENTIL_COLECTIVO = 95  # Percentil para considerar un salto colectivo "alto"
VENTANAS_BUSQUEDA_CRUCE = 5      # +- ventanas alrededor del cruce a considerar

logging.basicConfig(
    filename=os.path.join(resultados_dir, "afh_pipeline.log"),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def mutual_information(x, y, bins=N_BINS):
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)

def k_topo_proxy(corr_matrix):
    return np.nanmean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))

def phi_h_proxy(mi_matrix):
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
    return np.abs(lz1-lz2)

def signal_entropy(signal):
    hist, _ = np.histogram(signal, bins=N_BINS, density=True)
    return entropy(hist + 1e-10)

def calcular_indice_colectivo(df):
    d_k = np.diff(df['k_topo'].values)
    d_phi = np.diff(df['phi_h'].values)
    d_pci = np.diff(df['delta_pci'].values)
    return np.sqrt(d_k**2 + d_phi**2 + d_pci**2)

def detectar_cruces_horizonte(estados):
    # Retorna índices donde hay transición sueño<->vigilia
    transiciones = []
    for idx in range(1, len(estados)):
        if estados[idx] == 0 and estados[idx-1] in [1,2,3,4]:
            transiciones.append(('convergente', idx))  # DESPERTAR
        elif estados[idx] in [1,2,3,4] and estados[idx-1] == 0:
            transiciones.append(('divergente', idx))  # SE DUERME
    return transiciones

def buscar_singularidad_colectiva(indice_colectivo, cruce_idx, percentil_umbral=UMBRAL_PERCENTIL_COLECTIVO, ventanas=VENTANAS_BUSQUEDA_CRUCE):
    start = max(0, cruce_idx-ventanas)
    end = min(len(indice_colectivo), cruce_idx+ventanas)
    sub_ind = indice_colectivo[start:end]
    if np.all(np.isnan(sub_ind)):
        return np.nan, np.nan, np.nan
    idx_local = np.nanargmax(sub_ind) + start
    valor_max = indice_colectivo[idx_local]
    if np.any(~np.isnan(indice_colectivo)):
        umbral = np.percentile(indice_colectivo[~np.isnan(indice_colectivo)], percentil_umbral)
    else:
        umbral = np.nan
    return idx_local, valor_max, umbral

def asignar_etapa(desc):
    desc = desc.upper().strip()
    if 'SLEEP STAGE W' in desc or desc == 'W':
        return 0  # Vigilia
    elif 'SLEEP STAGE 1' in desc or desc == 'N1' or desc == '1':
        return 1  # N1
    elif 'SLEEP STAGE 2' in desc or desc == 'N2' or desc == '2':
        return 2  # N2
    elif 'SLEEP STAGE 3' in desc or desc == 'N3' or desc == '3':
        return 3  # N3
    elif 'SLEEP STAGE 4' in desc or desc == 'N4' or desc == '4':
        return 3  # En Sleep-EDF, N3 y N4 suelen considerarse juntos como N3
    elif 'SLEEP STAGE R' in desc or desc == 'REM' or desc == 'R':
        return 4  # REM
    elif '?' in desc or 'SLEEP STAGE ?' in desc:
        return -1  # Indeterminado/Artefacto
    else:
        return -1  # Cualquier otro valor desconocido

def procesar_sujeto(args):
    nfile, psg_path, total_archivos, canales_analizar = args
    base = os.path.basename(psg_path).replace("-PSG.edf", "")
    csv_path = os.path.join(resultados_dir, f"{base}_PAH_ventanas.csv")
    # --- CHECKPOINT: si ya existe el CSV de este sujeto, saltar ---
    if os.path.exists(csv_path):
        print(f"[{nfile+1}/{total_archivos}] {base}: resultado ya existe, saltando.")
        return csv_path

    hyp_path = None
    for f in os.listdir(os.path.dirname(psg_path)):
        if f.startswith(base[:-1]) and f.endswith("-Hypnogram.edf"):
            hyp_path = os.path.join(os.path.dirname(psg_path), f)
            break
    if not hyp_path:
        print(f"No hipnograma para {base}")
        return None
    print(f"[{nfile+1}/{total_archivos}] {base}")

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose='ERROR')
        raw.set_annotations(mne.read_annotations(hyp_path))
        sfreq = raw.info['sfreq']
        data = raw.get_data()
        if canales_analizar:
            idx_canales = [raw.ch_names.index(c) for c in canales_analizar]
            data = data[idx_canales]
        n_can, n_muestras = data.shape

        win_samples = int(VENTANA_S * sfreq)
        paso_samples = int(PASO_S * sfreq)
        n_ventanas = (n_muestras - win_samples) // paso_samples + 1

        hipnograma = np.full(n_muestras, -1)
        for ann in raw.annotations:
            ini = int(ann['onset'] * sfreq)
            fin = int((ann['onset']+ann['duration']) * sfreq)
            state = asignar_etapa(ann['description'])
            hipnograma[ini:fin] = state
        estados_ventana = [
            np.bincount(hipnograma[i*paso_samples:i*paso_samples+win_samples][hipnograma[i*paso_samples:i*paso_samples+win_samples]>=0]).argmax()
            if np.any(hipnograma[i*paso_samples:i*paso_samples+win_samples]>=0) else -1
            for i in range(n_ventanas)
        ]

        # Detectar transiciones
        transiciones = detectar_cruces_horizonte(estados_ventana)
        ventanas_despertar = [idx for tipo, idx in transiciones if tipo == "convergente"]
        ventanas_dormir = [idx for tipo, idx in transiciones if tipo == "divergente"]

        # Guardar ventanas de transiciones
        df_trans = pd.DataFrame({
            "ventana_despertar": ventanas_despertar,
            "ventana_dormir": ventanas_dormir + [None]*(len(ventanas_despertar)-len(ventanas_dormir))
        })
        df_trans.to_csv(os.path.join(resultados_dir, f"{base}_ventanas_transiciones.csv"), index=False)

        # --- Procesar solo las ventanas alrededor de todas las transiciones ---
        ventanas_a_procesar = set()
        rango = 5  # Puedes cambiar el rango de ventanas antes/después
        for idx in ventanas_despertar + ventanas_dormir:
            for i in range(max(0, idx-rango), min(n_ventanas, idx+rango+1)):
                ventanas_a_procesar.add(i)
        ventanas_a_procesar = sorted(list(ventanas_a_procesar))

        if not ventanas_a_procesar:
            print(f"[{base}] No se encontraron transiciones relevantes, no se procesa nada.")
            return None

        print(f"[{base}] Procesando ventanas alrededor de transiciones: {ventanas_a_procesar}")

        heartbeat_path = os.path.join(resultados_dir, f"alive_{os.getpid()}.txt")
        last_heartbeat = time.time()
        heartbeat_interval = 60

        resultados = []
        data_bin_prev = None
        for count, idx in enumerate(ventanas_a_procesar):
            ini = idx * paso_samples
            fin = ini + win_samples
            segmento = data[:, ini:fin]
            t_inicio = ini / sfreq
            corr_matrix = np.corrcoef(segmento)
            k_topo = k_topo_proxy(corr_matrix)
            mi_matrix = np.zeros((n_can, n_can))
            for i in range(n_can):
                for j in range(n_can):
                    mi_matrix[i,j] = mutual_information(segmento[i], segmento[j])
            phi_h = phi_h_proxy(mi_matrix)
            entropias = [signal_entropy(chan) for chan in segmento]
            data_bin = (segmento.flatten() > np.median(segmento)).astype(int)
            if data_bin_prev is not None:
                delta_pci_val = delta_pci(data_bin, data_bin_prev)
            else:
                delta_pci_val = np.nan
            data_bin_prev = data_bin
            resultados.append({
                'ventana': idx+1,
                't_inicio_s': t_inicio,
                'k_topo': k_topo,
                'phi_h': phi_h,
                'delta_pci': delta_pci_val,
                'estado': estados_ventana[idx],
                **{f'entropia_c{i+1}': e for i, e in enumerate(entropias)}
            })

            if (time.time() - last_heartbeat > heartbeat_interval):
                porcentaje = (100*(count+1)/len(ventanas_a_procesar))
                msg = f"[{base}] Sigo trabajando... ventana {idx+1}/{n_ventanas} (local {count+1}/{len(ventanas_a_procesar)}) ({porcentaje:.1f}%)"
                print(msg, flush=True)
                with open(heartbeat_path, "w") as f:
                    f.write(f"{msg} - {time.ctime()}\n")
                last_heartbeat = time.time()

        df = pd.DataFrame(resultados)
        indice_colectivo = calcular_indice_colectivo(df)
        # Buscar singularidades en cada transición
        singularidades = []
        for tipo, idx_cruce in transiciones:
            if idx_cruce not in ventanas_a_procesar:
                continue
            idx_local = ventanas_a_procesar.index(idx_cruce)
            idx_sing, valor, umbral = buscar_singularidad_colectiva(
                indice_colectivo, idx_local, percentil_umbral=UMBRAL_PERCENTIL_COLECTIVO, ventanas=VENTANAS_BUSQUEDA_CRUCE
            )
            desfase = idx_sing - idx_local if not np.isnan(idx_sing) else None
            if not np.isnan(valor) and not np.isnan(umbral) and valor > umbral:
                singularidades.append({
                    "tipo": tipo,
                    "indice_cruce": idx_local,
                    "indice_sing": idx_sing,
                    "desfase": desfase,
                    "tiempo_cruce_s": df.iloc[idx_local]['t_inicio_s'],
                    "tiempo_sing_s": df.iloc[idx_sing]['t_inicio_s'],
                    "valor_indice": valor,
                    "umbral_indice": umbral
                })
                print(f"[{base}] SINGULARIDAD {tipo.upper()} detectada en ventana {ventanas_a_procesar[idx_sing]} (cruce en {ventanas_a_procesar[idx_local]}) desfase {desfase}, valor {valor:.3f} > umbral {umbral:.3f}", flush=True)
            else:
                print(f"[{base}] No singularidad {tipo} en cruce {ventanas_a_procesar[idx_local]} (valor máx {valor:.3f} < umbral {umbral:.3f})", flush=True)

        # Guardar singularidades colectivas en archivo
        singu_path = os.path.join(resultados_dir, f"{base}_singularidades_COLECTIVAS.csv")
        with open(singu_path, "w") as f:
            f.write("tipo,indice_cruce,indice_sing,desfase,tiempo_cruce_s,tiempo_sing_s,valor_indice,umbral_indice\n")
            for d in singularidades:
                f.write(f"{d['tipo']},{d['indice_cruce']},{d['indice_sing']},{d['desfase']},{d['tiempo_cruce_s']},{d['tiempo_sing_s']},{d['valor_indice']},{d['umbral_indice']}\n")

        # ---- GUARDADO ----
        df.to_csv(csv_path, index=False)

        # ---- VISUALIZACIÓN ----
        tiempos = df['t_inicio_s'].values
        kappa = df['k_topo'].values
        phi = df['phi_h'].values
        pci = df['delta_pci'].values
        estados_hipno = df['estado'].values

        plt.figure(figsize=(15, 8))
        plt.subplot(2,1,1)
        plt.plot(tiempos, kappa, '-o', label='κ_topo')
        plt.plot(tiempos, phi, '-o', label='Φ_H')
        plt.plot(tiempos, pci, '-o', label='ΔPCI')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Valor')
        plt.legend()
        plt.title(f'{base} - Métricas individuales (ventanas transición)')

        plt.subplot(2,1,2)
        plt.plot(tiempos[1:], indice_colectivo, '-o', label='Índice colectivo (derivada conjunta)')
        for d in singularidades:
            plt.axvline(d['tiempo_cruce_s'], color='r', linestyle='--', label=f"Cruce {d['tipo']}")
            plt.scatter(d['tiempo_sing_s'], d['valor_indice'], color='black', s=180, marker='X', label=f"Sing. {d['tipo']} (desfase {d['desfase']})")
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Índice colectivo')
        plt.title('Índice colectivo de salto cualitativo (ventanas transición)')
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(resultados_dir, f"{base}_PAH_grafico_COLECTIVO.png")
        plt.savefig(fig_path)
        plt.close()

        # --- NUEVO: Analisis entre singularidades ---
        if len(singularidades) >= 2:
            df_segmentos = []
            for i in range(len(singularidades)-1):
                t1 = singularidades[i]['tiempo_sing_s']
                t2 = singularidades[i+1]['tiempo_sing_s']
                tipo1 = singularidades[i]['tipo']
                tipo2 = singularidades[i+1]['tipo']
                idx1 = singularidades[i]['indice_sing']
                idx2 = singularidades[i+1]['indice_sing']
                segmento = df[(df['t_inicio_s'] >= t1) & (df['t_inicio_s'] <= t2)]
                if len(segmento) < 2:
                    continue
                # Guardar CSV del segmento
                seg_filename = os.path.join(resultados_dir, f"{base}_entre_{tipo1}_{idx1}_y_{tipo2}_{idx2}.csv")
                segmento.to_csv(seg_filename, index=False)
                # Graficar
                plt.figure(figsize=(12,4))
                plt.plot(segmento['t_inicio_s'], segmento['k_topo'], label='κ_topo')
                plt.plot(segmento['t_inicio_s'], segmento['phi_h'], label='Φ_H')
                plt.plot(segmento['t_inicio_s'], segmento['delta_pci'], label='ΔPCI')
                plt.title(f"Evolución de métricas entre {tipo1} y {tipo2} ({base})")
                plt.xlabel("Tiempo (s)")
                plt.legend()
                seg_png = os.path.join(resultados_dir, f"{base}_entre_{tipo1}_{idx1}_y_{tipo2}_{idx2}.png")
                plt.savefig(seg_png)
                plt.close()
                # Estadísticas
                medias = segmento[['k_topo','phi_h','delta_pci']].mean()
                df_segmentos.append({
                    'sujeto': base,
                    'tipo1': tipo1,
                    'tipo2': tipo2,
                    'idx1': idx1,
                    'idx2': idx2,
                    't1': t1,
                    't2': t2,
                    'n_ventanas': len(segmento),
                    'k_topo_mean': medias['k_topo'],
                    'phi_h_mean': medias['phi_h'],
                    'delta_pci_mean': medias['delta_pci'],
                })
            if df_segmentos:
                pd.DataFrame(df_segmentos).to_csv(os.path.join(resultados_dir, f"{base}_resumen_segmentos_entre_singularidades.csv"), index=False)

        return csv_path

    except Exception as e:
        error_file = os.path.join(resultados_dir, f"error_{os.getpid()}_{base}.log")
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"ERROR procesando {base}: {repr(e)}", flush=True)
        return None

if __name__ == "__main__":
    print("\n========== PIPELINE CONFLUENCIA COLECTIVA (TRANSICIONES ETAPAS ROBUSTAS) ==========", flush=True)
    print("Directorio de resultados para esta corrida:", resultados_dir, flush=True)
    print("Archivos en carpeta base:", os.listdir(carpeta_base), flush=True)
    psg_files = [os.path.join(carpeta_base, f) for f in os.listdir(carpeta_base) if f.endswith("-PSG.edf")]
    print("Archivos PSG detectados:", [os.path.basename(f) for f in psg_files], flush=True)
    total_archivos = len(psg_files)
    print(f"Archivos PSG encontrados: {total_archivos}", flush=True)
    args = [(n, f, total_archivos, canals) for n, f in enumerate(psg_files)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        resultados_csv = list(executor.map(procesar_sujeto, args))
    dfs = []
    for csv_path in resultados_csv:
        if csv_path and os.path.exists(csv_path):
            dfs.append(pd.read_csv(csv_path))
    if dfs:
        df_total = pd.concat(dfs, ignore_index=True)
        global_csv = os.path.join(resultados_dir, "PAH_todas_ventanas_COLECTIVO.csv")
        df_total.to_csv(global_csv, index=False)
        print(f"Resultados globales guardados en: {global_csv}", flush=True)
    print("¡Pipeline AFH/PAH CONFLUENCIA COLECTIVA terminado!", flush=True)
