"""
=======================================================================
PRUEBA DE RASTREO DE FUENTE: DE CORRELACION DE MAGNITUD A
IDENTIFICABILIDAD DE CONTENIDO
=======================================================================

CONTEXTO

simulacion_acoplamiento_fase_temprana_tardia.py prueba si la MAGNITUD
resumen de la fase tardia covaria, ensayo a ensayo, con la magnitud de
la fase temprana, mas alla de un confusor compartido (arousal). Eso es
necesario para la Prediccion 6 pero no es exactamente lo que el
manuscrito pide bajo el nombre "rastreo de fuente": que la senal
tardia derive del MISMO PROCESAMIENTO CORTICAL que la entrada rapida
precedente -- una afirmacion sobre el CONTENIDO especifico de cada
ensayo, no solo sobre si dos numeros resumen (fuerza temprana, fuerza
tardia) suben y bajan juntos.

La diferencia importa porque son logicamente independientes:
    - Dos ensayos pueden tener magnitudes correlacionadas sin que el
      patron especifico de la fase tardia porte ninguna huella del
      patron especifico de la fase temprana de ESE ensayo (si la
      correlacion de magnitud viene solo de una ganancia global
      compartida).
    - El contenido puede ser rastreable (la fase tardia SI deriva del
      procesamiento de la fase temprana especifica) incluso si las
      magnitudes resumen no correlacionan -- por ejemplo, si la
      variacion entre ensayos esta sobre todo en QUE patron ocurrio,
      no en CUAN FUERTE fue.

Este script implementa la segunda prueba: identificabilidad por
emparejamiento, controlando el confusor por estratos. Y construye a
proposito un escenario donde la prueba de magnitud (la del script
anterior) es CIEGA -- para mostrar que no es redundante, llena un
hueco real. Ver Sec. 9.1 de AFH_modelo_matematico_pliegue.md para la
formalizacion completa.

QUE HACE:
    1. Genera, por ensayo, un vector de "contenido" temprano (una
       direccion aleatoria de radio ~constante -- lo que varia entre
       ensayos es EL PATRON, no la fuerza) y un vector de contenido
       tardio, bajo Tipo A (sin relacion especifica, solo un
       corrimiento aditivo compartido por arousal) o Tipo B (el
       tardio porta una rotacion fija del patron temprano, ademas del
       mismo corrimiento por arousal).
    2. Corre la prueba de MAGNITUD (identica en logica a la del script
       anterior) sobre ||e||, ||l|| -- y muestra que es ciega en este
       escenario, en AMBOS modelos.
    3. Corre la prueba de IDENTIFICABILIDAD POR EMPAREJAMIENTO: separa
       ensayos en entrenamiento/prueba, ajusta un mapeo lineal
       contenido-temprano -> contenido-tardio en entrenamiento
       (despues de descontar el confusor), y en prueba mide si el
       patron predicho para el ensayo i se parece mas a su propio
       patron tardio real que al de otros ensayos con arousal
       parecido (top-1 accuracy dentro de estratos de arousal). Nulo
       por permutacion DENTRO de cada estrato (no global -- permutar
       globalmente sesga el nulo, ver nota en la funcion).

INSTALACION: pip install numpy scipy
CORRER:      python "test_rastreo_de_fuente.py"
TIEMPO:      unos segundos
=======================================================================
"""

import numpy as np
from scipy import stats


# =====================================================================
# BLOQUE 1: GENERACION DE ENSAYOS CON CONTENIDO VECTORIAL
# =====================================================================

def generar_ensayos_contenido(n_trials, d=10, tipo="B", kappa=0.8,
                               r0=1.0, sigma_e=0.03, b_L=2.0,
                               sigma_l=0.25, sigma_meas=0.5, semilla=None):
    """
    e_i in R^d: contenido temprano. Direccion aleatoria unitaria por
    ensayo (el "patron", que es lo que porta la identidad del ensayo),
    radio ~constante r0 (con jitter minimo) -- a proposito, para que
    CASI TODA la varianza entre ensayos este en la direccion, no en la
    norma.

    l_i in R^d: contenido tardio. Un corrimiento aditivo compartido
    por el confusor (arousal, z) igual en las d dimensiones -- domina
    la norma de l_i -- mas, solo bajo Tipo B, una rotacion fija kappa*Q
    del contenido temprano de ESE ensayo.

    z: confusor latente. obsZ: su proxy observable y ruidoso (igual
    que en el script anterior).
    """
    rng = np.random.default_rng(semilla)
    z = rng.normal(size=n_trials)
    obsZ = z + sigma_meas * rng.normal(size=n_trials)

    raw = rng.normal(size=(n_trials, d))
    direcciones = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    e = direcciones * r0 + sigma_e * rng.normal(size=(n_trials, d))

    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))  # matriz ortogonal fija (Phi = kappa*Q)

    l = np.outer(b_L * z, np.ones(d)) + sigma_l * rng.normal(size=(n_trials, d))
    if tipo == "B":
        l = l + kappa * (e @ Q.T)

    return e, l, obsZ, z, Q


# =====================================================================
# BLOQUE 2: PRUEBA DE MAGNITUD (para comparar -- logica del script anterior)
# =====================================================================

def correlacion_parcial(x, y, control):
    n = len(x)
    r_xy, _ = stats.pearsonr(x, y)
    r_xc, _ = stats.pearsonr(x, control)
    r_yc, _ = stats.pearsonr(y, control)
    denom = np.sqrt((1 - r_xc**2) * (1 - r_yc**2))
    if denom < 1e-12:
        return np.nan, np.nan
    r_partial = (r_xy - r_xc * r_yc) / denom
    r_partial = max(-0.999999, min(0.999999, r_partial))
    df = n - 3
    if df <= 0:
        return r_partial, np.nan
    t = r_partial * np.sqrt(df / (1 - r_partial**2))
    p = 2 * (1 - stats.t.cdf(abs(t), df))
    return r_partial, p


def prueba_de_magnitud(e, l, obsZ):
    """La prueba del script anterior, aplicada a normas ||e||, ||l||."""
    E = np.linalg.norm(e, axis=1)
    L = np.linalg.norm(l, axis=1)
    r_raw, _ = stats.pearsonr(E, L)
    r_partial, p_partial = correlacion_parcial(E, L, obsZ)
    return r_raw, r_partial, p_partial


# =====================================================================
# BLOQUE 3: PRUEBA DE IDENTIFICABILIDAD POR EMPAREJAMIENTO
# =====================================================================

def deconfundir(X, proxy):
    """Residuos de X (n x d) tras regresion lineal columna a columna
    sobre [1, proxy]. Igual que restarle a cada dimension su mejor
    prediccion lineal a partir del proxy del confusor."""
    n = X.shape[0]
    A = np.column_stack([np.ones(n), proxy])
    coef, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def ridge_fit(X, Y, alpha=1.0):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)


def estratificar(proxy, n_estratos=5):
    cortes = np.quantile(proxy, np.linspace(0, 1, n_estratos + 1))
    return np.clip(np.digitize(proxy, cortes[1:-1]), 0, n_estratos - 1)


def top1_accuracy(Lhat, Lte, estrato):
    """Para cada ensayo de prueba, ¿el vecino mas cercano a su l
    predicho -- DENTRO de su propio estrato de arousal -- es su propio
    l verdadero? Devuelve la fraccion de aciertos."""
    aciertos, evaluados = 0, 0
    for e_id in np.unique(estrato):
        idxs = np.where(estrato == e_id)[0]
        if len(idxs) < 3:
            continue
        for i in idxs:
            dists = [np.sum((Lhat[i] - Lte[j])**2) for j in idxs]
            mejor = idxs[int(np.argmin(dists))]
            evaluados += 1
            if mejor == i:
                aciertos += 1
    return aciertos / evaluados if evaluados > 0 else np.nan


def permutar_dentro_de_estratos(Lte, estrato, rng):
    """Nulo por permutacion CORRECTO: mezcla identidades solo dentro
    de cada estrato de arousal, preservando la similitud que el
    confusor por si solo produciria. Permutar globalmente sesgaria el
    nulo -- ensayos de estratos distintos ya son diferentes por el
    confusor, así que un nulo global subestima la tasa de acierto
    esperada bajo Tipo A y vuelve la prueba anticonservadora."""
    Lte_perm = Lte.copy()
    for e_id in np.unique(estrato):
        idxs = np.where(estrato == e_id)[0]
        Lte_perm[idxs] = Lte[rng.permutation(idxs)]
    return Lte_perm


def prueba_identificabilidad(e, l, obsZ, n_estratos=5, alpha_ridge=1.0,
                              frac_train=0.5, n_perm=300, semilla=0):
    rng = np.random.default_rng(semilla)
    n = e.shape[0]
    idx = rng.permutation(n)
    n_train = int(n * frac_train)
    tr, te = idx[:n_train], idx[n_train:]

    e_res = deconfundir(e, obsZ)
    l_res = deconfundir(l, obsZ)

    A_hat = ridge_fit(e_res[tr], l_res[tr], alpha=alpha_ridge)
    Lhat = e_res[te] @ A_hat
    Lte = l_res[te]
    estrato = estratificar(obsZ[te], n_estratos=n_estratos)

    acc_obs = top1_accuracy(Lhat, Lte, estrato)

    accs_nulas = np.array([
        top1_accuracy(Lhat, permutar_dentro_de_estratos(Lte, estrato, rng), estrato)
        for _ in range(n_perm)
    ])
    p_val = np.mean(accs_nulas >= acc_obs)
    return acc_obs, np.nanmean(accs_nulas), p_val


# =====================================================================
# BLOQUE 4: DEMOSTRACION
# =====================================================================

def main():
    print("=" * 72)
    print("  PRUEBA DE RASTREO DE FUENTE (identificabilidad de contenido)")
    print("  vs. prueba de magnitud (correlacion de normas)")
    print("=" * 72)

    N = 400
    print(f"\n  N ensayos = {N}, d = 10 dimensiones de contenido\n")

    fila = "  {:<28s} {:>12s} {:>12s} {:>10s}"
    print(fila.format("Modelo", "r magnitud", "r parcial", "p parcial"))
    print("  " + "-" * 66)
    for tipo, nombre in [("A", "Tipo A (heterogenea)"), ("B", "Tipo B (reflexiva)")]:
        e, l, obsZ, z, Q = generar_ensayos_contenido(N, tipo=tipo, semilla=1)
        r_raw, r_part, p_part = prueba_de_magnitud(e, l, obsZ)
        marca = "*" if p_part < 0.05 else " "
        print(fila.format(nombre, f"{r_raw:.3f}", f"{r_part:.3f}{marca}", f"{p_part:.3f}"))

    print("\n  -> La prueba de MAGNITUD es ciega en los dos casos: casi toda la")
    print("     varianza entre ensayos esta en el PATRON, no en la fuerza, y la")
    print("     norma de la fase tardia esta dominada por el corrimiento del")
    print("     arousal. Esto es a proposito -- ilustra un caso real: un test")
    print("     de magnitud puede dar 'sin evidencia' incluso cuando SI hay")
    print("     rastreo de fuente genuino.")

    print("\n" + "-" * 72)
    print("  Prueba de IDENTIFICABILIDAD POR EMPAREJAMIENTO (misma simulacion)")
    print("-" * 72)
    fila2 = "  {:<28s} {:>14s} {:>14s} {:>10s}"
    print(fila2.format("Modelo", "top-1 obs.", "top-1 nulo", "p"))
    print("  " + "-" * 70)
    for tipo, nombre in [("A", "Tipo A (heterogenea)"), ("B", "Tipo B (reflexiva)")]:
        e, l, obsZ, z, Q = generar_ensayos_contenido(N, tipo=tipo, semilla=1)
        acc_obs, acc_null, p = prueba_identificabilidad(e, l, obsZ, semilla=2)
        marca = "*" if p < 0.05 else " "
        print(fila2.format(nombre, f"{acc_obs:.3f}", f"{acc_null:.3f}", f"{p:.4f}{marca}"))

    print("\n  -> Donde la magnitud no ve nada, la identificabilidad SI discrimina:")
    print("     top-1 al nivel del azar bajo Tipo A, muy por encima bajo Tipo B.")
    print("     Esta es la version fuerte de 'rastreo de fuente' -- el contenido")
    print("     especifico del ensayo, no solo su fuerza resumen, se puede")
    print("     recuperar de la fase tardia.")

    print("\n" + "=" * 72)
    print("  LECTURA PRACTICA")
    print("=" * 72)
    print("  Las dos pruebas no son redundantes -- responden preguntas distintas")
    print("  y pueden divergir. Un analisis real de la Prediccion 6 deberia")
    print("  correr AMBAS: la de magnitud (mas simple, mas familiar) y esta de")
    print("  identificabilidad (mas exigente, mas cercana a lo que 'rastreo de")
    print("  fuente' significa en el manuscrito). Si solo se corre la de")
    print("  magnitud, un resultado nulo no permite concluir que no hay Tipo B")
    print("  -- podria estar el efecto y la prueba, simplemente, no mirar donde")
    print("  esta.")
    print("=" * 72)


if __name__ == "__main__":
    main()
