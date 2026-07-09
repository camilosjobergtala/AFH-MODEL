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
    4. (Bloque 5) Muestra un segundo hueco, mas serio que el del
       arousal: si el paradigma real VARIA el estimulo entre ensayos
       (no lo repite identico), el Test 2 de los pasos 2-3 puede dar
       positivo bajo Tipo A puro -- porque el contenido temprano y el
       tardio pueden ser ambos predecibles a partir de "que estimulo
       fue", por vias independientes, sin ningun retorno reflexivo.
       Bloque 5 construye ese escenario, confirma que la version
       "ingenua" (solo arousal) falla, e implementa la correccion:
       condicionar tambien por identidad de estimulo (regresion + pool
       de emparejamiento restringido a la misma categoria), al costo
       de menos poder. Los pasos 1-3 de arriba son validos solo bajo
       el supuesto de estimulo unico o identico repetido -- ver Sec.
       9.1 de AFH_modelo_matematico_pliegue.md para la formalizacion
       completa de ambas versiones.

INSTALACION: pip install numpy scipy
CORRER:      python "test_rastreo_de_fuente.py"
TIEMPO:      unos segundos para Bloques 1-5; Bloque 6 (curvas FPR/poder
             vs. N, 80 repeticiones x 4 valores de N x 3 variantes)
             tarda unos 3-4 minutos -- top1_accuracy esta vectorizada
             con numpy precisamente para que esto sea viable.
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

def deconfundir(X, covariables):
    """Residuos de X (n x d) tras regresion lineal sobre `covariables`
    (columna del proxy de arousal, y desde Bloque 5 tambien columnas
    dummy de categoria de estimulo cuando aplica). Acepta covariables
    1D o 2D -- generalizacion de la version original, que solo
    aceptaba el proxy de arousal."""
    n = X.shape[0]
    covariables = np.asarray(covariables)
    if covariables.ndim == 1:
        covariables = covariables.reshape(-1, 1)
    A = np.column_stack([np.ones(n), covariables])
    coef, _, _, _ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def dummies_categoria(categoria, n_categorias):
    """Codificacion dummy (referencia = categoria 0) para condicionar
    por una variable categorica -- p.ej. identidad del estimulo."""
    D = np.zeros((len(categoria), n_categorias - 1))
    for k in range(1, n_categorias):
        D[:, k - 1] = (categoria == k).astype(float)
    return D


def ridge_fit(X, Y, alpha=1.0):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)


def estratificar(proxy, n_estratos=5):
    cortes = np.quantile(proxy, np.linspace(0, 1, n_estratos + 1))
    return np.clip(np.digitize(proxy, cortes[1:-1]), 0, n_estratos - 1)


def top1_accuracy(Lhat, Lte, estrato):
    """Para cada ensayo de prueba, ¿el vecino mas cercano a su l
    predicho -- DENTRO de su propio estrato de arousal -- es su propio
    l verdadero? Devuelve la fraccion de aciertos.

    Vectorizado con numpy (distancias por pares via broadcasting) en
    vez del doble loop en Python puro de la version original -- el
    cuello de botella real de las curvas de poder/FPR de Bloque 6, que
    necesitan cientos de repeticiones de esta funcion por celda."""
    aciertos, evaluados = 0, 0
    for e_id in np.unique(estrato):
        idxs = np.where(estrato == e_id)[0]
        m = len(idxs)
        if m < 3:
            continue
        Lhat_g, Lte_g = Lhat[idxs], Lte[idxs]
        sq_hat = np.sum(Lhat_g**2, axis=1)[:, None]
        sq_te = np.sum(Lte_g**2, axis=1)[None, :]
        dist2 = sq_hat + sq_te - 2 * (Lhat_g @ Lte_g.T)
        mejor_local = np.argmin(dist2, axis=1)
        aciertos += int(np.sum(mejor_local == np.arange(m)))
        evaluados += m
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


def prueba_identificabilidad(e, l, covariables_deconfundir, grupo_pool,
                              alpha_ridge=1.0, frac_train=0.5, n_perm=300,
                              semilla=0):
    """
    covariables_deconfundir: que se regresa fuera de e y l antes de
        ajustar el mapeo (arousal solamente en Bloque 3-4; arousal +
        categoria de estimulo en Bloque 5, ver nota de por que).
    grupo_pool: particion (n_trials,) de TODOS los ensayos dentro de
        la cual se busca el vecino mas cercano y se arma el nulo por
        permutacion. Bloque 3-4: estrato de arousal. Bloque 5
        (version corregida): categoria de estimulo exacta -- no basta
        con regresarla, hay que restringir el pool de comparacion a
        la MISMA categoria (mas robusto que solo regresion lineal si
        el efecto de estimulo no es puramente aditivo).
    """
    rng = np.random.default_rng(semilla)
    n = e.shape[0]
    idx = rng.permutation(n)
    n_train = int(n * frac_train)
    tr, te = idx[:n_train], idx[n_train:]

    e_res = deconfundir(e, covariables_deconfundir)
    l_res = deconfundir(l, covariables_deconfundir)

    A_hat = ridge_fit(e_res[tr], l_res[tr], alpha=alpha_ridge)
    Lhat = e_res[te] @ A_hat
    Lte = l_res[te]
    grupo_te = grupo_pool[te]

    acc_obs = top1_accuracy(Lhat, Lte, grupo_te)

    accs_nulas = np.array([
        top1_accuracy(Lhat, permutar_dentro_de_estratos(Lte, grupo_te, rng), grupo_te)
        for _ in range(n_perm)
    ])
    p_val = np.mean(accs_nulas >= acc_obs)
    return acc_obs, np.nanmean(accs_nulas), p_val


# =====================================================================
# BLOQUE 5: EL PROBLEMA DEL ESTIMULO COMPARTIDO, Y SU CORRECCION
# =====================================================================
#
# El Test 2 de Bloque 3 estratifica/descuenta por arousal, pero no por
# identidad de estimulo. Si el paradigma real VARIA el estimulo entre
# ensayos (no lo repite identico), eso es un problema serio, distinto
# del arousal: bajo Tipo A, el contenido temprano y el tardio pueden
# ser AMBOS predecibles a partir de "que estimulo fue", por dos vias
# completamente independientes, sin que exista ningun retorno
# reflexivo. Un mapeo e->l entrenado sobre datos con estructura de
# estimulo compartido puede lograr identificabilidad por encima del
# azar SOLO por reconocer el estimulo, no por rastrear linaje causal
# -- un falso positivo de Tipo B bajo Tipo A puro. Estratificar por
# arousal no toca esto: el confusor aqui es el estimulo, no el
# arousal.
#
# Este bloque construye ese escenario a proposito, muestra que el
# Test 2 "ingenuo" (que solo condiciona por arousal) en efecto falla
# -- da identificabilidad significativa incluso bajo Tipo A -- y que
# condicionar TAMBIEN por identidad de estimulo (regresion + pool de
# emparejamiento restringido a la misma categoria de estimulo, no solo
# regresion -- mas robusto si el efecto del estimulo no es puramente
# aditivo) restaura la validez, al costo de menos poder: la varianza
# discriminante que queda es solo la fluctuacion idiosincratica
# ensayo-a-ensayo, no la identidad del estimulo -- el mismo problema
# de bajo SNR de Bloque 3/`simulacion_acoplamiento_fase_temprana_tardia.py`,
# heredado con mas fuerza, no evitado.

def generar_ensayos_con_estimulo(n_trials, d=10, n_estimulos=5, tipo="B",
                                  kappa=0.8, fuerza_estimulo=1.3,
                                  fuerza_idiosync=0.4, sigma_e=0.05,
                                  b_L=2.0, sigma_l=0.25, sigma_meas=0.5,
                                  semilla=None):
    """
    Cada ensayo pertenece a una de n_estimulos categorias. El
    contenido temprano y el tardio estan CENTRADOS en un patron tipico
    de esa categoria -- compartido, pero por vias separadas para e y
    para l, que es justamente como luce "ambos codifican el mismo
    estimulo, sin lazo reflexivo" -- mas una fluctuacion idiosincratica
    por ensayo. Bajo Tipo B, SOLO la fluctuacion idiosincratica se
    transporta de e a l (via Phi=kappa*Q): es la unica parte que un
    retorno reflexivo real podria portar, porque la parte de "categoria
    de estimulo" en l ya esta explicada por una via cortical propia,
    independiente de lo que vuelve del talamo.
    """
    rng = np.random.default_rng(semilla)
    estimulo = rng.integers(0, n_estimulos, size=n_trials)
    z = rng.normal(size=n_trials)
    obsZ = z + sigma_meas * rng.normal(size=n_trials)

    patron_estim_e = rng.normal(size=(n_estimulos, d))
    patron_estim_e /= np.linalg.norm(patron_estim_e, axis=1, keepdims=True)
    patron_estim_l = rng.normal(size=(n_estimulos, d))  # via separada, misma categoria
    patron_estim_l /= np.linalg.norm(patron_estim_l, axis=1, keepdims=True)

    raw = rng.normal(size=(n_trials, d))
    idiosync = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    e = (fuerza_estimulo * patron_estim_e[estimulo]
         + fuerza_idiosync * idiosync
         + sigma_e * rng.normal(size=(n_trials, d)))

    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))

    l = (np.outer(b_L * z, np.ones(d))
         + fuerza_estimulo * patron_estim_l[estimulo]
         + sigma_l * rng.normal(size=(n_trials, d)))
    if tipo == "B":
        l = l + kappa * (idiosync @ Q.T)

    return e, l, obsZ, z, estimulo, Q


def estimulo_ruidoso(estimulo, n_estimulos, p_error, rng):
    """Proxy OBSERVADO e imperfecto de la identidad de estimulo: con
    probabilidad p_error, la etiqueta observada es una categoria
    distinta a la verdadera (elegida uniformemente entre las
    restantes). Es el analogo categorico del proxy ruidoso continuo
    que ya existe para arousal (obsZ = z + ruido).

    Sin esto, la "condicion oraculo" solo aislaba el proxy de arousal
    -- el estimulo, dummy-codificado y usado para restringir el pool
    de emparejamiento, siempre era la categoria VERDADERA, en toda
    version de la prueba. Eso no coincide con lo que describe el
    manuscrito (proxies ruidosos para AMBOS confusores) y esta funcion
    lo corrige: ahora existe una version genuinamente ruidosa de la
    identidad de estimulo para usar en la condicion "realista"."""
    n = len(estimulo)
    mal_clasificado = rng.random(n) < p_error
    obs = estimulo.copy()
    idxs = np.where(mal_clasificado)[0]
    if len(idxs) > 0:
        offset = rng.integers(1, n_estimulos, size=len(idxs))
        obs[idxs] = (estimulo[idxs] + offset) % n_estimulos
    return obs


def demostrar_confusion_por_estimulo(n_trials=600, n_estimulos=5,
                                      p_error_estimulo=0.08, semilla=1):
    print("\n" + "=" * 72)
    print("  EL PROBLEMA DEL ESTIMULO COMPARTIDO (y su correccion)")
    print("=" * 72)
    print(f"\n  (proxy de estimulo: {p_error_estimulo:.0%} de tasa de mala clasificacion)\n")

    fila = "  {:<42s} {:>10s} {:>10s} {:>10s}"
    print(fila.format("Modelo / version de la prueba", "top-1 obs", "top-1 null", "p"))
    print("  " + "-" * 76)

    rng = np.random.default_rng(semilla + 100)

    for tipo, nombre in [("A", "Tipo A"), ("B", "Tipo B")]:
        e, l, obsZ, z, estimulo, Q = generar_ensayos_con_estimulo(
            n_trials, n_estimulos=n_estimulos, tipo=tipo, semilla=semilla)
        obs_estimulo = estimulo_ruidoso(estimulo, n_estimulos, p_error_estimulo, rng)

        # prueba INGENUA: condiciona solo por arousal (igual que Bloque 3)
        estrato_arousal = estratificar(obsZ, n_estratos=5)
        acc_i, null_i, p_i = prueba_identificabilidad(e, l, obsZ, estrato_arousal, semilla=2)
        marca_i = "*" if p_i < 0.05 else " "
        print(fila.format(f"{nombre} - ingenua (solo arousal)",
                           f"{acc_i:.3f}", f"{null_i:.3f}", f"{p_i:.4f}{marca_i}"))

        # descomposicion 2x2: cual de los dos proxies (z, estimulo) es
        # ruidoso, para atribuir cualquier residuo a uno, al otro, o a
        # ambos -- en vez de mezclarlos en una sola condicion "realista".
        variantes = [
            ("z ruidoso, s verdadero",  obsZ, estimulo),
            ("z verdadero, s ruidoso",  z,    obs_estimulo),
            ("ambos ruidosos (realista)", obsZ, obs_estimulo),
            ("oraculo (ambos verdaderos)", z,  estimulo),
        ]
        for etiqueta, z_usado, estim_usado in variantes:
            dummies = dummies_categoria(estim_usado, n_estimulos)
            covars = np.column_stack([z_usado, dummies])
            acc, null, p = prueba_identificabilidad(e, l, covars, estim_usado, semilla=2)
            marca = "*" if p < 0.05 else " "
            print(fila.format(f"{nombre} - condicionada ({etiqueta})",
                               f"{acc:.3f}", f"{null:.3f}", f"{p:.4f}{marca}"))
        print()

    print("  -> La prueba INGENUA da positivo bajo Tipo A cuando hay estimulo")
    print("     compartido: identifica el estimulo, no el linaje causal.")
    print("     Condicionar por categoria de estimulo lo corrige. La")
    print("     descomposicion 2x2 muestra de donde viene cualquier residuo")
    print("     bajo Tipo A: si 'ambos verdaderos' (oraculo) esta limpio pero")
    print("     'ambos ruidosos' no, el residuo es enteramente de calidad de")
    print("     proxy -- no una falla de la logica de condicionamiento --, y")
    print("     las filas intermedias (un proxy ruidoso, el otro no) dicen")
    print("     cual de los dos confusores pesa mas en ese residuo.")


# =====================================================================
# BLOQUE 6: CURVA DE PODER (Tipo B) Y DE FALSOS POSITIVOS (Tipo A)
#           DEL TEST YA CORREGIDO POR ESTIMULO, EN FUNCION DE N
# =====================================================================
#
# Bloque 5 mostro un solo resultado por celda (una semilla). Eso alcanza
# para confirmar el MECANISMO del sesgo por estimulo y su correccion,
# pero no para dos preguntas distintas y mas importantes:
#   (a) SENSIBILIDAD: la version condicionada por estimulo colapsa la
#       varianza discriminante a la fluctuacion idiosincratica -- ¿eso
#       le cuesta tanto poder que deja de detectar Tipo B verdadero?
#   (b) TASA DE FALSOS POSITIVOS EN FUNCION DE N: el hallazgo central
#       de simulacion_acoplamiento_fase_temprana_tardia.py fue que, con
#       proxy ruidoso, la FPR de la prueba de MAGNITUD crece con N en
#       vez de estabilizarse. ¿La correccion por estimulo resuelve ese
#       problema, o solo tapa el del estimulo y deja vivo el del
#       proxy de arousal, que reaparece mas confiado (mas "signif-
#       icativo") cuantos mas ensayos se junten?
#
# Un solo p-valor (Bloque 5) no puede contestar (b): 0.037 es
# perfectamente compatible con una FPR nominal ~0.05 y una corrida con
# mala suerte. Hace falta la tasa de deteccion sobre MUCHAS repeticiones
# sinteticas, para cada N -- eso es lo que sigue.

def curva_fpr_y_poder_condicionada(n_trials_grid, n_estimulos=5,
                                    p_error_estimulo=0.08,
                                    n_repeticiones=80, n_perm=150,
                                    semilla=0):
    """
    Para cada N en n_trials_grid, corre n_repeticiones simulaciones
    independientes de la version YA CONDICIONADA por estimulo
    (regresion + pool restringido a la misma categoria) y devuelve:
        fpr_proxy   -- FPR bajo Tipo A, AMBOS confusores con proxy
                       ruidoso (arousal Y estimulo -- el caso realista)
        fpr_oraculo -- FPR bajo Tipo A, AMBOS confusores verdaderos
        poder_proxy -- poder bajo Tipo B, ambos confusores con proxy
                       ruidoso
    "Tasa" = proporcion de repeticiones con p<0.05. La version anterior
    de esta funcion solo hacia ruidoso el proxy de arousal y dejaba el
    estimulo siempre exacto -- no coincidia con lo que describe el
    manuscrito (proxies ruidosos para ambos confusores). Corregido.
    """
    rng_base = np.random.default_rng(semilla)
    filas = []
    for n in n_trials_grid:
        conteos = {"A_proxy": 0, "A_oraculo": 0, "B_proxy": 0}
        for rep in range(n_repeticiones):
            s = int(rng_base.integers(0, 2**31 - 1))
            rng_ruido = np.random.default_rng(s + 500)

            e, l, obsZ, z, estimulo, Q = generar_ensayos_con_estimulo(
                n, n_estimulos=n_estimulos, tipo="A", semilla=s)
            obs_estimulo = estimulo_ruidoso(estimulo, n_estimulos, p_error_estimulo, rng_ruido)

            dummies_proxy = dummies_categoria(obs_estimulo, n_estimulos)
            _, _, p_proxy = prueba_identificabilidad(
                e, l, np.column_stack([obsZ, dummies_proxy]), obs_estimulo,
                n_perm=n_perm, semilla=s + 1)
            if not np.isnan(p_proxy) and p_proxy < 0.05:
                conteos["A_proxy"] += 1

            dummies_oraculo = dummies_categoria(estimulo, n_estimulos)
            _, _, p_oraculo = prueba_identificabilidad(
                e, l, np.column_stack([z, dummies_oraculo]), estimulo,
                n_perm=n_perm, semilla=s + 1)
            if not np.isnan(p_oraculo) and p_oraculo < 0.05:
                conteos["A_oraculo"] += 1

            eB, lB, obsZB, zB, estimuloB, QB = generar_ensayos_con_estimulo(
                n, n_estimulos=n_estimulos, tipo="B", semilla=s + 2)
            obs_estimuloB = estimulo_ruidoso(estimuloB, n_estimulos, p_error_estimulo, rng_ruido)
            dummiesB_proxy = dummies_categoria(obs_estimuloB, n_estimulos)
            _, _, p_B = prueba_identificabilidad(
                eB, lB, np.column_stack([obsZB, dummiesB_proxy]), obs_estimuloB,
                n_perm=n_perm, semilla=s + 3)
            if not np.isnan(p_B) and p_B < 0.05:
                conteos["B_proxy"] += 1

        filas.append({
            "n": n,
            "fpr_proxy": conteos["A_proxy"] / n_repeticiones,
            "fpr_oraculo": conteos["A_oraculo"] / n_repeticiones,
            "poder_proxy": conteos["B_proxy"] / n_repeticiones,
        })
    return filas


def reportar_curva_fpr_y_poder():
    print("\n" + "=" * 72)
    print("  CURVA FPR-vs-N Y PODER-vs-N (test YA CONDICIONADO por estimulo,")
    print("  AMBOS confusores -- arousal Y estimulo -- con proxy ruidoso)")
    print("=" * 72)
    grid = [400, 800, 1600, 2400]
    print(f"\n  ({len(grid)} valores de N, 80 repeticiones c/u -- unos minutos)\n")
    filas = curva_fpr_y_poder_condicionada(grid, n_repeticiones=80, n_perm=150, semilla=7)

    encabezado = "  {:>8s} {:>16s} {:>16s} {:>16s}"
    print(encabezado.format("N", "FPR (proxy)", "FPR (oraculo)", "Poder Tipo B"))
    print("  " + "-" * 60)
    for f in filas:
        print("  {:>8d} {:>16.3f} {:>16.3f} {:>16.3f}".format(
            f["n"], f["fpr_proxy"], f["fpr_oraculo"], f["poder_proxy"]))

    fpr_proxy_vals = [f["fpr_proxy"] for f in filas]
    creciendo = fpr_proxy_vals[-1] > fpr_proxy_vals[0] + 0.05
    print()
    if creciendo:
        print("  -> La FPR con AMBOS proxies ruidosos SIGUE creciendo con N incluso")
        print("     despues de condicionar por estimulo: la correccion no evita que")
        print("     el proxy imperfecto (de cualquiera de los dos confusores) se")
        print("     comporte igual que en la prueba de magnitud. Mismo remedio de")
        print("     siempre: mejorar el proxy, no juntar mas datos.")
    else:
        print("  -> La FPR con AMBOS proxies ruidosos se mantiene aproximadamente")
        print("     estable alrededor de alpha=0.05 en este rango de N -- a diferencia")
        print("     de la prueba de magnitud, aqui condicionar por estimulo parece")
        print("     ser suficiente para no heredar el problema de crecimiento con N,")
        print("     incluso con los dos confusores medidos de forma imperfecta.")
    print("  La FPR con AMBOS confusores verdaderos (oraculo) deberia rondar 0.05")
    print("  en todos los casos -- es el punto de referencia para separar sesgo por")
    print("  proxy de cualquier otro problema.")
    print("  El poder de Tipo B deberia subir con N si el test tiene sensibilidad")
    print("  real tras condicionar -- un poder que no despega indicaria que la")
    print("  correccion por estimulo dejo tan poca señal que el test ya no sirve.")


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
        estrato = estratificar(obsZ, n_estratos=5)
        acc_obs, acc_null, p = prueba_identificabilidad(e, l, obsZ, estrato, semilla=2)
        marca = "*" if p < 0.05 else " "
        print(fila2.format(nombre, f"{acc_obs:.3f}", f"{acc_null:.3f}", f"{p:.4f}{marca}"))

    print("\n  -> Donde la magnitud no ve nada, la identificabilidad SI discrimina:")
    print("     top-1 al nivel del azar bajo Tipo A, muy por encima bajo Tipo B.")
    print("     IMPORTANTE -- esta demostracion asume ESTIMULO UNICO O IDENTICO")
    print("     repetido en todos los ensayos: no hay ninguna estructura de")
    print("     categoria de estimulo compartida entre e y l. Bloque 5 muestra")
    print("     por que esa suposicion es load-bearing, no un detalle.")

    demostrar_confusion_por_estimulo()
    reportar_curva_fpr_y_poder()

    print("\n" + "=" * 72)
    print("  LECTURA PRACTICA")
    print("=" * 72)
    print("  Las dos pruebas (magnitud, identificabilidad) no son redundantes.")
    print("  Pero la identificabilidad por si sola tampoco alcanza si el")
    print("  paradigma varia el estimulo entre ensayos: hay que condicionar por")
    print("  categoria de estimulo (Bloque 5), no solo por arousal, o el test")
    print("  identifica el estimulo en vez del linaje causal. La pregunta que")
    print("  decide todo esto ANTES de tocar datos: en el paradigma real, ¿los")
    print("  ensayos varian el estimulo, o repiten el mismo? Si varian, usar la")
    print("  version condicionada de Bloque 5 -- y aceptar que el poder baja,")
    print("  porque sólo queda la fluctuacion idiosincratica como señal.")
    print("=" * 72)


if __name__ == "__main__":
    main()
