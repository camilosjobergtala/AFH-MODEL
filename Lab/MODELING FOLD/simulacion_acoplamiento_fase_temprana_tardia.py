"""
=======================================================================
SIMULACION DE PODER ESTADISTICO: ACOPLAMIENTO ENSAYO-A-ENSAYO ENTRE
UNA FASE TEMPRANA TALAMO->CORTEZA Y UNA FASE TARDIA CORTEZA->TALAMO
=======================================================================

CONTEXTO

Este script simula un analisis propuesto (por el autor de AFH, en
correspondencia propia no reproducida aqui) para distinguir, dentro de
un paradigma de percepcion consciente con registro intracraneal
organizado por ensayos, si una fase tardia de retorno cortico-talamico
refleja elaboracion cortical de la fase temprana talamo-cortical
inmediatamente precedente (convergencia reflexiva, Tipo B) o si surge
por una via causalmente independiente que solo comparte factores
generales del ensayo -- arousal, enganche atencional, dinamica
pre-respuesta (convergencia heterogenea, Tipo A).

Es exactamente la distincion formalizada en
AFH_modelo_matematico_pliegue.md (Sec. 3-4, Prediccion 6), trasladada
de "senal continua con un unico retardo fijo tau" (el caso que
simulan 0.4.py y AFH_modelo_N_nodos.py) a "un par de metricas resumen
por ensayo, con la fase tardia ocurriendo despues de la temprana pero
sin retardo continuo fijo" -- el formato en que llega un registro
intracraneal real organizado por ensayos.

Las escalas temporales usadas mas abajo (~600 ms de fase temprana,
~400 ms de fase tardia antes de la respuesta) son ILUSTRATIVAS -- del
orden de magnitud razonable para una tarea de percepcion con respuesta
sacadica -- no se presentan como parametros confidenciales de ningun
conjunto de datos en particular.

LA COMPLICACION CENTRAL

Ambas fases de un mismo ensayo comparten factores generales del
ensayo (arousal, enganche, dinamica pre-respuesta). Por eso, una
CORRELACION CRUDA entre la fuerza de la fase temprana y la fuerza de
la fase tardia NO discrimina Tipo B de Tipo A: ambos modelos generan
correlacion cruda positiva si comparten ese confusor. Lo que
discrimina es si la correlacion SOBREVIVE a controlar por un proxy
observable del confusor (p.ej. pupila pre-estimulo, potencia basal,
tiempo de reaccion) -- de ahi que la prueba correcta sea una
CORRELACION PARCIAL, no la correlacion cruda. Este script hace ese
punto explicito y cuantifica cuanto "sesgo residual" queda cuando el
proxy del confusor es, como en la practica, ruidoso e imperfecto.

QUE HACE:
    1. Genera datos sinteticos ensayo-a-ensayo bajo dos modelos
       generativos (Tipo B: acoplamiento genuino mas alla del
       confusor compartido; Tipo A: solo el confusor compartido).
    2. Calcula correlacion cruda y parcial (controlando por un proxy
       ruidoso del confusor, como seria en la practica) en cada caso,
       y las compara contra la correlacion parcial "oraculo"
       (controlando por el confusor verdadero, no observable
       realmente -- solo calculable en la simulacion).
    3. Corre un analisis de poder: para una grilla de N ensayos y de
       fuerza de acoplamiento kappa, estima la tasa de deteccion
       (Tipo B) y la tasa de falsos positivos (Tipo A) de la prueba
       de correlacion parcial a alpha=0.05.
    4. Reporta cuantos ensayos harian falta para que la prueba tenga
       poder razonable -- la pregunta practica detras de proponer
       este analisis sobre datos ya registrados.

INSTALACION: pip install numpy scipy
CORRER:      python "simulacion_acoplamiento_fase_temprana_tardia.py"
TIEMPO:      unos segundos
=======================================================================
"""

import numpy as np
from scipy import stats


# =====================================================================
# BLOQUE 1: GENERACION DE ENSAYOS
# =====================================================================

def simular_ensayos(n_trials, kappa, a_E=0.0, b_E=0.6, sigma_E=1.0,
                     a_L=0.0, b_L=0.6, sigma_L=1.0, sigma_meas=0.6,
                     semilla=None):
    """
    Genera n_trials ensayos de:

        z_i      : confusor latente compartido (arousal/enganche),
                   no observable directamente.
        obsZ_i   : proxy observable y RUIDOSO de z_i (p.ej. pupila
                   pre-estimulo o potencia basal) -- lo unico con lo
                   que un analisis real podria controlar.
        E_i      : fuerza de la fase temprana talamo->PFC
                   (p.ej. transferencia de fase, ~600 ms post-estimulo).
        eps_E_i  : componente idiosincratico de E_i, propio de ese
                   ensayo, no explicado por el confusor -- lo que
                   Tipo B predice que "vuelve" en la fase tardia.
        L_i      : fuerza de la fase tardia PFC->talamo
                   (~400 ms antes de la respuesta).

    kappa = 0     -> Tipo A: L comparte solo el confusor con E.
    kappa > 0     -> Tipo B: L ademas rastrea el componente
                     idiosincratico de E (eps_E), mas alla de lo que
                     el confusor por si solo explicaria -- la firma
                     de un retorno que elabora la actividad propia
                     precedente, no una via paralela independiente.
    """
    rng = np.random.default_rng(semilla)
    z = rng.normal(size=n_trials)
    obsZ = z + sigma_meas * rng.normal(size=n_trials)

    eps_E = sigma_E * rng.normal(size=n_trials)
    E = a_E + b_E * z + eps_E

    eps_L = sigma_L * rng.normal(size=n_trials)
    L = a_L + b_L * z + kappa * eps_E + eps_L

    return E, L, obsZ, z


# =====================================================================
# BLOQUE 2: CORRELACION PARCIAL (control de un solo confusor)
# =====================================================================

def correlacion_parcial(x, y, control):
    """
    Correlacion parcial de x e y controlando por una sola variable,
    via la formula clasica a partir de las tres correlaciones
    simples, mas su prueba t (df = n-3).
    """
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


# =====================================================================
# BLOQUE 3: UNA CORRIDA ILUSTRATIVA (Tipo B vs Tipo A, lado a lado)
# =====================================================================

def analisis_ilustrativo(n_trials=150, kappa_B=0.45, semilla=0):
    print(f"  N ensayos = {n_trials}, kappa (Tipo B) = {kappa_B}\n")
    fila = "  {:<28s} {:>10s} {:>10s} {:>10s}"
    print(fila.format("", "r crudo", "r parcial", "r parcial"))
    print(fila.format("", "(E,L)", "(proxy)", "(oraculo)"))
    print("  " + "-" * 62)

    for kappa, nombre in [(0.0, "Tipo A (via independiente)"),
                           (kappa_B, "Tipo B (retorno reflexivo)")]:
        E, L, obsZ, z = simular_ensayos(n_trials, kappa, semilla=semilla)
        r_raw, p_raw = stats.pearsonr(E, L)
        r_part, p_part = correlacion_parcial(E, L, obsZ)
        r_orac, p_orac = correlacion_parcial(E, L, z)
        marca_part = "*" if p_part < 0.05 else " "
        marca_orac = "*" if p_orac < 0.05 else " "
        print(f"  {nombre:<28s} {r_raw:>9.3f}  {r_part:>8.3f}{marca_part} {r_orac:>8.3f}{marca_orac}")

    print("\n  (* p<0.05, prueba t sobre la correlacion parcial, df=N-3)")
    print("  Nota: la correlacion CRUDA es positiva en ambos casos -- por eso")
    print("  no discrimina. La parcial con el proxy ruidoso SI discrimina, pero")
    print("  con un sesgo residual visible frente a la parcial 'oraculo' (control")
    print("  perfecto del confusor, no disponible en la practica real).")


# =====================================================================
# BLOQUE 4: ANALISIS DE POTENCIA
# =====================================================================

def poder_deteccion(n_trials, kappa, n_montecarlo=800, alpha=0.05,
                     usar_proxy=True, semilla=0):
    """
    Proporcion de repeticiones Monte Carlo en que la correlacion
    parcial (E,L | control) es significativa y de signo positivo --
    "deteccion" de acoplamiento reflexivo bajo el modelo generado.

    Con kappa=0 (Tipo A) esto estima la TASA DE FALSOS POSITIVOS de
    la prueba; con kappa>0 (Tipo B) estima el PODER.
    """
    rng_base = np.random.default_rng(semilla)
    detecciones = 0
    for rep in range(n_montecarlo):
        s = int(rng_base.integers(0, 2**31 - 1))
        E, L, obsZ, z = simular_ensayos(n_trials, kappa, semilla=s)
        control = obsZ if usar_proxy else z
        r_part, p_part = correlacion_parcial(E, L, control)
        if not np.isnan(p_part) and p_part < alpha and r_part > 0:
            detecciones += 1
    return detecciones / n_montecarlo


def tabla_de_poder(n_trials_grid, kappas, n_montecarlo=600, semilla=0):
    print(f"  (proxy ruidoso del confusor, {n_montecarlo} repeticiones Monte Carlo por celda)\n")
    encabezado = "  {:>10s}" + "".join(f" | kappa={{:<5.2f}}".format(k)[:14] for k in kappas)
    print("  " + "N ensayos".rjust(10) + "".join(f" | k={k:<5.2f}" for k in kappas))
    print("  " + "-" * (12 + 12 * len(kappas)))
    for n in n_trials_grid:
        fila = f"  {n:>10d}"
        for k in kappas:
            p = poder_deteccion(n, k, n_montecarlo=n_montecarlo, semilla=semilla)
            fila += f" | {p:>9.2f}"
        print(fila)
    print()
    print("  kappa=0.00 es Tipo A: esa columna es la TASA DE FALSOS POSITIVOS")
    print("  (deberia rondar alpha=0.05; si es mayor, el proxy del confusor")
    print("  deja pasar sesgo residual -- ver analisis_ilustrativo()).")
    print("  Las columnas kappa>0 son el PODER para detectar Tipo B a ese")
    print("  tamano de efecto y ese N de ensayos.")


# =====================================================================
# BLOQUE 5: DEMOSTRACION
# =====================================================================

def main():
    print("=" * 72)
    print("  PARADIGMA FASE TEMPRANA (talamo->PFC) / FASE TARDIA (PFC->talamo)")
    print("  Prueba de acoplamiento ensayo-a-ensayo: Tipo B vs. Tipo A")
    print("=" * 72)

    print("\n[1] Corrida ilustrativa (N=150, un tamano de efecto moderado)")
    analisis_ilustrativo(n_trials=150, kappa_B=0.45, semilla=1)

    print("\n[2] Tasa de falsos positivos vs. calidad del proxy del confusor")
    print("    (N=150, Tipo A puro -- kappa=0 -- variando sigma_meas)")
    for sigma_meas in [0.0, 0.3, 0.6, 1.2]:
        rng = np.random.default_rng(2)
        detecciones = 0
        reps = 500
        for rep in range(reps):
            s = int(rng.integers(0, 2**31 - 1))
            E, L, obsZ, z = simular_ensayos(150, 0.0, sigma_meas=sigma_meas, semilla=s)
            r_part, p_part = correlacion_parcial(E, L, obsZ)
            if not np.isnan(p_part) and p_part < 0.05 and r_part > 0:
                detecciones += 1
        calidad = "perfecto" if sigma_meas == 0 else f"ruido={sigma_meas}"
        print(f"    proxy {calidad:<12s} -> falsos positivos = {detecciones/reps:.3f}"
              f"  (nominal esperado: 0.05)")

    print("\n[3] Poder de deteccion: N de ensayos x fuerza de acoplamiento")
    tabla_de_poder(
        n_trials_grid=[50, 100, 150, 200, 300],
        kappas=[0.0, 0.15, 0.30, 0.45],
        n_montecarlo=500,
        semilla=3,
    )

    print("=" * 72)
    print("  LECTURA PRACTICA")
    print("=" * 72)
    print("  El hallazgo mas importante de [2]-[3] no es el poder en si, sino")
    print("  que la tasa de FALSOS POSITIVOS crece con el ruido del proxy del")
    print("  confusor -- Y TAMBIEN CRECE CON N. Esto es contraintuitivo: mas")
    print("  ensayos no 'diluye' el sesgo de un control imperfecto, lo vuelve")
    print("  mas facil de declarar significativo. Una correlacion parcial con")
    print("  un proxy ruidoso de arousal, corrida sobre muchos ensayos, puede")
    print("  parecer una deteccion solida de Tipo B y ser enteramente un")
    print("  artefacto de deconfusion incompleta. Implicacion practica: el")
    print("  paso critico no es juntar mas ensayos, es tener un proxy de")
    print("  arousal/enganche lo mas fiel posible (o modelar el error de")
    print("  medicion explicitamente, p.ej. regresion con errores en variables)")
    print("  antes de interpretar cualquier correlacion parcial como evidencia")
    print("  de retorno reflexivo. El numero de ensayos necesario para poder")
    print("  razonable (~0.8) depende fuertemente de kappa; agregar sujetos")
    print("  (modelo jerarquico) en vez de agrupar ensayos crudos seria el")
    print("  siguiente paso natural, no simulado aqui.")
    print("=" * 72)


if __name__ == "__main__":
    main()
