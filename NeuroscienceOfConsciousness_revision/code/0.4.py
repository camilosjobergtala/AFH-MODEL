"""
=======================================================================
MODELO AFH v0.5 - Autopsychic Fold and Horizon
Dr. Camilo Alejandro Sjoberg Tala, M.D.
=======================================================================

CAMBIOS RESPECTO A v0.4:

DECISION METODOLOGICA:
    nabla normalizado fue abandonado.
    Razon: autocorrelacion de y > nabla bruto en este sistema,
    lo que produce nabla normalizado negativo. La correccion
    overcorrige y pierde la señal real.

NUEVA ARQUITECTURA DE METRICAS:

    METRICA 1: nabla bruto
        Rol: descriptiva, muestra acoplamiento general.
        Limitacion conocida: contaminada por autocorrelacion de y.
        Uso: visualizacion, barridos, mapa beta-tau.

    METRICA 2 (CENTRAL): Granger especifico en tau
        Rol: prueba causal principal.
        Pregunta: y(t-tau) predice x(t) mas alla del pasado de x?
        Resultado v0.4: B F=189, A F=1.59. Discriminacion perfecta.
        Uso: prueba principal de autorreferencialidad.

    METRICA 3: Funcion de correlacion cruzada completa
        Rol: muestra especificidad del pico en tau.
        Uso: figura central del paper.

    METRICA 4: Estabilidad temporal de nabla
        Rol: demuestra que el acoplamiento es sostenido.
        Uso: diferencia regimen dinamico de pico transitorio.

ESTRUCTURA DE FIGURAS:
    F1: Series temporales beta bajo vs alto
    F2: Funcion correlacion cruzada (pico en tau) por modo
    F3: Granger especifico — la figura mas importante
    F4: Barrido beta (nabla bruto + Granger)
    F5: Barrido tau (nabla bruto, pico en tau correcto)
    F6: Mapa beta x tau
    F7: Estabilidad temporal por ventanas
    F8: Lesion ILN

INSTALACION: pip install numpy matplotlib scipy
CORRER:      python afh_model_v05.py
TIEMPO:      25-35 minutos
=======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import lstsq


# =====================================================================
# BLOQUE 1: SIMULACION
# =====================================================================

def simular_AFH(
    T=10000, dt=0.5, tau_x=10.0, tau_y=10.0,
    alpha=1.0, beta=1.0, tau=500.0,
    sigma_F=0.2, mu_F=0.1, modo="B", semilla=None
):
    """
    Sistema talamo-corteza con retardo.

    dx/dt = [-x + tanh(alpha*F(t) + beta*retardada)] / tau_x
    dy/dt = [-y + tanh(x(t))]                        / tau_y

    Tipo B:    retardada = y(t-tau)  eco propio
    Tipo A:    retardada = E(t-tau)  ruido coloreado externo
    Ninguno:   retardada = 0
    """
    rng   = np.random.default_rng(semilla)
    n     = int(T / dt)
    k_tau = int(tau / dt)
    x     = np.zeros(n)
    y     = np.zeros(n)
    t     = np.arange(n) * dt
    F     = mu_F + sigma_F * rng.normal(size=n)

    if modo == "A":
        E = _ruido_coloreado(n, theta=0.05, sigma=sigma_F,
                             mu=mu_F, seed=rng.integers(int(1e9)))
    else:
        E = None

    x[0] = 0.01 * rng.normal()
    y[0] = 0.01 * rng.normal()

    for i in range(1, n):
        if i - k_tau >= 0:
            if   modo == "B":   retardada = y[i - k_tau]
            elif modo == "A":   retardada = E[i - k_tau]
            else:               retardada = 0.0
        else:
            retardada = 0.0

        x[i] = x[i-1] + dt * (-x[i-1] + np.tanh(alpha*F[i] + beta*retardada)) / tau_x
        y[i] = y[i-1] + dt * (-y[i-1] + np.tanh(x[i-1])) / tau_y

    return x, y, t


def simular_AFH_lesion(
    T=15000, dt=0.5, tau_x=10.0, tau_y=10.0,
    alpha=1.0, beta=1.5, tau=500.0,
    sigma_F=0.2, mu_F=0.1, t_lesion=7500, semilla=None
):
    """Beta -> 0 en t_lesion. Simula lesion ILN."""
    rng      = np.random.default_rng(semilla)
    n        = int(T / dt)
    k_tau    = int(tau / dt)
    n_lesion = int(t_lesion / dt)
    x        = np.zeros(n)
    y        = np.zeros(n)
    t        = np.arange(n) * dt
    F        = mu_F + sigma_F * rng.normal(size=n)
    x[0]     = 0.01 * rng.normal()
    y[0]     = 0.01 * rng.normal()

    for i in range(1, n):
        beta_i    = beta if i < n_lesion else 0.0
        retardada = y[i - k_tau] if i - k_tau >= 0 else 0.0
        x[i] = x[i-1] + dt * (-x[i-1] + np.tanh(alpha*F[i] + beta_i*retardada)) / tau_x
        y[i] = y[i-1] + dt * (-y[i-1] + np.tanh(x[i-1])) / tau_y

    return x, y, t


def _ruido_coloreado(n, theta=0.05, sigma=0.2, mu=0.0, seed=None):
    rng = np.random.default_rng(seed)
    E   = np.zeros(n)
    for i in range(1, n):
        E[i] = E[i-1] + theta*(mu - E[i-1]) + sigma*rng.normal()
    if E.std() > 1e-10:
        E = (E - E.mean()) / E.std() * sigma + mu
    return E


# =====================================================================
# BLOQUE 2: METRICA 1 — nabla bruto
# =====================================================================

def calcular_nabla(x, y, tau, dt, descartar_ms=2000):
    """
    nabla = corr[x(t), y(t-tau)]

    Metrica descriptiva del acoplamiento retardado.
    Limitacion conocida: contaminada por autocorrelacion de y.
    Se usa para visualizacion y barridos, no como prueba principal.
    """
    k = int(tau / dt)
    d = int(descartar_ms / dt)
    xe = x[d+k:]
    yr = y[d:-k] if k > 0 else y[d:]
    n  = min(len(xe), len(yr))
    if n < 20: return np.nan
    xe = xe[:n]; yr = yr[:n]
    if np.std(xe) < 1e-10 or np.std(yr) < 1e-10: return np.nan
    return np.corrcoef(xe, yr)[0, 1]


def correlacion_por_lags(x, y, dt, lag_min=10, lag_max=2000,
                          paso=25, descartar_ms=2000):
    """Funcion de correlacion cruzada para rango de retardos."""
    lags  = np.arange(lag_min, lag_max + paso, paso)
    corrs = []
    d     = int(descartar_ms / dt)
    for lag in lags:
        k = int(lag / dt)
        if k <= 0 or d+k >= len(x):
            corrs.append(np.nan); continue
        xe = x[d+k:]; yl = y[d:-k]
        n  = min(len(xe), len(yl))
        if n < 20: corrs.append(np.nan); continue
        xe = xe[:n]; yl = yl[:n]
        if np.std(xe) < 1e-10 or np.std(yl) < 1e-10:
            corrs.append(np.nan)
        else:
            corrs.append(np.corrcoef(xe, yl)[0, 1])
    return lags, np.array(corrs)


def calcular_z_surrogate(x, y, tau, dt, n_surr=80, descartar_ms=2000):
    """Z-nabla contra surrogate de fase (mas conservador)."""
    rng  = np.random.default_rng(42)
    nr   = calcular_nabla(x, y, tau, dt, descartar_ms)
    n    = len(y)
    surr = []
    for _ in range(n_surr):
        Yf    = np.fft.rfft(y)
        nf    = len(Yf)
        fases = rng.uniform(0, 2*np.pi, nf)
        fases[0] = 0
        if n % 2 == 0: fases[-1] = 0
        ys = np.fft.irfft(np.abs(Yf) * np.exp(1j*fases), n=n)
        v  = calcular_nabla(x, ys, tau, dt, descartar_ms)
        if not np.isnan(v): surr.append(v)
    if len(surr) < 5: return np.nan, np.nan, np.nan, surr
    mu  = np.mean(surr); sd = np.std(surr)
    z   = (nr - mu) / sd if sd > 1e-10 else np.nan
    return z, mu, sd, surr


def nabla_por_ventanas(x, y, tau, dt, ventana_ms=1000, descartar_ms=2000):
    """nabla en ventanas temporales sucesivas."""
    d   = int(descartar_ms / dt)
    w   = int(ventana_ms / dt)
    k   = int(tau / dt)
    ini = d + k
    vals = []
    while ini + w < len(x):
        xw = x[ini:ini+w]; yw = y[ini-k:ini-k+w]
        if np.std(xw) > 1e-10 and np.std(yw) > 1e-10 and len(xw)==len(yw):
            vals.append(np.corrcoef(xw, yw)[0, 1])
        ini += w
    vals = np.array(vals)
    prop = np.mean(vals > 0.3) if len(vals) > 0 else np.nan
    return vals, prop


# =====================================================================
# BLOQUE 3: METRICA 2 (CENTRAL) — Granger especifico en tau
# =====================================================================

def granger_en_tau(x, y, dt, tau, orden=5, descartar_ms=2000):
    """
    Granger especifico: solo y(t-tau) como predictor adicional de x(t).

    Modelo restringido:    x(t) ~ x(t-1..p)
    Modelo no restringido: x(t) ~ x(t-1..p) + y(t-tau)

    Pregunta concreta: el pasado cortical especifico en tau
    predice la actividad talamica actual, mas alla de lo que
    el propio pasado del talamo ya predice?

    En AFH: esto debe ocurrir en Tipo B (eco propio),
    no en Tipo A (señal externa).

    RETORNA:
        F:       estadistico F (mayor = mas causalidad en tau)
        p_val:   p-valor (< 0.05 = significativo)
        delta_R2: mejora en varianza explicada al agregar y(t-tau)
    """
    d     = int(descartar_ms / dt)
    p     = orden
    k_tau = int(tau / dt)
    xs    = x[d:]; ys = y[d:]
    n     = len(xs)
    ini   = max(p, k_tau)

    if n - ini < 3*p:
        return np.nan, np.nan, np.nan

    Xdep = xs[ini:]
    T    = len(Xdep)

    # Modelo restringido: lags de x
    Xr = np.column_stack([np.ones(T),
                          *[xs[ini-i-1:n-i-1] for i in range(p)]])
    cr, _, _, _ = lstsq(Xr, Xdep)
    rr   = Xdep - Xr @ cr
    RSSr = np.sum(rr**2)
    SST  = np.sum((Xdep - Xdep.mean())**2)
    R2r  = 1 - RSSr/SST if SST > 0 else np.nan

    # Modelo no restringido: lags de x + y(t-tau)
    ytau = ys[ini-k_tau:n-k_tau]
    if len(ytau) != T:
        return np.nan, np.nan, np.nan

    Xnr = np.column_stack([Xr, ytau])
    cnr, _, _, _ = lstsq(Xnr, Xdep)
    rnr   = Xdep - Xnr @ cnr
    RSSnr = np.sum(rnr**2)
    R2nr  = 1 - RSSnr/SST if SST > 0 else np.nan

    df1 = 1; df2 = T - p - 2
    if df2 <= 0 or RSSnr < 1e-15:
        return np.nan, np.nan, np.nan

    F     = ((RSSr - RSSnr)/df1) / (RSSnr/df2)
    p_val = 1 - stats.f.cdf(F, df1, df2)
    dR2   = R2nr - R2r

    return F, p_val, dR2


def comparar_granger_modos(beta=1.5, tau=500, repeticiones=15, orden=5):
    """Granger especifico comparado entre modos."""
    print(f"  [GRANGER tau={tau}ms orden={orden}] {repeticiones} rep.")
    res = {}
    for modo in ["B", "A", "ninguno"]:
        Fs=[]; pv=[]; dr=[]
        for rep in range(repeticiones):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo=modo)
            F, p, d = granger_en_tau(x, y, dt=0.5, tau=tau, orden=orden)
            if not np.isnan(F):
                Fs.append(F); pv.append(p); dr.append(d)
        res[modo] = {
            "F_med"    : np.mean(Fs)   if Fs else np.nan,
            "F_std"    : np.std(Fs)    if Fs else np.nan,
            "p_med"    : np.mean(pv)   if pv else np.nan,
            "prop_sig" : np.mean(np.array(pv)<0.05) if pv else np.nan,
            "dR2_med"  : np.mean(dr)   if dr else np.nan,
        }
        label = {"B":"Tipo B","A":"Tipo A","ninguno":"Sin feedback"}[modo]
        print(f"    {label}: F={res[modo]['F_med']:.1f}"
              f"  p={res[modo]['p_med']:.4f}"
              f"  prop_sig={res[modo]['prop_sig']:.2f}"
              f"  dR2={res[modo]['dR2_med']:.5f}")
    return res


def granger_barrido_beta(betas, tau=500, repeticiones=8, orden=5):
    """Granger especifico para cada beta. Busca H* en terminos causales."""
    print(f"  [GRANGER BARRIDO beta] {len(betas)} val x {repeticiones} rep.")
    res = []
    for i, beta in enumerate(betas):
        print(f"    beta={beta:.2f} ({i+1}/{len(betas)})   ", end="\r")
        Fs=[]; pv=[]
        for rep in range(repeticiones):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo="B")
            F, p, _ = granger_en_tau(x, y, dt=0.5, tau=tau, orden=orden)
            if not np.isnan(F): Fs.append(F); pv.append(p)
        res.append({
            "beta"     : beta,
            "F_med"    : np.mean(Fs)   if Fs else np.nan,
            "prop_sig" : np.mean(np.array(pv)<0.05) if pv else np.nan,
            "nabla_med": np.nan,  # se llenara en pipeline
        })
    print()
    return res


# =====================================================================
# BLOQUE 4: BARRIDOS nabla
# =====================================================================

def barrer_beta_nabla(betas, tau=500, repeticiones=8):
    print(f"  [BARRIDO nabla beta] tau={tau}ms {repeticiones} rep.")
    res = []
    for i, beta in enumerate(betas):
        print(f"    beta={beta:.2f} ({i+1}/{len(betas)})   ", end="\r")
        vals = []
        for rep in range(repeticiones):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo="B")
            v = calcular_nabla(x, y, tau=tau, dt=0.5)
            if not np.isnan(v): vals.append(v)
        res.append({"beta":beta,
                    "med":np.mean(vals) if vals else np.nan,
                    "std":np.std(vals)  if vals else np.nan})
    print()
    return res


def barrer_tau_nabla(taus, beta=1.5, repeticiones=8):
    print(f"  [BARRIDO nabla tau] beta={beta} {repeticiones} rep.")
    res = []
    for i, tau in enumerate(taus):
        print(f"    tau={tau}ms ({i+1}/{len(taus)})   ", end="\r")
        vals = []
        for rep in range(repeticiones):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo="B")
            v = calcular_nabla(x, y, tau=tau, dt=0.5)
            if not np.isnan(v): vals.append(v)
        res.append({"tau":tau,
                    "med":np.mean(vals) if vals else np.nan,
                    "std":np.std(vals)  if vals else np.nan})
    print()
    return res


def mapa_beta_tau(betas, taus, repeticiones=5):
    print(f"  [MAPA beta x tau] {len(betas)}x{len(taus)}x{repeticiones}")
    M   = np.full((len(betas), len(taus)), np.nan)
    tot = len(betas)*len(taus); cnt = 0
    for i, beta in enumerate(betas):
        for j, tau in enumerate(taus):
            cnt += 1
            print(f"    {cnt}/{tot}   ", end="\r")
            vals = []
            for rep in range(repeticiones):
                x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo="B")
                v = calcular_nabla(x, y, tau=tau, dt=0.5)
                if not np.isnan(v): vals.append(v)
            M[i,j] = np.mean(vals) if vals else np.nan
    print()
    return M


# =====================================================================
# BLOQUE 5: FIGURAS
# =====================================================================

def figura1_series_temporales(tau=500):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.suptitle("Figura 1: Series temporales — beta bajo vs alto",
                 fontsize=13, fontweight="bold")
    for col, (beta, label) in enumerate([
        (0.3, "beta BAJO — sin acoplamiento"),
        (1.8, "beta ALTO — acoplamiento autorreferencial")
    ]):
        x, y, t = simular_AFH(beta=beta, tau=tau, semilla=0, modo="B")
        nb = calcular_nabla(x, y, tau=tau, dt=0.5)
        F, p, _ = granger_en_tau(x, y, dt=0.5, tau=tau)
        ini, fin = int(2000/0.5), int(7000/0.5)

        axes[0,col].plot(t[ini:fin], x[ini:fin], color="#2563eb", lw=0.8)
        axes[0,col].set_title(f"{label}\nnabla={nb:.3f}  F_Granger={F:.1f}",
                              fontsize=10)
        axes[0,col].set_ylabel("Talamo x(t)")
        axes[0,col].set_xlabel("Tiempo (ms)")
        axes[0,col].grid(True, alpha=0.3)

        axes[1,col].plot(t[ini:fin], y[ini:fin], color="#dc2626", lw=0.8)
        axes[1,col].set_ylabel("Corteza y(t)")
        axes[1,col].set_xlabel("Tiempo (ms)")
        axes[1,col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figura1_series_temporales.png", dpi=150, bbox_inches="tight")
    print("  -> figura1_series_temporales.png")
    plt.show()


def figura2_correlacion_por_lags(tau=500, beta=1.5):
    """
    Funcion de correlacion cruzada por retardos para cada modo.
    El pico debe aparecer cerca de tau en Tipo B y no en Tipo A.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Figura 2: Correlacion cruzada por retardos — tau real={tau}ms\n"
        f"Pico debe aparecer en tau solo en Tipo B",
        fontsize=12, fontweight="bold"
    )
    modos   = ["B", "A", "ninguno"]
    nombres = ["Tipo B (autorreferencial)", "Tipo A (externo)", "Sin feedback"]
    colores = ["#16a34a", "#f59e0b", "#dc2626"]

    for ax, modo, nombre, color in zip(axes, modos, nombres, colores):
        todas = []
        for rep in range(5):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo=modo)
            lags, corrs = correlacion_por_lags(x, y, dt=0.5,
                                               lag_min=10, lag_max=1500, paso=25)
            todas.append(corrs)

        arr   = np.array(todas)
        media = np.nanmean(arr, axis=0)
        std   = np.nanstd(arr, axis=0)

        ax.plot(lags, media, color=color, lw=2, label=nombre)
        ax.fill_between(lags, media-std, media+std, alpha=0.2, color=color)
        ax.axvline(tau, color="black", lw=2, ls="--",
                   label=f"tau real={tau}ms")
        ax.axhline(0, color="gray", ls=":", lw=1)

        # Marcar pico
        if not np.all(np.isnan(media)):
            idx_pk = np.nanargmax(media)
            ax.scatter(lags[idx_pk], media[idx_pk], color=color,
                       s=100, zorder=5,
                       label=f"Pico en {lags[idx_pk]}ms")

        ax.set_xlabel("Retardo (ms)")
        ax.set_ylabel("Correlacion cruzada")
        ax.set_title(nombre, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.3, 1.05)

    plt.tight_layout()
    plt.savefig("figura2_correlacion_lags.png", dpi=150, bbox_inches="tight")
    print("  -> figura2_correlacion_lags.png")
    plt.show()


def figura3_granger_central(res_granger):
    """
    FIGURA CENTRAL DEL PAPER.
    Granger especifico en tau: separa perfectamente Tipo B de A.
    """
    modos   = ["B", "A", "ninguno"]
    nombres = ["Tipo B\n(autorreferencial)", "Tipo A\n(externo)", "Sin\nfeedback"]
    colores = ["#16a34a", "#f59e0b", "#dc2626"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle(
        "Figura 3 (CENTRAL): Causalidad de Granger especifica en tau\n"
        "y(t-tau) predice x(t) solo en Tipo B — no en A ni sin feedback",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: F estadistico
    for i, (modo, nombre, color) in enumerate(zip(modos, nombres, colores)):
        F    = res_granger[modo]["F_med"]
        Fstd = res_granger[modo]["F_std"]
        axes[0].bar(i, F, color=color, alpha=0.85, width=0.55, zorder=3)
        axes[0].errorbar(i, F, yerr=Fstd, fmt="none",
                         color="black", capsize=8, lw=2, zorder=4)
        axes[0].text(i, max(0, F+Fstd+2),
                     f"F={F:.0f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    axes[0].set_xticks([0,1,2]); axes[0].set_xticklabels(nombres, fontsize=10)
    axes[0].set_ylabel("Estadistico F de Granger", fontsize=11)
    axes[0].set_title("F de Granger en tau\n(mayor = mas causalidad)", fontsize=10)
    axes[0].axhline(4, color="gray", ls=":", lw=1.5, label="F=4 referencia")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2: proporcion significativa
    for i, (modo, nombre, color) in enumerate(zip(modos, nombres, colores)):
        prop = res_granger[modo]["prop_sig"]
        axes[1].bar(i, prop, color=color, alpha=0.85, width=0.55, zorder=3)
        axes[1].text(i, prop+0.02, f"{prop:.2f}",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    axes[1].set_xticks([0,1,2]); axes[1].set_xticklabels(nombres, fontsize=10)
    axes[1].set_ylabel("Proporcion p < 0.05", fontsize=11)
    axes[1].set_title("Fraccion de simulaciones\nsignificativas", fontsize=10)
    axes[1].axhline(0.05, color="gray", ls=":", lw=1.5, label="Azar (0.05)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_ylim(0, 1.1)

    # Panel 3: delta R2
    for i, (modo, nombre, color) in enumerate(zip(modos, nombres, colores)):
        dr = res_granger[modo]["dR2_med"]
        axes[2].bar(i, max(0, dr), color=color, alpha=0.85, width=0.55, zorder=3)
        axes[2].text(i, max(0, dr)+0.00005,
                     f"{dr:.5f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    axes[2].set_xticks([0,1,2]); axes[2].set_xticklabels(nombres, fontsize=10)
    axes[2].set_ylabel("delta R2", fontsize=11)
    axes[2].set_title("Mejora en varianza explicada\nal agregar y(t-tau)", fontsize=10)
    axes[2].axhline(0, color="gray", ls=":", lw=1); axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("figura3_granger_central.png", dpi=150, bbox_inches="tight")
    print("  -> figura3_granger_central.png")
    plt.show()


def figura4_barrido_beta_dual(res_nabla, res_granger_beta, mu_surr, sd_surr, tau=500):
    """
    Barrido beta: nabla (descriptivo) + Granger (causal).
    H* se define donde ambos cruzan umbral.
    """
    betas_n = [r["beta"] for r in res_nabla]
    med_n   = [r["med"]  for r in res_nabla]
    std_n   = [r["std"]  for r in res_nabla]

    betas_g = [r["beta"]     for r in res_granger_beta]
    Fs_g    = [r["F_med"]    for r in res_granger_beta]
    prop_g  = [r["prop_sig"] for r in res_granger_beta]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figura 4: Barrido de beta — tau={tau}ms\n"
                 f"nabla (descriptivo) y Granger especifico (causal)",
                 fontsize=12, fontweight="bold")

    # Panel 1: nabla
    axes[0].plot(betas_n, med_n, color="#2563eb", lw=2.5, label="nabla medio")
    axes[0].fill_between(betas_n,
                         np.array(med_n)-np.array(std_n),
                         np.array(med_n)+np.array(std_n),
                         alpha=0.2, color="#2563eb")
    umbral_n = mu_surr + 2*sd_surr
    axes[0].axhline(umbral_n, color="gray", ls=":", lw=1.5,
                    label=f"Umbral Z=2 ({umbral_n:.3f})")
    cruces_n = [b for b,m in zip(betas_n, med_n) if not np.isnan(m) and m > umbral_n]
    if cruces_n:
        axes[0].axvline(cruces_n[0], color="#dc2626", ls="--", lw=2,
                        label=f"H* nabla ~ beta={cruces_n[0]:.2f}")
    axes[0].set_xlabel("beta"); axes[0].set_ylabel("nabla")
    axes[0].set_title("nabla bruto (descriptivo)", fontsize=10)
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.05)

    # Panel 2: Granger
    axes[1].plot(betas_g, Fs_g, color="#16a34a", lw=2.5, label="F Granger medio")
    ax2 = axes[1].twinx()
    ax2.plot(betas_g, prop_g, color="#7c3aed", lw=2, ls="--",
             label="Prop. significativa")
    ax2.set_ylabel("Proporcion p<0.05", color="#7c3aed")
    ax2.set_ylim(0, 1.1)

    axes[1].axhline(4, color="gray", ls=":", lw=1.5, label="F=4 referencia")
    cruces_g = [b for b,F in zip(betas_g, Fs_g) if not np.isnan(F) and F > 4]
    if cruces_g:
        axes[1].axvline(cruces_g[0], color="#dc2626", ls="--", lw=2,
                        label=f"H* Granger ~ beta={cruces_g[0]:.2f}")
    axes[1].set_xlabel("beta"); axes[1].set_ylabel("F de Granger", color="#16a34a")
    axes[1].set_title("Granger especifico (causal)", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper left")
    ax2.legend(fontsize=9, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figura4_barrido_beta.png", dpi=150, bbox_inches="tight")
    print("  -> figura4_barrido_beta.png")
    plt.show()


def figura5_barrido_tau(res_tau, beta=1.5):
    """Barrido tau: nabla bruto con pico en tau real."""
    taus  = [r["tau"] for r in res_tau]
    meds  = [r["med"] for r in res_tau]
    stds  = [r["std"] for r in res_tau]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(taus, meds, color="#7c3aed", lw=2.5, label="nabla medio")
    ax.fill_between(taus,
                    np.array(meds)-np.array(stds),
                    np.array(meds)+np.array(stds),
                    alpha=0.2, color="#7c3aed")
    ax.axvspan(200, 600, alpha=0.1, color="#f59e0b",
               label="Ventana AFH propuesta (200-600ms)")

    validos = [(t,m) for t,m in zip(taus,meds) if not np.isnan(m)]
    if validos:
        t_opt, n_opt = max(validos, key=lambda x: x[1])
        ax.axvline(t_opt, color="#dc2626", ls="--", lw=2,
                   label=f"Pico nabla en tau={t_opt}ms")

    ax.set_xlabel("tau (ms)", fontsize=12)
    ax.set_ylabel("nabla", fontsize=12)
    ax.set_title(f"Figura 5: Barrido de tau — beta={beta}\n"
                 f"nabla como funcion del retardo funcional",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    plt.savefig("figura5_barrido_tau.png", dpi=150, bbox_inches="tight")
    print("  -> figura5_barrido_tau.png")
    plt.show()


def figura6_mapa_beta_tau(M, betas, taus):
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(M, aspect="auto", origin="lower",
                   extent=[taus[0],taus[-1],betas[0],betas[-1]],
                   cmap="YlOrRd", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("nabla", fontsize=11)
    ax.contour(np.array(taus), np.array(betas), M,
               levels=[0.5], colors=["white"], linewidths=[2], linestyles=["--"])
    ax.axvspan(200, 600, alpha=0.12, color="#2563eb",
               label="Ventana tau AFH (200-600ms)")
    ax.set_xlabel("tau (ms)", fontsize=12)
    ax.set_ylabel("beta", fontsize=12)
    ax.set_title("Figura 6: Mapa de fase beta x tau\nRegion H* (zona caliente = nabla alto)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("figura6_mapa_beta_tau.png", dpi=150, bbox_inches="tight")
    print("  -> figura6_mapa_beta_tau.png")
    plt.show()


def figura7_estabilidad_temporal(tau=500, beta=1.5):
    """nabla en ventanas: demuestra regimen sostenido vs pico accidental."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figura 7: Estabilidad temporal de nabla\n"
                 "Ventanas de 1000ms sucesivas",
                 fontsize=12, fontweight="bold")

    for ax, modo, nombre, color in zip(
        axes, ["B","A"],
        ["Tipo B (autorreferencial)","Tipo A (externo coloreado)"],
        ["#16a34a","#f59e0b"]
    ):
        todas=[]; props=[]
        for rep in range(8):
            x, y, t = simular_AFH(beta=beta, tau=tau, semilla=rep, modo=modo)
            vals, prop = nabla_por_ventanas(x, y, tau=tau, dt=0.5)
            todas.append(vals); props.append(prop)

        for vals in todas:
            ax.plot(vals, color=color, alpha=0.25, lw=1)

        min_len = min(len(v) for v in todas)
        arr     = np.array([v[:min_len] for v in todas])
        media   = np.nanmean(arr, axis=0)
        ax.plot(media, color=color, lw=2.5, label="Media (n=8)")
        ax.axhline(0.3, color="gray", ls=":", label="Umbral nabla=0.3")
        ax.axhline(0,   color="black", lw=0.8)

        prop_med = np.mean(props)
        ax.set_title(f"{nombre}\nFraccion ventanas > 0.3: {prop_med:.2f}", fontsize=10)
        ax.set_xlabel("Ventana (cada 1000ms)")
        ax.set_ylabel("nabla en ventana")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.05)

    plt.tight_layout()
    plt.savefig("figura7_estabilidad.png", dpi=150, bbox_inches="tight")
    print("  -> figura7_estabilidad.png")
    plt.show()


def figura8_lesion_ILN(tau=500, beta=1.5, t_lesion=7500):
    """nabla cae al lesionar ILN. Conecta con clinica."""
    print(f"  Simulando lesion ILN en t={t_lesion}ms...")
    x_l, y_l, t_l = simular_AFH_lesion(beta=beta, tau=tau,
                                         t_lesion=t_lesion, semilla=42)
    x_i, y_i, t_i = simular_AFH(T=15000, beta=beta, tau=tau,
                                  semilla=42, modo="B")

    def nabla_ventanas_t(x, y, tau, dt, ventana_ms=1000, descartar_ms=2000):
        d=int(descartar_ms/dt); w=int(ventana_ms/dt); k=int(tau/dt)
        tc=[]; vc=[]; ini=d+k
        while ini+w < len(x):
            xw=x[ini:ini+w]; yw=y[ini-k:ini-k+w]
            if np.std(xw)>1e-10 and np.std(yw)>1e-10 and len(xw)==len(yw):
                vc.append(np.corrcoef(xw,yw)[0,1])
                tc.append((ini+w/2)*dt)
            ini+=w
        return np.array(tc), np.array(vc)

    tc_l, vc_l = nabla_ventanas_t(x_l, y_l, tau, 0.5)
    tc_i, vc_i = nabla_ventanas_t(x_i, y_i, tau, 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(f"Figura 8: Simulacion de lesion ILN\n"
                 f"beta -> 0 en t={t_lesion}ms | nabla cae abruptamente",
                 fontsize=12, fontweight="bold")

    axes[0].plot(t_l/1000, x_l, color="#2563eb", lw=0.6,
                 label="Talamo (con lesion)")
    axes[0].plot(t_i/1000, x_i, color="#93c5fd", lw=0.6, alpha=0.5,
                 label="Talamo (intacto)")
    axes[0].axvline(t_lesion/1000, color="#dc2626", lw=2, ls="--",
                    label="Lesion ILN")
    axes[0].set_ylabel("Talamo x(t)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(tc_l/1000, vc_l, "o-", color="#16a34a", lw=2,
                 ms=6, label="nabla — con lesion")
    axes[1].plot(tc_i/1000, vc_i, "s--", color="#6b7280", lw=1.5,
                 ms=5, alpha=0.6, label="nabla — intacto")
    axes[1].axvline(t_lesion/1000, color="#dc2626", lw=2, ls="--",
                    label=f"Lesion (t={t_lesion}ms)")
    axes[1].axhline(0.3, color="gray", ls=":", lw=1, label="Umbral=0.3")
    axes[1].set_ylabel("nabla"); axes[1].set_xlabel("Tiempo (s)")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.15, 1.05)

    if len(vc_l) > 0:
        pre  = np.mean(vc_l[tc_l <= t_lesion])  if np.any(tc_l <= t_lesion)  else np.nan
        post = np.mean(vc_l[tc_l >  t_lesion])  if np.any(tc_l >  t_lesion)  else np.nan
        axes[1].annotate(
            f"Pre:  {pre:.3f}\nPost: {post:.3f}",
            xy=(t_lesion/1000+0.3, max(-0.1, post)),
            fontsize=9, color="#dc2626",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#dc2626", alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig("figura8_lesion_ILN.png", dpi=150, bbox_inches="tight")
    print("  -> figura8_lesion_ILN.png")
    plt.show()


# =====================================================================
# BLOQUE 6: PIPELINE PRINCIPAL
# =====================================================================

def main():
    print("=" * 65)
    print("  MODELO AFH v0.5")
    print("  Granger especifico como metrica central")
    print("  Dr. Camilo Alejandro Sjoberg Tala, M.D.")
    print("=" * 65)

    TAU  = 500
    BETA = 1.5

    # DEMO
    print("\n[DEMO] beta=1.5, tau=500ms...")
    x, y, t = simular_AFH(beta=BETA, tau=TAU, semilla=42, modo="B")
    nb       = calcular_nabla(x, y, tau=TAU, dt=0.5)
    F, p, dR = granger_en_tau(x, y, dt=0.5, tau=TAU)
    z, mu_s, sd_s, _ = calcular_z_surrogate(x, y, TAU, 0.5, n_surr=50)

    print(f"  nabla bruto    = {nb:.4f}")
    print(f"  Granger F      = {F:.1f}  p={p:.4f}  dR2={dR:.5f}")
    print(f"  Z surrogate    = {z:.2f}")

    # FIGURA 1
    print("\n[F1] Series temporales...")
    figura1_series_temporales(tau=TAU)

    # FIGURA 2
    print("\n[F2] Correlacion por lags...")
    figura2_correlacion_por_lags(tau=TAU, beta=BETA)

    # FIGURA 3 — CENTRAL
    print("\n[F3] Granger central (~3 min)...")
    res_granger = comparar_granger_modos(beta=BETA, tau=TAU,
                                          repeticiones=12, orden=5)
    figura3_granger_central(res_granger)

    # FIGURA 4
    print("\n[F4] Barrido beta dual (~5 min)...")
    betas      = np.arange(0.0, 2.6, 0.15)
    res_nb     = barrer_beta_nabla(betas, tau=TAU, repeticiones=8)
    res_gr_b   = granger_barrido_beta(betas, tau=TAU, repeticiones=6, orden=5)
    figura4_barrido_beta_dual(res_nb, res_gr_b, mu_s, sd_s, tau=TAU)

    # FIGURA 5
    print("\n[F5] Barrido tau (~3 min)...")
    taus   = [50,100,150,200,300,400,500,600,700,800,1000,1200,1500]
    res_tau = barrer_tau_nabla(taus, beta=BETA, repeticiones=8)
    figura5_barrido_tau(res_tau, beta=BETA)

    # FIGURA 6
    print("\n[F6] Mapa beta x tau (~8 min)...")
    betas_m = np.arange(0.0, 2.6, 0.25)
    taus_m  = [50, 150, 300, 500, 700, 1000, 1500]
    M = mapa_beta_tau(betas_m, taus_m, repeticiones=5)
    figura6_mapa_beta_tau(M, betas_m, taus_m)

    # FIGURA 7
    print("\n[F7] Estabilidad temporal (~2 min)...")
    figura7_estabilidad_temporal(tau=TAU, beta=BETA)

    # FIGURA 8
    print("\n[F8] Lesion ILN...")
    figura8_lesion_ILN(tau=TAU, beta=BETA, t_lesion=7500)

    # RESUMEN
    print("\n" + "=" * 65)
    print("  RESUMEN v0.5")
    print("=" * 65)
    print(f"\n  nabla bruto:      {nb:.4f}")
    print(f"  Z surrogate:      {z:.2f}")
    print(f"  Granger F (B):    {res_granger['B']['F_med']:.1f}"
          f"  prop={res_granger['B']['prop_sig']:.2f}")
    print(f"  Granger F (A):    {res_granger['A']['F_med']:.1f}"
          f"  prop={res_granger['A']['prop_sig']:.2f}")

    if (res_granger["B"]["prop_sig"] > 0.8
            and res_granger["A"]["prop_sig"] < 0.2):
        print("\n  RESULTADO PRINCIPAL:")
        print("  Granger especifico discrimina Tipo B de Tipo A.")
        print("  y(t-tau) causa x(t) solo cuando es eco propio.")
        print("  Esto es la firma causal del Pliegue Autopsiquico.")
    else:
        print("\n  REVISAR: discriminacion Granger B vs A incompleta.")

    print("\n  8 figuras guardadas.")
    print("\n  INTERPRETACION:")
    print("  nabla describe el acoplamiento.")
    print("  Granger especifico demuestra que ese acoplamiento")
    print("  es causal y especifico del feedback autorreferencial.")
    print("=" * 65)


if __name__ == "__main__":
    main()