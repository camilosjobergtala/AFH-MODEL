# MODELING FOLD — formalización matemática del Pliegue Autopsíquico

- **`0.4.py`** (internamente versionado "AFH v0.5" en su docstring — nombre de archivo y
  versión interna no coinciden, mantenido tal cual) — simulación de un lazo talamocortical
  escalar con retardo (tálamo `x`, corteza `y`). Implementa la distinción central del marco:
  Tipo B (retorno = eco propio del sistema) vs. Tipo A (retorno = ruido coloreado externo,
  emparejado en media/varianza/autocorrelación) vs. sin retroalimentación, discriminados vía
  causalidad de Granger específica en el retardo `tau`. Incluye barridos de `beta` y `tau`,
  mapa de fase, estabilidad temporal, y una simulación de lesión ILN. Corre en 25-35 min y
  genera 8 figuras.

- **`AFH_modelo_matematico_pliegue.md`** — el aparato formal completo: define H\*, ∇, R y V
  como funcionales computables de la dinámica de cualquier sistema (no sólo del cerebro),
  deriva analíticamente un umbral de bifurcación para H\* (ganancia de lazo crítica,
  independiente de todo retardo o constante de tiempo), formaliza la distinción Tipo A/Tipo B
  vía información de transferencia condicional, y traduce las cinco condiciones de
  universalidad y las siete predicciones del marco AFH a hipótesis estadísticas precisas.
  Incluye una tabla de correspondencia explícita entre esta formalización y `0.4.py`.
