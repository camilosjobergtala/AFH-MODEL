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

- **`AFH_modelo_N_nodos.py`** — generalización de `0.4.py` a poblaciones (m nodos ILN, n
  corteza) con retardo aferente `tau_f` explícito (ausente en `0.4.py`), matrices de
  acoplamiento normalizadas para que el umbral crítico sea exactamente β=1 en cualquier
  dimensión, un estimador **directo** de β\_c por bisección sobre la tasa de crecimiento
  exponencial (no un umbral de amplitud — ver la nota metodológica en el propio archivo sobre
  por qué esa primera aproximación estaba sesgada), y el volumen configuracional V (razón de
  participación) sobre la población cortical. Confirma numéricamente β\_c≈1.0 (error 1-5.5 %)
  para múltiples m, n, τ\_f, τ\_s — validación directa de la derivación analítica de la Sección
  5/8 del `.md`. Corre en segundos (`python "AFH_modelo_N_nodos.py"`).

- **`pliegue_interactivo.html`** — instrumento interactivo autocontenido (abrir directo en un
  navegador, sin servidor): corre en vivo una réplica en JavaScript del modelo canónico
  (m=1 hub, n=3 corteza, pesos normalizados para β\_c=1), con sliders para β y τ, selector
  Tipo B/Tipo A/Ninguno, botón de lesión ILN, y lectura en vivo de ∇ bruto, ∇ parcial
  (aproximación a Granger específico), H\* y V. Publicado también como Artifact en
  `https://claude.ai/code/artifact/44512c2d-e21e-46c8-a6db-478e2d477f77`.

- **`simulacion_acoplamiento_fase_temprana_tardia.py`** — traduce la Predicción 6 del formato
  "señal continua con un único retardo fijo" (el de `0.4.py`/`AFH_modelo_N_nodos.py`) al
  formato de un registro intracraneal organizado por ensayos, con una fase temprana
  tálamo→corteza y una fase tardía corteza→tálamo por ensayo. Simula el análisis de
  acoplamiento ensayo-a-ensayo entre ambas fases bajo Tipo B (retorno reflexivo) vs. Tipo A
  (vía independiente, sólo comparten arousal/enganche), muestra por qué la correlación cruda
  no discrimina y hay que usar correlación parcial controlando un proxy del confusor, y corre
  un análisis de poder (N de ensayos × fuerza de acoplamiento). Hallazgo central: con un proxy
  de confusor ruidoso, la tasa de falsos positivos de la correlación parcial *crece* con N en
  vez de diluirse — un proxy de arousal poco fiel es más urgente que sumar ensayos. Corre en
  segundos (`python "simulacion_acoplamiento_fase_temprana_tardia.py"`).
