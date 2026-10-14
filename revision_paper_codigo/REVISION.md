# Revisión: paper publicado vs. código depositado

- **Paper:** Sjöberg Tala, *Scientific Reports*, DOI 10.1038/s41598-026-74760-5 (Article in Press,
  14-10-2026).
- **Código:** `ScientificReports_revision/` en el commit `93339b4`, el que cita el paper (Zenodo
  10.5281/zenodo.22760496). SHA-256 de `eclipse_core.py`: `9d4320d2…b059b3`.
- **Alcance:** todo lo que el paper usa o describe: `engine/` (core y wrappers), `code/00–05`,
  `outputs/`, `README.md`, `STUDY4_specification.md` y `requirements_frozen.txt`. Quedaron fuera
  `Lab/`, `Ashes/` y `NeuroscienceOfConsciousness_revision/`, que no forman parte del paper.
- **Entorno:** Linux, Python con las versiones de `requirements_frozen.txt`.

El punto 11 (barras de la Fig. 4B) se retiró tras revisión: la leyenda publicada ya dice "5% by
construction", así que no es una discrepancia del paper. Los demás puntos conservan su número.

Cada punto indica cómo se verificó:
- **[ejecutado]**: lo reproduje corriendo código; el script está en esta carpeta.
- **[lectura]**: sale de leer el código.

---

## 1. Lo que está bien

| Comprobación | Resultado |
|---|---|
| Los 6 scripts (`00`–`05`) y `test_eis_wrapper.py` corren con `python -W error` | Todos terminan con código 0 [ejecutado] |
| Regenerar `outputs/` desde cero | Los CSV y JSON coinciden. Solo cambian el timestamp, `ms_per_script` y el fin de línea del CSV del Estudio 2b [ejecutado] |
| Figuras regeneradas | Mismos datos. Difieren en 1–2 píxeles de tamaño y en el antialiasing, por las fuentes del entorno (aquí "Inter"). Es esperable entre máquinas [ejecutado] |
| Estudio 1: 0.40 / 0.14 / 0.80–0.98 / media 0.92 / 1×10⁻¹⁶ / 0.12–0.97 / DE 0.17–0.32 / 41 % con N < 100 / ρ = 0.38 | Todo coincide [ejecutado] |
| Tablas 2, 3 y 4 (Estudio 2) | Coinciden con `outputs/study2/` [ejecutado] |
| Tabla 5 (Estudio 2b) | Coincide con `outputs/study2b/` y con `EMPIRICAL_SCREEN` [ejecutado] |
| Tablas 6 y 7 (Estudio 3) | Coinciden con `outputs/study3/` [ejecutado] |
| Fórmulas del EIS en §2.6.4 (S_pre, S_split, S_prot, S_leak, S_trans) | Coinciden con el código. Valores de S_leak: 0.98 / 0.88 / 0.50 / 0.12 [lectura + ejecutado] |
| STDS en §2.6.4 | sd poblacional, exclusión cuando σ = 0, max_z y mean_z: coincide [lectura] |
| Penalizaciones del auditor (30/15/5/2, 10 por defecto) y umbral 70 | Coincide [lectura] |
| Las 4 categorías adicionales del auditor que menciona el paper | Existen en el código [lectura] |
| Controles del auditor usados en los puntos 4 y 5 (`if accuracy > 0.8: evaluate(model, X_test)` y `print(X_test[0])`) | Ambos se detectan (`conditional_holdout_access`, `holdout_access`), así que los resultados negativos de los puntos 4 y 5 no se deben a un fallo del script de prueba [ejecutado] |
| `STUDY4_specification.md` vs. paper (modelos A–E, refs. 26–27, 556 modelos / 1.147 validaciones) | Coincide con el paper. **No** verifiqué las cifras contra los artículos citados |
| `run_unit_tests()` del core | 5/5 pasan [ejecutado] |

**Ningún resultado numérico de los Estudios 1–3 y 2b depende de los problemas de las secciones 2 y 3.**
Los estudios no usan `stage1`, `stage4`, `stage5` ni `verify_integrity` (verificado con grep).

---

## 2. El paper describe mecanismos que el código no tiene (sección de Métodos)

| # | Paper | Código | Evidencia |
|---|---|---|---|
| 1 | Etapa 1: el hash cubre los índices del holdout, la semilla, el ratio y un timestamp; modificar los índices cambia el hash | Solo cubre la semilla y el conjunto de IDs (`eclipse_core.py:2477`). Mover IDs entre desarrollo y holdout, cambiar el ratio o la fecha **no** se detecta | `verificar_p1_p4.py` [ejecutado] |
| 2 | Etapa 4: el archivo de holdout queda de solo lectura y los resultados se escriben al validar | No hay ningún archivo de holdout gestionado ni bloqueo de archivo. La Etapa 4 se puede repetir hasta que se ejecuta la Etapa 5, y `force=True` la permite después | `verificar_p1_p4.py` [ejecutado] |
| 3 | Etapa 5: "ECLIPSE automatically generates four outputs", incluido el *Code Audit Report*. La Fig. 1 muestra el auditor dentro de la Etapa 5 | `stage5_final_assessment` solo calcula el veredicto, el EIS y el STDS. El auditor hay que llamarlo aparte (`audit_code`) | `verificar_otros.py` A [ejecutado] |
| 4 | Auditor, nivel 2 (control de flujo): "loop-based optimization (e.g., for threshold in …: test_performance), and branch coverage anomalies" | El nivel 2 solo revisa sentencias `if` con nombres de métricas y acceso al holdout en el cuerpo (`:1436`). Un bucle de optimización de umbral no se marca, y no existe análisis de "branch coverage" | `verificar_otros.py` B [ejecutado] |
| 5 | Auditor, nivel 3 / Tabla 1: "Information leakage — Data-flow — `y=test_data.mean()`"; "cross-contamination between development and holdout" | El análisis de flujo de datos solo marca variables derivadas del holdout cuando entran en llamadas tipo `fit`, `train`, `optimize`, `tune` o `search` (`:1474`). El ejemplo de la Tabla 1 solo produce `holdout_access` (detección AST directa) | `verificar_otros.py` B [ejecutado] |
| 6 | Tabla 1, columna "Detection method": *threshold manipulation* AST+pattern, *multiple testing* AST+pattern, *test-set tuning* AST+pattern. Nivel 1: "function call analysis (e.g., GridSearchCV on a test set)" | *Threshold manipulation* es solo AST. *Multiple testing* y *test-set tuning* son solo regex. GridSearchCV se detecta por regex en el nivel 4, no por AST | [lectura] |
| 7 | Etapa 1: estratificado con clases desbalanceadas, temporal con series de tiempo | Solo barajado aleatorio (`:2468`) | [lectura] |
| 8 | Etapa 3: "A final model is then trained on all of D_dev" | `stage3_development` no lo hace; `stage4` recibe el modelo del usuario | [lectura] |

Los puntos 1 y 2 ya los conocías. Los 3–6 son nuevos. El 7 y el 8 se pueden leer como
recomendaciones, pero el paper los presenta dentro de "Implementation details" o como parte del
procedimiento.

---

## 3. Cifras del paper que el depósito no reproduce o que no cuadran

| # | Paper | Lo que da el depósito | Evidencia |
|---|---|---|---|
| 9 | §3.2: "Verified end to end through the engine, the realized honest-scenario rate is 4.4–5.0% across the sampled designs". El README dice lo mismo | **El rango publicado no es reproducible desde el depósito**, lo que es distinto de afirmar que esté mal. Ningún script, ni el actual ni ninguno del historial de git, lo produce. La frase entró al README en el commit `91db9d7` (14-09-2026), el mismo que creó `04_stds_screening_cutoffs.py`. Ese archivo tiene una sola versión en todo el historial. Su paso [2] calcula la tasa en 4 diseños por la ruta vectorizada, no por el motor, y solo la imprime: da 4.77–5.15 %, de forma determinista. En ningún commit hay un script de `code/` que mida tasas de marcado por el motor (uso de `risk_level` o `screen_cutoff`). Intento de reproducción por el motor completo: ver la sección 3b | `verificar_otros.py` D (historial de git) [ejecutado] |
| 10 | §3.2 y README: "against 5.1–54.4% under the fixed cutoff" | Mezcla dos fuentes. En el Estudio 2b (`stds_screening_cutoffs.csv`) el rango es **3.5–54.4 %**: la Fig. 4B muestra K = 20 / M = 1 en 3.5 %. En la Tabla 2 (Estudio 2) es 5.1–55.2 % | [ejecutado] |
| 12 | Estudio 1: el componente de fuga está "essentially uncorrelated with the others (\|r\| ≤ 0.08)" | El máximo es 0.086 (con preregistro), que redondea a 0.09. La propia Fig. 2A muestra −0.09 | [ejecutado] |
| 13 | §2.6.4: el máximo de 0.975 se obtiene "with a balanced split and a holdout at or below the development mean" | Con el holdout **en** la media, S_leak = 0.88 y el EIS máximo es ≈ 0.957. 0.975 solo se alcanza en el límite, con el holdout muy por debajo de la media de desarrollo (es un supremo). Con el 70/30 por defecto, el máximo es ≈ 0.955 (el paper no lo menciona) | [ejecutado] |

**Observación, no error:** las Tablas 2, 3 y 4 reportan 8.6 %, 11 % y 9.0 % para la misma
condición (K = 5, M = 1, σ = 0.05, δ = 0), porque vienen de simulaciones independientes. El
paper lo reconoce ("8–11%"). La diferencia entre 8.6 % y 11 % es de unos 2.7 errores estándar,
algo que un lector atento podría preguntar.

---

## 4. Textos del código que contradicen el encuadre del paper (ya conocidos)

| # | Problema |
|---|---|
| 14 | `_interpret_eis`: seis etiquetas ("EXCELLENT – Exceptional methodological rigor" … "VERY POOR"), y el reporte imprime "✅ Study meets high methodological standards". El paper usa tres bandas provisionales y dice "EIS does not measure methodological rigor objectively" |
| 15 | `WEIGHT_JUSTIFICATIONS` incluye cifras ("~60 %", "17/20") que no están en el paper |
| 16 | El modo `non_interactive` confirma la Etapa 4 con `"AUTOMATED"`; el paper solo describe la confirmación tecleada |

---

## 5. Paquete de reproducibilidad (Data availability y README)

| # | Paper / README | Realidad |
|---|---|---|
| 17 | "unit tests for the core wrappers" | Solo existe `test_eis_wrapper.py`, que imprime resultados sin ningún `assert` y cubre únicamente el wrapper del EIS. No hay tests de `stds_wrapper` ni de `auditor_wrapper`. Los tests con `assert` del core (`run_unit_tests`, 5/5) no se mencionan |
| 18 | "the script that regenerates Figs. 1–4" | El paper tiene 5 figuras. Los archivos se llaman `figure1`, `2`, `3`, `3b` y `4`: `figure3b` es la Fig. 4 del paper y `figure4` es la Fig. 5. Confunde al que intenta reproducir |
| 19 | Las 4 categorías adicionales del auditor están "documented in the repository" | Solo aparecen en el código, no en el README |
| 20 | §2.6.4 "transcribes the reference implementation" | La fuga omite las métricas con media de desarrollo = 0 (`dev_mean == 0`, `eclipse_core.py:516`). La definición formal no lo menciona |
| 21 | "Software implementation": "depends on NumPy and Pandas; scikit-learn; JSON; hashlib; ast" | También requiere SciPy (import obligatorio en el core) y matplotlib (figuras). Es un detalle menor |

---

## 6. Errores editoriales del texto (para las pruebas de imprenta)

| # | Ubicación | Problema |
|---|---|---|
| 22 | Leyenda de la Fig. 5 | Cita "Table 5" para el recall por categoría (debería ser la Tabla 6) y "Table 6" para los near-miss (debería ser la Tabla 7) |
| 23 | Varias referencias entre paréntesis apuntan a tablas o figuras sin relación con la frase | "(Table 1)" al final del párrafo del EIS en la Introducción (la Tabla 1 es la del auditor); "(Table 2)" en la frase de defensa en profundidad; "(Table 3)" en Related work; "(Table 4)" en el párrafo final de Scope; "(Table 5)" en el algoritmo de la Etapa 1 ("supporting tamper detection (Table 5)", y la Tabla 5 son los cortes del STDS); "(Table 6)" tras la definición de S_leak; "(Table 7)" tras la definición formal del STDS; "(Figs. 2, 3)" al cierre del Estudio 1 (la Fig. 3 es del STDS). Parecen anclas de posición de las tablas que quedaron en el texto |
| 24 | Tabla 1 vs. Tabla 6 | "Information leakage" (Tabla 1) y "Indirect holdout access (aliasing)" (Tabla 6) nombran la misma categoría de forma distinta |

---

## Prioridad sugerida

1. **Para el correo a la revista:** los puntos 1–6 (mecanismos descritos que no existen), el 9 y el 10
   (cifras no reproducibles o que mezclan fuentes) y el 22–23 (errores de
   referencias que pueden corregirse en las pruebas).
2. **Código (versión nueva, sin tocar la del paper):** 1, 2, 3, 14, 15 y 16; el 4 y el 5 si se
   quiere que el auditor haga lo que el paper describe, o si no, corregir el texto.
3. **Paquete de reproducibilidad:** 9 (depositar el cálculo que respalde la cifra publicada), 17, 18,
   19 y 20.
