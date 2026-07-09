# AFH — Materiales de revisión (Neuroscience of Consciousness)

Paquete autocontenido con todo lo necesario para evaluar el manuscrito **"La convergencia
temporal talámica intralaminar como mecanismo candidato de la presencia fenomenológica"**
(C. A. Sjöberg Tala), enviado a *Neuroscience of Consciousness*.

Esta carpeta es análoga a [`../ScientificReports_revision/`](../ScientificReports_revision/)
pero para este paper: reúne manuscrito, modelo formal, código y resultados en un solo lugar,
copiados desde el archivo exploratorio del autor en [`../Lab/`](../Lab/) (que se deja intacto,
sin modificar, como registro histórico de cómo evolucionó cada análisis — ver el README de
`Lab/` para el mapa completo de esa carpeta). Nada aquí depende de `Lab/`: todos los scripts
usan rutas relativas a esta carpeta.

## Mapa

```
manuscript/    manuscrito principal y Suplemento S1 (PDF, tal como se envían)
model/         aparato formal completo (H*, ∇, R, V) y el protocolo de la Predicción 6
code/          todo el código citado por el texto, listo para ejecutar (pip install -r code/requirements.txt)
instruments/   instrumento interactivo (HTML) y página de resultados ya corridos
outputs/       resultados ya generados: los que cita el texto directamente, y los del código de arriba
```

## `manuscript/`

- `AFH_manuscrito_principal.pdf` — el manuscrito.
- `AFH_suplemento_S1.pdf` — Suplemento S1 (validación en simulación del diseño de prueba).

## `model/`

- `AFH_modelo_matematico_pliegue.md` — define H\*, ∇, R y V como funcionales computables,
  deriva el umbral de bifurcación de H\* (Sec. 3.6 del manuscrito) y formaliza la distinción
  Tipo A/Tipo B (Sec. 3.6, 9.1). Incluye la tabla de correspondencia entre esta formalización
  y `code/0.4.py` / `code/AFH_modelo_N_nodos.py`.
- `DISENO_EXPERIMENTAL_prediccion6.md` — protocolo prospectivo puro (población, señales,
  orden de análisis, condiciones de retroceso) para la Predicción 6, la prueba decisiva del
  marco (Sec. 6.1 del manuscrito).

## `code/` — cómo correr cada cosa que el texto cita

```bash
pip install -r code/requirements.txt
```

| Script | Qué hace | Qué parte del texto reproduce |
|---|---|---|
| `AFH_modelo_N_nodos.py` | Estimación directa de β\_c por bisección (m nodos ILN × n corteza) y discriminación Tipo A/Tipo B vía F de Granger condicional | Tabla de Sec. 3.6 (β\_c≈1.0, error 1.5–5.5 % para (τ_f,τ_s) = 8/200, 30/400, 100/600 ms) y F=1088,9 (Tipo B) vs. F=1,2 (Tipo A) |
| `0.4.py` | Precursor escalar (m=n=1) de lo anterior, con barridos de β/τ, mapa de fase y simulación de lesión ILN | Base de la que `AFH_modelo_N_nodos.py` generaliza; no reemplazada, ver su propio docstring |
| `simulacion_acoplamiento_fase_temprana_tardia.py` | Simulación de poder estadístico del acoplamiento ensayo-a-ensayo (Predicción 6/7), con proxy de confusor ruidoso | Sec. 6.6: la FPR de la correlación cruda con proxy ruidoso crece de 0,08 a 0,32 entre N=50 y N=300 |
| `test_rastreo_de_fuente.py` | Prueba de identificabilidad por emparejamiento vs. prueba de magnitud; incluye la vulnerabilidad de estímulo compartido y su corrección | Suplemento S1 completo (Tablas S1 y S2): top-1 = 0,087/0,030/0,010 con p<0,0001/0,037/0,98; FPR 1 % vs. 8 % de error de etiquetado (0,20→0,89 entre N=400 y N=2400) |
| `simulate_afh_predictions.py` | Genera datos sintéticos Tipo B (`--regime afh_true`) o Tipo A (`--regime null_heterogeneous`) e implementa las 7 pruebas estadísticas de la Sección 6.1 | Sec. 6.1 completa (Predicciones 1–7); con semilla 42, `afh_true` pasa 7/7 (incluida P7 con el lag exacto de 300 ms) y `null_heterogeneous` falla 7/7 |
| `evaluate_real_sleepedf_predictions.py` | Re-analiza (no re-procesa EEG crudo) los resultados reales ya calculados sobre Sleep-EDF Cassette (153 sujetos) en `outputs/nivel3_pac_optimization/` y `outputs/deltatfold_sleepedf/` | Sec. 6.4: contexto real relacionado con los análogos de P3 (55,6 % vs. 50 % azar, p=6,5e-6, no alcanza el 70 % que el propio autor fijó) y Δt-Fold/P4 (d=0,219, p=0,021, FALSIFICADA 1/2 criterios) — con la salvedad explícita de que ninguno de los dos es un test oficial y pre-registrado de esas predicciones tal como están escritas en el texto |

Todos corren en segundos, salvo `test_rastreo_de_fuente.py` (7–8 min por el Bloque 7,
que repite la simulación FPR-vs-N en dos regímenes de calidad de etiquetado).

Nota de auditoría: los cinco scripts anteriores fueron ejecutados de cero como parte de la
revisión de este paquete (no solo leídos) y sus salidas coinciden, número a número, con las
cifras que el manuscrito y el Suplemento S1 citan.

## `instruments/`

- `pliegue_interactivo.html` — instrumento interactivo autocontenido (abrir directo en un
  navegador): réplica en vivo del modelo canónico con sliders de β/τ y lectura en vivo de
  ∇, H\* y V.
- `resultados.html` — página con los resultados reales (no ilustrativos) de correr
  `AFH_modelo_N_nodos.py` y `simulacion_acoplamiento_fase_temprana_tardia.py`.

## `outputs/`

- **`eureka_v3.7/`, `eureka_v4.1/`** — las dos implementaciones corticales exploratorias de
  H\* que el manuscrito cita en Sec. 6.4 (script `ECLIPSE 3.7.py` / `ECLIPSE 4.1.py`, criterios
  de aprobación pre-registrados, y el resultado final). F1 real = 0,0306 (v3.7, criterio propio
  ≥0,25) y F1 = 0,0367 (v4.1, criterio propio ≥0,60) — ambas muy por debajo de su propio umbral.
  Nota: el texto describe un único criterio "F1 ≥ 0,60" aplicado a ambas versiones; el criterio
  realmente pre-registrado para v3.7 fue ≥0,25 (más laxo), no ≥0,60 — no cambia la conclusión
  (ambas fallan por un orden de magnitud) pero vale la corrección para el revisor.
- **`deltatfold_sleepedf/`** — resultado real de Δt-Fold sobre Sleep-EDF (grupo de retardo de
  fase 2–9 Hz entre Fpz-Cz y Pz-Oz), citado en Sec. 6.4 como contexto relacionado a P4 (no como
  test oficial de P4).
- **`nivel3_pac_optimization/`** — CSVs/JSON del análisis de 153 sujetos reales de Sleep-EDF
  (144 tras exclusión por calidad) usado como análogo exploratorio de P3.
- **`predicciones_simulacion/`** — salidas de `simulate_afh_predictions.py` (regímenes
  `afh_true` y `null_heterogeneous`) y de `evaluate_real_sleepedf_predictions.py`, ya
  generadas; se regeneran solas al correr los scripts de `code/`.

## Alcance y honestidad epistémica (para el revisor)

- Ningún resultado en `outputs/` sustituye registro intracraneal real en los ILN: las
  Predicciones 6 y 7 (las decisivas) siguen pendientes, tal como el propio manuscrito declara
  en su Sec. 6.4.
- Los análisis de `nivel3_pac_optimization/` y `deltatfold_sleepedf/` son exploraciones previas
  del autor con hipótesis y métricas propias, no operacionalizaciones oficiales de las
  Predicciones 3/4 — cada script y este README lo marcan explícitamente para evitar sobre-lectura.
- La cifra específica "d de Cohen = 2,44–2,79" para PAC delta-gamma (N=153), que una versión
  anterior de este análisis buscó y no encontró archivada en ningún lado del repositorio, ya no
  aparece en el manuscrito actual: el texto vigente solo dice "tamaño de efecto grande,
  consistente con hallazgos previos en la literatura", sin cifra fija, y declara que ese
  análisis no quedó archivado formalmente — es decir, el texto ya se autocorrigió en ese punto.

License: Apache License 2.0 (ver `LICENSE` en la raíz del repositorio).
