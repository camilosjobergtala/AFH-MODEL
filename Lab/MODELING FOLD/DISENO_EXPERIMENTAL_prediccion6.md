# Diseño experimental: poner a prueba el origen de la señal de retorno (Predicción 6)

Documento de protocolo, no de matemática. Todo lo que sigue es ejecutable por alguien que
nunca vio `AFH_modelo_matematico_pliegue.md` — las fórmulas y derivaciones que lo
justifican están ahí, no acá. Este documento responde una sola pregunta: **si alguien
tuviera acceso a un registro intracraneal adecuado mañana, ¿qué exactamente habría que
hacer?**

Estado: especificación para ejecución prospectiva. Nada de esto se ha corrido sobre datos
reales todavía.

---

## 1. La pregunta, en una frase

Cuando el tálamo (ILN) recibe de vuelta, unos cientos de milisegundos después, una señal
desde la corteza — ¿esa señal es la elaboración cortical de su propia actividad reciente
(un eco genuino de sí mismo), o es una entrada tardía que sólo coincide en el tiempo con
la anterior? La primera opción es la que el marco AFH necesita para sostener su hipótesis
central; la segunda la refuta.

## 2. Población y criterios de inclusión

- Pacientes con electrodos intracraneales implantados por indicación clínica (sEEG o DBS)
  con **cobertura confirmada de los núcleos intralaminares** — específicamente
  centromediano-parafascicular (CM/Pf) y/o central lateral (CL). Esto **no está
  garantizado** en cohortes implantadas con objetivos clínicos (epilepsia, Parkinson,
  dolor) más que de investigación — es la primera condición a verificar, caso por caso,
  antes de cualquier otra cosa. Sin esta cobertura, el resto del diseño no es ejecutable
  sobre esos datos.
- Contactos corticales simultáneos con cobertura de las áreas relevantes a la tarea
  (prefrontal/parietal, según el paradigma elegido).
- Capacidad de realizar una tarea de percepción consciente con respuesta motora medible
  (ver §3). Excluir sedación profunda o deterioro cognitivo que impida completar la tarea.
- Consentimiento informado para uso de datos de investigación además del registro clínico.

## 3. Paradigma conductual

Cualquier tarea que produzca, dentro de un mismo ensayo, dos fases temporalmente
separadas y direccionalmente distintas:

1. **Fase temprana**: procesamiento ascendente tálamo→corteza, desencadenado por un
   estímulo con inicio preciso y medible (idealmente 0–600 ms post-estímulo).
2. **Fase tardía**: retorno cortico-talámico, ocurriendo antes de una respuesta motora
   medible (idealmente en una ventana de 200–600 ms antes de la respuesta).

Una tarea de detección o discriminación perceptual con respuesta sacádica cumple esto de
forma natural (inicio de estímulo preciso, sacada como marcador temporal claro del otro
extremo). No es la única opción — cualquier paradigma con las mismas dos anclas
temporales (inicio de estímulo, respuesta motora) sirve. Requisito de diseño: el paradigma
debe estar **balanceado** de forma que la fase tardía no pueda explicarse por una entrada
sensorial tardía independiente (p. ej., sin un segundo estímulo o retroalimentación
externa cerca de la ventana tardía) — si no, la Sección 6 (vía independiente) deja de ser
una alternativa real y la comparación pierde sentido.

**Decisión de diseño que hay que resolver antes de correr el Test 2 (§6): ¿el paradigma
repite un estímulo único/idéntico entre ensayos, o lo varía?** No es un detalle — determina
qué versión del Test 2 es válida. Si varía, registrar la identidad/categoría de estímulo
de cada ensayo como una medida más (§5): sin eso, el Test 2 no se puede corregir.

- **Estímulo repetido/idéntico**: la versión simple del Test 2 (§6) es válida tal cual.
- **Estímulo variable**: obligatorio usar la versión condicionada por estímulo del Test 2
  — de lo contrario el test puede confundir "ambas fases codifican el mismo estímulo,
  por vías separadas" con rastreo de fuente genuino, y dar positivo incluso sin ningún
  retorno reflexivo (ver §6 y la Sec. 9.1 de `AFH_modelo_matematico_pliegue.md`).

## 4. Señales a registrar

| Señal | Dónde | Para qué |
|---|---|---|
| LFP de alta frecuencia de muestreo | CM/Pf, CL | fase temprana y tardía (fuerza y contenido) |
| LFP simultáneo | corteza relevante a la tarea | idem, lado cortical del lazo |
| Proxy de arousal/enganche por ensayo | pupilometría, potencia espectral basal pre-estímulo, o tiempo de reacción | control de confusores (§7) — **esta señal es tan importante como las dos anteriores**, ver §8 |
| Marcadores conductuales | inicio de estímulo, tiempo y ocurrencia de respuesta, precisión | define las ventanas de fase temprana/tardía y filtra ensayos válidos |

## 5. Medidas por ensayo

Para cada ensayo válido, calcular:

- **Fuerza de la fase temprana** ($E$): una métrica direccional tálamo→corteza en la
  ventana temprana (p. ej. transferencia de fase, entropía de transferencia, o Granger
  direccional) resumida en un escalar.
- **Fuerza de la fase tardía** ($L$): la misma clase de métrica, dirección corteza→tálamo,
  ventana tardía.
- **Contenido de la fase temprana** ($e$): una featurización multivariada de la actividad
  temprana (forma espectral, forma de onda, o cualquier representación de mayor
  dimensión que un solo número) — necesaria sólo para el Test 2 (§6).
- **Contenido de la fase tardía** ($l$): análogo, mismo espacio de representación.
- **Proxy de arousal** ($\hat z$): un valor por ensayo de la señal elegida en §4.
- **Identidad/categoría de estímulo**: obligatoria si el paradigma varía el estímulo entre
  ensayos (§3) — sin esta medida, la versión condicionada del Test 2 (§6) no se puede
  ejecutar.
- **Diferencial de latencia**: separación temporal entre el pico/inicio de la fase
  temprana y el de la fase tardía.

## 6. Plan de análisis — en este orden

**Test 1 — Correlación de magnitud.** ¿$E$ y $L$ covarían más allá de lo que $\hat z$ ya
explica? Correr primero la correlación cruda (documentar que por sí sola no sirve de
evidencia — covariará bajo ambas hipótesis si comparten arousal) y después la correlación
parcial controlando $\hat z$. Éxito: parcial significativa y positiva. Fracaso: no
concluir todavía — pasar al Test 2 antes de descartar.

**Test 2 — Identificabilidad de contenido (rastreo de fuente).** Con los vectores de
contenido $e$, $l$: descontar $\hat z$ de ambos, ajustar un mapeo temprano→tardío en una
mitad de los ensayos, y en la otra mitad preguntar si el contenido tardío predicho para
el ensayo $i$ se parece más a su propio contenido tardío real que al de otros ensayos con
arousal similar (exactitud top-1, contra un nulo por permutación **dentro de cada estrato
de arousal**, nunca global). Éxito: exactitud por encima del nulo. Este test puede ser
positivo aunque el Test 1 no lo sea — no lo descarten sólo por eso.

**Cuál versión correr depende de la decisión de diseño de §3:**
- **Estímulo repetido/idéntico** → correr el Test 2 tal como se describe arriba.
- **Estímulo variable entre ensayos** → correr la versión **condicionada por estímulo**:
  incluir la categoría de estímulo (además de $\hat z$) en la regresión que descuenta el
  confusor, y restringir el pool de emparejamiento a ensayos de la **misma** categoría de
  estímulo, no sólo del mismo estrato de arousal. Sin esta corrección, un positivo en el
  Test 2 puede significar sólo "ambas fases codifican qué estímulo fue", no rastreo de
  fuente — un falso positivo bajo la hipótesis de vía independiente, no evidencia a favor
  de la reflexiva. Costo esperado: menor potencia, porque sólo la fluctuación
  idiosincrática ensayo-a-ensayo queda como señal discriminante.

  **Validado, no sólo esperado** (`test_rastreo_de_fuente.py`, Bloque 6): repitiendo la
  versión condicionada 80 veces por $N\in\{400,800,1600,2400\}$, la FPR con proxy ruidoso
  se mantiene entre 0.013–0.025 (sin la tendencia creciente con $N$ que sí muestra la
  prueba de magnitud, §8), y el poder bajo un acoplamiento reflexivo moderado es 1.000 en
  los cuatro tamaños — la corrección no destruye la sensibilidad, al menos para ese tamaño
  de efecto. No se caracterizó todavía el tamaño de efecto mínimo detectable tras
  condicionar por estímulo (haría falta repetir con acoplamientos más débiles).

**Test 3 — Covarianza H\*/∇.** Dentro de los ensayos donde el proxy de convergencia
talamocortical (H\*) está presente, ¿su fuerza covaría con la fuerza del retorno
específico (∇)? Umbral pre-registrado: $r>0.50$. **Sobre este número**: es la convención
de "efecto grande" que ya trae el manuscrito original (Predicción 4), no algo derivado
aquí de una propiedad específica de AFH — no hay una respuesta de primeros principios a
"por qué 0.50 y no 0.35". Tratarlo como convención pre-registrada heredada, y decirlo así
si un revisor pregunta, en vez de buscarle una justificación post hoc.

**Test 4 — Diferencial de latencia.** ¿La separación temprana/tardía cae en el rango
150–550 ms? Esto es descriptivo, no discrimina Tipo A de Tipo B por sí solo, pero es
parte del perfil esperado.

**Test 5 — Si hay datos de sueño/anestesia disponibles en la misma cohorte:** comparar
N3 contra anestesia profunda en las mismas métricas específicas de los ILN (disociación
esperada: H presente con volumen colapsado en N3 vs. H ausente en anestesia).

## 7. Qué significa cada combinación de resultados

**Tests 1 y 2 (el núcleo — origen de la señal).**

| Test 1 (magnitud) | Test 2 (contenido) | Lectura |
|---|---|---|
| Positivo | Positivo | Evidencia convergente fuerte a favor de Tipo B |
| Negativo | Positivo | Tipo B presente pero codificado en patrón, no en fuerza — **no** interpretar el Test 1 negativo como refutación |
| Positivo | Negativo | Covarianza de fuerza sin rastro de contenido específico — revisar si la covarianza es artefacto del confusor compartido |
| Negativo | Negativo | Evidencia en contra de Tipo B en esta cohorte/paradigma |

**Regla combinada para la familia completa de cinco tests — declarada antes de tocar
datos, no ajustada después de verlos.** Esto extiende la tabla de arriba a Tests 3–5 y
fija una postura sobre comparaciones múltiples, siguiendo la misma disciplina de
validación de un solo intento que ya rige en `ScientificReports_revision/` (ECLIPSE) —
cinco pruebas sueltas sin regla de combinación declarada por adelantado es exactamente el
jardín de bifurcaciones que ese marco existe para prevenir:

- **Corroboración fuerte**: Tests 1 y/o 2 positivos (según §6), **y** Test 3 ($r>0.50$)
  también positivo. Tests 4 y 5 son consistentes con el perfil esperado pero no
  necesarios para esta lectura — son descriptivos (Test 4) o condicionales a tener datos
  de sueño/anestesia (Test 5).
- **Corroboración parcial**: Tests 1 y/o 2 positivos, pero Test 3 no alcanza $r>0.50$.
  Reportar como evidencia de origen reflexivo sin confirmación de la covarianza H\*/∇ —
  no redondear hacia "corroboración fuerte".
- **Refutación**: Tests 1 y 2 ambos negativos (bajo la versión de Test 2 correcta para el
  paradigma usado, §6) — retroceder a interpretación causal o correlacional (§9),
  independientemente de qué digan Tests 3–5.
- **Comparaciones múltiples**: con cinco tests y varios umbrales, no corregir
  post-hoc — declarar antes de correr los datos cuáles de los cinco son confirmatorios
  (Tests 1–3, los que entran en la regla de arriba) y cuáles son exploratorios/descriptivos
  (Tests 4–5, que informan pero no mueven la aguja de corroboración/refutación por sí
  solos). Esa distinción, hecha por adelantado, es lo que evita tener que corregir por
  cinco compariones después — sólo Tests 1–3 compiten por el mismo presupuesto de error.

## 8. Tamaño muestral y calidad del proxy de arousal

Basado en la simulación de poder (`simulacion_acoplamiento_fase_temprana_tardia.py`,
parámetros ilustrativos, no calibrados a ningún dataset real):

- Con un proxy de arousal de calidad **moderada** y un acoplamiento reflexivo de tamaño
  moderado, se necesitan aproximadamente 150–200 ensayos por sujeto (o su equivalente
  agregado) para poder razonable (~0.8) en el Test 1.
- **Hallazgo que pesa más que el número anterior**: con un proxy de arousal ruidoso, la
  tasa de falsos positivos del Test 1 *crece* con el número de ensayos en vez de bajar.
  Antes de decidir cuántos ensayos juntar, hay que evaluar qué tan bueno es el proxy de
  arousal disponible — un proxy pobre no se arregla con más datos, se arregla con un mejor
  proxy o con un modelo que module explícitamente su error de medición.
- Recomendación concreta: reportar el Test 1 con al menos dos proxies de arousal distintos
  si están disponibles (p. ej. pupila y potencia basal), y tratar con escepticismo una
  significancia que sólo aparece con uno de los dos.

## 9. Condiciones de falsación (pre-registradas)

Retroceder de la posición constitutiva (Tipo B) a una interpretación causal o
correlacional si ocurre cualquiera de:

1. El Test 1 y el Test 2 (en la versión de §6 correcta para el paradigma usado —
   condicionada por estímulo si el estímulo varía) son ambos negativos.
2. La correlación H\*/∇ (Test 3) es menor a 0.50 de forma consistente.
3. Emergen disociaciones sistemáticas y replicables entre ∇ operativo (bajo cualquiera de
   los dos tests) y la presencia reportada, en cohortes con reporte fenomenológico
   disponible.

## 10. Limitaciones conocidas de este diseño

- La cobertura anatómica de ILN en cohortes implantadas por indicación clínica no está
  garantizada — verificar antes de comprometer análisis.
- El proxy de arousal es, en la práctica, siempre una medición imperfecta del confusor
  real — el Test 1 hereda ese límite (§8); el Test 2 lo mitiga parcialmente al estratificar,
  pero no lo elimina.
- **Si el paradigma varía el estímulo entre ensayos, el Test 2 sin condicionar por
  identidad de estímulo puede dar un falso positivo bajo la hipótesis de vía
  independiente** — el estímulo es causa común de ambas fases por vías separadas, y eso no
  lo resuelve estratificar sólo por arousal. Usar obligatoriamente la versión condicionada
  por estímulo en ese caso (§6); aceptar la pérdida de potencia que eso implica en vez de
  usar la versión sin condicionar porque "da resultados más claros".
- **Techo general, no específico del arousal ni del estímulo**: incluso el Test 2 en su
  versión mejor ejecutada sigue siendo una inferencia observacional que descansa en el
  supuesto de que ningún otro estado específico del ensayo —no sólo arousal o identidad de
  estímulo, sino cualquier otra causa común no medida (adaptación, atención encubierta,
  estado de red pre-estímulo)— module en paralelo ambas fases. Ninguna cantidad de
  condicionamiento observacional cierra esa posibilidad por completo; sólo una
  intervención (perturbar el tálamo y observar si el contenido tardío específico cambia)
  la rompería, y eso casi seguro excede lo disponible en una cohorte clínica. Esto es lo
  máximo que un diseño no-interventivo puede establecer sobre la Predicción 6 — nombrarlo
  ahora, no escalar a un sexto test si el resultado queda ambiguo.
- Este diseño prueba la Predicción 6. No prueba la hipótesis constitutiva completa (que la
  convergencia sea *idéntica* a la presencia, no sólo su correlato) — eso excede lo que
  cualquier diseño puramente neural puede establecer, y sigue dependiendo de reporte en
  primera persona bajo manipulaciones graduales (Predicción 1), no cubierto aquí.
- Ningún análisis aquí especificado se ha corrido sobre datos reales; los umbrales de
  tamaño muestral (§8) son ilustrativos hasta que se calibren con las propiedades reales
  del dataset disponible (varianza de las métricas, número de ensayos por sujeto, proxies
  de arousal efectivamente registrados).

---

**Referencias internas**: la justificación matemática de cada test está en
`AFH_modelo_matematico_pliegue.md` (§4 para ∇, §9 para las Predicciones, §9.1 para el
Test 2, §10 para las condiciones de falsación en su forma general). El código que
implementa y valida cada test sobre datos sintéticos: `simulacion_acoplamiento_fase_temprana_tardia.py`
(Test 1) y `test_rastreo_de_fuente.py` (Test 2).
