# ECLIPSE Workbench: diseño (iteración 1)

> **Estado:** documento de diseño. **No hay código del Workbench todavía.** Nada de lo descrito
> aquí existe hasta que se implemente y se pruebe según los criterios de aceptación de la §5.

Referencia del core: `ScientificReports_revision/engine/eclipse_core.py` (ECLIPSE v2.0 Core,
instrument version 3.1, `EclipseFramework.VERSION = "3.1.0"`), con dependencias fijadas en
`ScientificReports_revision/requirements_frozen.txt`.

---

## 0. Qué hace hoy el core (revisión previa)

Antes de diseñar, leí el core completo en las partes relevantes y ejecuté pruebas sobre él
(numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, las versiones de
`requirements_frozen.txt`). Un hallazgo marcado **[verificado]** se reprodujo ejecutando el
código. Uno marcado **[lectura]** sale de leer el código y no se ejecutó.

### 0.1 Mapa de lo existente

| Pieza | Ubicación | Qué hace |
|---|---|---|
| `EclipseConfig` | `eclipse_core.py:121` | Configuración: semilla, ratios 70/30, `n_folds_cv`, `non_interactive`, pesos del EIS, umbral del auditor |
| `FalsificationCriteria` | `eclipse_core.py:92` | Criterio: `name`, `threshold`, `comparison`, `description`, `is_required` |
| `EclipseFramework` | `eclipse_core.py:2345` | Orquestador original: `stage1_irreversible_split` … `stage5_final_assessment`, `verify_integrity` |
| `EclipseIntegrityScore` | `eclipse_core.py:254` | EIS, 5 componentes y pesos por defecto 0.25/0.25/0.20/0.15/0.15 |
| `StatisticalTestDataSnooping` | `eclipse_core.py:780` | STDS: z = (holdout − media_CV)/sd_CV con cortes empíricos `EMPIRICAL_SCREEN` (`:844`) |
| `CodeAuditor` / `StaticCodeAnalyzer` | `eclipse_core.py:1575` / `:1255` | Cribado estático por AST y patrones; admite `.ipynb` |
| `CryptographicCommitment` | `eclipse_core.py:1929` | Compromiso SHA-256 para el modo no interactivo |
| `EclipseReporter` | `eclipse_core.py:2008` | Reportes HTML/TXT |
| Wrappers | `engine/{eis,stds,auditor}_wrapper.py` | Adaptadores en memoria para los estudios; no reimplementan lógica |

El estado del core está en tres JSON en disco (`*_SPLIT_IMMUTABLE.json`,
`*_CRITERIA_BINDING.json`, `*_FINAL_RESULT.json`). Las banderas `_split_completed` y similares
se reconstruyen al instanciar, según qué archivos existen (`_load_existing_state`, `:2413`).

### 0.2 Hallazgos que condicionan el diseño

Ninguno se corrige tocando el core en esta propuesta. La capa de orquestación los compensa
(§1.3), y los cambios al core quedan como decisión del autor (§7, pregunta 3).

1. **El hash del split no compromete la asignación dev/holdout.** [verificado]
   `split_hash = sha256(f"{seed}_{sorted(data_identifiers)}")` (`:2477`) solo codifica la semilla
   y el *conjunto* de IDs. Intercambié un ID entre `development_ids` y `holdout_ids` en el JSON y
   `verify_integrity()` siguió devolviendo `all_valid: True`.
2. **IDs con tipos numpy rompen la verificación.** [verificado] Con numpy ≥ 2, `str(sorted(ids))`
   produce `np.int64(0)` si los IDs son escalares numpy (por ejemplo `list(np.arange(50))`), pero
   el JSON guarda enteros nativos. Resultado: `verify_integrity()` → `all_valid: False` en un
   split legítimo.
3. **El split depende del orden de entrada, no solo del conjunto.** [lectura]
   `np.random.seed(seed)` + `shuffle` sobre la lista en el orden recibido (`:2468`). La misma
   semilla con los IDs en otro orden da otro split. Además usa el RNG global de numpy (efecto
   lateral).
4. **El bloqueo de la Etapa 4 no existe hasta la Etapa 5.** [verificado] `stage4_single_shot_validation`
   solo se niega si existe `results_file` (`:2682`), pero ese archivo lo escribe
   `stage5_final_assessment`. Ejecuté la Etapa 4 tres veces seguidas antes de la Etapa 5 y las
   tres corrieron. Además `force=True` se salta el bloqueo, y `_validation_completed = True` se
   marca incluso si la validación falló (`:2754`).
5. **El modo no interactivo automatiza la confirmación.** [lectura] Con `non_interactive=True`
   la confirmación es la cadena `"AUTOMATED"` (`:2708`). Eso contradice el principio 3, así que el
   Workbench **nunca** usará ese modo.
6. **La Etapa 3 hace CV por filas, no por grupos.** [lectura] `KFold` sobre
   `range(len(development_data))` (`:2592`, `:2600`). Si hay varias filas por unidad (sujetos con
   muchas ventanas, como en `example_v3_demo`), filas del mismo sujeto caen en train y en
   validación del mismo fold. El split de la Etapa 1 sí es por unidad, pero el CV no.
7. **La Etapa 2 no es obligatoria antes de la 3.** [lectura] Solo emite un `warnings.warn`
   (`:2583`).
8. **El techo del EIS es 0.975, y con 70/30 es 0.955.** [verificado]
   - `compute_preregistration_score` da +0.4 solo si la validación *no* se ha completado (`:353`).
     En la Etapa 5 siempre se completó, así que se suma 0.3 (`:356`) y el máximo del componente
     es **0.9**. Eso da 0.25 × 0.1 = 0.025 de techo perdido.
   - Con split 50/50 y n ≥ 100, EIS = **0.9750**. Con el **70/30 por defecto del propio framework**,
     `split_strength` = 0.899 y el EIS máximo es **0.9548**.
9. **`S_leak` premia que el holdout salga peor que el desarrollo.** [verificado] El máximo de
   `S_leak` se alcanza con z → −∞. Con 700/300: holdout muy por debajo de la media de dev da
   EIS = 0.9548, y holdout igual a la media de dev da 0.9369. Es una vía de Goodhart concreta:
   un modelo peor en holdout sube el EIS.
10. **Dentro de una herramienta, 4 de los 5 componentes quedan casi constantes.** [lectura]
    Preregistro, adherencia, transparencia y el bono de hash del split (`:421`) premian que
    existan campos (`registration_date`, `criteria_hash`, `binding_declaration`,
    `integrity_verification`) que la herramienta escribe automáticamente. En un protocolo
    completado con el Workbench esos componentes estarán en su máximo o cerca. Lo único que
    variará será `split_strength` (ratio y n) y `S_leak`. Hay que decirlo en la interfaz.
11. **Textos del core que chocan con los principios 4 y 5.** [lectura]
    - `_interpret_eis` (`:676`) etiqueta "EXCELLENT / VERY GOOD / … / VERY POOR".
    - `generate_eis_report` escribe "✅ Study meets high methodological standards" si EIS ≥ 0.80
      (`:758`).
    - `EclipseReporter` pinta el veredicto verde/rojo (`:2032`).
    - `CodeAuditor` asigna `confidence=0.90` fijo a todo hallazgo (`:1718`); no es un valor empírico.
    - El auditor devuelve `passed` con un umbral de 70 (`:1684`), que es normativo.
12. **El auditor marca todos los casos legítimos con apariencia sospechosa.** En
    `outputs/study3/study3_summary.json`, `near_miss`: 50/50 marcados, especificidad 0.0, con
    los 5 tipos de trampa (`cv_fold_named_validation_set`, `stat_test_object_named_test`,
    `print_development_metric`, `default_then_set_threshold`, `unrelated_print_then_metric_var`).
    La especificidad 1.0 en scripts conformes es sobre un corpus diseñado (Estudio 3).
13. **STDS tiene una tabla de cortes acotada.** `EMPIRICAL_SCREEN` cubre K ∈ {3, 5, 10, 20} y
    M ∈ {1, 2, 3, 5}. Fuera de la tabla usa la entrada vecina conservadora (`cutoffs_exact_for_k_m`).
    Si no hay valores por fold asume K = 5 (`:966`). Las métricas con sd_CV = 0 se descartan en
    silencio. Con K = 3 el corte al 95 % es 4.06 (M = 1) y 9.94 (M = 5).
14. **El estado del framework depende del proceso.** [lectura] La Etapa 3 no persiste nada.
    Si las Etapas 3 y 5 corren en procesos distintos, `_development_completed` queda en `False`
    al llegar a la 5 y `compute_protocol_adherence` cae de 1.0 a 0.5 (`:427`). Un orquestador
    con estado en disco tiene que reconstruir esas banderas fielmente, o el EIS diferiría del
    que daría el core en una ejecución de un solo proceso.
15. **Hashes semánticos dependientes de Python.** El hash de criterios usa
    `json.dumps(..., sort_keys=True)` con separadores por defecto y `ensure_ascii=True` (`:2539`).
    El del split usa la `repr` de una lista de Python. Un verificador en R tendría que replicar
    ambos formatos, de ahí la capa de hashes por bytes de archivo (§2.6).

### 0.3 Limitaciones de fondo que ningún software local puede eliminar

- **El investigador siempre tiene sus datos originales.** El Workbench puede garantizar que *su
  propio código* no lee el holdout antes de la Etapa 4. No puede impedir que la persona abra el
  CSV original con otro programa.
- **Crear protocolos nuevos hasta obtener el resultado deseado** (seed shopping o protocol
  shopping) no se puede impedir localmente. Solo se puede hacer visible: un registro del espacio
  de trabajo y un anclaje externo del compromiso.
- **Los timestamps locales son el reloj de la máquina.** Una cadena de hashes prueba consistencia
  interna, no tiempo. Quien controla la máquina puede regenerar la cadena completa. La única
  garantía temporal frente a terceros es anclar un hash fuera (registro en OSF, OpenTimestamps,
  correo a un tercero, etc.) *antes* de liberar el holdout.

### 0.4 Contraste con el paper publicado

Fuente: Sjöberg Tala, *Scientific Reports*, DOI 10.1038/s41598-026-74760-5 (Article in Press,
publicado el 14-10-2026). El paper declara como versión de referencia la del commit
`93339b4c2d3f56db1dab87a5dafac7641cf921a7` (Zenodo 10.5281/zenodo.22760496). Es el mismo commit
`93339b4` sobre el que se hizo la revisión de §0.2. Las citas indican la sección del paper.

**Coincide con el código:**

- Fórmulas del EIS (§2.6.4): S_pre ≤ 0.9 tras la validación, techo de 0.975, κ(N) del split,
  valores posibles de S_prot, logística de S_leak (0.98 / 0.88 / 0.50 / 0.12 en z = −1 / 0 / +1 / +2)
  y pesos 0.25/0.20/0.25/0.15/0.15.
- STDS: z con sd poblacional, cortes de la Tabla 5 idénticos a `EMPIRICAL_SCREEN`, sin p-valores y
  con las dos limitaciones de la calibración (§2.6.4, §3.2).
- Auditor: penalizaciones 30/15/5/2, umbral recomendado ≥ 70 "heurístico, no estricto",
  especificidad 0 % en near-miss (Tabla 7) y solo Python (§4.4).

**El paper describe algo que el código no hace.** Esto es lo más importante que salió de la
lectura:

| # | Lo que dice el paper | Lo que hace el código | Consecuencia |
|---|---|---|---|
| P1 | Etapa 1, *Algorithm*: "A SHA-256 hash is computed over the indices of D_hold, the seed s, the split ratio, and a timestamp… Any subsequent modification to the split indices, seed, or sample identifiers changes the hash" | `sha256(f"{seed}_{sorted(todos_los_ids)}")` (`eclipse_core.py:2477`): no incluye qué IDs son holdout, ni el ratio, ni el timestamp | **Verificado:** mover un ID de holdout a dev no cambia el hash y `verify_integrity()` da válido (§0.2-1). La afirmación del paper no se cumple en la implementación |
| P2 | Etapa 1, *Implementation details*: con clases desbalanceadas, muestreo estratificado; con series temporales, split temporal (primer 70 % / último 30 %) | Solo barajado aleatorio con semilla (`:2468`) | El Workbench no puede ofrecer estratificado ni temporal sin una segunda implementación del split |
| P3 | Etapa 3: "A final model is then trained on all of D_dev" | `stage3_development` no entrena un modelo final. `stage4` recibe un `final_model` que provee el usuario | El Workbench debe hacerlo explícito en el paso "congelar" (§3.4) |
| P4 | Etapa 4: "File locking ensures that after validation, the holdout data file becomes read-only. Result immutability is achieved by writing validation results to a version-controlled file with timestamp" | No hay bloqueo de archivo. Los resultados de la Etapa 4 no se escriben hasta la Etapa 5. La Etapa 4 se puede repetir | **Verificado** (§0.2-4) |
| P5 | Etapa 4: confirmación tecleada "I ACCEPT SINGLE-SHOT VALIDATION" | Igual en modo interactivo, pero el modo `non_interactive` la sustituye por `"AUTOMATED"` (`:2708`). El paper no menciona ese modo | — |
| P6 | §2.6.1: tres bandas (0.80–1.00 "strong operationalized features", 0.60–0.79 moderate, < 0.60 weak), 0.70 umbral provisional; "EIS does not measure methodological rigor objectively" | `_interpret_eis` usa seis etiquetas, "EXCELLENT – Exceptional methodological rigor" … "VERY POOR – Critical methodological flaws". El reporte imprime "✅ Study meets high methodological standards" con EIS ≥ 0.80 | El texto del código contradice el encuadre del paper |
| P7 | §2.6.1: los pesos son "literature-informed heuristic defaults… not derived from meta-analysis", citando las refs. 17, 25, 8 y 18 sin cifras | `WEIGHT_JUSTIFICATIONS` incluye cifras ("~60 %", "17/20 surveyed ML studies") que **no aparecen en el paper** | Responde la antigua pregunta 8: la interfaz debe usar la redacción del paper, no la del código |
| P8 | Etapa 2, paso 4: el archivo "is marked immutable via version control" | El core no interactúa con git | El Workbench puede ofrecer un commit automático opcional |

El paper es explícito sobre los límites: "These mechanisms structurally reduce opportunities for
accidental iterative access but do not prevent deliberate circumvention" (Etapa 4). Las
discrepancias P1 y P4 no contradicen esa advertencia, pero sí son afirmaciones concretas sobre
mecanismos que hoy no existen en el código. El autor decide cómo tratarlas (§7, pregunta 0).

**Decisiones del diseño que el paper respalda:**

- §4.2, *Practical implications*: "Adopting the workflow… We recommend it. Using the scores — EIS
  in particular — … the evidence reported here does not support it." Esto justifica que el
  **modo solo disciplina sea el predeterminado** y que el modo con scores sea opcional (§2.1).
- §4.2, editores: "the appropriate analogue at this stage is deposition of the artifacts, not
  publication of the score." El producto central del Workbench es el bundle depositable, no el
  score.
- §4.1, *Goodhart*: "Researchers who learn that preregistration completeness, split entropy, and
  transparency indicators raise EIS may engineer compliance without engineering rigor." Una
  herramienta que rellena automáticamente esos campos materializa ese riesgo (§0.2-10), así que
  la interfaz tiene que decirlo.
- §4.4: solo confirmatorio ("does not target exploratory or hypothesis-generating research").
  Va en el onboarding.
- §2.6.3: lista de supuestos y condiciones límite del STDS (datos temporales, no i.i.d., clases
  muy desbalanceadas, K pequeño, etc.). El Workbench pregunta en la Etapa 1 por la estructura de
  los datos y muestra en la Etapa 5 las condiciones límite que aplican. Es la misma lista del
  paper, sin validación nueva.
- Etapa 1, *Implementation details*: "Sacred seed selection requires researchers to declare the
  seed before seeing data… The seed must be justified and recorded." Esto reemplaza mi propuesta
  anterior de derivar la semilla automáticamente (§3.2).

---

## 1. Arquitectura de alto nivel

### 1.1 Capas

```
┌──────────────────────────────────────────────────────────────────────┐
│ Interfaces            CLI (eclipse …)          Web local (FastAPI +  │
│ (sin lógica)                                    Jinja2/htmx)         │
├──────────────────────────────────────────────────────────────────────┤
│ Servicio de protocolo  ProtocolService: única API pública            │
│ (orquestación)         create / commit_split / register_criteria /   │
│                        run_development / freeze / release_holdout /  │
│                        assess / export / abandon                     │
├───────────────┬───────────────┬───────────────┬──────────────────────┤
│ Máquina de    │ Event log     │ Bundle store  │ Runners              │
│ estados       │ (append-only, │ (carpeta +    │ PythonRunner (subpro-│
│ (derivada del │ hash-chained) │ MANIFEST)     │ ceso), ExternalRunner│
│ log)          │               │               │ (R/notebooks, fase 4)│
├───────────────┴───────────────┴───────────────┴──────────────────────┤
│ core_bridge.py: ÚNICO módulo que importa eclipse_core                │
│ (mismo patrón que engine/*_wrapper.py)                               │
├──────────────────────────────────────────────────────────────────────┤
│ CORE (sin cambios): ScientificReports_revision/engine/eclipse_core.py│
└──────────────────────────────────────────────────────────────────────┘

        Fuera del árbol de dependencias:
┌──────────────────────────────────────────────────────────────────────┐
│ Verificador independiente: verify_bundle.py                          │
│ Solo stdlib. No importa ni el Workbench ni el core.                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.2 Componentes

| Componente | Responsabilidad | Notas |
|---|---|---|
| `core_bridge` | Llamar a `EclipseFramework`, `EclipseIntegrityScore`, `StatisticalTestDataSnooping` y `CodeAuditor` sin reimplementar su lógica | Registra el SHA-256 de `eclipse_core.py` al crear el protocolo y se niega a continuar si cambia a mitad de protocolo. Nunca pasa `force=True` ni `non_interactive=True` |
| `ProtocolService` | Toda transición pasa por aquí. Valida precondiciones, escribe el evento y produce artefactos | CLI y web son clientes delgados de este servicio |
| `StateMachine` | Calcula el estado **reproduciendo el log**. No hay un campo "estado" editable | Elimina la clase de bugs "el estado dice X pero los archivos dicen Y" |
| `EventLog` | `log.jsonl` append-only. Cada evento incluye `prev_hash` y `hash` | Ver §2.4 |
| `BundleStore` | Estructura de carpetas, `MANIFEST.sha256` y exportación `.zip` | Formato abierto; nada de bases de datos |
| `SealedHoldout` | Guarda la partición holdout en un archivo separado. La única función que la abre comprueba el estado `HOLDOUT_RELEASED` | Es fricción, no criptografía contra el propio usuario (§0.3). Modo custodio opcional en la fase 5 |
| `PythonRunner` | Ejecuta el código del usuario en un **subproceso**. En la Etapa 3 el subproceso solo recibe la ruta de los datos de desarrollo | La Etapa 4 lanza otro subproceso con el modelo congelado y el holdout |
| `ExternalRunner` (fase 4) | R y notebooks: el usuario sube métricas por fold y de holdout producidas fuera | Se marca en el bundle como `results_provenance: "self_reported"` |
| `ReportBuilder` | Reporte HTML/MD neutral (sin semáforos) | No usa `EclipseReporter` (§0.2, punto 11) |
| `verify_bundle.py` | Verificación independiente | Ver §2.6 |

### 1.3 Cómo compensa la orquestación los hallazgos sin tocar el core

| Hallazgo | Compensación en la capa de orquestación |
|---|---|
| 1. El hash no compromete la asignación | `stage1/split_commitment.json` con `assignment_hash = sha256(JSON canónico {seed, dev_ids ordenados, holdout_ids ordenados, ratio})`, más el SHA-256 del archivo de datos subido. Se conserva el `split_hash` del core por compatibilidad |
| 2. Tipos numpy | Los IDs se normalizan a `str` antes de llamar al core, con un charset restringido `[A-Za-z0-9._-]` para que la `repr` de Python sea trivial de replicar en otro lenguaje (o se mapean a claves opacas y se guarda la tabla de mapeo) |
| 3. Dependencia del orden | Los IDs se ordenan antes de `stage1_irreversible_split`. Así el split es función de (semilla, conjunto de IDs) y el verificador puede recalcularlo |
| 4. Bloqueo de la Etapa 4 | Evento `HOLDOUT_RELEASED` escrito **antes** de llamar a `stage4_single_shot_validation`, más el archivo `stage4/LOCK`. Un segundo intento se rechaza aunque el primero haya fallado |
| 5. Modo no interactivo | Prohibido. La confirmación la recoge el Workbench (TTY o formulario web con nonce de un solo uso) y se pasa al `input()` del core en un único punto controlado de `core_bridge` |
| 6. CV por filas | El Workbench advierte en la Etapa 1 si hay más de una fila por unidad. Arreglarlo exige cambiar `stage3_development` (§7, pregunta 3). No se inventa un CV agrupado dentro del Workbench, porque eso sería una segunda fuente de verdad |
| 7. Etapa 2 opcional | El Workbench la exige: no hay transición de `SPLIT_COMMITTED` a desarrollo |
| 8–10. EIS | Se muestra como perfil, con el techo real por componente y la advertencia de que 4 componentes reflejan artefactos generados por la herramienta |
| 11. Textos del core | `generate_reports=False` en la Etapa 5. Los TXT que `compute_integrity_metrics` escribe igual (`:2903`) se guardan en `stage5/raw_core_outputs/`, etiquetados como salida cruda; no se muestran en la interfaz |
| 14. Estado por proceso | `core_bridge` reconstruye las banderas del framework desde el log verificado (por ejemplo `_development_completed = True` solo si existe el evento `DEVELOPMENT_FROZEN`). Un test de equivalencia compara el resultado con una ejecución de un solo proceso del core |

### 1.4 Flujo de datos

```
datos.csv ──hash──► Etapa 1 ─► dev.parquet ─────────────────┐
                    │          holdout.sealed (no se lee)   │
                    ▼                                       ▼
          SPLIT_IMMUTABLE.json + split_commitment.json   Etapa 3 (subproceso:
                    │                                    solo dev) ─► N corridas
                    ▼                                       │  de CV (todas logueadas)
          Etapa 2: CRITERIA_BINDING.json (+ plan)           │
                    │   └─► [anclaje externo opcional]      ▼
                    └──────────────────────────────► FREEZE: código + modelo + hash
                                                            │ preflight sobre un fold de dev
                                                            ▼
                         confirmación tecleada ─► Etapa 4 (subproceso: modelo
                                                  congelado + holdout, UNA vez)
                                                            ▼
                                       Etapa 5: core.stage5 → FINAL_RESULT.json
                                       + EIS/STDS (core) + auditor sobre el código congelado
                                                            ▼
                                       report/ + MANIFEST.sha256 → bundle.zip
                                                            ▼
                                             verify_bundle.py (terceros)
```

**El auditor se ejecuta sobre el código que realmente corrió.** En el modo PythonRunner, el
código auditado es exactamente el código congelado y ejecutado, ligado por hash. Hoy el auditor
analiza cualquier archivo que se le pase, sin vínculo con lo ejecutado. En el modo ExternalRunner
ese vínculo no existe, y el reporte lo dice.

---

## 2. Modelo de estado del protocolo

### 2.1 Entidades

| Entidad | Campos clave | Mutabilidad |
|---|---|---|
| `Workspace` | ruta, `workspace_log.jsonl` (protocolos creados, sobre qué hash de datos) | Append-only |
| `Protocol` | `protocol_id` (ULID), título, investigador, `mode ∈ {discipline_only (predeterminado, §0.4), full}`, versión del Workbench, SHA-256 de `eclipse_core.py` y commit del core (referencia del paper: `93339b4`), `supersedes` (ID previo o nulo) | Inmutable tras la creación |
| `Dataset` | SHA-256 del archivo, formato, columna de unidad, n filas, n unidades | Inmutable tras la Etapa 1 |
| `SplitCommitment` | semilla, procedencia de la semilla, ratio, `assignment_hash`, `split_hash` del core | Inmutable |
| `CriteriaRegistration` | lista de `FalsificationCriteria`, `criteria_hash` (core), `n_folds_cv`, métricas declaradas, plan de análisis (texto, hasheado) | Inmutable |
| `ExternalAnchor` | tipo (osf, ots, manual), referencia, hash anclado | Append-only, opcional |
| `DevelopmentRun` | n.º de corrida, hash del código, parámetros, métricas por fold, agregados | Append-only; cualquier número |
| `FrozenModel` | corrida elegida, hash del código, hash del artefacto del modelo, resultado del preflight | Inmutable |
| `HoldoutRelease` | confirmación (frase, canal, nonce), timestamp | Única |
| `ValidationResult` | métricas, estado (success/failed), error | Inmutable |
| `Assessment` | `FINAL_RESULT.json` del core, veredicto, EIS/STDS (si `mode=full`), auditoría | Inmutable |
| `Annotation` | respuesta del investigador a un hallazgo del auditor o del STDS | Append-only |
| `Abandonment` | motivo (texto obligatorio) | Terminal |

### 2.2 Estados y transiciones

```
 DRAFT ──commit_split──► SPLIT_COMMITTED ──register_criteria──► CRITERIA_REGISTERED
                                                                       │
                                     (anchor: evento opcional, no cambia el estado)
                                                                       ▼
                                                        DEVELOPING ◄──┐ run_development
                                                           │      └───┘ (N veces)
                                                     freeze_model
                                                           ▼
                                                     MODEL_FROZEN
                                                           │ release_holdout
                                                           │ (frase tecleada; IRREVERSIBLE)
                                                           ▼
                                                   HOLDOUT_RELEASED
                                              ┌────────────┴────────────┐
                                              ▼                         ▼
                                         VALIDATED              VALIDATION_FAILED (terminal
                                              │ assess           para el holdout; ver §7, pregunta 2)
                                              ▼
                                          ASSESSED ──export──► ASSESSED (+ bundle exportado)

 Desde cualquier estado no terminal: abandon(motivo) ──► ABANDONED (terminal)
```

No hay transiciones hacia atrás. "Rehacer" significa `create_protocol(supersedes=<id>)`, que
crea un protocolo nuevo. El anterior conserva su estado, y el reporte del nuevo muestra la cadena
de protocolos sustituidos y en qué estado quedó cada uno. Si el anterior llegó a
`HOLDOUT_RELEASED`, el nuevo muestra la advertencia permanente: *"El holdout de un protocolo
previo sobre estos datos ya fue observado; este protocolo no es independiente de ese
resultado."*

### 2.3 Bloqueos

| Artefacto | Se bloquea en | Qué significa el bloqueo |
|---|---|---|
| Archivo de datos, split, semilla, ratio | `commit_split` | El hash queda en el log. Cualquier cambio de bytes invalida el `MANIFEST` y el servicio se niega a abrir el protocolo |
| Criterios, `n_folds_cv`, plan de análisis | `register_criteria` | Ídem. No existe la operación "editar criterios" |
| Código y modelo | `freeze_model` | Snapshot copiado dentro del bundle y hasheado |
| Holdout | `release_holdout` | Se abre una vez. El evento se escribe antes de abrirlo |
| Modo (`full` / `discipline_only`) | `create` | Pasar de `discipline_only` a ver scores solo es posible después de `ASSESSED`, mediante el evento `SCORES_REVEALED`, que queda registrado |

### 2.4 Event log

Cada línea de `log.jsonl`:

```json
{"seq": 7, "type": "CRITERIA_REGISTERED", "ts": "2026-10-14T15:02:11Z",
 "actor": "cli|web", "payload": {"criteria_hash": "…", "file_sha256": "…"},
 "prev_hash": "<hash del evento 6>", "hash": "sha256(JSON canónico del evento sin 'hash')"}
```

El JSON canónico es UTF-8, `sort_keys`, `separators=(",", ":")` y `ensure_ascii=False`. Está
especificado en `docs/BUNDLE_SPEC.md` para que cualquier lenguaje lo reproduzca.

### 2.5 Invariantes (cada uno con su test)

| ID | Invariante | Cómo se prueba |
|---|---|---|
| I1 | Ningún código del Workbench abre `holdout.sealed` antes del evento `HOLDOUT_RELEASED` | Test con `open`/`pandas.read_*` instrumentados en todas las rutas de las Etapas 1–3 y en el preflight. `SealedHoldout.open()` lanza excepción si el estado no es el correcto |
| I2 | Como mucho un `HOLDOUT_RELEASED` por protocolo | Test de servicio más una regla del verificador |
| I3 | Los bytes de los criterios coinciden con el hash registrado en todo estado posterior | Test de manipulación: editar un umbral hace fallar la carga y el verificador |
| I4 | La asignación dev/holdout coincide con `assignment_hash` | Test de manipulación: intercambiar un ID (el caso que el core no detecta, §0.2-1) |
| I5 | El log es una cadena válida, `seq` es contiguo y los timestamps no decrecen | Tests de reordenar, borrar o insertar eventos |
| I6 | `MODEL_FROZEN` precede a `HOLDOUT_RELEASED`, y el hash del modelo y del código usados en la Etapa 4 es el congelado | Test de servicio y regla del verificador |
| I7 | La confirmación existe, coincide con la plantilla específica del protocolo y su canal es `tty` o `web` con nonce válido | Test: la CLI con stdin que no es TTY se niega. No existe flag, variable de entorno ni campo de configuración que la provea |
| I8 | El SHA-256 de `eclipse_core.py` no cambia durante un protocolo | Test que modifica una copia del core |
| I9 | El Workbench nunca pasa `force=True` ni `non_interactive=True` al core | Test estático (AST sobre `eclipse_workbench/`) y test de runtime con un espía en `core_bridge` |
| I10 | En `discipline_only` ninguna respuesta de API, plantilla HTML ni salida de CLI contiene valores de EIS, STDS o puntaje del auditor | Test sobre todas las rutas |
| I11 | El estado se deriva del log. Si el log y los archivos discrepan, el protocolo no se abre | Test de consistencia |
| I12 | El EIS y el STDS producidos por el Workbench son idénticos a los del core en una ejecución de un solo proceso con las mismas entradas | Test de equivalencia (§0.2-14) |
| I13 | Desarrollo exige criterios registrados | Test de servicio |

### 2.6 Verificador independiente (`verify_bundle.py`)

Es un solo archivo que usa únicamente la biblioteca estándar de Python ≥ 3.9. No importa ni el
Workbench ni el core. Tiene tres niveles:

1. **Nivel 1, integridad** (stdlib): el `MANIFEST.sha256` coincide con los bytes, la cadena del
   log está intacta, el orden de eventos es válido (I2, I5, I6, I13), los hashes de criterios y
   del split que aparecen en el log coinciden con los archivos, y `assignment_hash` es correcto.
   El `split_hash` y el `criteria_hash` del core se recalculan replicando su receta exacta.
2. **Nivel 2, recálculo aritmético** (stdlib): evaluación de cada criterio (comparaciones),
   agregados de CV (media y sd poblacional) y z por métrica del STDS. Es aritmética de
   verificación, no un segundo motor de scoring; el EIS no se recalcula aquí.
3. **Nivel 3, re-ejecución opcional** (`--recompute-split`, que requiere numpy, y
   `--recompute-scores`, que requiere el core fijado): recalcula el split con
   `RandomState(seed).shuffle` y el EIS/STDS con el core. Este nivel ya no es independiente del
   software, y el verificador lo dice en su salida.

Salida: un informe por regla (`OK` / `FALLA` / `NO VERIFICABLE`) y, siempre, la lista de **lo que
el verificador no puede comprobar**: el acceso al holdout fuera del Workbench, otros protocolos
sobre los mismos datos fuera de este workspace, y el tiempo real si no hay anclaje externo.

---

## 3. UX de las cinco etapas

Principios de presentación:

- Tipografía neutra.
- Un solo color para todas las barras.
- Sin verde ni rojo.
- Sin ✅/❌.
- Sin "aprobado".
- Sin ranking de protocolos.

Las acciones irreversibles describen qué se bloquea; no felicitan.

### 3.0 Onboarding y panel

```
┌ ECLIPSE Workbench ───────────────────────────────────────────────┐
│ Qué hace: impone el orden split → criterios → desarrollo →       │
│ holdout único → evaluación, y deja un registro verificable.      │
│                                                                  │
│ Qué NO hace:                                                     │
│  · No impide que usted abra sus datos originales con otro        │
│    programa.                                                     │
│  · El EIS no está calibrado contra resultados externos; no mide  │
│    rigor.                                                        │
│  · STDS es heurístico y autocalibrado; su tasa real de falsas    │
│    alarmas es mayor al 5 % tabulado.                             │
│  · El auditor marcó el 100 % (50/50) de scripts legítimos con    │
│    apariencia sospechosa en el Estudio 3.                        │
│                                                     [Leer más]   │
│                                                                  │
│ Protocolos                                      [+ Nuevo]        │
│  P-01J…  "Predicción X"   Etapa 3 · Desarrollo   creado 12-oct   │
│  P-01H…  "Piloto"         Abandonado (motivo: …) creado 02-oct   │
│                        (sin columnas de score en esta vista)     │
└──────────────────────────────────────────────────────────────────┘
```

**Nuevo protocolo:** título, investigador y **modo**, que queda fijo:

- *Solo disciplina* (**predeterminado**): no calcula ni muestra scores; el bundle contiene lo
  necesario para que terceros los calculen. El paper recomienda adoptar el flujo y dice que la
  evidencia no respalda usar los scores (§4.2).
- *Completo* (opcional): muestra el EIS, el STDS y el auditor en la Etapa 5.

Cada modo va acompañado de la cita del paper sobre Goodhart (§4.1). El onboarding también dice
que ECLIPSE es para investigación confirmatoria, no exploratoria (§4.4 del paper).

### 3.1 Línea de tiempo (vista persistente)

```
 ● 1 Split          bloqueado  sha256 3f9a…  [recibo]
 ● 2 Criterios      bloqueado  sha256 a71c…  [recibo] [anclar externamente]
 ◐ 3 Desarrollo     4 corridas registradas
 ○ 4 Holdout        requiere: modelo congelado
 ○ 5 Evaluación
                                        [Abandonar protocolo…]
```

### 3.2 Etapa 1: Split

1. Subir el archivo (CSV o Parquet). Se calcula y muestra el SHA-256 del archivo.
2. Elegir la **columna de unidad** (sujeto o paciente). Si hay más de una fila por unidad, se
   muestra un aviso de que el CV de la Etapa 3 del core es por filas (§0.2-6), con un enlace a la
   explicación.
3. Elegir el ratio (por defecto 70/30), con la nota "el EIS penaliza ratios distintos de 50/50 en
   `split_strength`; no cambie el ratio por eso".
4. Semilla: **declarada por el investigador con justificación**, como prescribe el paper (Etapa 1,
   *Implementation details*: "declare the seed before seeing data… justified and recorded"; por
   ejemplo, una fecha de envío del manuscrito). Ambas se registran en el log. Una semilla derivada
   automáticamente queda como alternativa, también registrada.
4b. **Estructura de los datos** (la lista de §2.6.3 del paper): ¿temporales?, ¿varias filas por
   unidad?, ¿clases muy desbalanceadas? Las respuestas no bloquean nada. Se registran y determinan
   qué condiciones límite del STDS se muestran en la Etapa 5. Si los datos son temporales o
   desbalanceados, se advierte que el core solo hace un split aleatorio, aunque el paper
   recomienda uno temporal o estratificado (§0.4, P2).
5. **No se muestra una vista previa de qué unidades van a cada lado ni la distribución del
   outcome en el holdout.** Ver el holdout antes de comprometerse es justo lo que el protocolo
   prohíbe, y una vista previa con re-sorteo invita a buscar semillas.
6. Botón **[Comprometer split]**. El diálogo explica: "Después de esto, el split no se puede
   cambiar. Rehacerlo exige un protocolo nuevo, y el nuevo protocolo mostrará que este existió."
7. Resultado: n dev / n holdout, hashes y recibo descargable.

### 3.3 Etapa 2: Criterios

```
 Métrica       Comparación  Umbral  Obligatorio  Justificación (obligatoria)
 [f1_score ▾]  [>= ▾]       [0.60]  [x]          [Basado en … ]
 [+ criterio]
 Folds de CV (K): [5]   (se fija aquí; STDS tiene cortes tabulados para K = 3, 5, 10, 20)
 Plan de análisis (opcional, texto libre, se hashea): [ … ]
 Vista previa del JSON vinculante: { … }
                                                  [Registrar criterios]
```

- Las métricas se eligen de las que devuelve la plantilla de evaluación, o se escriben a mano.
  Si un criterio nombra una métrica que la evaluación no devuelve, el core la cuenta como **no
  cumplida** en la Etapa 5. La interfaz lo advierte aquí, y lo comprueba en la Etapa 3, antes del
  holdout.
- Tras registrar: hash, recibo PDF/MD para OSF y botón **[Anclar externamente]**, con la nota "es
  la única forma de que un tercero confíe en que este registro precede al holdout".

### 3.4 Etapa 3: Desarrollo

1. Descargar la plantilla `analysis.py` con `train(dev_df, train_idx, params)` y
   `evaluate(model, df, idx) -> dict[str, float]`. Subir el código.
2. **[Ejecutar CV]** corre en un subproceso que solo ve los datos de desarrollo. Muestra una tabla
   por fold y media ± sd (sd poblacional, como el core).
3. Se pueden hacer todas las corridas que se quiera. Todas quedan en el log ("Corrida 4"); explorar
   en desarrollo es legítimo.
4. **Auditor anticipado:** botón **[Revisar patrones en el código]**. Se ejecuta aquí, cuando
   todavía se puede corregir, con el aviso fijo del Estudio 3.
5. Comprobación: ¿las métricas de los criterios están en la salida?
6. **[Congelar modelo]**: elegir la configuración de una corrida → **entrenar el modelo final
   sobre todo D_dev**. El paper lo exige en la Etapa 3 ("A final model is then trained on all of
   D_dev") y el core no lo hace (§0.4, P3) → snapshot del código y del modelo + hash → **preflight**:
   ejecuta `evaluate` del modelo congelado sobre un fold de desarrollo para detectar errores
   técnicos antes de gastar el holdout.

### 3.5 Etapa 4: Validación de un solo disparo

```
┌ Validación única sobre holdout ──────────────────────────────────┐
│ Se ejecutará UNA vez:                                            │
│   modelo  sha256 9b2e…   código sha256 c41d…                     │
│   criterios sha256 a71c… (registrados 12-oct 15:02)              │
│   anclaje externo: no registrado  ← ver por qué importa          │
│ Si la ejecución falla, el holdout se considera usado igualmente. │
│                                                                  │
│ Escriba exactamente:                                             │
│   I ACCEPT SINGLE-SHOT VALIDATION FOR P-01JABCD                  │
│ [______________________________________________]                 │
│                                       [Ejecutar validación]      │
└──────────────────────────────────────────────────────────────────┘
```

- La frase es la del paper ("I ACCEPT SINGLE-SHOT VALIDATION") seguida de "FOR <ID>", para que
  no sirva una frase genérica guardada. `core_bridge` le entrega al `input()` del core la frase
  exacta que este espera.
- Web: el campo bloquea el pegado y el envío lleva un nonce de un solo uso emitido al
  renderizar. Bloquear el pegado es fricción de intención, no seguridad, y la documentación lo
  dice.
- CLI: lee la frase de la TTY (`/dev/tty`). Si stdin no es una terminal, se niega. No existe un
  flag `--yes`.
- Resultado: métricas de holdout y tabla de criterios (métrica, comparación, umbral, valor
  observado, "cumplido / no cumplido" en texto). Veredicto en los términos del core: `VALIDATED`
  o `FALSIFIED (k/n criterios obligatorios cumplidos)`, con la nota "una falsificación es un
  resultado científico válido".

### 3.6 Etapa 5: Evaluación (solo en modo completo)

**Perfil EIS:**

```
 Componente (peso)            valor   máx. alcanzable en esta etapa
 Preregistro (0.25)           ▇▇▇▇▇▇▇▇▇░  0.90   0.90  ¹
 Adherencia (0.25)            ▇▇▇▇▇▇▇▇▇▇  1.00   1.00  ²
 Fuerza del split (0.20)      ▇▇▇▇▇▇▇▇▇░  0.90   1.00 (solo con 50/50 y n ≥ 100)
 Fuga, S_leak (0.15)          ▇▇▇▇▇▇▇░░░  0.71   →1 cuando holdout ≪ dev ³
 Transparencia (0.15)         ▇▇▇▇▇▇▇▇▇▇  1.00   1.00  ²
 Agregado (texto pequeño): 0.912 · techo teórico 0.975 · techo con 70/30: 0.955
 ¹ El core da como máximo 0.9 a este componente una vez completada la validación.
 ² Este Workbench escribe automáticamente los campos que este componente premia; su valor
   refleja el uso de la herramienta, no evidencia independiente de rigor.
 ³ S_leak sube cuando el holdout rinde PEOR que el desarrollo. No intente optimizarlo.
 El EIS no está calibrado contra resultados externos (manuscrito §3–§4.4).
 Bandas del paper (§2.6.1, provisionales, no calibradas): ≥0.80 / 0.60–0.79 / <0.60, a leer
 contra el techo. No se usan las etiquetas "EXCELLENT…" del código (§0.4, P6).
 Pesos: "literature-informed heuristic defaults… not derived from meta-analysis" (paper),
 sin las cifras de WEIGHT_JUSTIFICATIONS (§0.4, P7).
```

**STDS:** z por métrica, max z, cortes al 95 y al 99 % para (K, M) y si son exactos o vecinos. El
nivel se muestra como texto: "no marcado / marcado (notable) / marcado (alto)". Caja de
advertencia siempre visible, con el texto de la docstring del core: no es una prueba de hipótesis,
la calibración es autoconsistente, y la tasa en un estudio real es mayor (el 5 % es una cota
inferior).

**Auditor:** lista de patrones con `archivo:línea`, fragmento, categoría y explicación. Encabezado
fijo: "En el Estudio 3, 50 de 50 scripts legítimos con apariencia sospechosa fueron marcados
(especificidad 0 %)". No se muestran `passed`, `adherence_score` ni `confidence` en la vista
principal. Hay una sección plegable "salida cruda del core" donde aparecen, con la nota de que el
umbral de 70 es normativo y `confidence=0.90` es una constante, no una estimación. El
investigador puede **anotar** cada hallazgo; la anotación queda en el log.

### 3.7 Exportar y verificar

**[Exportar bundle]** genera un `.zip`. Por defecto **no incluye los datos crudos**
(privacidad), solo sus hashes; se pueden incluir de forma opcional. La pantalla incluye
instrucciones para terceros (`python verify_bundle.py bundle.zip`) y un botón **[Verificar
ahora]** que ejecuta el mismo verificador como subproceso.

---

## 4. Estructura de carpetas

Propuesta para añadir **sin tocar** `ScientificReports_revision/`:

```
AFH-MODEL/
├── ScientificReports_revision/        # SIN CAMBIOS (core 3.1 + estudios del paper)
│   └── engine/eclipse_core.py         # fuente única de verdad
└── eclipse_workbench/
    ├── README.md                      # honesto, limitaciones primero
    ├── DESIGN.md                      # este documento
    ├── pyproject.toml                 # paquete eclipse-workbench; entry point `eclipse`
    ├── docs/
    │   ├── BUNDLE_SPEC.md             # formato del bundle, JSON canónico, reglas del verificador
    │   ├── LIMITATIONS.md             # versión extendida, citando el manuscrito y los Estudios 1–3
    │   └── R_AND_NOTEBOOKS.md         # contrato del ExternalRunner (fase 4)
    ├── src/eclipse_workbench/
    │   ├── core_bridge.py             # único import de eclipse_core; pinning por hash
    │   ├── service.py                 # ProtocolService
    │   ├── state.py                   # máquina de estados (derivada del log)
    │   ├── eventlog.py                # log append-only con cadena de hashes
    │   ├── bundle.py                  # BundleStore, MANIFEST, exportación
    │   ├── sealed.py                  # SealedHoldout
    │   ├── canonical.py               # JSON canónico y normalización de IDs
    │   ├── runners/
    │   │   ├── python_runner.py       # subproceso; dev o holdout según la etapa
    │   │   └── external_runner.py     # fase 4
    │   ├── report/                    # plantillas neutrales (HTML/MD)
    │   ├── cli.py                     # argparse; sin --yes
    │   └── web/
    │       ├── app.py                 # FastAPI; solo llama a service
    │       ├── templates/             # Jinja2 + htmx, sin paso de build
    │       └── static/
    ├── verifier/
    │   └── verify_bundle.py           # solo stdlib, sin imports del paquete
    ├── templates/
    │   └── analysis_template.py       # plantilla train/evaluate para usuarios
    ├── examples/
    │   └── synthetic_walkthrough/     # protocolo completo con datos sintéticos
    └── tests/
        ├── core_characterization/     # fija el comportamiento ACTUAL del core (incl. §0.2)
        ├── test_invariants.py         # I1–I13
        ├── test_tamper.py             # manipulaciones que el verificador debe detectar
        ├── test_equivalence.py        # I12
        ├── test_verifier_isolation.py # el verificador no importa nada del paquete
        └── test_web_flows.py          # Playwright (fase 2)
```

Bundle de un protocolo:

```
P-01JABCD/
├── protocol.json
├── log.jsonl
├── MANIFEST.sha256
├── core/                         # exactamente lo que escribe el core (nombres del core)
│   ├── P-01JABCD_SPLIT_IMMUTABLE.json
│   ├── P-01JABCD_CRITERIA_BINDING.json
│   └── P-01JABCD_FINAL_RESULT.json
├── stage1/split_commitment.json
├── stage2/analysis_plan.md
├── stage3/runs/001.json …  frozen/{code/, model.bin, model_commitment.json}
├── stage4/{confirmation.json, VALIDATION_RESULT.json, LOCK}
├── stage5/{integrity.json, audit.json, annotations.jsonl, raw_core_outputs/}
├── report/{report.html, report.md}
└── data/                         # excluida de la exportación por defecto
    ├── development.parquet
    └── holdout.sealed
```

Para no romper la compatibilidad, `core_bridge` importa el core desde su ubicación actual.
Opción recomendada: añadir `ScientificReports_revision/engine` al `sys.path`, igual que hacen los
scripts de estudio con `HERE.parent / "engine"`. Así no se copia ni se mueve el archivo, porque
copiarlo crearía una segunda fuente de verdad.

---

## 5. Plan por fases

| Fase | Contenido | Criterios de aceptación |
|---|---|---|
| **F0: Base** | Paquete y `pyproject`. Tests de caracterización del core (incluyendo los hallazgos 1–4, 8 y 9 como comportamiento actual documentado). `core_bridge` mínimo | `pytest` en verde. `eclipse_core.py` byte a byte idéntico (hash en el test). Re-ejecutar `code/01`–`04` produce `outputs/` idénticos (diff vacío) |
| **F1: Motor + CLI + verificador** | `eventlog`, `state`, `bundle`, `sealed`, `PythonRunner`, `service`, CLI completa, `verify_bundle.py` niveles 1–2, ejemplo sintético | Flujo completo por CLI sobre datos sintéticos. El verificador da OK. Tests de manipulación (editar criterio, intercambiar un ID dev/holdout, segundo `HOLDOUT_RELEASED`, reordenar el log, cambiar el core a mitad de protocolo) → todos FALLA. I1–I13 en verde. El verificador corre en un entorno sin numpy |
| **F2: Web local** | FastAPI + Jinja2/htmx sobre el mismo servicio | Flujo completo en el navegador (Playwright). Test de I10 en todas las rutas. Test de que el CSS no define verdes ni rojos de estado. La confirmación web rechaza un nonce reutilizado |
| **F3: Reporte y anclaje** | Reporte neutral, `LIMITATIONS.md`, recibos, anclaje externo (manual u OSF; OpenTimestamps si se aprueba) | El reporte pasa una lista de comprobación de limitaciones (cada advertencia de la §3.6 presente). Revisión del autor |
| **F4: R y notebooks** | `ExternalRunner`, `BUNDLE_SPEC` independiente del lenguaje, prototipo de verificación nivel 1 en R | Un bundle producido con métricas autorreportadas se marca como tal en el reporte y en el verificador. Un script en R verifica el `MANIFEST` y la cadena del log |
| **F5: Uso real** | Pruebas con 3–5 investigadores; modo custodio opcional (holdout cifrado con la clave de un tercero) | Registro de fricciones; ningún participante llega a la Etapa 4 sin entender que es irreversible (entrevista posterior) |

Fuera de alcance de esta iteración: modelos incorporados ("elige un clasificador"), métricas de
regresión propias y cualquier score nuevo.

---

## 6. README

Ver `eclipse_workbench/README.md`.

---

## 7. Preguntas antes de empezar

0. **Discrepancias entre el paper y el código (§0.4, P1–P8).** P1 (el hash no cubre la asignación
   del holdout) y P4 (no hay bloqueo de archivo y la Etapa 4 se puede repetir) son afirmaciones
   del paper publicado que el código de referencia no cumple. ¿Cómo quiere tratarlas: una
   corrección en el core (versión nueva), una nota en el README del paper, una consulta a la
   revista sobre una fe de erratas, o solo la compensación en el Workbench? Recomiendo, como
   mínimo, documentarlas en `ScientificReports_revision/README.md`. La decisión es suya.
1. **Modo de ejecución de modelos.** ¿Priorizo el PythonRunner (el Workbench ejecuta el código
   del usuario y puede garantizar el orden), el ExternalRunner (R o notebooks con métricas
   autorreportadas, con garantías más débiles), o ambos desde el inicio? Recomiendo empezar por
   el PythonRunner.
2. **Fallo técnico en la Etapa 4.** Si `evaluate` lanza una excepción en holdout, ¿el holdout se
   considera consumido (estricto, mi recomendación junto con el preflight), o se permite un
   reintento con el *mismo* hash de modelo y código, registrado?
3. **¿Cambios al core?** Los hallazgos 1, 2, 4, 6 y 7 se arreglarían mejor en el core (por
   ejemplo como 3.1.1 o 3.2), pero eso cambia el instrumento descrito en el paper. Recomiendo
   dejar el 3.1 congelado para la reproducibilidad del manuscrito, compensar en la orquestación
   (§1.3) y discutir un 3.2 aparte. El CV agrupado (punto 6) solo se resuelve bien en el core.
4. **Semilla.** Ya está resuelto con el paper: la declara el investigador con justificación. ¿Le
   parece bien ofrecer la semilla derivada solo como alternativa?
5. **Custodia del holdout.** ¿Basta el modo local (fricción), o quiere el modo custodio en el
   alcance temprano?
6. **Ubicación.** El `CLAUDE.md` describe este repo como colección de investigación, no como
   producto, y mezcla trabajo no relacionado (`Lab/`, `NeuroscienceOfConsciousness_revision/`).
   ¿Lo creo aquí en `eclipse_workbench/` o en un repositorio propio que dependa del core por
   versión o commit? Para usuarios externos recomiendo un repo propio.
7. **Anclaje externo.** ¿Qué mecanismos acepta: instrucciones manuales para OSF, OpenTimestamps
   (requiere red), u otro?
8. **Citas de los pesos del EIS.** El paper no contiene las cifras de `WEIGHT_JUSTIFICATIONS`
   ("~60 %", "17/20"); usa una justificación cualitativa (§0.4, P7). La interfaz usará la
   redacción del paper. ¿Quiere además corregir esas cadenas en el core en una versión futura?
   No verifiqué las cifras contra los artículos originales.
8b. **Split estratificado y temporal (P2).** El paper los recomienda y el core no los implementa.
   ¿Los dejamos fuera del Workbench (solo advertencia, mi recomendación para no crear una segunda
   implementación del split) hasta que existan en el core?
9. **Tipo de tareas.** ¿Solo datos tabulares con métricas de clasificación binaria al inicio
   (`EclipseValidator.binary_classification_metrics` existe en el core), o también regresión?
10. **Idioma.** ¿Interfaz y README en inglés (público internacional), en español, o en ambos?
