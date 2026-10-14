# ECLIPSE Workbench

> **Estado: diseño. Todavía no hay software ejecutable aquí.** Este README describe lo que el
> Workbench *hará* y, sobre todo, lo que *no puede hacer*. Las secciones marcadas como
> "planificado" no existen hasta que se implementen y se prueben (ver `DESIGN.md` §5).

El Workbench es una capa de orquestación local (CLI + interfaz web) para aplicar el protocolo
ECLIPSE descrito en:

> Sjöberg Tala, C. A. *ECLIPSE v2.0: a reproducible scaffold for severe confirmatory testing
> through preregistration, single-shot validation, and static-analysis screening.* Scientific
> Reports. DOI: 10.1038/s41598-026-74760-5

Usa sin modificarla la implementación de referencia en
`ScientificReports_revision/engine/eclipse_core.py` (instrument version 3.1; commit `93339b4`,
archivado en Zenodo, DOI 10.5281/zenodo.22760496).

**Solo para investigación confirmatoria.** Como dice el paper, ECLIPSE no está pensado para
investigación exploratoria o generadora de hipótesis, donde la flexibilidad analítica es necesaria.

**Adoptar el flujo y usar los scores son cosas distintas.** El paper recomienda adoptar el flujo
de trabajo y dice que la evidencia no respalda usar los scores, en particular el EIS, para
evaluar estudios. Por eso el modo predeterminado del Workbench es *solo disciplina*.

---

## Lea esto antes de usarlo

### Lo que ECLIPSE y este Workbench **no** son

- **El EIS (Eclipse Integrity Score) no es una medida objetiva de rigor.** No está calibrado
  contra resultados externos (replicación, validez posterior). Sus pesos son una elección
  normativa. Los Estudios 1–3 del manuscrito son comprobaciones de constructo en condiciones
  sintéticas, no validez externa.
  - El **techo real es 0.975**: el componente de preregistro vale como máximo 0.9 una vez hecha
    la validación. Con el split 70/30 por defecto, el techo es **≈ 0.955**.
  - Dentro de esta herramienta, **cuatro de los cinco componentes quedan casi en su máximo para
    cualquier protocolo completado**, porque premian campos (timestamps, hashes, declaración
    vinculante) que la propia herramienta escribe. Su valor refleja el uso de la herramienta, no
    evidencia independiente.
  - El componente de fuga (`S_leak`) **sube cuando el holdout rinde peor** que el desarrollo.
    Optimizar el EIS puede premiar un peor modelo. Por eso existe el modo *solo disciplina*.
- **El STDS es heurístico y autocalibrado; no es una prueba de hipótesis.** Sus cortes son los
  percentiles 95 y 99 de max z bajo un escenario honesto *simulado por el mismo modelo
  generativo* (calibración autoconsistente, no independiente). La simulación trata los folds
  como independientes y los folds reales de CV se solapan, así que la tasa de alarmas en un
  estudio honesto real **será mayor que el 5 % tabulado**: ese 5 % es una cota inferior. Una
  marca es un candidato a revisión humana, no evidencia de mala conducta.
- **El auditor de código es un cribado de patrones con una tasa alta de falsos positivos.** En
  el Estudio 3 marcó **50 de 50** scripts legítimos escritos para parecerse a una violación
  (especificidad 0 %). La especificidad del 100 % en los scripts conformes corresponde a un
  corpus diseñado, no a código real. Una marca significa "este patrón aparece", no "hubo una
  violación". El valor `confidence=0.90` que produce el core es una constante, no una
  estimación.

### Lo que el Workbench **no puede** garantizar

- **Que usted no mire el holdout por otra vía.** Usted conserva sus datos originales. El
  Workbench garantiza que *su propio código* no abre la partición de holdout antes de la Etapa 4;
  no puede controlar otros programas.
- **Que no se creen protocolos nuevos hasta obtener el resultado deseado.** Rehacer exige un
  protocolo nuevo, y el registro del espacio de trabajo lo deja visible. Ese registro es local,
  y quien controla la máquina puede borrarlo.
- **El momento real de los eventos.** Los timestamps son del reloj local. La cadena de hashes
  prueba consistencia interna, no tiempo. Para que un tercero confíe en que los criterios se
  registraron antes de ver el holdout, **ancle el hash de la Etapa 2 externamente** (por
  ejemplo, un preregistro en OSF) antes de la Etapa 4.

---

## Qué hará (planificado)

Las cinco etapas de ECLIPSE, cada una con un bloqueo real:

| Etapa | Qué ocurre | Qué queda bloqueado |
|---|---|---|
| 1. Split | Partición dev/holdout (70/30 por defecto) a nivel de unidad, con semilla registrada y compromiso SHA-256 | Datos, semilla, ratio y asignación. Rehacer = protocolo nuevo |
| 2. Criterios | Criterios de falsación con umbral y justificación, K de CV y plan de análisis | Criterios (no existe "editar") |
| 3. Desarrollo | k-fold CV **solo sobre dev**, tantas corridas como quiera (todas registradas). Congelado de código y modelo | Código y modelo congelados |
| 4. Holdout | Validación **una sola vez**. Requiere teclear `I ACCEPT SINGLE-SHOT VALIDATION FOR <ID>`; no existe un flag para saltarlo | Holdout consumido, incluso si la ejecución falla |
| 5. Evaluación | Veredicto según los criterios preregistrados. En modo completo, también perfil EIS (5 componentes), STDS y auditor como señales de cribado | Resultado final |

Además:

- **Modo solo disciplina:** no calcula ni muestra el EIS, el STDS ni el puntaje del auditor. El
  modo se elige al crear el protocolo y no se puede cambiar.
- **Bundle verificable:** una carpeta o `.zip` con JSON, log append-only encadenado por hashes y
  `MANIFEST.sha256`. Sin bases de datos. Por defecto no incluye los datos crudos, solo sus
  hashes.
- **Verificador independiente** (`verifier/verify_bundle.py`): un solo archivo, solo biblioteca
  estándar de Python, sin importar el Workbench ni el core. Comprueba la integridad, el orden de
  las etapas, que los criterios no cambiaron, la asignación del split y un único acceso al
  holdout. Recalcula la evaluación de los criterios y los z del STDS. Siempre lista lo que *no*
  puede comprobar.

## Diferencias con usar `eclipse_core.py` directamente

El Workbench corrige en la capa de orquestación, sin modificar el core, comportamientos que
encontré al revisarlo (detalle y evidencia en `DESIGN.md` §0.2):

- El `split_hash` del core no compromete qué IDs van a dev y cuáles a holdout, aunque el paper
  describe un hash sobre los índices del holdout, la semilla, el ratio y un timestamp. El
  Workbench añade un `assignment_hash` que sí cubre la asignación.
- En el core, la Etapa 4 puede repetirse antes de la Etapa 5, y `force=True` se salta el bloqueo.
  Tampoco existe el bloqueo del archivo de holdout que describe el paper. El Workbench escribe el
  bloqueo antes de abrir el holdout y nunca usa `force`.
- El core no entrena el modelo final sobre todo el conjunto de desarrollo, como describe el paper.
  El Workbench lo hace explícito al congelar el modelo.
- El modo `non_interactive` del core confirma automáticamente. El Workbench no lo usa.
- El Workbench no muestra las etiquetas "EXCELLENT … VERY POOR" del EIS ni el veredicto en
  verde/rojo del reporte HTML del core.

**Limitaciones que el Workbench no corrige:**

- La Etapa 3 del core hace CV por filas. Si hay varias filas por sujeto, pueden caer filas del
  mismo sujeto en entrenamiento y validación del mismo fold.
- El paper recomienda un split estratificado (clases desbalanceadas) o temporal (series de
  tiempo), y el core solo hace un split aleatorio.

El Workbench advierte de ambas, pero corregirlas requiere cambios en el core.

**Auditor:** solo analiza código Python. El código en R o MATLAB no se audita.

## Instalación y uso

*Planificado. Todavía no hay paquete.*

## Licencia

Apache License 2.0 (ver `LICENSE` en la raíz del repositorio).
