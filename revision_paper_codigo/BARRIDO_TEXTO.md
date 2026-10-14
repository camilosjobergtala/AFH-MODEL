# Barrido del texto: afirmaciones sobre hash, inmutabilidad, sellado y detección de manipulación

- **Fuente:** PDF publicado (Article in Press, 14-10-2026), con el texto extraído mediante `pdftotext`, más
  las etiquetas de la Fig. 1, que salen de `code/05_figures.py`.
- **Código:** `eclipse_core.py` en el commit `93339b4`.

Las afirmaciones nuevas se comprobaron con `verificacion_windows/p11_afirmaciones_integridad.py`
(bloques T1–T8), cuyo control positivo pasa: editar un umbral sin recalcular el hash **sí** se
detecta. Los puntos P1 y P4 remiten a `verificar_p1_p4.py`.

**Resultados de p11 en los que se apoya esta tabla:**

| Bloque | Qué se hizo | Resultado |
|---|---|---|
| T1 | Criterios editados y hash recalculado con la receta del core | `verify_integrity` = válido |
| T2 | `registration_date` cambiada a 2020-01-01 | válido |
| T3 | Re-registro de criterios con `force=True` | sobrescribe el archivo; válido; no queda copia del anterior |
| T4 | Re-split con `force=True` y otra semilla | 25 de 30 IDs del holdout cambian; válido; no queda copia |
| T5 | Llamada a `stage1` | devuelve al código que la llama la lista de los 30 IDs del holdout |
| T6 | Llamada a `stage3` con los 100 IDs | se ejecuta sin error, aunque incluyen los 30 del holdout |
| T7 | Edición de `FINAL_RESULT.json` | `verify_integrity` solo comprueba `split` y `criteria`; el resultado editado da `all_valid: True` |
| T8 | Búsqueda de `registration_date` | solo se comprueba su **presencia** (líneas 359 y 592) y se escribe (2545); ningún código la compara con la fecha de validación |

**Nota general sobre T1:** ningún hash calculado y guardado localmente impide que quien controla
los archivos los edite y recalcule el hash. Solo un anclaje externo, como un preregistro público o
un sello de tiempo, lo haría. T1 no es un defecto de implementación, sino el límite de cualquier
esquema local. Importa porque varias frases del paper afirman más que eso.

Leyenda: **V** verdadera · **F** falsa dado el código · **I** imprecisa o exagerada.

---

## Abstract

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| A1 | "a computational scaffold combining preregistration, cryptographically verified data splitting (SHA-256), single-shot holdout validation, …" | **I** | El split no se verifica criptográficamente: lo hace un barajado con semilla. El SHA-256 cubre solo la semilla y el conjunto de IDs, no la asignación (P1) | "…combining preregistration, seeded data splitting with a SHA-256 record of the seed and sample set, single-shot holdout validation, …" |

## Introduction — "The replication crisis and its discontents", 4.º párrafo

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| B1 | "Preregistration … provides no technical enforcement mechanism. Researchers can deviate from protocols without detection if access logs are unavailable" | **I (implícita)** | Contrasta con lo que ECLIPSE aportaría, pero el código no lleva ningún registro de accesos: no registra re-splits, re-registros ni validaciones repetidas (T3, T4, P4) | Sin cambio en la frase. Evitar en otras secciones afirmar que ECLIPSE aporta registros de acceso |
| B2 | "reviewers cannot practically … verify that holdout data remained untouched" | **I (implícita)** | Presenta un problema que el lector supone que ECLIPSE resuelve. El código no puede verificar que el holdout no se tocó: lo entrega al usuario en la Etapa 1 (T5) | Sin cambio en la frase. Ver C2, G1 y G2 |

## Introduction — "ECLIPSE: a computational scaffold"

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| C1 | párrafo del EIS | "split strength (entropy-based balance, cryptographic verification)" | **I** | El componente comprueba la **presencia** del campo de hash, no lo verifica. La definición formal (§2.6.4) sí lo dice bien: "I_verified = 1 when a cryptographic hash is present" | "…(entropy-based balance, presence of a hash record)" |
| C2 | párrafo "These components integrate…" | "Stage 1 implements irreversible data splitting via SHA-256 cryptographic hashing" | **I** | El split lo produce la semilla, no el hash. La "irreversibilidad" consiste en que la Etapa 1 reutiliza el archivo si existe, y `force=True` lo rehace sin dejar rastro (T4) | "Stage 1 performs a seeded data split, records a SHA-256 hash of the seed and sample set, and reuses the stored split on subsequent calls" |
| C3 | ídem | "Stage 2 establishes binding preregistration of success criteria with timestamp and hash" | **I** | Timestamp y hash existen. "Binding" no se impone: `force=True` re-registra sin rastro (T3). Ver también E3 | "…with a timestamp and a SHA-256 hash of the criteria" |
| C4 | ídem | "Stage 3 enforces development on training data only via k-fold cross-validation" | **F** | La Etapa 3 acepta cualquier dato que se le pase, incluido el holdout (T6) | "Stage 3 runs k-fold cross-validation on the development data supplied by the researcher" |
| C5 | ídem | "Stage 4 implements single-shot validation requiring explicit researcher confirmation" | **V**, con matiz | La confirmación existe en modo interactivo. El modo `non_interactive` la sustituye por "AUTOMATED" | Opcional: "…(in interactive mode)" |
| C6 | párrafo "prevention-first philosophy" | "technical constraints reduce opportunities for accidental protocol violations through cryptographic verification and single-shot enforcement" | **I** | La verificación no cubre la asignación (P1). El disparo único no se impone entre las Etapas 4 y 5 (P4) | "…through a hash record of the split and criteria and a confirmation step before validation" |

## Related work

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| D1 | "Preregistration quality and adherence" | "operationalizes … temporal precedence (hash and timestamp before validation)" | **I** | El EIS comprueba que los campos existan y que la validación no esté completa **en el momento de calcular**. No compara fechas (T8). Además, en el flujo normal el EIS se calcula después de la validación | "…temporal precedence (presence of a hash and timestamp, and registration recorded before validation in the framework's stage flags)" |
| D2 | "Computational provenance and reproducibility" | "ECLIPSE's cryptographic verification and immutable record contribute one specific provenance primitive (tamper-evident splits and criteria)" | **I** | Split: la asignación no queda protegida (P1) y el re-split no deja rastro (T4). Criterios: detecta ediciones sin recalcular el hash, pero no el re-registro (T3) ni la edición con hash recalculado (T1). Los archivos no son inmutables (permisos `rw`, P4) | "…a hash-based record of the split seed and sample set and of the registered criteria, which detects edits made without recomputing the hash" |

## Introduction — "Scope and contributions", párrafo "Fourth"

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| D3 | "a five-stage operational discipline (cryptographic splitting, binding preregistration, clean development, single-shot validation, automated assessment)" | **I** | Mismos motivos que A1, C3, C4 y P4 | "(seeded splitting with a hash record, timestamped and hashed preregistration, cross-validated development, confirmation-gated validation, automated assessment)" |

## Methods — "ECLIPSE framework overview"

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| E1 | 1.er párrafo | "irreversible data splitting using cryptographically-verified partitioning" | **F** | La partición no se verifica (P1) y puede rehacerse (T4) | "seeded data splitting with a SHA-256 record of the seed and sample set" |
| E2 | 1.er párrafo | "binding performance thresholds are declared before validation with cryptographic verification" | **I** | El hash verifica que el contenido no cambió respecto del hash guardado. No verifica que se declarara antes de la validación (T2, T8) | "…declared before validation, with a SHA-256 hash of the criteria and a registration timestamp" |
| E3 | 1.er párrafo | "The third stage enforces clean development through model training via k-fold cross-validation on development data only" | **F** | T6 | "The third stage runs k-fold cross-validation on the development data supplied by the researcher" |
| E4 | 2.º párrafo | "Reduction of opportunity is achieved through Stages 1–4, which structurally reduce opportunities for accidental protocol violations through enforced separation and single-shot validation" | **I** | La separación no se impone: el holdout se entrega al usuario en la Etapa 1 (T5) y la Etapa 3 acepta datos del holdout (T6). El disparo único es parcial (P4) | "…through a recorded split, registered criteria and a confirmation step before validation" |

## Methods — Stage 1

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| F1 | Rationale | "ECLIPSE prevents accidental leakage and provides tamper-evident records of the original split through cryptographic verification." | **I / F** | "Tamper-evident records of the original split" es falso para la asignación (P1) y para un re-split con `force=True` (T4). "Prevents accidental leakage" no lo impone el código (T5, T6) | "ECLIPSE reduces opportunities for accidental leakage by recording the split, and stores a SHA-256 hash of the seed and sample set that detects changes to either" |
| F2 | Algorithm | "A SHA-256 hash is computed over the indices of D_hold, the seed s, the split ratio, and a timestamp … Any subsequent modification to the split indices, seed, or sample identifiers changes the hash, supporting tamper detection" | **F** | **P1** (ya reportado) | Ver el correo, punto 1 |

## Fig. 1 (figura y leyenda)

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| G1 | Leyenda, 1.ª frase | "The scaffold enforces separation between development and validation across five sequential stages with automated integrity checks." | **F** | T5 y T6. Las comprobaciones de integridad no son automáticas entre etapas: `verify_integrity` es una llamada aparte y solo cubre split y criterios (T7) | "The scaffold organizes development and validation into five sequential stages; integrity records are written at Stages 1–2 and can be checked with verify_integrity()" |
| G2 | Figura, recuadro del holdout | "Sealed — no access until Stage 4" | **F** | El holdout no está sellado: la Etapa 1 devuelve sus IDs al usuario (T5) | "Holdout — not to be accessed until Stage 4 (by protocol)" |
| G3 | Figura, flecha discontinua | "prohibited: holdout access during development" | **V** como norma / **I** como mecanismo | Es una prohibición del protocolo, no un bloqueo del código (T6). El pie de figura habla de "enforced workflow" | Mantener, cambiando "enforced workflow" por "intended workflow" en la leyenda |
| G4 | Figura, Etapa 1 | "Split SHA-256 verified" | **I** | P1 | "Split (SHA-256 record of seed and sample set)" |
| G5 | Figura (Etapa 4) y leyenda | "(locked after use)"; "single-shot validation on the untouched holdout (one evaluation, locked after use)" | **F** | El bloqueo se activa solo tras la Etapa 5, y `force=True` lo salta (P4). "Untouched" no se verifica (T5) | "(refused after the Stage 5 assessment)"; "single-shot validation on the holdout (confirmation required; further calls refused after Stage 5)" |
| G6 | Leyenda | "Stage 2: binding preregistration with timestamp verification" | **F** | Ningún código verifica la fecha. Cambiarla no se detecta (T2, T8) | "Stage 2: preregistration with a timestamp and a SHA-256 hash of the criteria" |
| G7 | Leyenda | "the diagram is a schematic of the enforced workflow" | **I** | Ver G1–G3 | "…a schematic of the intended workflow" |
| G8 | Figura y leyenda, Etapa 5 | el auditor dentro de la Etapa 5 | **F** | Punto 3 (ya reportado) | Ver el correo, punto 3 |

## Methods — Stage 2

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| H1 | Rationale | "ECLIPSE requires a priori declaration of success criteria with timestamp and hash, supporting binding-when-followed preregistration with cryptographic verification of temporal precedence." | **F** (última parte) | Un hash no prueba precedencia temporal. Ningún código compara fechas (T8), y cambiar la fecha no se detecta (T2) | "…with a timestamp and a SHA-256 hash, supporting binding-when-followed preregistration. Temporal precedence can be established only by depositing the hash with an external timestamping or registration service." |
| H2 | Registration process, paso 4 | "Fourth, the file is marked immutable via version control." | **F** | El código no interactúa con ningún control de versiones. El archivo queda escribible | "Fourth, the researcher should commit the file to version control or deposit its hash externally." |
| H3 | Registration process, paso 5 | "Fifth, any post-registration modification invalidates the analysis under the framework's discipline." | **V** como norma / **I** como mecanismo | Es normativo. Pero el código permite re-registrar con `force=True` sin dejar rastro (T3), y una edición con hash recalculado no se detecta (T1) | Mantener, añadiendo: "The implementation detects edits made without recomputing the hash; it does not record re-registration." |

## Methods — Stage 4

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| I1 | "File locking ensures that after validation, the holdout data file becomes read-only. Result immutability is achieved by writing validation results to a version-controlled file with timestamp." | **F** | **P4** (ya reportado). Además, el archivo de resultados puede editarse después y `verify_integrity` no lo comprueba (T7) | Ver el correo, punto 2 |

## Methods — Stage 5, EIS

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| J1 | "S_pre assesses temporal precedence, immutability, and threshold specificity; … S_trans aggregates indicators for documentation files, timestamps, hashes, and immutability flags." | **I**, leve | "Immutability" es la presencia de `registration_date` + `criteria_hash` (S_pre) y de una declaración de texto `binding_declaration` (S_trans), no inmutabilidad real. La definición formal de §2.6.4 es precisa | "S_pre assesses registration timing, the presence of a timestamp and hash, and threshold specificity; … S_trans … timestamps, hashes, and a binding declaration" |

## Methods — "Software implementation"

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| K1 | "JSON for immutable record storage" | **I** | Archivos JSON normales, con permisos de escritura (P4); el core los sobrescribe con `force=True` | "JSON for record storage" |

## Discussion — "Prevention-first philosophy"

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| L1 | 2.º párrafo | "data splits cannot be silently modified without changing the hash" | **F** | La asignación puede cambiar sin alterar el hash (P1). Un re-split con `force=True` produce un hash nuevo válido sin rastro del anterior (T4) | "changes to the split seed or to the set of sample identifiers alter the stored hash" |
| L2 | 2.º párrafo | "criteria cannot be silently adjusted post-hoc" | **F** | Re-registro con `force=True` sin rastro (T3). Edición con hash recalculado no detectable (T1, límite de cualquier hash local) | "edits to the registered criteria that do not recompute the hash are detected" |
| L3 | 2.º párrafo | "validation occurs exactly once under the framework's discipline" | **I** | Como norma, sí. Como mecanismo, no: P4 | "validation is gated by an explicit confirmation and refused once the final assessment has been written" |
| L4 | 2.º párrafo | "they raise the threshold for accidental error and create audit trails" | **I** | No hay registro de re-splits, re-registros ni validaciones repetidas (T3, T4, P4) | "…and leave records (split, criteria and results files) that can be inspected" |
| L5 | 4.º párrafo | "once data are contaminated or criteria are silently adjusted, no amount of retrospective scrutiny can recover the original test. Operational discipline avoids these irreversible compromises before they occur." | **I** | Argumento general, válido como razonamiento. "Avoids" atribuye al software una prevención que no impone (T3–T6) | "…Operational discipline is intended to make these compromises less likely before they occur." |

## Discussion — "Comparison to alternative approaches"

| # | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|
| M1 | "ECLIPSE adds technical enforcement (cryptographic verification, single-shot validation)" | **I** | P1, P4, T5, T6 | "ECLIPSE adds technical support (hash records of the split and criteria, a confirmation-gated validation step)" |
| M2 | "ECLIPSE favors a simpler operational discipline (one validation attempt, no exceptions)" | **I** | "No exceptions" es la norma; el código tiene `force=True` y permite repetir antes de la Etapa 5 (P4) | "(one validation attempt by protocol)" |
| M3 | "ECLIPSE contributes specific provenance primitives (tamper-evident splits and criteria)" | **I** | Igual que D2 | Igual que D2 |

## Conclusion

| # | Ubicación | Texto literal | Veredicto | Motivo | Reemplazo propuesto |
|---|---|---|---|---|---|
| N1 | 1.er párrafo | "combining preregistration, cryptographic data splitting, single-shot validation, …" | **I** | Igual que A1 | "…seeded data splitting with hash records, …" |
| N2 | 2.º párrafo | "The five-stage operational discipline structures confirmatory work to reduce opportunities for accidental protocol violations and to create audit trails." | **I** (solo "audit trails") | Igual que L4 | "…and to leave inspectable records." |

---

## Afirmaciones correctas que conviene no tocar

- **§2.6.4, S_split:** "I_verified = 1 when a cryptographic hash is present". Exacto: es presencia, no verificación.
- **Stage 4:** "These mechanisms … do not prevent deliberate circumvention; the framework's discipline depends on adoption rather than coercion". Exacto.
- **Discussion:** "These arrangements do not prevent determined deception". Exacto.
- **Discussion, Goodhart:** el párrafo sobre "engineer compliance without engineering rigor" es coherente con lo observado.

## Resumen

Hay 42 afirmaciones listadas (A1–N2), sin contar las correctas de arriba:
- **Falsas dado el código:** C4, E1, E3, F2, G1, G2, G5, G6, G8, H1 (última parte), H2, I1, L1 y L2.
  Además, F1 es "imprecisa o falsa".
- **Imprecisas:** las 24 restantes, incluidas las dos implícitas de la Introducción (B1, B2).
- **Verdaderas:** C5, G3 y H3, como normas del protocolo; G3 y H3 son imprecisas como descripción de un mecanismo.

No saco conclusiones sobre la tesis del paper.
