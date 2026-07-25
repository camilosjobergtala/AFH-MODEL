# Ashes — Admisibilidad de afirmaciones constitutivas sobre la presencia

Paquete para el manuscrito **"An admissibility test for constitutive claims about phenomenal
presence: necessary conditions, asymmetric falsification, and a conjunctive return-compatible
criterion"** (C. A. Sjöberg Tala), artículo teórico/metodológico. Propone una disciplina de
admisibilidad asimétrica para teorías constitutivas de la conciencia (§2) y desarrolla, como
ejemplo trabajado, un criterio observacional conjuntivo de cuatro componentes para el problema
recurrencia-vs-causa-común (§3), validado por simulación sobre doce arquitecturas causales
conocidas (§4).

Es un proyecto distinto del de [`../NeuroscienceOfConsciousness_revision/`](../NeuroscienceOfConsciousness_revision/)
(el manuscrito AFH sobre convergencia talámica intralaminar) — comparte el mismo problema
metodológico de fondo (recurrencia temprana-tardía vs. confusores) y por eso el ejemplo trabajado
reutiliza la misma lógica de test que
`NeuroscienceOfConsciousness_revision/code/test_rastreo_de_fuente.py`, pero aquí el explanandum es
genérico (cualquier teoría constitutiva de la presencia), no específico del marco AFH.

## Estado: segunda versión del ejemplo trabajado

`manuscript/` contiene la versión actual del manuscrito (inglés, criterio conjuntivo de cuatro
componentes con test de cadena episódica sobre doce arquitecturas) y, en `manuscript/archive/`,
el borrador anterior (`Admisibilidad_presencia_v5.pdf`, español, el par de tests Test 1/Test 2
sobre siete arquitecturas descrito más abajo bajo "Código del borrador anterior"). El ejemplo
trabajado cambió de diseño entre versiones: la versión actual generaliza el par Test 1/Test 2 a
una conjunción de cuatro componentes (dos ganancias predictivas fuera de muestra, una prueba de
identidad episódica a nivel de cadena, y un requisito de diseño de misma-población-fuente) y
amplía la caracterización de siete a doce arquitecturas causales. El código nuevo (`code/sim_return.py`
en adelante) implementa esta versión; `simulacion_recurrencia_vs_causa_comun.py` y
`barrido_confusor_latente.py` documentan el diseño del borrador anterior y no se han eliminado
porque siguen siendo código auditado y ejecutado, pero no reproducen la Tabla 6 en adelante del
manuscrito actual.

## Mapa

```
manuscript/          el paper actual (PDF) y su Supplementary Methods
manuscript/archive/   el borrador anterior (v5, español)
code/                 código de simulación; ver mapa de scripts abajo
outputs/              tablas (JSON) y figuras (PNG/PDF) generadas por code/run_*.py y code/make_fig*.py
notes/                hallazgos de revisión sobre el borrador anterior, no incorporados al texto
```

## `manuscript/`

- `Main_manuscript.pdf` — el manuscrito actual.
- `Supplementary_Methods.pdf` — especificación generativa completa, parámetros, semillas, entorno
  computacional y mapeo script-a-resultado (S1-S11) citados por el código de `code/`.
- `archive/Admisibilidad_presencia_v5.pdf` — el borrador anterior (español, Test 1/Test 2).

## `code/` — mapa de scripts (Supplementary Methods, Tabla S6)

Reproduce las Tablas 6-9 y las Figuras 1-4 del manuscrito actual. Los dos primeros son módulos
núcleo compartidos (se importan, no se ejecutan); los siguientes cuatro corren las simulaciones y
escriben resultados a `../outputs/`; los últimos cuatro generan las figuras a partir de esos
resultados.

```bash
pip install -r code/requirements.txt   # numpy==2.4.4, matplotlib==3.10.8 (Supplementary Methods S1)
```

| Script | Produce | Comando | Tiempo aprox. |
|---|---|---|---|
| `sim_return.py` | módulo núcleo: diseño, ridge, R² OOS, estratos, Wilson, once de las doce arquitecturas | se importa | — |
| `chain_return.py` | estimandos de segmento, nulos condicionales, emparejamiento de cadena, arquitectura de disrupción de token (la 12ª) | se importa | — |
| `run_chain.py` | Tabla 6, datos de Figura 3 | `python3 run_chain.py 0:12 200` | ~20-30 min |
| `run_outcomes.py` | Tabla 7, datos de Figura 4 | `python3 run_outcomes.py 0:12 200` | ~15-30 min |
| `run_sweep.py` | Tabla 8 | `python3 run_sweep.py 250,500,1000,2000,4000 200` | ~10-15 min |
| `run_thresh.py` | Tabla 9 | `python3 run_thresh.py 0.005 150` (y 0.01, 0.02) | ~5 min c/u |
| `make_fig1.py` | Figura 1 (flujo de decisión) | `python3 make_fig1.py` | <1 min |
| `make_fig2.py` | Figura 2 (diagrama de arquitecturas) | `python3 make_fig2.py` | <1 min |
| `make_fig3.py` | Figura 3 (tasas por componente, lee `outputs/chain_*.json`) | `python3 make_fig3.py` | <1 min |
| `make_fig4.py` | Figura 4 (mapa de calor de desenlaces, lee `outputs/out_*.json`) | `python3 make_fig4.py` | <1 min |

Nota de auditoría: los cuatro scripts `run_*.py` fueron ejecutados de cero (no solo leídos) como
parte de este depósito. Los resultados obtenidos reproducen el patrón cualitativo y, en la mayoría
de las celdas, el valor numérico exacto reportado en las Tablas 6-9 del manuscrito (p. ej. "Latent
feedforward state" 100/100/100/100 en la Tabla 6, o "Absent return, adequate sample" 100%
contradicted en la Tabla 7, coinciden exactamente); algunas celdas — sobre todo en la Tabla 8 y en
componentes del test de cadena con arquitecturas de acoplamiento débil — difieren de la tabla
publicada en unos pocos puntos porcentuales, dentro del rango esperable de una reimplementación
independiente a partir de la especificación matemática del Supplementary Methods (no del código
original del autor, que este depósito es el primero en escribir). El mecanismo de control positivo
(S10, umbral "recovery > 0.5·δmin") no está especificado por el Supplementary Methods más allá de
su criterio de paso — `run_outcomes.py` documenta en su docstring el canal de calibración sintético
concreto que este depósito eligió para instanciarlo.

### `code/` — código del borrador anterior (v5, Test 1/Test 2)

- `simulacion_recurrencia_vs_causa_comun.py` — Test 1 (dependencia condicionada) y Test 2
  (identificabilidad por ensayo, naive y residualizada) sobre siete arquitecturas; reproduce
  número a número la Tabla 2 del borrador v5.
- `barrido_confusor_latente.py` — barrido de intensidad del confusor latente (kappa=0 a 0.9) para
  la arquitectura `confusor_latente` de la tabla anterior; ver `notes/adenda_limitacion4.md`.

## `notes/`

- `adenda_limitacion4.md` — hallazgos de revisión sobre el borrador v5 (Test 1 vs Test 2
  residualizado como par sensibilidad/especificidad); no incorporados al texto del manuscrito
  actual, que ya reemplazó ese par de tests por el criterio conjuntivo de cuatro componentes.

License: Apache License 2.0 (ver `LICENSE` en la raíz del repositorio).
