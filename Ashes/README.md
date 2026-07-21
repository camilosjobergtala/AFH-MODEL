# Ashes — Admisibilidad de afirmaciones constitutivas sobre la presencia

Paquete para el manuscrito **"Poner a prueba afirmaciones constitutivas sobre la presencia
fenomenológica: falsación asimétrica y control de causas comunes"** (C. A. Sjöberg Tala),
artículo teórico/metodológico. Propone una disciplina de admisibilidad asimétrica para
teorías constitutivas de la conciencia (§4) y desarrolla, como ejemplo trabajado, un diseño
observacional de dos tests para el problema recurrencia-vs-causa-común (§6), validado por
simulación.

Es un proyecto distinto del de [`../NeuroscienceOfConsciousness_revision/`](../NeuroscienceOfConsciousness_revision/)
(el manuscrito AFH sobre convergencia talámica intralaminar) — comparte el mismo problema
metodológico de fondo (recurrencia temprana-tardía vs. confusores) y por eso el ejemplo
trabajado de §6 reutiliza la misma lógica de test que
`NeuroscienceOfConsciousness_revision/code/test_rastreo_de_fuente.py`, pero aquí el
explanandum es genérico (cualquier teoría constitutiva de la presencia), no específico del
marco AFH.

## Mapa

```
manuscript/    el paper (PDF)
code/          código de simulación citado en §6.4, y su extensión (barrido de intensidad)
notes/         hallazgos de revisión que no están todavía incorporados al texto del paper
```

## `manuscript/`

- `Admisibilidad_presencia_v5.pdf` — el manuscrito.

## `code/` — cómo correr cada cosa

```bash
pip install -r code/requirements.txt   # solo numpy
```

| Script | Qué hace | Qué parte del texto reproduce |
|---|---|---|
| `simulacion_recurrencia_vs_causa_comun.py` | Genera datos bajo 7 arquitecturas causales con verdad-base conocida y corre Test 1 (dependencia condicionada) y Test 2 (identificabilidad, naive y residualizada) sobre 40 réplicas c/u | Tabla 2, §6.4 — verificado número a número contra el paper |
| `barrido_confusor_latente.py` | Extiende `confusor_latente` de la Tabla 2 barriendo la intensidad del acoplamiento del confusor (kappa=0 a 0.9) en vez de un solo punto | No citado todavía en el texto — ver `notes/adenda_limitacion4.md` |

`simulacion_recurrencia_vs_causa_comun.py` tarda ~4-5 min; `barrido_confusor_latente.py`
~5 min (7 valores de kappa x 40 réplicas x 3 tests c/u).

Nota de auditoría: ambos scripts fueron ejecutados de cero (no solo leídos) como parte de
la revisión de este paquete; la salida de `simulacion_recurrencia_vs_causa_comun.py`
coincide, número a número, con la Tabla 2 del manuscrito.

## `notes/`

- `adenda_limitacion4.md` — el barrido de `barrido_confusor_latente.py` muestra que la falla
  ante confusión latente (§6.4) es gradual, no de umbral, y caracteriza a los dos tests como
  un par sensibilidad/especificidad (el Test 1 dispara con confusor más débil que el Test 2
  residualizado). Documenta la decisión del autor de **mantener** el Test 1 como primario —es
  el test de la condición necesaria del §4— y afinar el caveat: texto propuesto para §8 y una
  aclaración para §6.2 de que "primario" no implica que su positivo sea más diagnóstico de
  linaje que el del Test 2 residualizado.

## Huecos pendientes entre el texto y el material suplementario

- El paper cita una "Figura 2" (tasa de positivos por arquitectura, gráfico de barras) que
  ningún script de `code/` genera todavía — solo hay tablas de texto.
- §6.2 describe el Test 1 como evaluado "con validación cruzada"; la implementación actual
  usa un único split 50/50 por réplica (no k-fold). Falta decidir si se ajusta el código o
  la redacción.

License: Apache License 2.0 (ver `LICENSE` en la raíz del repositorio).
