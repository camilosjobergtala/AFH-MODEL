# Adenda propuesta — §8 Limitación 4 (y nota abierta sobre §6.2-6.3)

Motivada por `../code/barrido_confusor_latente.py`, que extiende la validación de §6.4
(Tabla 2) más allá del único punto de intensidad de confusor (kappa_U=0.9) que la tabla
reporta para la arquitectura `confusor_latente`. No incorporada todavía al texto del
manuscrito — es una propuesta para que el autor decida, no una edición aplicada.

## Resultado que motiva la adenda

```
kappa_U |  T1 dR2   T1+ | T2n dAcc  T2n+ | T2r dAcc  T2r+
   0.00 |  -0.002    5% |    0.009   62% |    0.000    5%
   0.15 |   0.001   82% |    0.009   75% |    0.001    5%
   0.30 |   0.028  100% |    0.015   90% |    0.004   30%
   0.45 |   0.087  100% |    0.025  100% |    0.014   88%
   0.60 |   0.168  100% |    0.047  100% |    0.031  100%
   0.75 |   0.257  100% |    0.073  100% |    0.057  100%
   0.90 |   0.344  100% |    0.107  100% |    0.087  100%
```

(La fila kappa_U=0.90 es la fila `confusor_latente` de la Tabla 2 del paper.)

En kappa_U=0 (sin confusor), Test 1 y Test 2 residualizado calibran al nivel nominal (~5%),
confirmando que la falla no es un artefacto de implementación. Pero la falla no es de
umbral: crece de forma gradual y, sobre todo, **asimétrica** — el Test 1 supera 50% de
falsos positivos ya en kappa_U≈0.15-0.30 (una fracción de la intensidad de la Tabla 2),
mientras el Test 2 residualizado se mantiene calibrado hasta ese mismo rango y recién
después se dispara.

**Razón estructural** (no solo un patrón numérico): el Test 1 acumula covarianza residual
sobre toda la muestra y las 6 dimensiones de L, así que cualquier fuga no nula de un
confusor no medido se vuelve estadísticamente detectable con N suficiente, por diminuta que
sea — es el problema conocido de que casi ninguna hipótesis nula puntual ("la covarianza es
exactamente cero") es cierta con datos continuos y N grande. El Test 2 residualizado exige
algo más duro: que el aporte incremental de E alcance para identificar, ENSAYO POR ENSAYO,
cuál L residual le corresponde contra sus vecinos del mismo estrato — un criterio de
discriminabilidad individual que una fuga débil y difusa no satisface aunque sea
estadísticamente detectable en agregado.

## Texto propuesto para agregar a la Limitación 4 (§8)

Texto actual del paper:

> "Cuarta, la validación del ejemplo es sintética: establece la sensibilidad y especificidad
> del diseño frente a arquitecturas conocidas —incluida su falla sistemática ante factores
> latentes— pero no sustituye su aplicación a datos reales ni la variación exógena que §6.5
> exige."

Adenda propuesta (a continuación del texto anterior, misma viñeta):

> "La Tabla 2 reporta esa falla sistemática en un único punto de intensidad de acoplamiento;
> un barrido posterior (material suplementario, `barrido_confusor_latente.py`) muestra que no
> es un fenómeno de umbral: la tasa de falsos positivos crece de forma gradual con la fuerza
> del confusor, y lo hace de forma asimétrica entre los dos tests — el Test 1 (dependencia
> condicionada, declarado primario en §6.2) se satura al 100% con una fracción de la
> intensidad que requiere el Test 2 residualizado para hacer lo mismo. La razón es
> estructural: el Test 1 acumula covarianza residual sobre toda la muestra, de modo que
> cualquier fuga no nula de un confusor no medido termina siendo detectable con N suficiente,
> mientras que el Test 2 residualizado exige identificabilidad ensayo por ensayo, un criterio
> que una fuga débil no alcanza a satisfacer. En consecuencia, ante confusión latente débil el
> Test 1 es el más vulnerable de los dos, no el más robusto, pese a su designación como
> primario."

## Pregunta abierta: ¿mantener el Test 1 como "primario"? (§6.2-6.3)

No se resuelve acá — es una decisión editorial del autor sobre la arquitectura argumental
del paper, no algo para aplicar unilateralmente. Los argumentos:

**A favor de mantener a Test 1 como primario:**
- Mayor poder ante recurrencia genuina y ante confusores fuertes (Tabla 2, fila
  `recurrencia`: ambos tests detectan al 100%, pero el Test 1 es más simple de computar e
  interpretar — una ganancia de R², no un procedimiento de emparejamiento).
- Es más fácil de exponer y de preregistrar como criterio de falsación único (§6.3 ya lo
  usa como condición (a) del criterio conjunto).

**A favor de revertir la jerarquía (Test 2 residualizado como primario, Test 1 como
corroborativo):**
- El Test 2 residualizado aproxima mejor el estimando que el propio §6.1 declara
  ("continuidad informacional temprana-tardía condicionada", no "existe alguna covarianza
  residual, sea cual sea su origen").
- Es más conservador precisamente en el régimen más realista y más insidioso: un confusor
  latente *fuerte* que domina la señal es, en la práctica, más fácil de sospechar (y quizás
  de medir y mover a X); uno *débil y difuso* —el que este barrido muestra que rompe el
  Test 1 primero— es el que pasa desapercibido en un registro real.

**Recomendación de mínima intervención:** incorporar la adenda de arriba a la Limitación 4
(defendible sin reescribir la arquitectura argumental del paper) y, si el autor quiere ir
más allá sin invertir la jerarquía, agregar una frase en §6.2 aclarando que "primario"
refiere a simplicidad de cómputo/interpretación, no a robustez ante confusión latente débil
— evitando así que el texto prometa, incluso implícitamente, una jerarquía de robustez que
este barrido contradice.
