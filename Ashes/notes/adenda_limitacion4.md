# Adenda propuesta — §8 Limitación 4 y aclaración en §6.2

Motivada por `../code/barrido_confusor_latente.py`, que extiende la validación de §6.4
(Tabla 2) más allá del único punto de intensidad de confusor (kappa_U=0.9) que la tabla
reporta para la arquitectura `confusor_latente`. No incorporada todavía al texto del
manuscrito — es una propuesta para el autor, no una edición aplicada.

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
confirmando que lo que la Tabla 2 muestra en kappa=0.9 no es un artefacto de implementación.
La falla no es de umbral: crece de forma gradual con la fuerza del confusor, y de forma
asimétrica entre los dos tests — el Test 1 supera 50% de positivos ya en kappa_U≈0.15,
mientras el Test 2 residualizado se mantiene calibrado hasta kappa_U≈0.30-0.45 y recién
después se dispara.

## Cómo hay que leer esa asimetría (importante — no es "un test más frágil que el otro")

Una primera lectura tentadora es "el test primario es el más frágil, habría que invertir la
jerarquía". Es incorrecta, y por una razón que toca el núcleo del §4:

1. **El Test 1 es el test de la condición necesaria.** En el aparato del §4, refutar "linaje
   E→L necesario" exige mostrar independencia condicional E ⊥ L | X — que es exactamente la
   condición (a) del §6.3, y lo que el Test 1 evalúa. El Test 2 mide algo *más fuerte* que la
   condición necesaria (identificabilidad por ensayo). Por construcción, entonces, el Test 1
   es el falsador primario; el Test 2 corrobora especificidad por ensayo. Invertir pondría al
   test de la propiedad más-que-necesaria como falsador principal, rompiendo la estructura
   falsacionista sobre la que se apoya todo el §4.

2. **Bajo confusor latente ninguno de los dos "falla como test".** E y L son *genuinamente*
   dependientes dado el X observado (serían independientes recién dado X *y* U). El Test 1
   detecta correctamente esa dependencia; lo "falso" está solo en la sobre-lectura como
   linaje, no en el test. Lo mismo vale para el Test 2r. Ambos positivos son correctos
   respecto de lo que operacionalmente miden y engañosos solo como evidencia de linaje —
   exactamente el punto que §6.4-6.5 ya declara.

3. **La asimetría es sensibilidad/especificidad, no robustez.** El Test 1 es el detector
   *más sensible* de dependencia condicional: dispara con acoplamiento más débil, venga de
   linaje o de confusor. El Test 2r es *más específico*: exige que la señal sea lo bastante
   fuerte para sostener identificabilidad por ensayo. Que el Test 1 dispare antes bajo
   confusor débil es el reverso de su mayor poder, no un defecto — la curva
   sensibilidad/especificidad de manual. Razón estructural: el Test 1 acumula covarianza
   residual sobre toda la muestra y todas las dimensiones de L, de modo que cualquier fuga no
   nula de un confusor no medido se vuelve detectable con N suficiente; el Test 2r exige que
   el aporte incremental de E identifique, ensayo por ensayo, cuál L residual le corresponde,
   y una fuga débil se pierde en el ruido idiosincrático antes de discriminar nada.

**Decisión (autor, resuelta):** mantener el Test 1 como primario —es el test de la condición
necesaria— y afinar el caveat, no invertir ni disolver la jerarquía.

## Texto propuesto para §8 (Limitación 4)

Texto actual:

> "Cuarta, la validación del ejemplo es sintética: establece la sensibilidad y especificidad
> del diseño frente a arquitecturas conocidas —incluida su falla sistemática ante factores
> latentes— pero no sustituye su aplicación a datos reales ni la variación exógena que §6.5
> exige."

Adenda propuesta (a continuación, misma viñeta):

> "La Tabla 2 reporta esa falla en un único punto de intensidad de acoplamiento; un barrido
> posterior (material suplementario, `barrido_confusor_latente.py`) muestra que no es un
> fenómeno de umbral sino gradual: la tasa de positivos bajo confusor latente crece de forma
> continua con la fuerza del confusor. El barrido caracteriza además a los dos tests como un
> par sensibilidad/especificidad — el Test 1 detecta dependencia condicional con un
> acoplamiento más débil que el que el Test 2 residualizado necesita para alcanzar
> identificabilidad por ensayo—. Conviene subrayar que, bajo confusor latente, ninguno de los
> dos incurre en error respecto de lo que operacionalmente mide: E y L son genuinamente
> dependientes dado el X observado, y el positivo es engañoso solo como evidencia de linaje,
> que es el punto de §6.4-6.5. Por ello un positivo del Test 1, por ser el más sensible, no es
> por eso más diagnóstico de linaje que uno del Test 2 residualizado."

## Aclaración propuesta para §6.2 (una cláusula)

Donde el texto introduce el Test 1 como "(primario)", agregar (o nota al pie):

> "'Primario' refiere a que el Test 1 opera la condición necesaria del §4 —dependencia
> condicional E ⊥̸ L | X, cuya ausencia refuta el linaje— y a su mayor sensibilidad, no a que
> su positivo sea más diagnóstico de linaje que el del Test 2 residualizado: por §6.4 y el
> barrido suplementario, ambos positivos son igualmente compatibles con una causa común
> latente."

Con estas dos incorporaciones, el rótulo "primario/corroborativo" deja de sugerir —aunque sea
implícitamente— una jerarquía de robustez que el barrido contradice, sin tocar la lógica
falsacionista de §6.3 (que sigue siendo conjuntiva y bien justificada: se falsa solo si el
test sensible *y* el específico dan negativo con potencia adecuada).
