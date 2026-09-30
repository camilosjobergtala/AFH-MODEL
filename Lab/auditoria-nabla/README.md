# Auditoría de ∇ y de H\* (modelo formal AFH)

Verificación ejecutada del aparato formal que cita el manuscrito de *Neuroscience of
Consciousness* (Sec. 3.6) y `NeuroscienceOfConsciousness_revision/model/AFH_modelo_matematico_pliegue.md`
(Sec. 3–5, 8). Usa el código original de `NeuroscienceOfConsciousness_revision/code/AFH_modelo_N_nodos.py`
**sin modificarlo**: la única adición es un "espía" que captura el proceso exógeno E que
`simular_AFH_N()` genera internamente en Tipo A.

```bash
python auditoria_tipo_AB.py        # ~40 s  -> resultados_tipo_AB.txt
python auditoria_umbral_beta_c.py  # ~1,5 min con 4 núcleos -> resultados_umbral_beta_c.txt
```

Ambos son deterministas (semillas fijas) y reproducen primero las cifras originales
(F = 1088,85 vs 1,20; error de β_c = 1,47 / 3,14 / 5,53 %) antes de testear otra cosa.

## Hallazgo 1: ∇ definido como TE(retorno → hub) no separa Tipo A de Tipo B

Semilla de matrices 2 (la del bloque [3] original), β = 1,5, 10 repeticiones:

| Test | Tipo B | Tipo A |
|---|---|---|
| T1 original: C(t−400 ms) → U(t), 2,5 ms de pasado del hub (F) | 1088,85 | 1,20 |
| T2 definición formal literal: retorno real r(t−400 ms) → U(t) (F) | 1230,19 | **10792,45** |
| T3 = T1 condicionando en el pasado del hub a la latencia del lazo (F) | **1,90** | 0,70 |
| T4 reflexividad: R² fuera de muestra de r(t) desde el pasado de U | **1,00** | **−0,23** |

Con las semillas 1 y 3 el patrón es el mismo (ver `resultados_tipo_AB.txt`).

**Por qué.** Si el retorno es r(t−τ) = φ(pasado de u) + η(t−τ), entonces, condicionando en el
pasado completo de u, φ(·) es una constante y
I(u(t); r(t−τ) | pasado de u) = I(u(t); η(t−τ) | pasado de u).
Con condicionamiento completo, la transferencia de información desde el retorno sólo ve la parte
del retorno que **no** viene del hub (η): mide heterogeneidad, no reflexividad.
- En Tipo B puro (η = 0) vale 0. T3 lo muestra: F pasa de 1088,85 a 1,90.
- En Tipo A puro (φ = 0) es positiva cuando β ≠ 0. T2 lo muestra.
- T1 separa los tipos por dos motivos:
  - en la simulación de Tipo A, la señal cortical testeada no entra al hub;
  - la ventana autorregresiva (2,5 ms) es mucho más corta que el lazo (420 ms).

La cantidad que sí separa los tipos va en la dirección inversa, **fuente → retorno** (T4). Es el
criterio P(R | E, Z) ≠ P(R | Z) que el propio manuscrito enuncia en Sec. 3.6, la prueba de rastreo
de fuente (modelo .md Sec. 9.1) y el componente A1 → B del criterio conjuntivo de `Ashes/`.

**Corrección propuesta (conjuntiva).** ∇ requiere las dos condiciones:
1. **Reflexividad.** El retorno es predecible desde el propio pasado del hub:
   I(r(t); u(≤ t−τ_f)) > 0. En el caso lineal-gaussiano, es el R² de T4. Tipo A falla esta.
2. **Eficacia.** El retorno influye sobre el hub más allá de una ventana declarada W < τ_s:
   TE_W(r → u) > 0. Leída así, es información del pasado del hub, de fuera de la ventana presente,
   que vuelve a entrar (una lectura formal de la retención). Un sistema sin lazo falla esta.

La misma identidad implica una advertencia para el componente C2 de `Ashes/` (B → A2 | A1, X): sólo
es identificable a través de la variabilidad propia de B que A1 no explica. Un relevo puramente
determinista daría C2 = 0. En las arquitecturas simuladas de `Ashes/`, B siempre tiene ruido
propio, así que esto no cambia esos resultados, pero conviene declararlo como supuesto.

## Hallazgo 2: la cifra "tres órdenes de magnitud" depende de la semilla

La semilla 2 tiene autovalor dominante del lazo L = B@C igual a **−1**. Eso es un régimen
oscilatorio, fuera de la derivación de bifurcación estática. Con autovalor +1 (semillas 1 y 3),
T1 da F = 13,45 y 15,99 (Tipo B) contra 0,72 y 0,56 (Tipo A). La separación sigue existiendo, pero
es de ~20–30×, no de ~1000×. En la semilla 1, además, sólo 6 de 10 repeticiones de Tipo B son
significativas.

## Hallazgo 3: el umbral β_c = 1

- **La derivación exige que el autovalor dominante de L sea real y positivo.** Con
  `construir_matrices()` eso ocurre en 316 de 1000 semillas. Las otras 684 tienen autovalor
  dominante real negativo (332) o complejo (352). En esos casos la inestabilidad es
  oscilatoria, no hay puntos fijos no triviales, y el umbral depende (poco) de los retardos:
  β_c = 1,0190 / 1,0049 / 1,0019 para μ = −1 y D = 208 / 430 / 700 ms.
  - Arreglo: imponer acoplamientos excitatorios (B, C ≥ 0), que además es lo biológicamente
    correcto para las proyecciones talamocorticales y corticotalámicas. Perron–Frobenius
    garantiza entonces un autovalor dominante real y no negativo, y β_c = 1 es exacto.
- **El error de 1,5–5,5 % no es discretización.** El umbral del sistema discretizado es
  exactamente 1 (ganancia estática del lazo). Al reducir dt el error **sube**
  (1,47 → 3,91 %; 5,53 → 14,44 %), y al alargar T de 20 a 80 s **baja** (→ 0,80 %; 2,91 %).
  Es un sesgo del estimador por tiempo finito. Cerca del umbral, el crecimiento
  λ ≈ (β−1)/(τ_u+τ_c+β·D) es más lento cuanto mayor es el retardo D, y por eso el error crece
  con los retardos.

## Qué no muestra esta auditoría

Nada de esto afecta la pregunta empírica central (P6/P7, P1), que sigue sin datos. Tampoco
invalida el diseño de rastreo de fuente de Sec. 6.6 ni de `Ashes/`: al contrario, identifica
ese diseño como el que corresponde a la definición de cierre reflexivo. Lo que corrige es la
definición formal de ∇ como transferencia de información (manuscrito Sec. 3.6, modelo .md
Sec. 4) y las cifras presentadas como su demostración.
