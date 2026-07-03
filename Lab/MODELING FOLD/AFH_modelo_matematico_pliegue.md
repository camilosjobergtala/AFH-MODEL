# Modelo matemático del Pliegue Autopsíquico (∇)

Formalización del marco AFH (*Autopsychic Fold and Horizon*) descrito en Sjöberg Tala,
"Presencia fenomenológica y el tálamo intralaminar" (2026). Este documento traduce las
definiciones verbales de H\*, ∇, R y V (Sección 3.4 del manuscrito) y las cinco condiciones
de universalidad (Sección 5.2) a un formalismo matemático único, computable sobre la
dinámica de cualquier sistema — cerrando exactamente el hueco que el propio manuscrito
señala como pendiente: *"la operacionalización independiente del sustrato de las cinco
condiciones no existe todavía, ni para AFH ni... para ningún marco de este tipo"* (§5.2).

No es una teoría nueva ni reemplaza al manuscrito: es su aparato formal. Todo lo que aquí
se define debe leerse como *candidato operacional*, con el mismo estatus de riesgo empírico
que el manuscrito atribuye a la hipótesis constitutiva (§4.1) — ver §11 (Límites) al final.

**Relación con el código del repositorio.** `Lab/MODELING FOLD/0.4.py` (versión interna
"v0.5") ya implementa el caso más simple (escalar, dos nodos) de la realización canónica
descrita en §7, incluyendo la distinción Tipo A/Tipo B y el test de Granger específico que
aquí se generaliza como estimador de ∇ (§4). `Lab/AFH vs ECLIPSE/FOLD/NIVEL 4/` contiene
intentos empíricos previos de operacionalizar H\* y V sobre EEG real (theta/beta ratio,
complejidad de Lempel-Ziv). §10 traza la correspondencia explícita entre ese código y las
definiciones de abajo.

---

## 1. Sistema dinámico general

Sea un sistema descrito únicamente por su dinámica, sin referencia a sustrato — esto es
lo que la condición C1 de universalidad exige (§5.2 del manuscrito, "definida sobre la
dinámica del sistema").

Estado: $s(t) \in \mathcal{S} \subseteq \mathbb{R}^n$.

Dinámica (posiblemente estocástica, con retardo):

$$
\dot{s}(t) = F\big(s(t),\, s(t-\tau_1), \dots, s(t-\tau_k),\, \xi(t)\big)
$$

donde $\xi(t)$ es ruido y $\tau_1,\dots,\tau_k$ son los retardos estructurales del sistema
(en el cerebro: latencias de conducción y de procesamiento; en cualquier otro sistema,
lo que sea que retarde la propagación de una señal a través de él).

**Trayectoria en ventana:** $s_{[t-W,t]} := \{s(u) : u \in [t-W, t]\}$. Como se verá en §5,
las cinco condiciones de universalidad están definidas sobre $s_{[t-W,t]}$, no sobre
$s(t)$ solo — esta es la diferencia estructural con la Teoría de la Información Integrada
(IIT), que opera sobre una única configuración causal instantánea (§5.1, §5.3 del
manuscrito).

---

## 2. Descomposición rápida/lenta y el hub

AFH propone un *hub* de convergencia temporal (H\*, realizado biológicamente por los ILN
— Sección 3.2 del manuscrito). Se descompone el estado:

$$
s(t) = \big(u(t),\, c(t)\big), \qquad u(t)\in\mathbb{R}^{m}\ \text{(hub, "ILN")},\quad
c(t)\in\mathbb{R}^{n}\ \text{("elaboración cortical")}
$$

El hub recibe dos entradas cualitativamente distintas:

1. **Entrada rápida** $f(t-\tau_f)$: una vía aferente de latencia corta,
   $\tau_f \in [8,30]\,\text{ms}$ para el componente reticular más temprano, o
   $\tau_f \in [50,100]\,\text{ms}$ para la ventana talamocortical rápida más amplia
   relevante a ∇ específicamente (el manuscrito distingue ambas escalas en §3.2 y aclara
   que no compiten por una misma cantidad).
2. **Retorno** $r(t-\tau_s)$, $\tau_s \in [200,600]\,\text{ms}$: no es una entrada externa
   nueva, sino una transformación cortical de la propia salida pasada del hub — el
   contenido matemático exacto de "copia eferente" (§3.3 del manuscrito).

$$
\tau_u\, \dot u(t) = -u(t) + \tanh\!\big(\alpha f(t-\tau_f) + \beta\, r(t-\tau_s)\big), \qquad
\tau_c\, \dot c(t) = -c(t) + \tanh\!\big(u(t-\tau_f)\big)
$$

con $r(t) := c(t)$ (el retorno *es* el estado cortical, retardado). $\beta$ es la ganancia
del lazo de retorno — el parámetro de control cuyo umbral crítico se identificará con H\*
en §6.

Esta es exactamente la ecuación que `simular_AFH()` en `0.4.py` implementa para el caso
escalar $m=n=1$, con `beta`=$\beta$, `tau`=$\tau_s$, `tau_x`=$\tau_u$, `tau_y`=$\tau_c$, y
$\tau_f\to dt\approx 0$ (el código no modela explícitamente el retardo aferente rápido; ver
§10).

---

## 3. Cierre reflexivo vs. heterogéneo: la distinción formal

Esta es la afirmación de mayor riesgo del marco (§3.3, §4.1 del manuscrito) y la que debe
formalizarse con más cuidado, porque de ella depende toda la Predicción 6.

**Definición (Convergencia heterogénea, Tipo A).** El retorno es un proceso exógeno
estadísticamente independiente de la historia propia del sistema:

$$
r^{A}(t) = E(t), \qquad \sigma\big(E(t)\big) \perp \sigma\big(\{u(s): s < t\}\big)
$$

donde $E(t)$ puede estar *emparejado en momentos estadísticos* con el retorno reflexivo
(misma media, varianza, escala de autocorrelación) — precisamente para que la
discriminación no sea trivial. Esto es lo que `_ruido_coloreado()` en `0.4.py` construye:
un proceso Ornstein–Uhlenbeck con los mismos $\mu,\sigma$ que el retorno cortical real,
pero generativamente desconectado de $u$.

**Definición (Convergencia reflexiva, Tipo B).** El retorno es una función medible de la
propia trayectoria pasada del sistema:

$$
r^{B}(t) = \varphi\big(u(s) : s \le t\big) \quad \text{para algún } \varphi \text{ intrínseco al sistema}
$$

En el hub canónico de §2, $\varphi$ es literalmente la elaboración cortical:
$r^B(t-\tau_s) = c(t-\tau_s) = \tanh(u(t-\tau_s-\tau_f))$.

**Por qué esto captura "cierre reflexivo" y no "retroalimentación genérica".** Un
termostato satisface $r(t)=g(\text{mundo}(t))$ donde mundo cambia por la propia salida —
retroalimentación de *consecuencia externa*. Aquí, en cambio, lo que retorna nunca sale del
sistema: $\varphi$ actúa sobre $u$ mismo, no sobre un efecto de $u$ en un entorno. Esta es
precisamente la distinción que el manuscrito traza en §3.3 ("retroalimentación de
continuación causal" vs. "de consecuencia externa"), y es lo que hace que Tipo A y Tipo B
sean generativamente distintos incluso cuando son estadísticamente indistinguibles en sus
momentos de primer y segundo orden.

**Consecuencia empírica (Predicción 6 formalizada).** La distinción Tipo A/B no puede leerse
de la varianza o la correlación cruzada bruta (ambas están, por diseño, emparejadas). Lo
único que las separa es una propiedad *causal/informacional* de tercer orden — esto motiva
el estimador de §4.

---

## 4. ∇ operacional: de Granger específico a información de transferencia

El manuscrito dice que ∇ debe ser "rastreable... a la elaboración cortical de la entrada
rápida inmediatamente precedente, no una corriente independiente" (Predicción 6). La forma
general y sustrato-independiente de esa afirmación es la **información de transferencia
condicional** (Schreiber, 2000):

$$
\nabla(t;\tau) := I\big(u(t)\,;\, r(t-\tau) \,\big|\, u(t-1),\dots,u(t-p)\big)
$$

— cuánta información sobre $u(t)$ aporta el retorno retardado $r(t-\tau)$, más allá de lo
que el propio pasado autorregresivo de $u$ ya explica. Esta cantidad es cero por
construcción bajo Tipo A puro sin acoplamiento hacia adelante, y estrictamente positiva
bajo Tipo B con $\beta>0$, porque en Tipo B $r(t-\tau)$ *es* una función determinista del
pasado de $u$ que la parte autorregresiva lineal no captura completamente (por la no
linealidad $\tanh$ y por combinar información de dos retardos distintos).

**Estimador lineal-gaussiano (lo que el código ya calcula).** Cuando $u,r$ son
aproximadamente gaussianos y la dependencia es lineal, $\nabla(t;\tau)$ se estima mediante
la mejora de varianza explicada al comparar dos modelos anidados — exactamente
`granger_en_tau()` en `0.4.py`:

$$
\text{Modelo restringido:}\quad u_t = \textstyle\sum_{i=1}^p a_i u_{t-i} + \varepsilon_t
\qquad
\text{Modelo no restringido:}\quad u_t = \textstyle\sum_{i=1}^p a_i u_{t-i} + b\, r_{t-\tau} + \varepsilon_t
$$

$$
\widehat{\nabla}(\tau) := \Delta R^2 = R^2_{\text{no restr.}} - R^2_{\text{restr.}}, \qquad
F = \frac{(RSS_r - RSS_{nr})/1}{RSS_{nr}/(T-p-2)} \sim F_{1,\,T-p-2}
$$

Esto es la causalidad de Granger específica en $\tau$ del código, y es exactamente el
caso especial lineal-gaussiano de $\nabla(t;\tau)$ arriba. La ventaja de presentarlo como
información de transferencia condicional es que la definición general no depende de que
la relación sea lineal ni de que las variables sean gaussianas — requisito de C1
(independencia de sustrato).

**Criterio de Tipo B vs. Tipo A (operacional).** Rechazar $H_0: \nabla(\tau)=0$
consistentemente bajo Tipo B, y no rechazar bajo Tipo A pese a momentos estadísticos
idénticos, es la prueba decisiva. `0.4.py` reporta esto como
`F_Tipo B ≫ F_Tipo A` con `prop_sig(B) > 0.8`, `prop_sig(A) < 0.2` en el resumen de
`main()` — una instancia concreta y ya ejecutable de la Predicción 6.

---

## 5. H\* como bifurcación: derivación analítica del umbral

El manuscrito describe H\* como "un umbral hipotético de convergencia organizacional
talamocortical" (§3.4) pero señala que su "métrica operacional [está] pendiente" (Tabla 1).
Aquí se deriva un candidato: **H\* es la ganancia crítica de lazo a la cual el punto fijo
desacoplado deja de ser estable** — un umbral de bifurcación bien definido, computable
puramente de $F$ (condición C1), sin referencia anatómica.

**Linealización.** Cerca de $u=c=0$, con $\tanh(z)\approx z$, y absorbiendo la entrada
rápida en un término de forzamiento que no afecta la estabilidad del lazo homogéneo:

$$
\tau_u\, \dot u(t) = -u(t) + \beta\, c(t-\tau_s), \qquad
\tau_c\, \dot c(t) = -c(t) + u(t-\tau_f)
$$

Con ansatz $u(t)=U e^{\lambda t}$, $c(t)=C e^{\lambda t}$:

$$
U(1+\tau_u\lambda) = \beta C e^{-\lambda\tau_s}, \qquad C(1+\tau_c\lambda) = U e^{-\lambda\tau_f}
$$

Sustituyendo, la **ecuación característica** del lazo es:

$$
(1+\tau_u\lambda)(1+\tau_c\lambda) = \beta\, e^{-\lambda(\tau_s+\tau_f)}
$$

**Umbral estático ($\lambda=0$):**

$$
\boxed{\ \beta_c = 1\ }
$$

*independientemente de $\tau_u,\tau_c,\tau_f,\tau_s$.* Esto es un hecho general de teoría de
control: un retardo puro tiene ganancia unitaria en $\lambda=0$ ($e^{-0\cdot\tau}=1$), así
que no participa del umbral de ganancia de continua — sólo afecta la posibilidad de
bifurcaciones oscilatorias adicionales en $\lambda=i\omega$ (Hopf, dependientes de $\tau$;
candidatas naturales para conectar con la resonancia gamma ~40 Hz de Llinás, aunque esto es
una extensión no derivada aquí, no una afirmación del marco).

**Interpretación.** Para $\beta<1$, el origen es el único punto fijo estable: el sistema no
sostiene un estado de cierre — régimen "sin Pliegue". Para $\beta>1$, el origen se
desestabiliza (bifurcación de tipo pitchfork, por la simetría impar de $\tanh$) y emergen
dos puntos fijos no triviales simétricos: el lazo se autosostiene. Se propone:

$$
H^\ast := \{\beta = 1\} \quad\text{(en las unidades normalizadas donde } \tanh'(0)=1\text{)}
$$

Esto cumple exactamente la forma que el manuscrito exige: un umbral "independiente del
sustrato", derivado sólo de la estructura del lazo (ganancia unitaria), no de una escala de
tiempo particular ni de una anatomía particular — los ILN son *un* realizador de esta
estructura, no su definición (§3.1 del manuscrito).

**Consistencia con el código.** El barrido `barrer_beta_nabla()` / `granger_barrido_beta()`
en `0.4.py` explora $\beta \in [0, 2.6]$ y usa $\beta=0.3$ como "bajo" y $\beta=1.5$ (por
encima de 1) como "alto" en la figura de demostración — consistente con $\beta_c=1$ derivado
arriba. El cruce empírico observado en las figuras 4/6 del código no caerá exactamente en
$\beta=1$ porque la simulación incluye forzamiento estocástico $F(t)$ y la no linealidad
completa (no sólo la linealización); ese desplazamiento inducido por ruido es él mismo un
objeto de estudio (teoría de bifurcaciones estocásticas), no una discrepancia del modelo.

---

## 6. R (Residuo) y V (Volumen Configuracional)

**R (Residuo).** El manuscrito es explícito en que R "no postula ningún sustrato adicional"
(§3.4): bajo la hipótesis de identidad, medición externa y reporte fenomenológico son dos
modos de acceso al mismo evento. Formalmente esto **no** es una función $R(\nabla, H^\ast)$
adicional — es una identidad:

$$
R(t) \ \equiv\ \mathbb{1}\big[\beta > \beta_c\big] \wedge \big[\nabla(t;\tau) \text{ estable y } >0 \text{ sobre una ventana de persistencia}\big]
$$

donde "$\equiv$" es identidad, no igualdad causal: la afirmación constitutiva (§4.1 del
manuscrito) es que el lado izquierdo y el derecho son el *mismo evento* descrito desde dos
modos de acceso (tercera persona / primera persona), no que uno produzca al otro. Esta
distinción de notación importa: escribir $R=f(\nabla,H^\ast)$ presupondría ya la posición
causal que el marco trata como una de tres hipótesis en competencia (§4.1), no como dada.

**V (Volumen Configuracional).** Se propone la dimensionalidad efectiva del subsistema
cortical, vía razón de participación sobre el espectro de covarianza en ventana deslizante:

$$
\Sigma(t) := \operatorname{Cov}\big(c(t')\big)_{t' \in [t-W, t]} \in \mathbb{R}^{n\times n}, \qquad
\lambda_1(t) \ge \dots \ge \lambda_n(t) \ \text{(autovalores de } \Sigma(t))
$$

$$
\text{PR}(t) := \frac{\big(\sum_i \lambda_i(t)\big)^2}{\sum_i \lambda_i(t)^2} \in [1, n], \qquad
V(t) := \frac{\text{PR}(t) - 1}{n - 1} \in [0, 1]
$$

$V\to 0$ cuando la actividad cortical colapsa a un único modo dominante (varianza
concentrada — p. ej. oscilación lenta dominante), $V\to 1$ cuando la varianza se reparte
uniformemente entre los $n$ modos (campo rico y diferenciado). Esto reproduce exactamente
la descripción cualitativa del manuscrito: "V varía de forma continua desde casi cero...
hasta el máximo en la vigilia alerta" (§2.1), y está *definido sólo cuando* $\beta>\beta_c$
(H presente) — coherente con la Figura 2 del manuscrito, donde V es indefinido bajo el
umbral.

Para señales binarizadas/simbólicas (EEG discretizado), un estimador alternativo y
complementario es la complejidad de Lempel–Ziv normalizada, ya usada en
`Lab/AFH vs ECLIPSE/FOLD/NIVEL 4/CASCADA*.py` — ver §10.

---

## 7. Las cinco condiciones de universalidad, formalizadas

El manuscrito (§5.2) lista cinco condiciones que un candidato constitutivo debe satisfacer
para ser universal (definible sin apelar a un observador externo). Se mapean aquí sobre las
cantidades definidas arriba:

| # | Condición (manuscrito) | Formalización |
|---|---|---|
| C1 | Definida sobre la dinámica | $\nabla,H^\ast,V$ son funcionales de $F$ y de $s_{[t-W,t]}$ solamente — ningún término hace referencia a "tálamo" o "corteza", sólo a $u,c,\tau,\beta$ genéricos. |
| C2 | Cierre de largo alcance | Existe $\tau_s \gg \tau_u,\tau_c$ (el retardo del lazo excede la escala de relajación intrínseca del hub) tal que $\nabla(t;\tau_s)$ es persistentemente $>0$ — distingue el lazo de un simple filtro pasa-bajos interno. |
| C3 | Auto-interno | $r^B(t)=\varphi(u(s){:}s\le t)$ es función exclusivamente de la propia historia del sistema (§3, condición de independencia de Tipo B vs. Tipo A) — excluye termostatos y controladores que miden un mundo externo cambiado. |
| C4 | De dos escalas temporales | El hub es función conjunta de $f(t-\tau_f)$ **y** $r(t-\tau_s)$ con $\tau_f \ne \tau_s$ simultáneamente — no una integración de un solo grano temporal (a diferencia de $\Phi$ de IIT, §5.1 del manuscrito). |
| C5 | Irreducible | $\nabla(t;\tau)$ es un funcional de la ley conjunta $(u_t, r_{t-\tau}, u_{t-1},\dots)$, no del núcleo de transición de un solo paso $P(s(t+dt)\mid s(t))$ evaluado en un instante — requiere la ventana extendida $[t-\tau,t]$ para computarse, no se descompone en un cómputo por-instante. |

**Nota de honestidad epistémica (heredada del manuscrito, §5.3).** Si C5 se redefiniera
como irreducibilidad de una *configuración causal instantánea*, esto colapsaría a una
medida del tipo $\Phi$ de IIT y la distinción con esa teoría se perdería. La respuesta —
igual que en el manuscrito — es que las cinco condiciones, incluida C5, se definen aquí
como propiedades que *requieren* una ventana temporal extendida para computarse por
construcción ($\nabla$ depende de dos instantes separados por $\tau$, nunca de uno solo);
esto es una decisión de diseño explícita, no una demostración de que ninguna
reformulación instantánea pudiera capturar la misma información.

Caracterizado así, el Pliegue es una **propiedad candidata universal**: un procedimiento
que, dado $F$ y una trayectoria observada, calcula las cinco condiciones y devuelve un
veredicto — negativo para un termostato o una red simple sin lazo de dos escalas
(C2/C3/C4 fallan trivialmente), positivo para el hub canónico de §2 con $\beta>1$.

---

## 8. Realización canónica generalizada (N nodos)

Generalización directa de §2 a poblaciones ($u\in\mathbb{R}^m$ ILN, $c\in\mathbb{R}^n$
corteza, matrices de acoplamiento $A,B,C$):

$$
\tau_u\, \dot u(t) = -u(t) + \tanh\!\big(A f(t-\tau_f) + B\, c(t-\tau_s)\big), \qquad
\tau_c\, \dot c(t) = -c(t) + \tanh\!\big(C\, u(t-\tau_f)\big)
$$

El escalar de §5 es el caso $m=n=1$, $B=\beta$. La derivación del umbral estático se
generaliza reemplazando $\beta$ por el **radio espectral del lazo linealizado**
$\rho(BC)$:

$$
H^\ast := \{\rho(BC) = 1\}
$$

(ganancia de lazo unitaria en el modo dominante) — mismo argumento de §5: los retardos
puros no alteran la ganancia en $\lambda=0$, sólo la estructura de bifurcaciones
oscilatorias en $\lambda=i\omega$, que ahora sí depende de $\tau_f+\tau_s$ y podría
relacionarse con bandas de frecuencia específicas (una dirección de trabajo futuro, no
derivada aquí).

---

## 9. Las siete Predicciones (§6 del manuscrito), como hipótesis formales

| Predicción | Formalización |
|---|---|
| **P1** — correlación fenomenológica gradada | $\rho(t)$ = reporte en primera persona (escala continua); manipulación gradada de $\beta$ (titulación farmacológica, tACS delta, DBS). Se predice $\operatorname{corr}(\beta, \rho) > 0$ significativo, rastreando $\operatorname{corr}(\beta, \widehat\nabla)$. |
| **P2** — disociación N3/anestesia | N3 $:=\{\beta>\beta_c,\ V\approx 0\}$ (H presente, volumen colapsado); anestesia/lesión bilateral ILN $:=\{\beta<\beta_c\}$ (H ausente, V indefinido). Se predice $\widehat\nabla_{\text{N3}}$ no distinguible de vigilia en métricas específicas de los ILN, mientras $\widehat\nabla_{\text{anestesia}}\approx 0$. |
| **P3** — ordenamiento temporal en emergencia | Durante la recuperación de anestesia, el cruce $\beta(t)$ por $\beta_c$ (proxy de H\*) debe preceder temporalmente el cruce de $\widehat\nabla(t;\tau)$ por su criterio, y ambos preceder la evidencia conductual de presencia. |
| **P4** — covarianza H\*/∇ | Dentro de $\{\beta>\beta_c\}$: $\operatorname{corr}(\hat H^\ast, \widehat\nabla) > 0.50$. |
| **P5** — covarianza bajo manipulación de la retención | Manipular $\tau_s$ (duración de retención) debe desplazar $\widehat\nabla$ en la banda lenta (200–600 ms) covariando con vividez retencional reportada, sin afectar la banda rápida (8–30 / 50–100 ms) de control. |
| **P6** — origen reflexivo (prueba decisiva) | Test formal de §3: rechazar $H_0$: $r(t-\tau) \perp \{u(s):s<t-\tau\}$ (Tipo A) en favor de $r(t-\tau)=\varphi(u_{<t-\tau})$ (Tipo B), vía $\widehat\nabla(\tau)=\Delta R^2$/F de Granger específico sobre registro intracraneal real. |
| **P7** — diferencial de latencia | $\tau_s - \tau_f \in [150, 550]\,\text{ms}$, directamente medible como la diferencia entre el retardo aferente rápido (8–30 ms) y el retorno lento (200–600 ms de lazo completo). Consistente con los rangos de §2. |

---

## 10. Condiciones de falsación (§6.4), como regla de decisión

Sea $\text{Falla}(k)\in\{0,1\}$ el indicador de que la condición $k$ de §6.4 del
manuscrito se cumple (evidencia en contra):

1. El acoplamiento fase-amplitud delta-gamma no replica como métrica sensible a $V$ en
   $\ge 2$ conjuntos de datos públicos independientes.
2. Dentro de $\{\beta>\beta_c\}$: $\operatorname{corr}(\hat H^\ast,\widehat\nabla) < 0.50$.
3. $\operatorname{corr}(\widehat\nabla, \rho) < 0.15$ a través de múltiples paradigmas de
   manipulación gradada.
4. N3 indistinguible de anestesia profunda en $\ge 4$ de $5$ operacionalizaciones
   planificadas específicas de los ILN.
5. Emergen disociaciones sistemáticas y replicables entre $\widehat\nabla$ operativo (bajo
   todos los criterios medibles) y la presencia reportada.

**Regla:**

$$
\text{Rechazar la posición constitutiva} \iff \big[\text{Falla}(1)\wedge\text{Falla}(2)\big] \ \lor\ \text{Falla}(3) \ \lor\ \text{Falla}(4) \ \lor\ \text{Falla}(5)
$$

— cumplir dos o más de (1)–(4), o (5) por sí sola, motiva retroceder a una interpretación
causal o correlacional (exactamente el umbral que el manuscrito fija por adelantado en
§6.4).

---

## 11. Qué demuestra este documento y qué no

- **Demuestra** que las cinco condiciones de universalidad (§5.2 del manuscrito) admiten
  una operacionalización concreta, computable únicamente a partir de $F$ y de una
  trayectoria observada, sin apelar a anatomía — cerrando parcialmente el hueco que el
  propio manuscrito marca como pendiente en esa sección.
- **Demuestra** que la realización canónica de §2/§8 tiene un umbral de bifurcación bien
  definido ($\rho(BC)=1$), independiente de todas las constantes de tiempo y retardos —
  un candidato matemáticamente no trivial para H\*.
- **Demuestra** que Tipo A y Tipo B (convergencia heterogénea vs. reflexiva) son
  generativamente distinguibles mediante información de transferencia condicional /
  Granger específico, incluso cuando están emparejados en momentos estadísticos de primer
  y segundo orden — dando contenido formal preciso a la Predicción 6.
- **No demuestra** la hipótesis constitutiva. Que $\beta>\beta_c$ y $\nabla$ estable
  *coocurran* con la presencia fenomenológica sigue siendo, exactamente como en el
  manuscrito (§4.1, §8), una apuesta empírica —no derivable de la matemática del lazo—.
  Nada en este formalismo explica *por qué* el cierre reflexivo debería ir acompañado de
  experiencia; sólo especifica con precisión qué tendría que medirse para sostener o
  refutar que van juntos.
- **No demuestra** que los ILN biológicos instancien Tipo B en lugar de Tipo A — eso sigue
  siendo, como señala §6.3 del manuscrito, la brecha empírica mayor, pendiente de registro
  intracraneal directo (Predicción 6).
- El umbral analítico $\beta_c=1$ (§5, §8) es una propiedad de *esta* realización canónica
  bajo linealización sin ruido; no es una afirmación de que la ganancia biológica real del
  lazo talamocortical valga exactamente 1 en ninguna unidad particular. Extender la
  derivación a la bifurcación bajo forzamiento estocástico (que desplaza el umbral
  observado empíricamente, como en los barridos de `0.4.py`) es trabajo pendiente.

---

## 12. Correspondencia con el código existente

| Símbolo aquí | En `Lab/MODELING FOLD/0.4.py` | En `Lab/AFH vs ECLIPSE/FOLD/NIVEL 4/` |
|---|---|---|
| $u(t)$ (hub) | `x` (tálamo) | señal talámica/ILN extraída de EEG |
| $c(t)$ (elaboración) | `y` (corteza) | señal cortical |
| $\beta$ (ganancia de lazo) | `beta` | no modelado explícitamente (datos reales, no simulación) |
| $\tau_s$ (retorno lento) | `tau` | ventana de análisis PAC / Granger |
| Tipo B / Tipo A | `modo="B"` / `modo="A"` (ruido coloreado emparejado) | no implementado — el registro real no permite el contrafactual "sin acoplamiento" |
| $\widehat\nabla(\tau)$, estimador lineal | `granger_en_tau()` → `F`, `ΔR²` | ninguno directo; `HCTSA*.py`/`hctsa discovery.py` buscan proxies (Hjorth, DFA) |
| $\beta_c$ (umbral H\*) | cruce empírico en `barrer_beta_nabla()`/`granger_barrido_beta()`, $\beta\in[0,2.6]$ | "H\* v2" (`NIVEL 4/3.py`): proxy exploratorio `theta_beta_ratio`, sin derivación analítica |
| $V(t)$ (razón de participación) | no implementado | complejidad de Lempel-Ziv en `CASCADA*.py` (estimador alternativo, señal simbólica) |

Esta tabla es la ruta concreta para extender el código existente: `0.4.py` ya calcula
$\widehat\nabla$ vía Granger específico y compara Tipo B/A; le falta el retardo aferente
$\tau_f$ explícito (§2, §8) y una estimación directa de $\beta_c$ vía ajuste del punto de
bifurcación en lugar de sólo inspección visual del barrido. `NIVEL 4/` opera sobre datos
reales (Sleep-EDF) pero sin el contrafactual Tipo A/B que sólo la simulación puede ofrecer
limpio — de ahí que la Predicción 6 siga, correctamente, marcada como pendiente en el
manuscrito (§6.3): requiere registro intracraneal con cobertura ILN adecuada, no disponible
en las cohortes usadas hasta ahora en este repositorio.
