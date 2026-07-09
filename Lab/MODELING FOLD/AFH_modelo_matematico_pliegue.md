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

## 9.1 De correlación de magnitud a identificabilidad de contenido: la prueba de rastreo de fuente

La Predicción 6 tal como se formaliza en §3–4 opera sobre una cantidad **escalar**: la
fuerza (magnitud) de la señal lenta en un retardo $\tau$. Eso es necesario pero no agota lo
que el manuscrito llama "rastreo de fuente": que la señal retardada "derive... del mismo
procesamiento cortical desencadenado por la entrada rápida" es una afirmación sobre el
**contenido específico** de cada instancia de convergencia, no sólo sobre si dos números
resumen (fuerza temprana, fuerza tardía) covarían. Ambas nociones son lógicamente
independientes:

- Dos ensayos pueden tener magnitudes correlacionadas sin que el patrón específico de la
  fase tardía porte ninguna huella del patrón específico de la fase temprana de *ese*
  ensayo — si la correlación de magnitud viene sólo de una ganancia global compartida
  (p. ej. arousal modulando la fuerza de ambas fases por igual).
- El contenido puede ser rastreable —la fase tardía *sí* deriva del procesamiento de la
  fase temprana específica— incluso si las magnitudes resumen no correlacionan, si la
  variación entre ensayos está sobre todo en **qué patrón** ocurrió, no en **cuán fuerte**
  fue.

Un análisis que sólo pruebe magnitud puede reportar "sin evidencia de Tipo B" cuando el
acoplamiento reflexivo genuino está presente pero codificado en el patrón, no en la fuerza
— un falso negativo estructural, no una falla del sistema. Esta sección formaliza la
prueba más exigente.

**Generalización vectorial.** Sea $e_i \in \mathbb{R}^d$ el contenido de la fase temprana
del ensayo $i$ (una featurización multivariada — espectral, de forma de onda, o cualquier
representación de la actividad talamocortical rápida de ese ensayo) y $l_i \in
\mathbb{R}^d$ el contenido de la fase tardía, en el mismo espacio de representación (o uno
relacionado por una transformación conocida). Como en §3, $z_i$ es el confusor latente
(arousal) y $\hat z_i$ su proxy observable y ruidoso.

$$
\textbf{Tipo A (heterogénea):}\quad l_i = \mu_L + B z_i \mathbf{1} + \eta_i, \qquad \eta_i \perp e_i
$$

$$
\textbf{Tipo B (reflexiva):}\quad l_i = \mu_L + B z_i \mathbf{1} + \Phi(e_i) + \eta_i, \qquad \Phi:\mathbb{R}^d\to\mathbb{R}^d \text{ fija, independiente del ensayo}
$$

Bajo Tipo A, ningún mapeo ajustado de $e$ a $l$ puede identificar de qué ensayo proviene
$l_i$ mejor que el azar, una vez descontado $z$ — porque $\eta_i$ es, por construcción,
independiente de la identidad de $e_i$. Bajo Tipo B, $\Phi(e_i)$ es una huella específica
del ensayo que sobrevive a controlar por $z$.

**La prueba: identificabilidad por emparejamiento, estratificada por confusor.**

1. Descontar el confusor de ambos contenidos por regresión lineal contra el proxy $\hat
   z$: $\tilde e_i := e_i - \widehat{\mathbb{E}}[e_i \mid \hat z_i]$, análogamente
   $\tilde l_i$.
2. Ajustar $\hat\Phi$ (regresión ridge $\tilde l \sim \tilde e$) sólo sobre ensayos de
   entrenamiento.
3. Sobre ensayos de prueba, calcular $\hat l_i := \hat\Phi(\tilde e_i)$ y, **dentro de cada
   estrato de $\hat z$** (para no comparar ensayos con niveles de confusor distintos),
   buscar cuál $\tilde l_j$ del mismo estrato está más cerca de $\hat l_i$. Estadístico:
   *exactitud top-1* — proporción de ensayos donde el más cercano es el propio $\tilde
   l_i$.
4. Nulo por permutación **dentro de cada estrato** (nunca global — permutar entre estratos
   subestima la tasa de acierto esperada bajo Tipo A, porque ensayos de estratos distintos
   ya son disímiles por el confusor solo, y eso vuelve la prueba anticonservadora; el
   mismo tipo de sesgo que el proxy ruidoso introduce en la Sec. de falsos positivos de
   `simulacion_acoplamiento_fase_temprana_tardia.py`).

**Relación con ∇.** Esta exactitud top-1 estratificada es la generalización
multivariada, a nivel de contenido, de $\nabla(t;\tau)$ (§4): mismo principio
—información sobre el presente que proviene específicamente de la propia trayectoria
pasada, más allá de lo que factores compartidos ya explican— aplicado a un vector de
contenido en vez de a un escalar de magnitud, y controlando por un confusor medido en vez
de por el propio pasado autorregresivo.

**Validación numérica (`test_rastreo_de_fuente.py`).** Con $N=400$ ensayos, $d=10$, y un
diseño donde el contenido temprano varía casi enteramente en dirección (patrón) y casi
nada en norma (fuerza): la prueba de magnitud da $r_{\text{parcial}}\approx 0.03$–$0.04$
($p>0.4$) en **ambos** modelos generativos — ciega, tal como se predice arriba. La prueba
de identificabilidad, sobre los mismos datos, da exactitud top-1 al nivel del azar bajo
Tipo A ($0.030$ vs. nulo $0.025$, $p=0.37$) y muy por encima del azar bajo Tipo B ($0.130$
vs. nulo $0.026$, $p<0.001$). Esto confirma que las dos pruebas no son redundantes: un
resultado nulo en la prueba de magnitud no permite concluir ausencia de Tipo B si no se
corrió también la de identificabilidad.

**El problema del estímulo compartido — más serio que el del arousal.** El modelo de
Tipo A de arriba asume $\eta_i \perp e_i$: el ruido del contenido tardío es independiente
de la *identidad específica* del contenido temprano. Eso es cierto si $e_i$ varía de
ensayo a ensayo sin ninguna estructura compartida con $l_i$ más allá de $z$. Pero si el
paradigma presenta **estímulos que varían entre ensayos** (no un estímulo único o
idéntico repetido), esa independencia falla incluso bajo Tipo A puro: el estímulo es una
causa común de $e_i$ y de $l_i$ por *dos vías anatómicamente separadas* —la vía rápida que
genera $e$, y cualquier procesamiento cortical descendente de $l$ que no tenga nada que
ver con un retorno reflexivo—, y ambas vías codifican de qué estímulo se trató. Un mapeo
$\hat\Phi$ ajustado sobre datos así puede lograr exactitud top-1 muy por encima del azar
**identificando el estímulo, no rastreando linaje causal** — un falso positivo de Tipo B
bajo Tipo A puro. Estratificar por arousal (como en el procedimiento de arriba) no
resuelve esto: el confusor relevante aquí es la identidad del estímulo, no el arousal, y
son necesidades de control distintas.

La corrección: si el paradigma varía el estímulo, condicionar la prueba **también** por
identidad de estímulo — (i) incluir la categoría de estímulo (codificada como variables
indicadoras) junto con $\hat z$ en la regresión que descuenta el confusor, y (ii)
restringir el pool de emparejamiento del paso 3 a ensayos de la **misma** categoría de
estímulo, no sólo del mismo estrato de arousal (más robusto que sólo la regresión si el
efecto del estímulo no es puramente aditivo). El precio: la varianza discriminante que
queda tras condicionar por estímulo es sólo la fluctuación idiosincrática ensayo-a-ensayo
—no la identidad del estímulo—, que tiene menor relación señal/ruido. El problema de
poder de la Sec. 8 de `DISENO_EXPERIMENTAL_prediccion6.md` no se evita condicionando por
estímulo; se hereda agravado.

**Validación numérica del problema y su corrección (`test_rastreo_de_fuente.py`, Bloque
5).** Con estímulo compartido entre $e$ y $l$ por vías separadas (5 categorías, $N=600$):
la prueba "ingenua" (sólo arousal) da exactitud top-1 significativamente por encima del
azar bajo **Tipo A puro** ($0.087$ vs. nulo $0.017$, $p<0.0001$) — el falso positivo
predicho arriba, confirmado. La prueba condicionada por estímulo lo corrige cuando ambos
confusores se conocen con exactitud ("oráculo": $0.010$ vs. nulo $0.017$, $p=0.98$,
limpio). Bajo Tipo B, la versión condicionada sigue detectando el efecto con claridad en
las cuatro variantes de la descomposición 2×2 siguiente ($p<0.0001$ en todas), con
exactitud top-1 menor que en el escenario sin estímulo compartido — el costo de poder
anticipado arriba.

**Un matiz que la primera versión de esta sección no cubría.** "Oráculo" no es una única
condición: hay que descontar el confusor verdadero de arousal *y* usar la identidad de
estímulo verdadera. La descomposición 2×2 (¿cuál de los dos confusores es ruidoso?) aísla
de dónde viene cualquier residuo bajo Tipo A puro, con $N=600$:

| Variante (Tipo A) | top-1 obs. | top-1 nulo | $p$ |
|---|---|---|---|
| $z$ ruidoso, estímulo verdadero | 0.030 | 0.017 | 0.037 (residuo marginal) |
| $z$ verdadero, estímulo ruidoso (8%) | 0.043 | 0.017 | <0.0001 |
| Ambos ruidosos (realista) | 0.043 | 0.016 | <0.0001 |
| Ambos verdaderos (oráculo) | 0.010 | 0.017 | 0.98 (limpio) |

El resultado no es simétrico: ruido de clasificación en la identidad de estímulo pesa
*más* que ruido continuo de medición en arousal, incluso a una tasa de error moderada
(8%). Tiene sentido estructuralmente — el ruido continuo se diluye parcialmente en una
regresión lineal; el error de clasificación categórico mueve un ensayo *fuera* del grupo
de comparación correcto por completo, rompiendo la premisa misma del emparejamiento.

**FPR-vs-N: la fragilidad es proporcional a la calidad del etiquetado, no binaria
(`test_rastreo_de_fuente.py`, Bloque 7).** Repitiendo la prueba condicionada 80 veces por
$N \in \{400, 800, 1600, 2400\}$, comparando dos regímenes de calidad del proxy de
estímulo:

| $N$ | FPR, error 1% | FPR, error 8% | FPR oráculo | Poder Tipo B |
|---|---|---|---|---|
| 400 | 0.037 | 0.200 | 0.037 | 1.000 |
| 800 | 0.037 | 0.400 | 0.025 | 1.000 |
| 1600 | 0.062 | 0.787 | 0.025 | 1.000 |
| 2400 | 0.062 | 0.887 | 0.062 | 1.000 |

Con un error de clasificación de estímulo del 1% (plausible si la identidad de estímulo
la registra directamente el software de la tarea, no algo inferido post-hoc), la FPR se
mantiene plana y cerca de $\alpha=0.05$ — el resultado que la versión anterior de esta
sección reportaba, pero que sólo había probado variando la calidad del proxy de *arousal*,
dejando la identidad de estímulo siempre exacta. Con un error del 8% (no exótico si la
identidad de estímulo se infiere post-hoc o hay codificación manual), la FPR **se dispara
con N** — de $0.20$ a $0.89$ entre $N=400$ y $N=2400$, peor que el problema original de la
prueba de magnitud en `simulacion_acoplamiento_fase_temprana_tardia.py`. El poder bajo
Tipo B se mantiene en $1.000$ en ambos regímenes — la corrección no pierde sensibilidad,
pero eso no compensa una FPR que se dispara si el etiquetado de estímulo es poco fiable.

**Consecuencia para la Predicción 6.** La fiabilidad del etiquetado de estímulo es una
condición de validez de primer orden para la versión condicionada del test — auditable y,
en principio, controlable (a diferencia del ruido de arousal, que es en parte irreducible
por ser una variable latente). Antes de confiar en un resultado del Test 2 condicionado
sobre datos reales, verificar la tasa de error de codificación de estímulo del propio
registro, no asumirla despreciable. Honestidad sobre el límite de esta validación: el
poder saturado en $1.0$ confirma que la prueba retiene sensibilidad, pero no traza una
curva de poder informativa (no hay pendiente que mostrar cuando ya se está en el techo) —
para caracterizar el punto donde el poder empieza a caer haría falta repetir esto con
$\kappa$ menor, no hecho aquí.

**La pregunta que decide el diseño, antes de tocar datos.** ¿El paradigma real repite un
estímulo único/idéntico, o lo varía entre ensayos?
- Si se **repite**: el procedimiento original de esta sección (sin condicionar por
  estímulo) es válido tal como está — no hay causa común vía estímulo que remover.
- Si **varía**: es obligatorio condicionar también por identidad de estímulo, como se
  describe arriba, y reportar el costo de poder que eso implica.

**El techo que ninguna de las dos versiones rompe.** Incluso la versión condicionada por
estímulo sigue siendo una inferencia causal observacional, apoyada en el supuesto de
*no-causa-común-no-medida*: que ningún otro estado específico del ensayo —adaptación,
atención encubierta, estado de red pre-estímulo— module en paralelo el contenido temprano
y el tardío. El arousal es una de esas causas posibles; la identidad del estímulo es otra,
ya corregida arriba; puede haber una tercera no identificada. Ninguna cantidad de
condicionamiento observacional puede cerrar esa posibilidad por completo — sólo la
intervención (perturbar el tálamo y observar si el contenido tardío específico cambia en
consecuencia) rompe ese techo, y eso excede lo que una cohorte puramente observacional
puede ofrecer. La prueba de identificabilidad, en cualquiera de sus dos versiones, es
**corroborativa bajo supuestos declarados**, no una prueba concluyente — y esto vale para
la Predicción 6 en general, no sólo para esta sección (ver §11).

**Predicción 6, versión fuerte.** Un análisis real debería reportar ambas pruebas
(magnitud e identificabilidad), la versión de identificabilidad correspondiente a si el
paradigma repite o varía el estímulo, y declarar explícitamente el supuesto de
no-causa-común-no-medida bajo el cual se interpreta el resultado. La posición
constitutiva se fortalece si la identificabilidad —en su versión correcta para el
paradigma usado— es significativa incluso cuando la magnitud no lo es (el patrón
encontrado en la validación numérica); se debilita si ninguna de las dos lo es.

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
- **Ninguna versión de la prueba de rastreo de fuente (§9.1) es una prueba causal
  concluyente**, ni siquiera en su versión corregida para estímulo compartido. Ambas
  descansan en un supuesto de no-causa-común-no-medida que ningún condicionamiento
  observacional puede cerrar por completo (§9.1) — son corroborativas bajo supuestos
  declarados. Esto no es específico del arousal ni del estímulo: es el techo general de
  cualquier diseño no-interventivo para la Predicción 6, y conviene declararlo por
  adelantado en vez de tratarlo como un problema a resolver agregando más condicionantes
  cuando el resultado sea ambiguo.

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
| Realización N-nodos, $\beta_c$ estimado directamente | `AFH_modelo_N_nodos.py`: bisección sobre tasa de crecimiento (§5/8), confirma $\beta_c\approx1.0$ | — |
| $\widehat\nabla$ ensayo-a-ensayo (magnitud) | `simulacion_acoplamiento_fase_temprana_tardia.py`: correlación parcial controlando proxy de arousal | — |
| Identificabilidad de contenido (§9.1) | `test_rastreo_de_fuente.py`: exactitud top-1 estratificada, nulo por permutación intra-estrato | — |

Esta tabla es la ruta concreta para extender el código existente: `0.4.py` ya calcula
$\widehat\nabla$ vía Granger específico y compara Tipo B/A; le falta el retardo aferente
$\tau_f$ explícito (§2, §8) y una estimación directa de $\beta_c$ vía ajuste del punto de
bifurcación en lugar de sólo inspección visual del barrido. `NIVEL 4/` opera sobre datos
reales (Sleep-EDF) pero sin el contrafactual Tipo A/B que sólo la simulación puede ofrecer
limpio — de ahí que la Predicción 6 siga, correctamente, marcada como pendiente en el
manuscrito (§6.3): requiere registro intracraneal con cobertura ILN adecuada, no disponible
en las cohortes usadas hasta ahora en este repositorio.
