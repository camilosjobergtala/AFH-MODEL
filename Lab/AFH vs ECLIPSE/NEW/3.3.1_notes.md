\begin{table}[ht]
\centering
\caption{State-dependent reconfiguration of multivariate predictors}
\label{tab:multivariate_final}
\begin{tabular}{lccccc}
...
\end{tabular}
\vspace{0.3cm}
{\small \textit{Note:} ...}
\end{table}
```

**Verificado:** Sin dependencias de paquetes problemáticos.

---

## 🔬 VERIFICACIÓN DE JSON

**Multicolinealidad confirmada como EXCELENTE:**

**Wake:**
- Max VIF: 1.64 (muy por debajo de 5)
- Max |correlation|: 0.498 (moderada, acceptable)

**N3:**
- Max VIF: 1.12 (prácticamente ideal)
- Max |correlation|: 0.278 (baja)

**Conclusión:** Coeficientes son **ESTABLES** y **CONFIABLES**.

---

# 🎯 INTERPRETACIÓN CIENTÍFICA FINAL

## **HALLAZGO CENTRAL (submission-proof):**

### 1. **Phase Stability (B2) - Sign Reversal**
```
Wake: β = -0.018 (null)
N3:   β = +0.224 (moderate positive)
Change: CI(Δβ) = [+0.016, +0.485] excludes 0 ✓
```

**Significado biológico:**
- **Wake:** Phase stability NO predice complexity
- **N3:** Phase stability SÍ predice complexity (positivamente)
- **Implicación:** En N3, sistemas con mayor estabilidad de fase muestran mayor complejidad residual → contrarresta el efecto de slow-wave synchrony

### 2. **Spectral Variability (B3) - Reweighting**
```
Wake: β = -0.644 (dominant, strong negative)
N3:   β = -0.441 (moderate negative)
Change: CI(Δβ) = [+0.002, +0.379] excludes 0 ✓
```

**Significado biológico:**
- **Ambos estados:** Mayor variabilidad espectral → menor complexity (como esperado)
- **Cambio:** Pierde fuerza predictiva en N3 (de dominante a moderada)
- **Implicación:** En N3, otros factores (B1, B2) cobran relevancia relativa

### 3. **Temporal Variability (B1) - Stable Positive**
```
Wake: β = +0.122 (weak)
N3:   β = +0.295 (moderate)
Change: CI(Δβ) = [-0.099, +0.427] includes 0 ✗