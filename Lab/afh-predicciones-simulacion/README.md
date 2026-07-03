# Simulacion de las predicciones AFH (Seccion 6.1 del paper)

`simulate_afh_predictions.py` genera datos sinteticos que instancian el mecanismo
propuesto por AFH (via rapida 8-30 ms + retorno lento tipo copia eferente
200-600 ms, convergencia refleja en los ILN) y aplica sobre ellos las 7 pruebas
estadisticas que el texto especifica en la Seccion 6.1 ("Presencia fenomenologica
y el talamo intralaminar").

Esto **no** es un analisis sobre registros ILN reales -- el propio paper marca las
Predicciones 6 y 7 como pendientes de registro intracraneal (Seccion 6.3). Sirve
para verificar que las 7 predicciones son computables y genuinamente falsables:
el script puede generar tanto el caso en que AFH es correcto como un caso nulo en
el que la convergencia es meramente heterogenea (Tipo A), y las pruebas deben
distinguir ambos casos.

## Uso

```bash
pip install -r ../../requirements.txt
python "simulate_afh_predictions.py" --regime afh_true --plots
python "simulate_afh_predictions.py" --regime null_heterogeneous --plots
```

`--regime afh_true` genera la senal lenta como una transformacion causal
retardada de la envolvente de la senal rapida (convergencia refleja, Tipo B).
`--regime null_heterogeneous` genera la senal lenta como una fuente
independiente (convergencia heterogenea, Tipo A) -- el escenario que, segun el
propio texto, reclasificaria Nabla y devolveria el argumento del zombi a su
forma estandar.

Con la semilla por defecto (`--seed 42`): `afh_true` confirma las 7/7
predicciones (incluida la P6 decisiva, con el lag exacto de 300 ms recuperado
en la P7); `null_heterogeneous` falla las 7/7. Los reportes se guardan en
`outputs/` como JSON (y PNG si se pasa `--plots`).

## Correspondencia con el texto

| Script | Prediccion (Seccion 6.1) |
|---|---|
| `predict_1` | Correlacion fenomenologica gradada |
| `predict_2` | Disociacion cualitativa N3/anestesia |
| `predict_3` | Ordenamiento temporal durante la emergencia |
| `predict_4` | Covarianza H*/Nabla dentro de estados con horizonte presente (r > 0.50, umbral explicito del texto) |
| `predict_5` | Covarianza bajo manipulaciones de la retencion |
| `predict_6` | Origen reflexivo de la senal lenta (la prueba decisiva) |
| `predict_7` | Diferencial de latencia rapido-lento en los ILN (150-550 ms) |

Los umbrales de paso/fallo de P4 y P7 estan tomados literalmente del texto; los
de P1, P2, P3, P5 y P6 son umbrales ilustrativos razonables (el texto especifica
la direccion y forma de la prediccion pero no siempre un numero de corte fijo).

## Version con datos reales de Sleep-EDF

`evaluate_real_sleepedf_predictions.py` re-analiza (no re-descarga) los
resultados que el autor ya calculo sobre Sleep-EDF Cassette real (153 sujetos)
en `Lab/AFH vs ECLIPSE/FOLD/NIVEL 3 PAC/nivel3b_v3_optimization/` y
`NIVEL 0 FOLD/eclipse_v2/`. Este entorno de ejecucion no tiene ruta de red
hacia physionet.org (bloqueada por el proxy) ni PSG crudo cacheado, asi que
no puede descargar ni reprocesar EEG desde cero -- pero puede reportar
honestamente lo que los datos reales ya computados dicen:

```bash
python "evaluate_real_sleepedf_predictions.py"
```

**Importante -- ninguno de los dos analisis de abajo es un test oficial de las
Predicciones 3 o 4 tal como estan escritas en el paper.** Son exploraciones
previas del propio autor en `Lab/` (separadas del manuscrito, segun el README
raiz del repo), tematicamente relacionadas pero con hipotesis y metricas
propias, no la operacionalizacion que el paper mismo llama "pendiente" en su
Seccion 6.3:

- **Analogo de P3** (`raw_results_full.csv`, NIVEL 3 PAC): testea si un proxy
  H* (hjorth_complexity+dfa+petrosian_fd, umbralizado) cambia antes que la
  transicion de etapa anotada en el hipnograma -- misma logica que Prediccion 3,
  pero no una instancia oficial de ella. Resultado real: precede en 55.6% de
  1542 transiciones validas (153 sujetos) vs 50% azar, p = 6.5e-6 --
  significativo pero muy por debajo del 70% que el propio autor fijo como
  umbral practico en `OPTIMIZATION_SUMMARY.json`.
- **Contexto Delta-t-Fold** (`DeltaTFold_SleepEDF_RealData_*.json`, NIVEL 0
  FOLD): **no mide covarianza H*/Nabla** (eso seria Prediccion 4). Mide el
  group delay (retardo de fase, banda 2-9Hz) entre los canales EEG Fpz-Cz y
  Pz-Oz bajo la hipotesis Delta-t-Fold_vigilia > Delta-t-Fold_sueno. Bajo sus
  propios criterios pre-registrados (d>=0.30 y p<=0.05, ambos requeridos), el
  holdout dio d=0.22 (falla) y p=0.021 (pasa) -> FALSIFICADA por su propia
  regla de pre-registro. Se reporta como contexto relacionado, no como test
  de P4.
- **P1, P2, P5, P6, P7 -- y, en su forma exacta, tambien P3 y P4 -- no son
  testables con Sleep-EDF**, sin importar cuantos sujetos se agreguen: faltan
  estructuralmente manipulacion experimental graduada, reporte en primera
  persona, un brazo de anestesia, y sobre todo cualquier canal que resuelva
  actividad de los ILN (nucleo subcortical profundo, inaccesible a EEG de
  cuero cabelludo).
- El numero exacto que el paper cita en su Seccion 6.3 (F1=0,031 vs criterio
  >=0,60; d de Cohen=2,44-2,79 para PAC delta-gamma) **no se encontro en este
  repositorio** -- probablemente proviene de un analisis no archivado aqui.
