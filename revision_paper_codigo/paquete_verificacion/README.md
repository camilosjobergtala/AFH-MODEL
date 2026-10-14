# Paquete de verificación: ECLIPSE v2.0 (Scientific Reports, DOI 10.1038/s41598-026-74760-5)

Scripts que comprueban, contra el código depositado (commit `93339b4`), cada punto del correo de
corrección enviado a la revista, junto con la salida que deben producir.

## Requisitos

- Un clon del repositorio en el commit citado:
  ```
  git clone https://github.com/camilosjobergtala/AFH-MODEL
  cd AFH-MODEL
  git checkout 93339b4
  ```
- Las dependencias de `ScientificReports_revision/requirements_frozen.txt`: numpy 2.4.4,
  pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8.
- Los scripts se guardan **fuera** del clon y se ejecutan **desde la raíz del clon**:
  ```
  python RUTA/scripts/p3_etapa5_auditor.py
  ```

## Cómo funcionan

- **Hash:** cada script empieza imprimiendo el SHA-256 de `eclipse_core.py`. Si la versión
  normalizada a LF no coincide con `9d4320d225375fb5ed326b1cc282779125595ef500ff81bda81d7dd528b059b3`,
  el script se detiene.
- **Controles:** toda afirmación negativa va acompañada de un control positivo, es decir, un caso
  que sí debe detectarse. Si un control falla, el script lo dice y el hallazgo no vale.
- **Sin efectos en el clon:** los scripts trabajan en carpetas temporales y no generan
  `__pycache__`. Después de ejecutarlos, `git status` no muestra cambios.

## Correspondencia con el correo

| Punto del correo | Script | Resultado esperado (resumen) | Ejecutado en |
|---|---|---|---|
| 1. Alcance del hash del split | `p1_p4_hash_y_bloqueo.py` (bloque P1) | Intercambiar IDs, reemplazar el holdout, cambiar el ratio o la fecha: VÁLIDO. Controles (semilla, eliminar un ID): DETECTADO | Linux; Windows (versión anterior del script) |
| 1. Orden de entrada | `p12_orden_y_avisos.py` (O1) | Orden inverso: 18 de 30 IDs del holdout distintos, mismo `split_hash`. El archivo no guarda el orden | Linux |
| 2. Separación no impuesta | `p11_afirmaciones_integridad.py` (T5, T6), `p12_orden_y_avisos.py` (O2) | La Etapa 1 devuelve los 30 IDs del holdout; la Etapa 3 acepta los 100 IDs sin warnings, logs ni mensajes | Linux; Windows (p11) |
| 2. Modelo final (si se incluye) | `p6_etapa3_modelo_final.py` | Tamaños de entrenamiento [56, 56, 56, 56, 56], nunca 70; sin modelo en la salida | Linux; Windows |
| 3. Disparo único | `p1_p4_hash_y_bloqueo.py` (bloque P4) | La Etapa 4 se ejecuta 3 veces antes de la 5; no hay archivo de holdout; archivos escribibles; rechazada tras la Etapa 5 salvo `force=True` | Linux; Windows (versión anterior del script) |
| 4. Re-ejecutar no deja rastro | `p11_afirmaciones_integridad.py` (T1, T3, T4) | Re-registro: umbral 0.7 → 0.5, válido, sin copia. Re-split: 25 de 30 IDs cambian, válido, sin copia | Linux; Windows |
| 5. Precedencia temporal | `p11_afirmaciones_integridad.py` (T2, T8) | Cambiar la fecha a 2020-01-01: no detectado. `registration_date` solo aparece en las líneas 359, 592 y 2545 | Linux; Windows |
| 6. Alcance de `verify_integrity` | `p11_afirmaciones_integridad.py` (T7) | Solo comprueba `split` y `criteria`; los resultados editados dan `all_valid: True` | Linux; Windows |
| 7. Salidas de la Etapa 5 | `p3_etapa5_auditor.py` | 0 llamadas al auditor; control: 1 llamada manual | Linux; Windows |
| 7. Auditor, nivel 2 | `p4_auditor_nivel2.py` | Ningún bucle de umbral da `controlflow`; con `X_test` solo `holdout_access` y `model_selection_bias`; no hay método de cobertura | Linux; Windows |
| 7. Auditor, Tabla 1 | `p5_auditor_nivel3.py` | `y = test_data.mean()` solo da `('holdout_access', 'ast')`; el control de nivel 3 sí da `semantic` | Linux; Windows |
| 8. "4.4–5.0 %" | `p7_reproducir_rango.py` o `p7b_reproducir_rango_paralelo.py` | (a) 4.73–5.21 %, (b) 4.84–5.21 %; conteos exactos abajo | Linux; Windows (p7b) |
| 9. "5.1–54.4 %" | `p8_rango_corte_fijo.py` | Study 2b: 3.53–54.45 %; Study 2: 5.15–55.25 % | Linux; Windows |
| 10. Techo del EIS | `p9_techo_eis.py` | 0.9571 con el holdout en la media; 0.9750 muy por debajo; 0.9548 con 70/30 | Linux; Windows |
| 11. Correlación de la fuga | `p10_correlacion_fuga.py` | Máximo \|r\| = 0.0858 (−0.09 con 2 decimales) | Linux; Windows |

**Conteos exactos del punto 8** (casos marcados de 60.000, semilla 20250805):
- **(a) Tabla 5:** K=3: 3123, 3058, 2941 · K=5: 2935, 2836, 2991 · K=10: 2909, 2979, 2874 ·
  K=20: 3019, 3081, 3001 (|M| = 1, 3, 5).
- **(b) Paso [2] de `04`:** (3,1) 3123 · (5,3) 2966 · (10,1) 2906 · (20,5) 2926.

`p7` y `p7b` dan exactamente los mismos conteos. `p7b` reparte las celdas entre procesos y añade
dos controles propios: C4, que todos los casos sean válidos, y C5, que cada proceso use su propio
archivo temporal. `p7` tardó unos 11 minutos en Linux; `p7b` tardó unos 36 minutos en el Windows
del autor, con 15 procesos.

## Salidas esperadas

En `salidas_esperadas/` está la salida completa de cada script, capturada al ejecutarlo sobre un
clon limpio en `93339b4` en Linux. En otra máquina pueden diferir, sin que eso importe:

- **El hash "tal cual":** en Windows, Git puede convertir los saltos de línea a CRLF. Lo que debe
  coincidir es el hash normalizado a LF.
- **Fechas, tiempos y rutas temporales:** cambian en cada ejecución.
- **Formato de permisos (`p1_p4`):** Linux muestra `-rw-r--r--`; Windows suele mostrar `-rw-rw-rw-`.
  En ambos casos el archivo es escribible, que es lo que importa.
- **Orden de las líneas "terminada" en `p7b`:** depende de qué proceso acabe antes. El resumen final
  aparece ordenado.
- **Redondeo en `p7b`:** la línea de avance de K=5, |M|=5 puede mostrar 4.98 % o 4.99 %. El conteo,
  2991, es el mismo.

## Archivos

```
scripts/
  p1_p4_hash_y_bloqueo.py         puntos 1 y 3
  p3_etapa5_auditor.py            punto 7 (Etapa 5)
  p4_auditor_nivel2.py            punto 7 (nivel 2)
  p5_auditor_nivel3.py            punto 7 (Tabla 1)
  p6_etapa3_modelo_final.py       modelo final sobre D_dev
  p7_reproducir_rango.py          punto 8 (secuencial)
  p7b_reproducir_rango_paralelo.py punto 8 (paralelo, mismos resultados)
  p8_rango_corte_fijo.py          punto 9
  p9_techo_eis.py                 punto 10
  p10_correlacion_fuga.py         punto 11
  p11_afirmaciones_integridad.py  puntos 2, 4, 5, 6 (T1–T8)
  p12_orden_y_avisos.py           puntos 1 y 2 (orden de entrada; avisos de la Etapa 3)
salidas_esperadas/
  <nombre_del_script>.txt         salida completa de cada script
```

## Afirmaciones del correo que no salen de un script

Estas se comprobaron leyendo el código o el historial del repositorio:

- **Nivel 2 del auditor:** solo revisa sentencias `if`. Ver `_control_flow_analysis`,
  `eclipse_core.py:1436`. `p4` lo respalda listando los métodos del analizador.
- **Control de versiones:** el framework no lo usa. Ninguna parte del core llama a git.
- **Historial del "4.4–5.0 %":** la frase entró en el commit `91db9d7` (14-09-2026), el mismo que
  creó `04_stds_screening_cutoffs.py`, que tiene una sola versión en todo el historial. En ningún
  commit hay un script de `code/` que mida tasas con los cortes empíricos por el motor. Se comprueba
  con:
  ```
  git log --all --format="%h %cs %s" -G "4\.4%|4\.4.?[–-].?5\.0"
  git log --all --format="%h %cs %s" -- ScientificReports_revision/code/04_stds_screening_cutoffs.py
  ```
- **El 4.77–5.15 % de `04`:** se obtiene ejecutando `code/04_stds_screening_cutoffs.py`. Cuidado:
  ese script **reescribe** `outputs/study2b/`, así que conviene correrlo en una copia del clon o
  restaurar después con `git checkout -- ScientificReports_revision/outputs`.
