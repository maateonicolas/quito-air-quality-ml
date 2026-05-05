## Resultados preliminares con datos sintéticos

Se entrenaron cinco modelos de clasificación para predecir la categoría de calidad del aire: regresión logística, árbol de decisión, random forest, red neuronal y red neuronal con pesos de clase.

Los resultados muestran que la regresión logística obtuvo el mayor accuracy general, mientras que Random Forest logró el mejor equilibrio entre clases según Macro F1-score y recall de la clase minoritaria `buena`.

Estos resultados son preliminares porque se obtuvieron sobre un dataset sintético usado para validar el pipeline. La siguiente fase consiste en reemplazar estos datos por registros reales de calidad del aire de Quito.
![accuracy_comparison.png](reports/figures/accuracy_comparison.png)
![macro_f1_comparison.png](reports/figures/macro_f1_comparison.png)
![recall_buena_comparison.png](reports/figures/recall_buena_comparison.png)

## Estado actual del proyecto

El proyecto ya cuenta con dos fases implementadas:

1. Validación del pipeline con datos sintéticos.
2. Construcción de un dataset real a partir de reportes mensuales de calidad del aire de Quito.

### Datos sintéticos

Primero se generó un dataset sintético de 1000 registros horarios para validar el flujo completo de Machine Learning. Este dataset permitió probar la arquitectura del proyecto, el preprocesamiento, la creación de variables, el entrenamiento de modelos baseline y la implementación de una red neuronal.

Se entrenaron los siguientes modelos:

- Logistic Regression
- Decision Tree
- Random Forest
- Neural Network
- Neural Network + Class Weights

Los resultados preliminares mostraron que Logistic Regression obtuvo el mayor accuracy general, mientras que Random Forest logró el mejor equilibrio entre clases según Macro F1-score y recall de la clase minoritaria.

### Datos reales de Quito

Posteriormente se descargaron reportes mensuales reales de la red de monitoreo de calidad del aire de Quito. Los reportes originales venían separados por variable y en formato mensual ancho. Se implementó un script para transformar estos archivos a formato largo horario y unirlos por fecha y hora.

El dataset real generado contiene:

- 816 registros horarios
- 12 columnas
- variables de contaminantes atmosféricos y meteorológicas

Variables disponibles:

- PM2.5
- PM10
- NO2
- O3
- SO2
- CO
- temperatura
- humedad relativa
- precipitación
- velocidad del viento

El archivo final generado es:

`data/raw/quito_air_quality_real.csv`

### Próximo paso

Como el dataset real no incluye directamente una columna AQI, se definirá una variable objetivo inicial usando PM2.5 como referencia:

- `buena`: PM2.5 <= 15
- `moderada`: 15 < PM2.5 <= 35
- `mala`: PM2.5 > 35

Con esta variable objetivo se entrenarán nuevamente los modelos baseline y la red neuronal sobre datos reales.
## Dataset real de Quito

Se construyó un dataset real usando reportes mensuales horarios de calidad del aire de Quito. Los archivos originales estaban separados por variable ambiental y en formato ancho mensual. Para convertirlos en un formato útil para Machine Learning se implementó un proceso ETL que:

1. Lee cada reporte mensual CSV.
2. Extrae las mediciones horarias.
3. Convierte el formato ancho a formato largo.
4. Agrupa registros duplicados por fecha y hora.
5. Une todas las variables usando `datetime`.
6. Genera un dataset integrado.

El dataset real procesado contiene 805 registros horarios y 17 columnas.

Variables disponibles:

- PM2.5
- PM10
- NO2
- O3
- SO2
- CO
- temperatura
- humedad relativa
- precipitación
- velocidad del viento
- variables temporales: hora, día de la semana, mes y fin de semana

Como el dataset no incluye directamente una columna AQI, se definió una variable objetivo inicial usando PM2.5 como referencia:

- `buena`: PM2.5 <= 15
- `moderada`: 15 < PM2.5 <= 35
- `mala`: PM2.5 > 35

La distribución de clases fue:

| Clase | Registros |
|---|---:|
| buena | 490 |
| moderada | 299 |
| mala | 16 |

Esto muestra un desbalance fuerte, especialmente en la clase `mala`.

Para evitar fuga de información, PM2.5 no se usa como predictor en el entrenamiento de modelos reales, ya que la variable objetivo fue construida a partir de PM2.5. En su lugar, los modelos usan otros contaminantes, variables meteorológicas y variables temporales para predecir la categoría de calidad del aire.
## Resultados con datos reales

Se entrenaron tres modelos baseline sobre el dataset real procesado de Quito:

- Logistic Regression
- Decision Tree
- Random Forest

Para evitar fuga de información, la variable `pm25` no fue usada como predictor, ya que el target fue construido a partir de PM2.5. Los modelos usaron como entrada PM10, NO2, O3, SO2, CO, temperatura, humedad, precipitación, velocidad de viento y variables temporales.

La distribución de clases fue:

| Clase | Registros |
|---|---:|
| buena | 490 |
| moderada | 299 |
| mala | 16 |

Esto muestra un fuerte desbalance, especialmente para la clase `mala`.

### Resultados multicategoría

| Modelo | Accuracy | Observación |
|---|---:|---|
| Logistic Regression | 0.5537 | Detectó algunos casos de clase `mala`, pero con muchos falsos positivos |
| Decision Tree | 0.5992 | No logró detectar casos de clase `mala` |
| Random Forest | 0.7438 | Obtuvo el mejor accuracy general, pero no detectó la clase `mala` |

Random Forest obtuvo el mejor desempeño general, con accuracy de 0.7438. Sin embargo, el modelo no logró detectar la clase `mala`, lo cual se explica por la baja cantidad de registros disponibles para esa categoría. En el conjunto de prueba solo existían 5 registros de clase `mala`.

### Conclusión preliminar

Los resultados muestran que el problema multiclase está limitado por el desbalance de datos. Aunque Random Forest clasifica razonablemente bien las clases `buena` y `moderada`, la clase `mala` requiere más datos históricos para ser aprendida correctamente.

Como siguiente mejora metodológica, se propone reformular el problema como clasificación binaria:

- `buena`
- `no_buena` = moderada o mala

Esta formulación puede ser más estable para un dataset pequeño y desbalanceado.

## Resultados finales con datos reales

Después de validar el pipeline con datos sintéticos, se entrenaron modelos usando datos reales horarios de calidad del aire de Quito.

El problema multiclase original (`buena`, `moderada`, `mala`) presentó un fuerte desbalance, especialmente en la clase `mala`, que tenía únicamente 16 registros. Por esta razón, se reformuló el problema como clasificación binaria:

- `buena`
- `no_buena`: combinación de las clases `moderada` y `mala`

Esta reformulación permite construir un modelo más estable con la cantidad actual de datos disponibles.

### Distribución de clases

| Clase | Registros |
|---|---:|
| buena | 490 |
| no_buena | 315 |

### Modelos evaluados

Se entrenaron tres modelos baseline:

- Logistic Regression
- Decision Tree
- Random Forest

Para evitar fuga de información, `pm25` no fue usado como predictor, ya que la variable objetivo fue construida a partir de PM2.5. Los modelos usaron como entrada:

- PM10
- NO2
- O3
- SO2
- CO
- temperatura
- humedad relativa
- precipitación
- velocidad de viento
- hora
- día de la semana
- mes
- indicador de fin de semana
- estación

### Resultados

| Modelo | Accuracy | F1 buena | F1 no_buena | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7066 | 0.75 | 0.65 | 0.70 |
| Decision Tree | 0.6653 | 0.67 | 0.66 | 0.67 |
| Random Forest | 0.7603 | 0.80 | 0.70 | 0.75 |

El mejor modelo fue **Random Forest**, con accuracy de 0.7603 y Macro F1-score de 0.75.

### Matriz de confusión del mejor modelo

| | Predicho buena | Predicho no_buena |
|---|---:|---:|
| Real buena | 116 | 31 |
| Real no_buena | 27 | 68 |

### Conclusión

Random Forest fue el modelo más robusto para el dataset real, ya que obtuvo el mejor desempeño general y el mejor equilibrio entre las clases. La clasificación binaria resultó más adecuada que la clasificación multiclase debido al bajo número de registros de la clase `mala`.

Los resultados sugieren que es posible estimar si la calidad del aire será `buena` o `no_buena` usando contaminantes secundarios, variables meteorológicas y variables temporales, sin utilizar PM2.5 directamente como predictor.

### Limitaciones

- El dataset real corresponde a un período limitado.
- La clase `mala` tiene muy pocos registros, lo que dificulta el aprendizaje multiclase.
- La variable objetivo fue construida usando umbrales simplificados de PM2.5.
- Para mejorar la generalización, se recomienda descargar más meses de datos históricos.

### Próximas mejoras

- Incorporar más meses de datos reales.
- Evaluar validación temporal en lugar de separación aleatoria train/test.
- Ajustar hiperparámetros de Random Forest.
- Probar técnicas de balanceo como oversampling o SMOTE.
- Comparar contra una red neuronal entrenada con datos reales.

![real_binary_accuracy.png](reports/figures/real_binary_accuracy.png)
![real_binary_f1_by_class.png](reports/figures/real_binary_f1_by_class.png)
![real_binary_macro_f1.png](reports/figures/real_binary_macro_f1.png)

### Red neuronal binaria

Se entrenó una red neuronal densa para la clasificación binaria `buena` vs `no_buena`. La arquitectura utilizada fue:

- Capa densa de 64 neuronas con activación ReLU
- Dropout de 0.2
- Capa densa de 32 neuronas con activación ReLU
- Dropout de 0.2
- Capa de salida con una neurona y activación sigmoid

La red fue entrenada con `binary_crossentropy`, optimizador Adam, early stopping y pesos de clase para compensar el desbalance.

El modelo obtuvo:

| Modelo | Accuracy | F1 buena | F1 no_buena | Macro F1 |
|---|---:|---:|---:|---:|
| Neural Network | 0.7355 | 0.78 | 0.67 | 0.72 |

La matriz de confusión fue:

| | Predicho buena | Predicho no_buena |
|---|---:|---:|
| Real buena | 114 | 33 |
| Real no_buena | 31 | 64 |

La red neuronal logró un desempeño competitivo, pero no superó a Random Forest. Esto sugiere que, para este dataset tabular y relativamente pequeño, Random Forest es más adecuado que una red neuronal densa.

### Modelo final seleccionado

El modelo final recomendado es **Random Forest**, ya que obtuvo el mejor desempeño en datos reales binarios:

- Accuracy: 0.7603
- F1 buena: 0.80
- F1 no_buena: 0.70
- Macro F1: 0.75

Aunque la red neuronal logró resultados competitivos, Random Forest presentó mejor balance entre clases y mayor capacidad predictiva con el tamaño actual del dataset.