# Sistema Predictivo de Intención de Compra para E-commerce

Este proyecto implementa una solución de inteligencia de negocios para detectar de forma temprana si un usuario realizará una compra en un sitio de comercio electrónico.

## 1. Contexto del Negocio
El objetivo es permitir que la tienda muestre dinámicamente productos de mayor valor a usuarios con alta intención de compra, aumentando así los ingresos generales (revenue).

## 2. Objetivo del Proyecto
Desarrollar un pipeline de ciencia de datos completo, entrenar un modelo con una precisión (accuracy) cercana al 90% y desplegarlo mediante una API REST.

## 3. Descripción del Dataset
El dataset contiene 12,330 sesiones de usuarios con:
- **Atributos de navegación**: Administrative, Informational, Product Related (y sus duraciones).
- **Métricas de Google Analytics**: Bounce Rate, Exit Rate, Page Value.
- **Atributo temporal**: Special Day (proximidad a fechas festivas).
- **Datos técnicos y demográficos**: Operating Systems, Browser, Region, Traffic Type, Visitor Type, Weekend, Month.
- **Variable Objetivo**: `Revenue` (booleano).

## 4. Pipeline de Machine Learning

### A. Preprocesamiento y Limpieza
1.  **Mapeo de Variables**: Se transformó el mes de texto a numérico.
2.  **Codificación**: Se aplicó `One-Hot Encoding` para la variable `VisitorType`.
3.  **Conversión de Tipos**: Las variables booleanas (`Weekend`, `Revenue`) se convirtieron a enteros.
4.  **Eliminación de Duplicados**: Se eliminaron registros duplicados para evitar sesgos.
5.  **Tratamiento de Leakage**: Se eliminó la columna `ExitRates` ya que, según el análisis, introducía información del futuro de la sesión que no estaría disponible al momento de la predicción temprana. También se eliminó `VisitorType_New_Visitor` por redundancia.

### B. Ingeniería de Características y Escalado
Se utilizó un `ColumnTransformer` para aplicar:
- **RobustScaler**: Para variables con valores atípicos significativos (`PageValues`, duraciones).
- **StandardScaler**: Para variables con distribución más uniforme.

### C. Entrenamiento del Modelo
Se optó por utilizar Deep Learning por la naturaleza compleja (no lineal) del problema. 
- **Optimización**: Se realizó una búsqueda aleatoria de hiperparámetros optimizando el `AUC`.
- **Mejores Parámetros**: `units_1: 320, dropout_1: 0.2, num_layers: 2, units_2: 96, 
learning_rate: 0.01, units_3: 64, units_4: 64`.

### D. Evaluación
El modelo alcanzó los siguientes resultados en el conjunto de prueba:
- **Accuracy**: 89.39%
- **F1-Score**: 0.6174
- **ROC AUC**: 0.9200

## 5. API REST
Desarrollada con Flask, permite predicciones en tiempo real a través del endpoint `/predict`.

### Ejemplo de "data" como JSON en el Body de la petición HTTP:
```json
{
    "Administrative": 3,
    "Administrative_Duration": 61.0000,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 504.506000,
    "BounceRates": 0.00000,
    "ExitRates": 0.026984,
    "PageValues": 11.535959,
    "SpecialDay": 0.0,
    "Month": "Apr",
    "OperatingSystems": 1,
    "Browser": 2,
    "Region": 4,
    "TrafficType": 4,
    "VisitorType": "Returning_Visitor",
    "Weekend": True
}
```

### Ejemplo de Respuesta JSON:
```json
{
  "clasificacion": "Comprar",
  "mensaje": "El usuario va a: Comprar. Con un 52.43% de probabilidades, lo que lo hace bastante probable.",
  "probabilidad": 0.5243004560470581
}
```

## 6. Ejecución del Proyecto
1. Tener Python 3.10 - 3.13 para la compatibilidad con `tensorflow`
2. Instalar dependencias: `pip install pandas scikit-learn flask joblib tensorflow keras-tuner matplotlib.pyplot seaborn`
3. Iniciar la API: `python api.py`
4. Entrenar el modelo: `python entrenamiento.py`
5. Probar: `python test_api.py`

## 7. Justificación de Decisiones
- **Manejo de Probabilidades**: La API devuelve la probabilidad específica de la clase predicha para dar mayor claridad al departamento de IT, tal como se solicitó.

## 8. Cuaderno Jupyter
El análisis de datos y el entrenamiento del modelo se realizaron formalmente en el siguiente cuaderno Jupyter: 
https://colab.research.google.com/drive/1wuk98k_Eer7rKLQJAm_fhCaYbCbxzvUb?usp=sharing

Se descargó el modelo y se dejó en el repositorio. "entrenamiento.py" e "investigacion.py" tienen el mismo contenido que el
cuaderno pero estructurado de otra forma para que sea utilizable como proyecto python independiente