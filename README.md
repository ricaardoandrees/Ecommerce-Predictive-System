# Sistema Predictivo de Intención de Compra para E-commerce

**Integrantes**

- Daniella Cruz (C.I 28404632)
- Mariadelia Finizola (C.I. 30693341)
- Claudia López (C.I. 30715112)
- Giovanni Libertella (C.I. 30078875)
- Ricardo Mejias (C.I. 30886262)
- Alessandro Ventresca (C.I: 30282309)

## 1. Contexto y Objetivo del Negocio

En el entorno actual del comercio electrónico, identificar a los usuarios con alta probabilidad de realizar una compra es fundamental para optimizar estrategias de ventas. Por lo cual, este proyecto implementa una solución de Inteligencia de Negocios que evalúa el comportamiento de navegación de un usuario en tiempo real. Al detectar de forma temprana una alta intención de compra, el sistema permite a la tienda web mostrar dinámicamente productos de mayor valor, maximizando así los ingresos generales (revenue). El objetivo principal fue desarrollar un pipeline de ciencia de datos, entrenar un modelo con una precisión cercana al 90% y desplegarlo mediante una API REST para consumo del departamento de IT

## 2. Descripción del Dataset

El conjunto de datos base consta de 12,330 sesiones de usuarios, compuestas por atributos que describen la interacción del cliente con la plataforma:

- **Navegación:** Cantidad y duración de visitas a páginas administrativas, informativas y relacionadas con productos
- **Métricas de Google Analytics:** Tasa de rebote (Bounce Rate), tasa de salida (Exit Rate) y valor de la página (Page Value)
- **Contexto Temporal:** Cercanía a días festivos (Special Day) y mes del año
- **Demografía y Tecnología:** Sistema operativo, navegador, región, tipo de tráfico y tipo de visitante (nuevo o recurrente)
- **Variable Objetivo (Etiqueta):** Revenue (Booleano que indica si la sesión terminó en compra)

## 3. Pipeline de Machine Learning y Decisiones de Diseño

Para garantizar que el modelo sea robusto, matemáticamente estable y verdaderamente aplicable al mundo real del comercio electrónico, se construyó un pipeline de procesamiento dividido en las siguientes etapas clave:

### A. Preprocesamiento, Limpieza y Análisis Exploratorio (EDA)

Antes de entrenar cualquier algoritmo, fue indispensable analizar la distribución de los datos y adaptarlos, ya que las redes neuronales solo comprenden relaciones numéricas. Las decisiones críticas en esta etapa fueron:

- **Mapeo y Codificación (One-Hot Encoding):** Los algoritmos no pueden procesar texto. Por ello, los meses del año se estandarizaron a un formato numérico. Para variables categóricas sin un orden jerárquico, como VisitorType, se aplicó *One-Hot Encoding*. Esta técnica transforma las categorías en columnas binarias (1 o 0) independientes. Esto es vital para evitar que el modelo asuma falsamente que una categoría es "mayor" o "mejor" que otra simplemente por tener un número más alto asignado

- **Eliminación de Duplicados:** Al existir registros idénticos, la red neuronal podría memorizar patrones repetidos en lugar de aprender a generalizar, lo que generaría un sesgo en sus predicciones. Por ello, se eliminaron del dataset

- **Prevención de Data Leakage (Fuga de Información):** Este es uno de los pasos más importantes del diseño. Se eliminó la variable ExitRates porque representa el porcentaje de veces que una página fue la última en la sesión. En un escenario real, esta métrica solo se conoce cuando el usuario ya ha abandonado el sitio web. Entrenar al modelo con este dato le daría "información del futuro", invalidando por completo su capacidad para hacer predicciones de compra tempranas mientras el usuario aún navega

- **Reducción de Multicolinealidad:** Se eliminó la columna VisitorType_New_Visitor. Al tener columnas que indican si un usuario es "Recurrente" o "De otro tipo", incluir explícitamente a los "Nuevos" es redundante (si no es de los primeros dos, por descarte es nuevo). Eliminar redundancias hace que el modelo entrene más rápido y sea más preciso, que es justamente lo que buscamos

### B. Ingeniería de Características y Escalado

En el Deep Learning, la magnitud de los números importa, por ende, si combinamos variables que se miden en miles (como la duración de una sesión en segundos) con variables que se miden en fracciones (como la tasa de rebote entre 0 y 1), la red neuronal le dará un peso desproporcionado a los números grandes. Para igualar su importancia, se unificaron las escalas usando un ColumnTransformer:

- RobustScaler: Se aplicó exclusivamente a variables con valores atípicos severos (ej. tiempos de navegación extremadamente altos de usuarios que dejaron la pestaña abierta, o picos en el PageValues). A diferencia del escalado tradicional, este método utiliza la mediana y el rango intercuartílico, logrando que estos valores extremos y poco comunes no distorsionen el aprendizaje del modelo

- StandardScaler: Se aplicó al resto de las variables numéricas con distribuciones más regulares. Este escalador utiliza la ecuación \$z = \\frac{x - \\mu}{\\sigma}\$ para reajustar los datos, garantizando que tengan una media de cero y una varianza unitaria. Esto acelera matemáticamente la convergencia de la red neuronal durante el entrenamiento

### C. Modelado y Entrenamiento

Debido a la naturaleza altamente no lineal del comportamiento de navegación humano, los modelos estadísticos simples no eran suficientes. Por lo cual, se optó por una arquitectura de Deep Learning (Redes Neuronales utilizando la API de TensorFlow/Keras) para capturar estas interacciones complejas

- **Optimización Estratégica:** Se empleó la herramienta Keras Tuner (mediante el algoritmo Hyperband) para realizar una búsqueda automatizada y exhaustiva de los hiperparámetros de la red. La decisión clave aquí fue optimizar la métrica AUC (Área Bajo la Curva ROC) en lugar de la precisión (*accuracy*) tradicional. Dado que en el e-commerce hay muchos más "no compradores" que "compradores" (clases desbalanceadas), optimizar el AUC obliga al modelo a ser igualmente bueno detectando a la minoría que realmente compra, en lugar de predecir "no compra" todo el tiempo para inflar su porcentaje de acierto

- **Arquitectura Final Seleccionada**: El modelo ganador se compone de una secuencia de capas densas (Dense) que reducen su tamaño progresivamente para destilar la información. Se implementó Batch Normalization (Normalización por Lotes) para evitar que los cálculos internos se desestabilicen entre capas, y Dropout, una técnica de regularización que "apaga" aleatoriamente un porcentaje de las neuronas (en este caso el 20%) durante el entrenamiento. Esto fuerza a la red a no memorizar los datos de entrenamiento y previene el sobreajuste (*overfitting*)

- **Parámetros Finales:** El ajuste determinó que la mejor configuración es una tasa de aprendizaje iterativo (learning_rate) de 0.01, operando sobre 2 bloques principales de capas ocultas con 320 y 96 neuronas respectivamente

## 4. Evaluación y Métricas de Rendimiento

El modelo final fue evaluado sobre un conjunto de datos de prueba (20% del dataset original que el modelo nunca había visto):

- **Accuracy (Precisión Global):** 89.39%
- **ROC AUC:** 0.9200 (Demuestra una excelente capacidad del modelo para distinguir entre las clases de compra y no compra)
- **F1-Score:** 0.6174 (Métrica de equilibrio que confirma que el modelo no solo predice ceros, sino que identifica correctamente las compras reales)

## 5. API REST y Despliegue

El modelo se desplegó utilizando **Flask**. La API recibe la información de la sesión en tiempo real, aplica exactamente las mismas transformaciones del pipeline de entrenamiento a través del archivo preprocessor.pkl, y utiliza model.keras para generar la respuesta

El departamento de IT requiere información clara. Por ello, la API calcula la probabilidad de la clase dominante y retorna un JSON estrictamente estructurado con tres elementos que son fundamentales para el negocio: la clasificación final, la probabilidad matemática exacta y un mensaje dinámico legible para humanos

**Ejemplo de Petición (POST a /predict):**

Método: POST

URL: https://tractor-avid-extended.ngrok-free.dev/predict

JSON

{

"Administrative": 3,

"Administrative_Duration": 61.0000,

"Informational": 0,

"Informational_Duration": 0.0,

"ProductRelated": 15,

"ProductRelated_Duration": 504.506000,

"BounceRates": 0.00000,

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

**Ejemplo de Respuesta JSON:**

JSON

{

"clasificación": "Comprar",

"mensaje": "El usuario va a: Comprar. Con un 52.43% de probabilidades, lo que lo hace bastante probable.",

"probabilidad": 0.5243004560470581

}

## 6. Instrucciones de Ejecución y Notas Importantes

Para replicar y probar este proyecto localmente, siga los siguientes pasos:

- **Requisito del Sistema:** Python 3.10 - 3.13 (necesario para la compatibilidad con TensorFlow)

- **Instalación de Dependencias:** Ejecute en su terminal: pip install pandas scikit-learn flask joblib tensorflow keras-tuner matplotlib seaborn

- **Entrenamiento (Opcional):** Para generar los archivos base desde cero, ejecute python entrenamiento.py.

- **Inicio del Servidor Web:** Levante la API ejecutando python api.py

**NOTAS IMPORTANTES PARA LA PRUEBA (POSTMAN):** \* El análisis de datos y el entrenamiento del modelo se realizaron formalmente en el [Cuaderno Jupyter en Google Colab](https://colab.research.google.com/drive/1wuk98k_Eer7rKLQJAm_fhCaYbCbxzvUb?usp=sharing)

- Si está evaluando el despliegue a través del entorno en la nube o Colab, recuerde que debe darle "Play" para activar la conexión al último cuadro. Solo después de hacer esto, el servidor estará escuchando y podrá enviar las peticiones exitosamente desde Postman

- Luego de la ejecución del primer colab, se crean dos archivos, los cuales sirven de entrada para que el segundo colab funcione
