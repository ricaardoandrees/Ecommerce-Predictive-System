import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, callbacks
import keras_tuner as kt
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def preprocesar_datos(df):
    """Limpia los datos y aplica las transformaciones."""
    print("--- INICIANDO PREPROCESAMIENTO ---")
    
    # 1. Transformaciones de la vez pasada
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_map)
    df = pd.get_dummies(df, columns=['VisitorType'])
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    if 'VisitorType_Returning_Visitor' in df.columns:
        df['VisitorType_Returning_Visitor'] = df['VisitorType_Returning_Visitor'].astype(int)
    if 'VisitorType_Other' in df.columns:
        df['VisitorType_Other'] = df['VisitorType_Other'].astype(int)
        
    # 2. LIMPIEZA NUEVA (Lo que me acabas de pasar)
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # Eliminando columnas que aporten leakage o no tengan valor
    def eliminarColumnas(columnas, dataset):
        for i in range(len(columnas)):
            if columnas[i] in dataset.columns:
                dataset.drop(columnas[i], axis=1, inplace=True)
                
    cols = ['ExitRates', 'VisitorType_New_Visitor']
    eliminarColumnas(cols, df)
    
    # 3. Separación de Variables
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    return X, y

def build_model(hp):
    # (OJO: para que X_train_final.shape[1] funcione aquí, puedes pasarle el input_shape como parámetro, 
    # o simplemente poner el número de columnas que te quedó tras la limpieza. 
    # Para hacerlo dinámico, lo leeremos directamente de los datos)
    pass # La definimos dentro de entrenar_modelo para tener acceso a X_train_final

def entrenar_modelo(X_train_final, y_train, X_valid_final, y_valid):
    """Entrena la red neuronal buscando los mejores hiperparámetros."""
    print("\n--- BUSCANDO LOS MEJORES HIPERPARÁMETROS ---")
    
    # Definimos la arquitectura aquí adentro para que lea el shape correcto
    def build_model_internal(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train_final.shape[1],)))

        # Bloque 1: Ajuste de neuronas y dropout inicial
        hp_units_1 = hp.Int('units_1', min_value=128, max_value=512, step=64)
        model.add(layers.Dense(units=hp_units_1, activation="relu"))
        model.add(layers.BatchNormalization())
        hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
        model.add(layers.Dropout(rate=hp_dropout_1))

        # Bloque 2: Capas intermedias variables
        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(layers.Dense(units=hp.Int(f'units_{i+2}', 32, 128, step=32), activation="relu"))
            model.add(layers.BatchNormalization())

        # Capas de refinamiento finales
        model.add(layers.Dense(16, activation="relu"))
        model.add(layers.Dense(1, activation="sigmoid"))

        # Optimización
        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        optimizer = keras.optimizers.Adam(learning_rate=hp_lr)

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[
                metrics.BinaryAccuracy(name='accuracy'),
                metrics.Precision(name='precision'),
                metrics.Recall(name='recall'),
                metrics.AUC(name='auc')
            ]
        )
        return model

    # Configuración del Tuner
    tuner = kt.Hyperband(
        build_model_internal,
        objective='val_auc',
        max_epochs=50,
        factor=3,
        directory='kt_dir',
        project_name='market_predict'
    )

    # Callback de parada temprana
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    # Búsqueda
    tuner.search(
        X_train_final, y_train,
        epochs=20,
        validation_data=(X_valid_final, y_valid),
        callbacks=[early_stop],
        verbose=1
    )

    # Recuperar y entrenar el mejor modelo
    print("\n--- ENTRENANDO EL MEJOR MODELO ENCONTRADO ---")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model_final = tuner.hypermodel.build(best_hps)

    history = model_final.fit(
        X_train_final, y_train,
        epochs=30,
        validation_data=(X_valid_final, y_valid),
        callbacks=[early_stop],
        verbose=0
    )
    
    return model_final, tuner, history

def ejecutar_entrenamiento():
    # 1. Cargar datos
    df = pd.read_csv('ecommerce_dataset.csv')
    
    # 2. Preprocesar y limpiar
    X, y = preprocesar_datos(df)
    
    # 3. Separación (¡Doble split como en el Colab!)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    print("\nDatos listos para el entrenamiento.")
    
    # 4. Configurar el Escalador (ColumnTransformer)
    cols_robust = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues', 'BounceRates']
    cols_estandar = ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'TrafficType']

    preprocessor = ColumnTransformer(
        transformers=[
            ('robust', RobustScaler(), cols_robust),
            ('std', StandardScaler(), cols_estandar)
        ],
        remainder='passthrough'
    )
    
    # 5. Aplicar el escalado a los datos
    X_train_final = preprocessor.fit_transform(X_train)
    X_valid_final = preprocessor.transform(X_valid)
    X_test_final = preprocessor.transform(X_test)
    print(f"Forma del set de entrenamiento: {X_train_final.shape}")
    
    # 6. ¡GUARDAR EL PREPROCESSOR PARA LA API!
    print("Guardando el preprocesador...")
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("¡preprocessor.pkl generado con éxito!")
    
    # 7. Entrenar el modelo (Asegúrate de pasarle los datos de validación también)
    model_final, tuner, history = entrenar_modelo(X_train_final, y_train, X_valid_final, y_valid)
    
    # 8. Evaluación del Modelo (Con el test set)
    print("\n--- EVALUACIÓN FINAL ---")
    best_model = tuner.get_best_models(num_models=1)[0]
    y_probs = best_model.predict(X_test_final).ravel()
    y_pred = (y_probs > 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"F1-Score en Test: {f1:.4f}")
    print(f"ROC AUC en Test: {roc_auc:.4f}")

    # 9. Mostrar las gráficas
    # Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Modelo Final')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

    # Curvas de aprendizaje
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Evolución de la Pérdida')
    ax1.legend()
    
    ax2.plot(history.history['auc'], label='Train AUC')
    ax2.plot(history.history['val_auc'], label='Val AUC')
    ax2.set_title('Evolución del AUC')
    ax2.legend()
    plt.show()

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Compra', 'Compra'])
    disp.plot(cmap='Purples', values_format='d')
    plt.title('Matriz de Confusión Final')
    plt.show()

    # 10. ¡GUARDAR EL MODELO PARA LA API!
    print("\n--- GUARDANDO RED NEURONAL ---")
    model_final.save('model.keras')
    print("¡model.keras exportado exitosamente!")
    print("¡Pipeline completado! Ya tienes preprocessor.pkl y model.keras listos para la API.")