import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers, metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

def preprocesar_datos(df):
    """Limpia los datos y aplica las transformaciones."""
    print("--- INICIANDO PREPROCESAMIENTO ---")
    
    # 1. Transformaciones
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_map)
    
    # Manejar VisitorType
    df = pd.get_dummies(df, columns=['VisitorType'])
    
    # Asegurar que las columnas booleanas sean int
    df['Weekend'] = df['Weekend'].astype(int)
    df['Revenue'] = df['Revenue'].astype(int)
    
    # Manejar posibles columnas faltantes tras get_dummies si es necesario
    for col in ['VisitorType_Returning_Visitor', 'VisitorType_Other', 'VisitorType_New_Visitor']:
        if col in df.columns:
            df[col] = df[col].astype(int)
        
    # 2. LIMPIEZA
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)

    # Eliminando columnas que aporten leakage o no tengan valor (según código original)
    cols_to_drop = ['ExitRates', 'VisitorType_New_Visitor']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    # 3. Separación de Variables
    X = df.drop('Revenue', axis=1)
    
    # Reordenar columnas para asegurar consistencia con la API
    expected_order = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'PageValues', 
        'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
        'Weekend', 'VisitorType_Other', 'VisitorType_Returning_Visitor'
    ]
    
    # Asegurar que todas las columnas existan (por si acaso)
    for col in expected_order:
        if col not in X.columns:
            X[col] = 0
            
    X = X[expected_order]
    y = df['Revenue']
    
    return X, y

def entrenar_modelo(X_train_final, y_train, X_valid_final, y_valid):
    print("\n--- BUSCANDO LOS MEJORES HIPERPARÁMETROS CON RANDOM SEARCH ---")
    
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
    
    print(f"Mejores parámetros: {tuner.get_best_hyperparameters(num_trials=1)[0]}")
    return tuner.get_best_models(num_models=1)[0]

def ejecutar_entrenamiento():
    # 1. Cargar datos
    dataset_path = 'Dataset/dataset shop.csv'
    df = pd.read_csv(dataset_path)
    
    # 2. Preprocesar y limpiar
    X, y = preprocesar_datos(df)
    
    # 3. Separación
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
    
    # 4. Configurar el Escalador (ColumnTransformer)
    # Ajustamos las columnas según las disponibles tras el preprocesamiento
    cols_robust = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'PageValues', 'BounceRates']
    cols_estandar = ['Administrative', 'Informational', 'ProductRelated', 'SpecialDay', 'TrafficType']

    # Verificar que las columnas existan antes de aplicar
    cols_robust = [c for c in cols_robust if c in X_train_full.columns]
    cols_estandar = [c for c in cols_estandar if c in X_train_full.columns]

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
    
    # 7. Entrenar el modelo
    model_final = entrenar_modelo(X_train_final, y_train, X_valid_final, y_valid)
    
    # 8. Evaluación del Modelo (Con el test set)
    print("\n--- EVALUACIÓN FINAL ---")
    y_probs = model_final.predict(X_test_final).ravel()
    y_pred = (y_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy en Test: {acc:.4f}")
    print(f"F1-Score en Test: {f1:.4f}")
    print(f"ROC AUC en Test: {roc_auc:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # 10. ¡GUARDAR EL MODELO PARA LA API!
    print("\n--- GUARDANDO MODELO ---")
    model_final.save('model.keras')
    print("¡model.keras exportado exitosamente!")
    print("¡Pipeline completado! Ya tienes preprocessor.pkl y model.keras listos para la API.")

    # Visualización de la Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Modelo Final')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == "__main__":
    ejecutar_entrenamiento()
