from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Cargar el modelo y el preprocesador
try:
    model = load_model('model.keras')
except Exception as e:
    print(f"Error al cargar model.keras: {e}. Asegúrate de ejecutar entrenamiento.py primero.")
    model = None

preprocessor = joblib.load('preprocessor.pkl')

def preprocesar_entrada(data):
    """Aplica el mismo preprocesamiento que en el entrenamiento."""
    df = pd.DataFrame([data])
    
    # 1. Transformaciones
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    if 'Month' in df.columns and isinstance(df['Month'].iloc[0], str):
        df['Month'] = df['Month'].map(month_map)
    
    # Manejar VisitorType (get_dummies manual para asegurar consistencia con el entrenamiento)
    df['VisitorType_Returning_Visitor'] = 1 if df['VisitorType'].iloc[0] == 'Returning_Visitor' else 0
    df['VisitorType_Other'] = 1 if df['VisitorType'].iloc[0] == 'Other' else 0
    # No incluimos VisitorType_New_Visitor porque fue droppeada
    
    # Asegurar tipos
    if 'Weekend' in df.columns:
        df['Weekend'] = df['Weekend'].astype(int)
        
    # Eliminar columnas no deseadas
    cols_to_drop = ['ExitRates', 'VisitorType', 'Revenue']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            
    # Reordenar columnas para que coincidan con el orden esperado por el preprocessor
    expected_order = [
        'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'PageValues', 
        'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 
        'Weekend', 'VisitorType_Other', 'VisitorType_Returning_Visitor'
    ]
    
    # Asegurar que todas las columnas existan, si no, poner 0
    for col in expected_order:
        if col not in df.columns:
            df[col] = 0
            
    df = df[expected_order]
    
    return df

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Preprocesar datos
        input_df = preprocesar_entrada(data)
        
        # Aplicar el escalador
        input_final = preprocessor.transform(input_df)
        
        # Predicción
        prob_class_1 = model.predict(input_final, verbose=0).ravel()
        prediction = (prob_class_1 > 0.5).astype(int)

        if prediction == 1:
            label = "Comprar"
            final_prob = prob_class_1[0]
        else:
            label = "NO Comprar"
            final_prob = 1 - prob_class_1[0]

        # Mensaje amigable
        prob_percent = round(final_prob * 100, 2)

        if final_prob > 0.8:
            sentiment = "lo que lo hace muy probable"
        elif final_prob > 0.5:
            sentiment = "lo que lo hace bastante probable"
        elif final_prob > 0.2:
            sentiment = "lo que lo hace poco probable"
        else:
            sentiment = "lo que lo hace muy poco probable"

        message = f"El usuario va a: {label}. Con un {prob_percent}% de probabilidades, {sentiment}."

        return jsonify({
            "clasificacion": label,
            "probabilidad": float(final_prob),
            "mensaje": message
        })

        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
