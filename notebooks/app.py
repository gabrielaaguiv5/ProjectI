from flask import Flask, request, jsonify
import pickle
import json
import numpy as np

# Carga scaler, modelo y parámetros
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)
with open('params.json', 'r') as f:
    params = json.load(f)
columns = params['columns']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Ordenar los valores en el mismo orden de columnas
    X_input = np.array([data[col] for col in columns]).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    cluster = int(kmeans.predict(X_scaled)[0])
    return jsonify({'cluster': cluster})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True)