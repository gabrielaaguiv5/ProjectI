from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load model, scaler, and category names
with open("opt_kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("opt_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("category_names.pkl", "rb") as f:
    category_names = pickle.load(f)

# Define profile names based on clusters
profile_names = [
    "Amante de la Naturaleza y la Cultura",   # Cluster 0
    "Viajero Urbano y Social",                # Cluster 1
]

app = Flask(__name__)

@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json()
    # Ensure the categories are in the correct order
    input_array = np.array([[data.get(cat, 0.0) for cat in category_names]])
    data_scaled = scaler.transform(input_array)
    cluster = int(kmeans.predict(data_scaled)[0])
    perfil = profile_names[cluster] if cluster < len(profile_names) else f"Perfil {cluster}"
    return jsonify({"cluster": cluster, "perfil": perfil})

if __name__ == "__main__":
    app.run(debug=True)


print("Starting Flask API...")