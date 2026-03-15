from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib

from pdf_csv import convert_pdf_to_csv
from preprocessing import preprocess_transactions
from feature_engineering import create_features

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return {"status": "Prediction backend is running"}


# ---------------- UPLOAD PDF ----------------
@app.route("/upload", methods=["POST"])
def upload_pdf():

    if "file" not in request.files or "user_id" not in request.form:
        return jsonify({"error": "file and user_id required"}), 400

    file = request.files["file"]
    user_id = request.form["user_id"]

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    pdf_path = f"uploads/{user_id}.pdf"
    csv_path = f"data/{user_id}_raw.csv"

    file.save(pdf_path)

    # Step 1: Convert PDF → CSV
    convert_pdf_to_csv(pdf_path, csv_path)

    return jsonify({
        "message": "PDF uploaded and converted to CSV",
        "csv_file": csv_path
    })


# ---------------- PREDICTION ----------------
@app.route("/predict/<user_id>", methods=["GET"])
def predict(user_id):

    raw_csv = f"data/{user_id}_raw.csv"
    clean_csv = f"data/{user_id}_clean.csv"
    feature_csv = f"data/{user_id}_features.csv"

    if not os.path.exists(raw_csv):
        return jsonify({"error": "CSV not found for user"}), 404

    # Step 2: Preprocess dataset
    preprocess_transactions(raw_csv, clean_csv)

    # Step 3: Feature engineering
    create_features(clean_csv, feature_csv)

    # Step 4: Load feature dataset
    df = pd.read_csv(feature_csv)

    if len(df) == 0:
        return jsonify({"error": "Not enough transaction data"}), 400

    # Step 5: Load trained model
    model = joblib.load("models/spending_model.pkl")

    # Step 6: Take latest row
    latest = df.tail(1)

    features = [
        "lag1",
        "lag2",
        "lag3",
        "rolling_avg",
        "rolling_std",
        "trend",
        "momentum",
        "month_sin",
        "month_cos"
    ]

    X = latest[features]

    # Step 7: Prediction
    prediction = model.predict(X)

    result = {
        "predicted_next_month_spending": float(prediction[0]),
        "category": latest["category"].values[0]
    }

    return jsonify(result)


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)