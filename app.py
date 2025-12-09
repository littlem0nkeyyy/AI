from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model + preprocessor bundle
BUNDLE_PATH = "XgBoost_SalePrediction.joblib"
bundle = joblib.load(BUNDLE_PATH)

model = bundle["model"]
preprocessor = bundle["preprocessor"]
numeric_features = bundle["numeric_features"]
categorical_features = bundle["categorical_features"]

feature_order = numeric_features + categorical_features


@app.get("/health")
def health():
    return jsonify({"status": "ok", "message": "XGBoost seasonal API is running"}), 200


@app.post("/predict")
def predict():
    """
    Expects JSON body:
    - Either a single object:
        {
          "Product ID": 123,
          "Price": 29.99,
          "Rating": 4.5,
          "Time_year": 2025,
          "Time_month": 3,
          "Time_day": 21,
          "Time_dayofweek": 4,
          "Promotion": "Yes",
          "Product Category": "Tops",
          "Seasonal": "Spring",
          "Terms": "Standard",
          "Section": "WOMAN",
          "season": "Spring"
        }
    - Or a list of such objects.
    """
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON payload provided"}), 400

    # Normalize to list
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return jsonify({"error": "Invalid JSON format. Must be object or list of objects."}), 400

    try:
        df_input = pd.DataFrame(records)

        # Check that required features exist
        missing = [col for col in feature_order if col not in df_input.columns]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing_features": missing
            }), 400

        X = df_input[feature_order]
        X_processed = preprocessor.transform(X)
        preds = model.predict(X_processed)

        # Build response, optionally echo back Product ID / Name if present
        response_records = []
        for i, y_hat in enumerate(preds):
            rec = {"prediction": float(y_hat)}
            if "Product ID" in df_input.columns:
                rec["Product ID"] = df_input.loc[i, "Product ID"]
            if "Name" in df_input.columns:
                rec["Name"] = df_input.loc[i, "Name"]
            if "season" in df_input.columns:
                rec["season"] = df_input.loc[i, "season"]
            response_records.append(rec)

        return jsonify({
            "n_samples": len(response_records),
            "results": response_records
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local development only; in production use gunicorn/uwsgi
    app.run(host="0.0.0.0", port=8000, debug=True)
