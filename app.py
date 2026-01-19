from flask import Flask, request, render_template
import joblib
import os
import numpy as np

MODEL_PATH = os.path.join("model", "house_price_model.pkl")

app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Model not found. Run: python model_development.py (it will create model/house_price_model.pkl)"
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=FEATURES, result=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect inputs in the exact feature order used in training
        values = []
        for f in FEATURES:
            raw = request.form.get(f, "").strip()
            if raw == "":
                raise ValueError(f"Missing value for {f}")
            values.append(float(raw))

        x = np.array(values, dtype=np.float64).reshape(1, -1)
        pred = float(model.predict(x)[0])

        return render_template("index.html", features=FEATURES,
                               result=f"${pred:,.0f}", error=None)
    except Exception as e:
        return render_template("index.html", features=FEATURES,
                               result=None, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
