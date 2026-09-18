"""Local REST prediction API representing the deployed victim."""

import argparse

from flask import Flask, jsonify, request

from victim import victim

parser = argparse.ArgumentParser(description="Run the victim model API")
parser.add_argument("--decimals", type=int, choices=[2, 3, 4, 5], default=None, help="Defender-side rounding precision")
args = parser.parse_args()

app = Flask(__name__)
app.config["ROUNDING_DECIMALS"] = args.decimals


@app.post("/predict")
def predict():
    body = request.get_json(silent=True) or {}
    try:
        features = body["features"]
        result = victim.predict(features, decimals=app.config.get("ROUNDING_DECIMALS"))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "features": victim.feature_count})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
