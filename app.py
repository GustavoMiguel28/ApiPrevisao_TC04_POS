from flask import Flask, request, jsonify
from utils import load_model_and_scaler, preprocess_values, TIME_STEPS

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API de predicao LSTM funcionando. Use /predict com POST."
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Validações básicas
    if not data or "ticker" not in data or "values" not in data:
        return jsonify({"error": "JSON deve conter 'ticker' e 'values'"}), 400

    ticker = data["ticker"]
    values = data["values"]

    if len(values) != TIME_STEPS:
        return jsonify({"error": f"'values' deve conter exatamente {TIME_STEPS} valores"}), 400

    # Carrega modelo e scaler
    model, scaler = load_model_and_scaler(ticker)
    if model is None:
        return jsonify({"error": f"Modelo para o ticker {ticker} não encontrado"}), 404

    # Preprocessa valores
    X = preprocess_values(values, scaler)

    # Predição
    prediction_scaled = model.predict(X)
    prediction = scaler.inverse_transform(prediction_scaled)

    return jsonify({
        "ticker": ticker,
        "prediction": float(prediction[0][0])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)