import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

TIME_STEPS = 60
MODEL_BASE_PATH = "model"

def load_model_and_scaler(ticker: str):
    """
    Carrega modelo LSTM e scaler do ticker.
    Retorna (model, scaler) ou (None, None) se não existir.
    """
    model_path = os.path.join(MODEL_BASE_PATH, ticker, "lstm_model.h5")
    scaler_path = os.path.join(MODEL_BASE_PATH, ticker, "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_values(values: list, scaler):
    """
    Converte a lista de valores em array numpy normalizado e reshape para LSTM.
    """
    values = np.array(values).reshape(-1, 1)
    values_scaled = scaler.transform(values)
    X = values_scaled.reshape(1, TIME_STEPS, 1)
    return X