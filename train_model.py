import os
import yfinance as yf
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error

tickers = ['ITUB4.SA', 'PETR4.SA', 'ABEV3.SA']
start_date = '2020-01-01'
end_date = '2025-12-30'
time_steps = 60

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

for ticker in tickers:
    print(f"Treinando modelo para {ticker}")

    # 1. Coleta
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']].dropna()

    # 2. Normalização
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    # 3. Sequências
    X, y = create_sequences(scaled_data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 4. Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 5. Modelo
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 6. Treinamento
    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # 7. Avaliação
    y_pred = model.predict(X_val)

    y_pred_inv = scaler.inverse_transform(y_pred)
    y_val_inv = scaler.inverse_transform(y_val)

    mae = mean_absolute_error(y_val_inv, y_pred_inv)
    rmse = mean_squared_error(y_val_inv, y_pred_inv)
    mape = (abs((y_val_inv - y_pred_inv) / y_val_inv)).mean() * 100

    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%")

    # 7. Salvamento
    model_dir = f"model/{ticker}"
    os.makedirs(model_dir, exist_ok=True)

    model.save(f"{model_dir}/lstm_model.h5")
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    print(f"Modelo salvo para {ticker}\n")

#########################################################
########## AVALIAÇÕES DO MODELO TREINADO ################
#########################################################

#ITUB4.SA -> MAE: 0.49 | RMSE: 0.37 | MAPE: 1.59%
#PETR4.SA -> MAE: 0.72 | RMSE: 0.90 | MAPE: 2.35%
#ABEV3.SA -> MAE: 0.20 | RMSE: 0.08 | MAPE: 1.68%