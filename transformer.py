'''
Author: pang-lee
Date: 2024-12-12 13:18:43
LastEditTime: 2024-12-12 13:24:47
LastEditors: LAPTOP-22MC5HRI
Description: In User Settings Edit
FilePath: \stock\transformer.py
'''
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Fetch historical stock data
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

# Step 2: Prepare data for time series prediction
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Step 3: Define Transformer model
def build_transformer_model(input_shape, d_model=64, num_heads=4, ff_dim=128, dropout_rate=0.1):
    inputs = Input(shape=input_shape)
    
    # Multi-Head Attention
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention = Dropout(dropout_rate)(attention)
    attention = Add()([inputs, attention])
    attention = LayerNormalization(epsilon=1e-6)(attention)
    
    # Feed-Forward Network
    ff = Dense(ff_dim, activation="relu")(attention)
    ff = Dense(input_shape[-1])(ff)
    ff = Dropout(dropout_rate)(ff)
    ff = Add()([attention, ff])
    outputs = LayerNormalization(epsilon=1e-6)(ff)

    # Output Layer
    outputs = Dense(1)(outputs[:, -1, :])  # Use the last time step's representation

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

# Step 5: Plot predictions vs actual values
def plot_predictions(actual, predicted, title="Actual vs Predicted"):
    plt.figure(figsize=(14, 7))
    plt.plot(actual, label="Actual", color="blue", alpha=0.7)
    plt.plot(predicted, label="Predicted", color="red", alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Stock Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    n_steps = 30

    # Load data
    data = get_stock_data(ticker, start_date, end_date)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Prepare training data
    X, y = prepare_data(data_scaled, n_steps)

    # Build model
    model = build_transformer_model(input_shape=(n_steps, 1))

    # Train model
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X, y, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping])

    # Make predictions
    predictions = model.predict(X)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))  # Rescale predictions
    y_rescaled = scaler.inverse_transform(y.reshape(-1, 1))  # Rescale true values

    # Plot the predictions vs actual values
    plot_predictions(y_rescaled.flatten(), predictions_rescaled.flatten(), title=f"{ticker} Stock Price Prediction")