import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Ignore TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Function to download stock data from yfinance
def fetch_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    data = data.fillna(method='ffill')
    return data

# Function to normalize and scale the data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler

# Function to create sequences (X as input features, y as target values)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dense(output_size))  # Output layer for each stock in the portfolio
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to plot the actual vs predicted stock prices
def plot_predictions(y_test_rescaled, predicted_prices_rescaled, tickers):
    plt.figure(figsize=(10, 6))
    for i, ticker in enumerate(tickers):
        plt.subplot(len(tickers), 1, i+1)
        plt.plot(y_test_rescaled[:, i], label=f'Actual Price for {ticker}')
        plt.plot(predicted_prices_rescaled[:, i], label=f'Predicted Price for {ticker}')
        plt.legend()
        plt.tight_layout()
    plt.show()

# Function to calculate portfolio value
def calculate_portfolio_value(prices, initial_investment):
    portfolio_value = np.sum(prices * initial_investment)
    return portfolio_value

# Step 1: Set parameters for stocks and data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']  # Tickers including Amazon
start_date = '2020-01-01'
end_date = '2023-01-01'
seq_length = 10  # Increased sequence length to capture more information

# Step 2: Fetch and prepare data
data = fetch_stock_data(tickers, start_date, end_date)
data_scaled, scaler = normalize_data(data)

# Step 3: Prepare sequences for training/testing
X, y = create_sequences(data_scaled, seq_length)

# Step 4: Split data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 5: Build and compile the LSTM model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape, len(tickers))

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Step 6: Train the model with more epochs and early stopping
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Step 7: Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Step 8: Make predictions
predicted_stock_prices = model.predict(X_test)

# Step 9: Rescale the predicted and actual values back to the original scale
predicted_stock_prices_rescaled = scaler.inverse_transform(predicted_stock_prices)
y_test_rescaled = scaler.inverse_transform(y_test)

# Step 10: Plot actual vs predicted stock prices for each stock
plot_predictions(y_test_rescaled, predicted_stock_prices_rescaled, tickers)

# Step 11: Calculate initial and final portfolio values

# Initial portfolio value based on actual prices at the start of the testing period
initial_prices = y_test_rescaled[0]  # First day in the test data
initial_investment = np.array([25, 25, 25, 25])  # 25 units of each stock
initial_portfolio_value = calculate_portfolio_value(initial_prices, initial_investment)
print(f'Initial Portfolio Value: {initial_portfolio_value}')

# Final portfolio value based on predicted prices at the end of the testing period
final_prices = predicted_stock_prices_rescaled[-1]  # Last predicted day in the test data
final_portfolio_value = calculate_portfolio_value(final_prices, initial_investment)
print(f'Final Portfolio Value: {final_portfolio_value}')

# Step 12: Calculate profit or loss
profit_loss = final_portfolio_value - initial_portfolio_value
print(f'Profit/Loss: {profit_loss}')
