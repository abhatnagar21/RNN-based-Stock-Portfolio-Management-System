# RNN-based-Stock-Portfolio-Management-System

RNN-based Stock Portfolio Management System
This project is a deep learning-based stock portfolio management system. It leverages an LSTM (Long Short-Term Memory) model to predict the future prices of multiple stocks and calculates the portfolio's performance over a given period. The project uses data fetched from Yahoo Finance and focuses on Apple (AAPL), Google (GOOGL), Microsoft (MSFT), and Amazon (AMZN).

Table of Contents
Overview
Project Structure
Features
Installation
Usage
Model and Parameters
Results
License
Overview
This project demonstrates how recurrent neural networks (RNNs), specifically LSTM networks, can be applied to time series forecasting in stock price prediction. The system is designed to:

Fetch stock price data from Yahoo Finance.
Train a deep learning model to predict stock prices.
Calculate the value of a stock portfolio over time and provide an estimate of profit/loss.
Technologies Used
Python: Core language.
TensorFlow/Keras: For building and training the LSTM model.
yFinance: For fetching stock price data.
scikit-learn: For data preprocessing (normalization).
Matplotlib: For visualizing stock prices and predictions.


Results
Initial Portfolio Value: The value of the portfolio at the start of the testing period (based on actual prices).
Final Portfolio Value: The portfolio's value based on predicted stock prices at the end of the testing period.
Profit/Loss: The overall performance of the portfolio.
