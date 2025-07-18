"""
Air Pollution Forecasting with Multivariate LSTM & MLOps Pipeline
IEEE Research Implementation

Title: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"
DOI: 10.1109/INOCON60754.2024.10511808
Conference: 2024 3rd International Conference for Innovation in Technology (INOCON)

Author: Tirumala Manav
GitHub: https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps
Email: Contact via GitHub Issues
Date: July 2024

Description: [Add specific module description here]

License: MIT License
Copyright (c) 2024 Tirumala Manav

This module implements [specific functionality] as part of the complete
air pollution forecasting system with production-ready MLOps pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import tensorflow as tf
from sklearn.metrics import mean_squared_error

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

# Load the pre-trained Keras model
keras_model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
keras_model = tf.keras.models.load_model(keras_model_path)

def make_prediction(input_data):
    # Convert input_data to DataFrame
    data = pd.DataFrame(input_data)

    # Ensure input_data contains the required features
    if not set(config.FEATURES).issubset(data.columns):
        raise ValueError("Input data must contain the following features: " + ", ".join(config.FEATURES))

    # Prepare the data for LSTM
    data = data[config.FEATURES].values
    data = data.reshape((data.shape[0], 1, data.shape[1]))  # Reshape to (samples, timesteps, features)

    # Make predictions using the Keras model
    prediction = keras_model.predict(data)

    return prediction

def plot_predictions_and_evaluate(actual_values, predicted_values):
    # Plot the graph between actual vs predicted values
    plt.figure(figsize=(10,6))
    plt.plot(predicted_values[:100], color='green', label='Predicted Pollution Level')
    plt.plot(actual_values[:100], color='red', label='Actual Pollution Level')
    plt.title("Air Pollution Prediction (Multivariate)")
    plt.xlabel("Sample Index")
    plt.ylabel("Pollution Level")
    plt.legend()
    plt.savefig('graph.png')
    plt.show()

    # Calculate Mean Absolute Percentage Error (MAPE)
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-8  # Small constant to avoid division by zero
        return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    mape = mean_absolute_percentage_error(actual_values, predicted_values)
    print('MAPE:', mape)

    # Calculate Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    print('RMSE:', rmse)
    print("Mean of Actual Values:", np.mean(actual_values))

if __name__ == '__main__':
    # Load test data
    test_data = pd.read_csv(config.DATAPATH + '/pollution.csv')
    test_features = test_data[config.FEATURES]
    actual_values = test_data[config.TARGET].values

    # Make predictions
    predictions = make_prediction(test_features)

    # Plot and evaluate predictions
    plot_predictions_and_evaluate(actual_values, predictions)
    print("Predictions:", predictions[:10])
    print("Actual Values:", actual_values[:10])
    print("Prediction Shape:", predictions.shape)
    print("Actual Values Shape:", actual_values.shape)

