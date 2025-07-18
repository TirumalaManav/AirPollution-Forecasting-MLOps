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

from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from prediction_model.processing.data_handling import load_dataset, save_model, load_model, save_pipeline, load_pipeline
from prediction_model.processing.reshape_transformer import ReshapeForLSTM

def build_model(input_shape):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Second LSTM layer
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Third LSTM layer
    model.add(LSTM(units=100))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def create_lstm_model():
    return build_model((None, len(config.FEATURES)))


classification_pipeline = Pipeline(
    [
        # ('categorical_imputer',
        #  pp.CategoricalImputer(variables=config.CATEGORICAL_FEATURES)),
        ('numerical_imputer',
         pp.NumericalImputer(variables=config.NUMERICAL_FEATURES)),
        ('drop_columns',
         pp.DropColumns(variables_to_drop=config.DROP_FEATURES,
                        reference_variable=config.TARGET)),
        ('scaler', MinMaxScaler()),
        ('reshape', ReshapeForLSTM(time_steps=1)),  # Ensure correct reshaping
        ('lstm_model', KerasRegressor(build_fn=create_lstm_model, epochs=50, batch_size=32, verbose=0))
    ]
)



