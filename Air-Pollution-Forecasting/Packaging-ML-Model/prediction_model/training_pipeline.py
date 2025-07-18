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
from pathlib import Path
import os
import sys
import joblib


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import LSTM, Dropout, BatchNormalization, Dense
from prediction_model.config import config
from prediction_model.processing import preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset, save_model, save_pipeline,load_model,load_pipeline
from prediction_model.processing.preprocessing import SomePreprocessingClass
from prediction_model.processing.reshape_transformer import ReshapeForLSTM
from prediction_model.pipeline import create_lstm_model

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

def perform_training():
    # Load the dataset
    train_data = load_dataset(config.TRAIN_FILE)

    # Prepare target variable
    train_y = train_data[config.TARGET]

    # Prepare X_train
    required_columns = list(config.NUMERICAL_FEATURES)
    X_train = train_data[required_columns].values.reshape((train_data.shape[0], 1, len(required_columns)))

    # Initialize preprocessing
    preprocessor = SomePreprocessingClass(variables=required_columns)
    X_train_prepared = preprocessor.fit_transform(X_train)

    # Define the Scikit-learn pipeline
    classification_pipeline = Pipeline(
        [
            ('numerical_imputer', pp.NumericalImputer(variables=config.NUMERICAL_FEATURES)),
            ('drop_columns', pp.DropColumns(variables_to_drop=config.DROP_FEATURES, reference_variable=config.TARGET)),
            ('scaler', MinMaxScaler()),
            ('reshape', ReshapeForLSTM(time_steps=1))
        ]
    )

    # Fit the Scikit-learn pipeline
    classification_pipeline.fit(X_train_prepared, train_y)

    # Save the Scikit-learn pipeline as a .pkl file
    pipeline_path = os.path.join(config.SAVE_MODEL_PATH, 'pollution.pkl')
    joblib.dump(classification_pipeline, pipeline_path)

    # Transform the data for the Keras model
    reshaped_X_train = classification_pipeline.named_steps['reshape'].transform(X_train_prepared)

    # Create and train the Keras model
    keras_model = create_lstm_model()
    keras_model.fit(reshaped_X_train, train_y, epochs=50, batch_size=1024)

    # Save the Keras model as an .h5 file
    keras_model_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    keras_model.save(keras_model_path)

if __name__ == "__main__":
    perform_training()
