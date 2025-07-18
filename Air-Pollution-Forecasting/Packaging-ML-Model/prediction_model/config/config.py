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

import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TRAIN_FILE = 'pollution.csv'
TEST_FILE = 'pollution.csv'

MODEL_NAME = 'pollution_model.h5'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TARGET = 'pollution'

FEATURES = ['pollution','dew', 'temp', 'pressure', 'w_speed', 'snow', 'rain']


NUMERICAL_FEATURES = ['pollution', 'dew', 'temp', 'pressure', 'w_speed', 'snow', 'rain']

CATEGORICAL_FEATURES = []

FEATURES_TO_ENCODE = []  # Encoding typically applies to categorical features

DROP_FEATURES = ['dew', 'temp', 'pressure', 'w_speed', 'snow', 'rain']

