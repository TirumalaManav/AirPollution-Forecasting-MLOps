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

import os
import joblib
import tensorflow as tf
import pandas as pd
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config

# Loading the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data

# Serialization for Keras models
def save_model(model_to_save, model_name="model.h5"):
    save_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
    model_to_save.save(save_path)  # Save the Keras model as .h5
    print(f"Model has been saved at: {save_path}")

# Deserialization for Keras models
def load_model(model_name="model.h5"):
    load_path = os.path.join(config.SAVE_MODEL_PATH, model_name)
    print(f"Loading model from: {load_path}")
    model_loaded = tf.keras.models.load_model(load_path)  # Load the Keras model
    return model_loaded

# Serialization for scikit-learn pipelines
def save_pipeline(pipeline_to_save, pipeline_name="pipeline.pkl"):
    save_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_name)
    joblib.dump(pipeline_to_save, save_path)  # Save the scikit-learn pipeline
    print(f"Pipeline has been saved at: {save_path}")

# Deserialization for scikit-learn pipelines
def load_pipeline(pipeline_name="pipeline.pkl"):
    load_path = os.path.join(config.SAVE_MODEL_PATH, pipeline_name)
    print(f"Loading pipeline from: {load_path}")
    pipeline_loaded = joblib.load(load_path)  # Load the scikit-learn pipeline
    return pipeline_loaded

