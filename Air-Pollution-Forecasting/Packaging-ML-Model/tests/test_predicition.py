import pytest
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import numpy as np

# Set up the path to include the project root directory
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# Importing the necessary modules
from prediction_model.config import config
from prediction_model.processing.data_handling import load_dataset
from prediction_model.predict import make_prediction

def test_make_prediction():
    # Given
    test_data = load_dataset(config.TEST_FILE)  # Load the dataset from the config
    input_data = test_data[config.FEATURES]
    expected_prediction = test_data[config.TARGET]
    
    # When
    prediction = make_prediction(input_data)
    
    # Generate and save the plot
    plt.figure(figsize=(10, 6))
    plt.plot(prediction, color='green', label='Predicted Pollution Level')
    plt.plot(expected_prediction.values, color='red', label='Actual Pollution Level')
    plt.title("Air Pollution Prediction vs Actual")
    plt.xlabel("Index")
    plt.ylabel("Pollution Level")
    plt.legend()
    plt.savefig('prediction_vs_actual.png')
    plt.close()
    
    # Optionally, assert that the predictions are reasonable (e.g., within some tolerance)
    assert len(prediction) == len(expected_prediction), "Prediction and actual values length mismatch"
    assert prediction is not None, "Prediction is None"
    assert not np.isnan(prediction).any(), "Prediction contains NaN values"
    assert prediction.size > 0, "Prediction is empty"
    
    # Additional assertions can be added based on specific criteria
    # For example, checking if predictions are within a reasonable range of expected values
    # assert (prediction - expected_prediction).abs().max() < threshold, "Predictions out of expected range"

if __name__ == "__main__":
    pytest.main()
