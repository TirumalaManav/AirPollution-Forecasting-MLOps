#Importing Libraries
import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to prevent tkinter-related errors
import matplotlib.pyplot as plt
import io
import base64

# import logging

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Constants
TRAIN_FILE = 'data/pollution.csv'
MODEL_NAME = 'pollution_model.h5'
SAVE_MODEL_PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))), 'trained_models')
TARGET = 'pollution'
FEATURES = ['pollution', 'dew', 'temp', 'pressure', 'w_speed', 'snow', 'rain']

# Flask app initialization
app = Flask(__name__)

# Model training function
def build_model(input_shape):
    model = Sequential()
    
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(LSTM(units=100))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Load data
def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Preprocess data
def preprocess_data(df, features, target):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[features])
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape for LSTM (samples, timesteps, features)
    y = df[target].values
    return X, y, scaler

plot_urls = None

# Train the model and serve it via Flask
def train_and_serve():
    global plot_urls
    df = load_data(TRAIN_FILE)
    if df is None:
        return

    X, y, scaler = preprocess_data(df, FEATURES, TARGET)
    
    model = build_model((X.shape[1], X.shape[2]))  # Input shape (timesteps, features)
    
    history = model.fit(X, y, epochs=100, batch_size=2048, verbose=1, validation_split=0.2)
    
    model_path = os.path.join(SAVE_MODEL_PATH, MODEL_NAME)
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    model.save(model_path)
    
    predictions = model.predict(X)
    plot_urls = plot_predictions_and_evaluate(y, predictions, history)

    # Serve the model via Flask
    serve_model(model_path, plot_urls)

# Plot predictions and evaluate
def plot_predictions_and_evaluate(actual_values, predicted_values, history):
    # Prediction vs Actual Plot
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_values[:100], color='green', label='Predicted Pollution Level')
    plt.plot(actual_values[:100], color='red', label='Actual Pollution Level')
    plt.title("Air Pollution Prediction (Multivariate)")
    plt.xlabel("Sample Index")
    plt.ylabel("Pollution Level")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    prediction_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    # Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    loss_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    # RMSE Plot (Accuracy Equivalent)
    rmse_per_epoch = [np.sqrt(mse) for mse in history.history['loss']]
    plt.figure(figsize=(10, 6))
    plt.plot(rmse_per_epoch, label='Training RMSE')
    plt.title('Model RMSE (Training Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    rmse_plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    # Calculate and print final RMSE for the whole dataset
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    print('Final RMSE:', rmse)
    
    return {'prediction_plot_url': prediction_plot_url, 
            'loss_plot_url': loss_plot_url, 
            'rmse_plot_url': rmse_plot_url, 
            'rmse': rmse}

# Preprocess input for the web app
def preprocess_input(data, scaler):
    data = scaler.transform(data)
    data = data.reshape((data.shape[0], 1, data.shape[1]))  # Reshape for LSTM (samples, timesteps, features)
    return data

# Flask route to serve predictions
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in FEATURES]
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        processed_data = preprocess_input(input_df, scaler)

        model = load_model(MODEL_PATH)
        prediction = model.predict(processed_data)
        prediction_value = prediction[0][0]

        return render_template('index.html', 
                               prediction_text=f'Predicted Pollution Level: {prediction_value:.2f}',
                               rmse_text=f'Final RMSE: {plot_urls["rmse"]:.2f}',
                               prediction_plot_url=plot_urls['prediction_plot_url'],
                               loss_plot_url=plot_urls['loss_plot_url'],
                               rmse_plot_url=plot_urls['rmse_plot_url'])
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error occurred: {str(e)}')

# Serve model using Flask
def serve_model(model_path, plot_urls):
    global MODEL_PATH, scaler
    MODEL_PATH = model_path
    # Reload scaler to ensure it's available in Flask route
    df = load_data(TRAIN_FILE)
    _, _, scaler = preprocess_data(df, FEATURES, TARGET)
    app.run(host='0.0.0.0', port=5000)



# def serve_model(model_path, plot_urls):
#     global MODEL_PATH, scaler
#     MODEL_PATH = model_path
#     df = load_data(TRAIN_FILE)
#     _, _, scaler = preprocess_data(df, FEATURES, TARGET)
    
#     # Modified to ensure proper Docker networking
#     app.run(
#         host='0.0.0.0',  # Binds to all interfaces
#         port=5000,
#         debug=False,     # Disable debug mode in production
#         use_reloader=False  # Prevent double execution in Docker
#     )

def main():
    train_and_serve()  # Train the model and start the Flask server

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     logger.info("Starting Flask application...")
#     try:
#         app.run(
#             host='0.0.0.0',  # This is crucial for Docker
#             port=5000,
#             debug=True,      # Enable debug mode for more information
#             use_reloader=False  # Disable reloader in Docker
#         )
#     except Exception as e:
#         logger.error(f"Failed to start Flask app: {e}")



#Author: Tirumala Manav
