# 🌍 Air Pollution Forecasting with Multivariate LSTM & MLOps Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![MLOps](https://img.shields.io/badge/MLOps-Complete-green.svg)](https://github.com/PAPPULASANDEEPKUMAR/Air-Pollution-Forecasting-LSTM-MLOps)

## 📊 **Research Foundation**
> **IEEE Publication**: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"  
> **DOI**: [10.1109/INOCON60754.2024.10511808](https://doi.org/10.1109/INOCON60754.2024.10511808)

*Production-ready air pollution forecasting system with 3-layer LSTM, complete MLOps pipeline, and Docker deployment.*

---

## 🎯 **What You'll Build**

- **LSTM Model**: 3-layer deep learning network with Batch Normalization
- **MLOps Pipeline**: Complete CI/CD with testing and packaging
- **Docker Deployment**: Containerized application ready for production
- **Web Interface**: Real-time predictions with visualizations
- **Professional Code**: Interview-ready repository structure

---

## 🚀 **Quick Start - 3 Ways to Run**

### 1️⃣ **Docker (Recommended - 2 minutes)**

```bash
# Clone repository
git clone https://github.com/PAPPULASANDEEPKUMAR/Air-Pollution-Forecasting-LSTM-MLOps.git
cd Air-Pollution-Forecasting-LSTM-MLOps

# Build and run
docker build -t air-pollution-forecasting .
docker run -p 5000:5000 air-pollution-forecasting

# Open browser
http://localhost:5000
```

### 2️⃣ **Local Development (5 minutes)**

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run application
python docker_train.py

# Access at http://localhost:5000
```

### 3️⃣ **MLOps Package (1 minute)**

```bash
pip install prediction-model
```

```python
from prediction_model.predict import make_prediction
result = make_prediction(input_data)
```

---

## 🏗️ **Project Structure - What Goes Where**

```
Air-Pollution-Forecasting-LSTM-MLOps/
├── 📄 docker_train.py              # Main application (run this)
├── 📄 Dockerfile                   # Docker configuration
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # This file
├── 📄 LICENSE                      # MIT License
├── 📄 .gitignore                   # Git ignore rules
├── 📁 data/
│   └── 📄 pollution.csv            # Dataset (7 features)
├── 📁 templates/
│   └── 📄 index.html               # Web interface
├── 📁 trained_models/
│   └── 📄 pollution_model.h5       # Trained LSTM model
├── 📁 .github/workflows/
│   └── 📄 main.yaml                # CI/CD pipeline
└── 📁 Air-Pollution-Forecasting/   # MLOps package
    └── 📁 Packaging-ML-Model/
        ├── 📁 prediction_model/    # Package source
        ├── 📁 tests/               # Test suite
        └── 📁 dist/                # Built packages
```

---

## 💻 **Step-by-Step Implementation**

### **Step 1: Setup Your Environment**

```bash
# Create project directory
mkdir Air-Pollution-Forecasting-LSTM-MLOps
cd Air-Pollution-Forecasting-LSTM-MLOps

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow keras scikit-learn pandas numpy matplotlib flask
```

### **Step 2: Create Main Application**

```python
# docker_train.py - Your main file
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.preprocessing import MinMaxScaler

# Your LSTM model architecture
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

# Run: python docker_train.py
```

### **Step 3: Create Docker Configuration**

```dockerfile
# Dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "docker_train.py"]
```

### **Step 4: Set Up MLOps Pipeline**

```yaml
# .github/workflows/main.yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.8
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest tests/ -v
    - name: Build Docker image
      run: docker build -t air-pollution-forecasting .
```

### **Step 5: Create Package Structure**

```bash
# Create MLOps package structure
mkdir -p Air-Pollution-Forecasting/Packaging-ML-Model/prediction_model
mkdir -p Air-Pollution-Forecasting/Packaging-ML-Model/tests
mkdir -p Air-Pollution-Forecasting/Packaging-ML-Model/prediction_model/config
mkdir -p Air-Pollution-Forecasting/Packaging-ML-Model/prediction_model/processing
mkdir -p Air-Pollution-Forecasting/Packaging-ML-Model/prediction_model/trained_models
```

---

## 🔧 **How to Customize for Your Project**

### **Change the Model Architecture**

```python
# In docker_train.py, modify build_model function
def build_model(input_shape):
    model = Sequential()
    # Change LSTM units (default: 100)
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    # Change dropout rate (default: 0.3)
    model.add(Dropout(0.2))
    # Add/remove layers as needed
    return model
```

### **Use Your Own Dataset**

```python
# Replace data/pollution.csv with your CSV file
# Update FEATURES list in docker_train.py
FEATURES = ['your_feature1', 'your_feature2', 'your_feature3']
TARGET = 'your_target_variable'
```

### **Modify Training Parameters**

```python
# In docker_train.py, change these values
BATCH_SIZE = 1024    # Default: 2048
EPOCHS = 50          # Default: 100
VALIDATION_SPLIT = 0.1  # Default: 0.2
```

---

## 🐳 **Docker Commands You'll Use**

```bash
# Build image
docker build -t air-pollution-forecasting .

# Run container
docker run -p 5000:5000 air-pollution-forecasting

# Run in background
docker run -d -p 5000:5000 --name pollution-app air-pollution-forecasting

# Stop container
docker stop pollution-app

# Remove container
docker rm pollution-app

# View logs
docker logs pollution-app

# Access container shell
docker exec -it pollution-app /bin/bash
```

---

## 🧪 **Testing Your Application**

### **Test the Web Interface**

```bash
# Start application
python docker_train.py

# Open browser and go to:
http://localhost:5000

# Test with sample data:
# pollution: 129.0, dew: -16.0, temp: -4.0
# pressure: 1016.0, w_speed: 1.79, snow: 0.0, rain: 0.0
```

### **Test API Endpoint**

```python
import requests

# Test prediction endpoint
data = {
    'pollution': 129.0,
    'dew': -16.0,
    'temp': -4.0,
    'pressure': 1016.0,
    'w_speed': 1.79,
    'snow': 0.0,
    'rain': 0.0
}

response = requests.post('http://localhost:5000/predict', data=data)
print(response.text)
```

### **Run Unit Tests**

```bash
# Install pytest
pip install pytest

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=prediction_model
```

---

## 📤 **How to Push to GitHub**

### **Step 1: Create Repository**

```bash
# On GitHub, create new repository:
# Name: Air-Pollution-Forecasting-LSTM-MLOps
# Description: IEEE Research Implementation with MLOps Pipeline
# Public/Private: Your choice
# Initialize: Don't initialize (you have local files)
```

### **Step 2: Initialize Git**

```bash
# In your project directory
git init
git add .
git commit -m "Initial commit: IEEE Air Pollution Forecasting System"
```

### **Step 3: Connect to GitHub**

```bash
# Add remote repository
git remote add origin https://github.com/PAPPULASANDEEPKUMAR/Air-Pollution-Forecasting-LSTM-MLOps.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### **Step 4: Verify Upload**

```bash
# Check your GitHub repository
# All files should be visible
# CI/CD pipeline should start automatically
```

---

## 🔍 **Troubleshooting Common Issues**

### **Docker Issues**

```bash
# Port already in use
docker run -p 5001:5000 air-pollution-forecasting

# Out of memory
docker run -m 4g -p 5000:5000 air-pollution-forecasting

# Permission denied
sudo docker run -p 5000:5000 air-pollution-forecasting
```

### **Python Issues**

```bash
# Module not found
pip install -r requirements.txt

# TensorFlow issues
pip install tensorflow==2.12.0

# Flask issues
export FLASK_APP=docker_train.py
```

### **Git Issues**

```bash
# Authentication failed
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Large files
git lfs track "*.h5"
git add .gitattributes
```

---

## 🎯 **Model Performance Details**

### **Architecture Overview**
- **Input**: 7 features (pollution, dew, temp, pressure, w_speed, snow, rain)
- **LSTM Layers**: 3 layers with 100 units each
- **Regularization**: Dropout (0.3) + Batch Normalization
- **Output**: Single pollution value prediction

### **Training Configuration**
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error
- **Batch Size**: 2048
- **Epochs**: 100
- **Validation Split**: 20%

### **Performance Metrics**
- **RMSE**: Calculated and displayed in real-time
- **Training/Validation Loss**: Plotted automatically
- **Prediction Accuracy**: Visual comparison graphs

---

## 🔧 **Environment Configuration**

### **Required Dependencies**

```txt
# Core requirements
tensorflow==2.12.0
keras==2.12.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
flask==2.3.3
```

### **Environment Variables**

```bash
# Optional configuration
export FLASK_HOST=0.0.0.0
export FLASK_PORT=5000
export MODEL_NAME=pollution_model.h5
export BATCH_SIZE=2048
export EPOCHS=100
```

---

## 🏆 **IEEE Research Implementation**

### **Published Paper Details**
- **Title**: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"
- **Authors**: Tirumala Manav, et al.
- **Conference**: 2024 3rd International Conference for Innovation in Technology (INOCON)
- **DOI**: [10.1109/INOCON60754.2024.10511808](https://doi.org/10.1109/INOCON60754.2024.10511808)

### **Key Research Contributions**
- **Batch Normalization**: Improves LSTM training stability
- **Multivariate Analysis**: Uses 7 environmental factors
- **Production Implementation**: Complete MLOps pipeline
- **Real-world Application**: Deployable air quality system

---

## 🤝 **Contributing & Support**

### **How to Contribute**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Create Pull Request

### **Get Help**
- 📋 **Issues**: [GitHub Issues](https://github.com/PAPPULASANDEEPKUMAR/Air-Pollution-Forecasting-LSTM-MLOps/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/PAPPULASANDEEPKUMAR/Air-Pollution-Forecasting-LSTM-MLOps/discussions)

---

## 📞 **Contact**

**Tirumala Manav**
- 🐙 GitHub: [@PAPPULASANDEEPKUMAR](https://github.com/PAPPULASANDEEPKUMAR)
- 📧 Email: Contact via GitHub Issues

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 **Star this repository if you find it useful!** 🌟

**Made with ❤️ by [Tirumala Manav](https://github.com/PAPPULASANDEEPKUMAR)**

</div>

---

*Professional IEEE research implementation with complete MLOps pipeline - ready for technical interviews and production deployment.*
