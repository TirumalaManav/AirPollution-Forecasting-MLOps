# 🌍 Air Pollution Forecasting with Multivariate LSTM & MLOps Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Available-blue.svg)](https://www.docker.com/)
[![MLOps](https://img.shields.io/badge/MLOps-Complete-green.svg)](https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps)

## 📊 **Research Foundation**
> **IEEE Publication**: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"  
> **DOI**: [10.1109/INOCON60754.2024.10511808](https://doi.org/10.1109/INOCON60754.2024.10511808)  
> **Conference**: 2024 3rd International Conference for Innovation in Technology (INOCON)  
> **Author**: Tirumala Manav

*Production-ready air pollution forecasting system implementing IEEE research with complete MLOps pipeline and Docker deployment capabilities.*

---

## 🎯 **What This Project Delivers**

- **🧠 Advanced LSTM Model**: 3-layer architecture with Batch Normalization
- **📊 Multivariate Analysis**: 7 environmental features for accurate prediction
- **🔄 Complete MLOps Pipeline**: CI/CD, testing, packaging, and deployment
- **🐳 Docker Deployment**: Production-ready containerized application
- **🌐 Web Interface**: Real-time predictions with interactive visualizations
- **📈 Performance Monitoring**: RMSE tracking and model evaluation

---

## 🔬 **Research Background & Theory**

### **Problem Statement**
Air pollution has become a significant environmental concern affecting public health and air quality standards. Traditional forecasting methods often fail to capture the complex temporal dependencies and multivariate relationships in environmental data. This research addresses these limitations by implementing a sophisticated LSTM-based approach.

### **Key Research Contributions**
Based on the IEEE publication, this implementation provides:

1. **Batch Normalization Integration**: Stabilizes LSTM training and accelerates convergence by normalizing layer inputs
2. **Multivariate Time Series Analysis**: Incorporates seven environmental factors (pollution, dew, temp, pressure, wind speed, snow, rain) for comprehensive prediction
3. **Long Short-Term Memory Networks**: Captures long-term dependencies crucial for accurate air quality forecasting
4. **Production Implementation**: Transforms research into deployable system with MLOps best practices

### **Technical Innovation**
- **LSTM Architecture**: Addresses vanishing gradient problem in traditional RNNs
- **Batch Normalization**: Reduces internal covariate shift, enabling faster training
- **Multivariate Approach**: Considers interdependencies between environmental variables
- **Regularization**: Dropout layers prevent overfitting in deep networks

---

## 🏗️ **System Architecture**

### **📊 Model Architecture Flowchart**

```
                    ┌─────────────────────────────────────┐
                    │         INPUT LAYER                 │
                    │   7 Features: pollution, dew,      │
                    │   temp, pressure, w_speed,         │
                    │   snow, rain                        │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │    DATA PREPROCESSING               │
                    │  • MinMaxScaler (0-1 range)        │
                    │  • Reshape for LSTM input           │
                    │  • Sequence preparation             │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     LSTM LAYER 1                    │
                    │  • 100 units                        │
                    │  • return_sequences=True            │
                    │  • Input shape: (timesteps, 7)     │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     DROPOUT LAYER 1                 │
                    │  • Rate: 0.3                        │
                    │  • Regularization                   │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │   BATCH NORMALIZATION 1             │
                    │  • Normalize layer inputs           │
                    │  • Accelerate convergence           │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     LSTM LAYER 2                    │
                    │  • 100 units                        │
                    │  • return_sequences=True            │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     DROPOUT LAYER 2                 │
                    │  • Rate: 0.3                        │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │   BATCH NORMALIZATION 2             │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     LSTM LAYER 3                    │
                    │  • 100 units                        │
                    │  • return_sequences=False           │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     DROPOUT LAYER 3                 │
                    │  • Rate: 0.3                        │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │   BATCH NORMALIZATION 3             │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │      DENSE OUTPUT LAYER             │
                    │  • 1 unit (pollution prediction)    │
                    │  • Linear activation                │
                    └─────────────┬───────────────────────┘
                                  │
                    ┌─────────────▼───────────────────────┐
                    │     PREDICTION OUTPUT               │
                    │  • Single pollution value           │
                    │  • RMSE evaluation                  │
                    └─────────────────────────────────────┘
```

### **🔄 MLOps Pipeline Architecture**

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                            MLOPS PIPELINE                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
            │
    ┌───────▼──────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │     DATA     │    │    MODEL     │    │   TESTING    │    │  DEPLOYMENT  │
    │  INGESTION   │───▶│  TRAINING    │───▶│ & VALIDATION │───▶│ & SERVING    │
    │              │    │              │    │              │    │              │
    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
            │                    │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │ pollution.csv│    │ LSTM Training│    │ pytest Suite│    │ Docker Image │
    │ 7 features   │    │ 100 epochs   │    │ Unit Tests   │    │ Flask App    │
    │ Preprocessing│    │ Batch: 2048  │    │ Integration  │    │ Port: 5000   │
    │ MinMaxScaler │    │ Adam Optim.  │    │ Coverage     │    │ Health Check │
    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
            │                    │                    │                    │
    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐    ┌───────▼──────┐
    │ Data Quality │    │ Model Saving │    │ Performance  │    │ Monitoring   │
    │ Validation   │    │ Versioning   │    │ Metrics      │    │ & Logging    │
    │ Schema Check │    │ Artifacts    │    │ RMSE Track   │    │ Alerts       │
    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
            │                    │                    │                    │
            └────────────────────┼────────────────────┼────────────────────┘
                                 │                    │
                    ┌────────────▼────────────────────▼───────────┐
                    │         GITHUB ACTIONS CI/CD              │
                    │  • Automated Testing                      │
                    │  • Docker Build & Push                    │
                    │  • Model Validation                       │
                    │  • Deployment Automation                  │
                    └───────────────────────────────────────────┘
```

---

## 🚀 **Quick Start Guide**

### **🐳 Docker Deployment (Recommended)**
```bash
# Clone repository
git clone https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps.git
cd AirPollution-Forecasting-MLOps

# Build and run
docker build -t air-pollution-forecasting .
docker run -p 5000:5000 air-pollution-forecasting

# Access: http://localhost:5000
```

### **💻 Local Development**
```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run application
python docker_train.py
```

### **📦 MLOps Package**
```bash
# Install package
pip install prediction-model

# Use in Python
from prediction_model.predict import make_prediction
```

---

## 🏗️ **Complete Project Structure**

```
AirPollution-Forecasting-MLOps/
├── 📄 README.md                                    # Project documentation
├── 📄 LICENSE                                      # MIT License
├── 📄 .gitignore                                   # Git ignore rules
├── 📄 docker_train.py                             # Main application file
├── 📄 Dockerfile                                  # Docker configuration
├── 📄 requirements.txt                            # Python dependencies
├── 📄 docker-compose.yml                          # Development environment
├── 📄 setup.py                                    # Package installation
├── 📄 MANIFEST.in                                 # Package manifest
├── 📄 .dockerignore                               # Docker ignore rules
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 main.yaml                           # CI/CD pipeline
├── 📁 data/
│   ├── 📄 pollution.csv                           # Training dataset
│   └── 📄 README.md                               # Data documentation
├── 📁 templates/
│   ├── 📄 index.html                              # Web interface
│   └── 📄 base.html                               # Base template
├── 📁 static/
│   ├── 📁 css/
│   │   └── 📄 style.css                           # Styling
│   ├── 📁 js/
│   │   └── 📄 main.js                             # JavaScript
│   └── 📁 images/
│       └── 📄 logo.png                            # Assets
├── 📁 trained_models/
│   ├── 📄 pollution_model.h5                      # Trained LSTM model
│   └── 📄 model_metadata.json                     # Model information
├── 📁 config/
│   ├── 📄 __init__.py
│   ├── 📄 config.py                               # Configuration settings
│   └── 📄 logging.conf                            # Logging configuration
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 model_training.py                       # Training pipeline
│   ├── 📄 data_preprocessing.py                   # Data processing
│   ├── 📄 model_evaluation.py                     # Model evaluation
│   └── 📄 utils.py                                # Utility functions
├── 📁 Air-Pollution-Forecasting/
│   └── 📁 Packaging-ML-Model/
│       ├── 📄 setup.py                            # Package setup
│       ├── 📄 pyproject.toml                      # Build configuration
│       ├── 📄 README.md                           # Package documentation
│       ├── 📁 prediction_model/
│       │   ├── 📄 __init__.py
│       │   ├── 📄 VERSION                         # Version file
│       │   ├── 📄 pipeline.py                     # ML pipeline
│       │   ├── 📄 predict.py                      # Prediction module
│       │   ├── 📄 training_pipeline.py            # Training pipeline
│       │   ├── 📄 graph.png                       # Model visualization
│       │   ├── 📁 config/
│       │   │   ├── 📄 __init__.py
│       │   │   └── 📄 config.py                   # Package config
│       │   ├── 📁 datasets/
│       │   │   ├── 📄 __init__.py
│       │   │   └── 📄 pollution.csv               # Package data
│       │   ├── 📁 processing/
│       │   │   ├── 📄 __init__.py
│       │   │   ├── 📄 data_handling.py            # Data utilities
│       │   │   ├── 📄 preprocessing.py            # Preprocessing
│       │   │   └── 📄 reshape_transformer.py      # Data transformation
│       │   └── 📁 trained_models/
│       │       ├── 📄 __init__.py
│       │       ├── 📄 pollution.pkl               # Serialized model
│       │       └── 📄 pollution_model.h5          # Keras model
│       ├── 📁 tests/
│       │   ├── 📄 __init__.py
│       │   ├── 📄 conftest.py                     # Test configuration
│       │   ├── 📄 pytest.ini                      # pytest settings
│       │   ├── 📄 test_prediction.py              # Prediction tests
│       │   ├── 📄 test_data_handling.py           # Data tests
│       │   ├── 📄 test_preprocessing.py           # Preprocessing tests
│       │   └── 📄 prediction_vs_actual.png        # Test results
│       ├── 📁 build/                              # Build artifacts
│       ├── 📁 dist/                               # Distribution packages
│       │   ├── 📄 prediction_model-1.0.0.tar.gz  # Source distribution
│       │   └── 📄 prediction_model-1.0.0-py3-none-any.whl  # Wheel
│       └── 📁 prediction_model.egg-info/          # Package metadata
├── 📁 tests/
│   ├── 📄 __init__.py
│   ├── 📄 conftest.py                             # Test configuration
│   ├── 📄 test_main.py                            # Main app tests
│   ├── 📄 test_model.py                           # Model tests
│   ├── 📄 test_api.py                             # API tests
│   └── 📄 test_docker.py                          # Docker tests
├── 📁 docs/
│   ├── 📄 API.md                                  # API documentation
│   ├── 📄 DEPLOYMENT.md                           # Deployment guide
│   ├── 📄 DEVELOPMENT.md                          # Development guide
│   └── 📄 ARCHITECTURE.md                         # Architecture details
├── 📁 scripts/
│   ├── 📄 train_model.py                          # Training script
│   ├── 📄 evaluate_model.py                       # Evaluation script
│   ├── 📄 deploy.py                               # Deployment script
│   └── 📄 run_tests.py                            # Test runner
└── 📁 logs/
    ├── 📄 app.log                                 # Application logs
    ├── 📄 training.log                            # Training logs
    └── 📄 error.log                               # Error logs
```

---

## 🎯 **Technical Specifications**

### **Model Architecture**
- **Type**: 3-Layer LSTM with Batch Normalization
- **Input Features**: 7 environmental variables
- **LSTM Units**: 100 per layer
- **Dropout Rate**: 0.3 for regularization
- **Optimizer**: Adam with MSE loss
- **Training**: 100 epochs, batch size 2048

### **Performance Metrics**
- **RMSE**: Root Mean Square Error tracking
- **Training/Validation Loss**: Convergence monitoring
- **Prediction Accuracy**: Visual comparison plots
- **Model Evaluation**: Real-time performance assessment

### **MLOps Components**
- **CI/CD Pipeline**: GitHub Actions automation
- **Testing**: Comprehensive pytest suite
- **Packaging**: Professional Python package
- **Monitoring**: Performance tracking and logging
- **Deployment**: Docker containerization

---

## 🔧 **Environment Setup**

### **Prerequisites**
- Python 3.8+
- Docker & Docker Compose
- Git

### **Installation Steps**
```bash
# Clone repository
git clone https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps.git

# Setup environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 **Testing & Validation**

### **Test Categories**
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Model accuracy and speed
- **Docker Tests**: Container deployment verification

### **Run Tests**
```bash
# Complete test suite
pytest tests/ -v --cov=prediction_model

# Specific test categories
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

---

## 📤 **GitHub Deployment**

### **Repository Setup**
```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit: IEEE Air Pollution Forecasting System"

# Connect to GitHub
git remote add origin https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps.git
git branch -M main
git push -u origin main
```

### **Automated CI/CD**
- GitHub Actions workflow automatically triggers on push
- Runs tests, builds Docker image, and validates deployment
- Provides continuous integration and deployment capabilities

---

## 🏆 **IEEE Research Implementation**

### **Research Paper Details**
- **Title**: "Multivariate Time Series Analysis and Batch Normalization for Air Quality Prediction in Long Short-Term Memory Networks"
- **Author**: Tirumala Manav
- **Conference**: 2024 3rd International Conference for Innovation in Technology (INOCON)
- **DOI**: [10.1109/INOCON60754.2024.10511808](https://doi.org/10.1109/INOCON60754.2024.10511808)

### **Key Contributions**
- Novel application of Batch Normalization in LSTM networks for air quality prediction
- Comprehensive multivariate analysis using environmental factors
- Production-ready implementation with MLOps best practices
- Scalable architecture for real-world deployment

---

## 🤝 **Contributing**

### **How to Contribute**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m 'Add new feature'`
4. Push branch: `git push origin feature/new-feature`
5. Create Pull Request

### **Development Guidelines**
- Follow PEP 8 coding standards
- Add comprehensive tests for new features
- Update documentation accordingly
- Maintain 90%+ test coverage

---

## 📞 **Contact & Support**

### **Author**
**Tirumala Manav**
- 🐙 GitHub: [@TirumalaManav](https://github.com/TirumalaManav)
- 📧 Email: Contact via GitHub Issues
- 🔗 Repository: [AirPollution-Forecasting-MLOps](https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps)

### **Support Channels**
- 📋 **Issues**: [GitHub Issues](https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/TirumalaManav/AirPollution-Forecasting-MLOps/discussions)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- IEEE INOCON 2024 for publishing the research
- Open Source Community for tools and frameworks
- Environmental Science Community for data insights
- MLOps Community for best practices

---

<div align="center">

### 🌟 **Star this repository if you find it useful!** 🌟

**Made with ❤️ by [Tirumala Manav](https://github.com/TirumalaManav)**

</div>

---

*This project represents a complete implementation of IEEE research with professional MLOps practices, suitable for academic citations, technical interviews, and production deployment.*
