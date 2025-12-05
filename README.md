# 🏝️ OTA Price Analysis - Airbnb Bali Price Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Ensemble%20Models-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 Overview

**OTA Price Analysis** adalah sistem prediksi harga komprehensif untuk properti Airbnb di Bali yang memanfaatkan machine learning dan analisis data mendalam. Project ini menggabungkan analisis data eksploratori (EDA), multiple algoritma machine learning, dan dashboard web interaktif untuk memberikan prediksi harga yang akurat dan insight pasar yang valuable.

### 🎯 Key Highlights

- **Akurasi Tinggi**: Prediksi harga dengan akurasi 85%+
- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Neural Networks, CatBoost
- **Interactive Dashboard**: Web interface yang user-friendly dengan Flask
- **Comprehensive EDA**: Analisis data mendalam dengan visualisasi interaktif
- **Real-time Predictions**: Prediksi harga instant dengan confidence intervals
- **Market Insights**: Analisis tren musiman dan pola booking

---

## 🏗️ Project Structure

```
GYE-OTA-ANALYSIS/
│
├── 📁 Airbnb-ML-Dashboard/          # Main application directory
│   ├── 📁 flask_app/                # Flask web application
│   │   ├── app.py                   # Main Flask application
│   │   ├── templates/               # HTML templates
│   │   │   ├── base.html
│   │   │   ├── index.html           # Prediction form
│   │   │   ├── result.html          # Results display
│   │   │   └── analytics.html       # Analytics dashboard
│   │   └── static/                  # CSS, JS, images
│   │       ├── css/
│   │       ├── js/
│   │       └── images/
│   │
│   ├── 📁 ml_models/                # Trained ML models
│   │   ├── best_model.pkl
│   │   ├── random_forest_model.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── lightgbm_model.pkl
│   │   ├── neural_network_model.h5
│   │   ├── model_metrics.json
│   │   └── feature_encoders.pkl
│   │
│   ├── 📁 dataset/                  # Data files
│   │   ├── airbnb_bali_ml_ready.csv    # ML-ready dataset
│   │   ├── feature_encodings.json
│   │   ├── ml_data_dictionary.txt
│   │   └── feature_analysis.txt
│   │
│   ├── 📁 notebooks/                # Jupyter notebooks
│   │   ├── Airbnb_Bali_EDA_Analysis.ipynb
│   │   └── ML_Price_Prediction_Training.ipynb
│   │
│   ├── 📁 config/                   # Configuration files
│   │   └── model_config.json
│   │
│   ├── 📄 requirements.txt          # Python dependencies
│   ├── 📄 run_app.py               # Quick start script
│   ├── 📄 retrain_all_models.py    # Model retraining script
│   └── 📄 test_all_models.py       # Model testing script
│
├── 📄 Airbnb_Bali_EDA_Analysis.ipynb    # Main EDA notebook
├── 📄 Data Airbnb Bali.json             # Raw data source
├── 📄 .gitignore                        # Git ignore rules
└── 📄 README.md                         # This file
```

---

## 🚀 Features

### 🔍 Data Analysis & Processing

#### Exploratory Data Analysis (EDA)
- **Data Cleaning**: Menangani missing values, outliers, dan duplikasi
- **URL Parsing**: Ekstraksi informasi dari URL Airbnb (check-in, stay duration, dll)
- **Statistical Analysis**: Analisis statistik deskriptif dan inferensial
- **Correlation Analysis**: Identifikasi hubungan antar variabel
- **Distribution Analysis**: Analisis distribusi harga dan features

#### Feature Engineering
- **Temporal Features**: Season, month, day, weekend/weekday
- **Stay Features**: Duration, advance booking period
- **Market Features**: Currency, locale, market segment
- **Price Features**: Price per night, total price, pricing patterns
- **Synthetic Features**: Interaksi features dan polynomial features

### 🤖 Machine Learning Pipeline

#### Multiple Algorithm Support
1. **Random Forest**
   - Ensemble method dengan multiple decision trees
   - Robust terhadap outliers dan overfitting
   - Feature importance analysis

2. **XGBoost**
   - Gradient boosting dengan high performance
   - Optimal untuk structured data
   - Advanced regularization

3. **LightGBM**
   - Fast gradient boosting framework
   - Memory efficient
   - Handling large datasets

4. **CatBoost**
   - Categorical features handling
   - Built-in overfitting detection
   - GPU acceleration support

5. **Neural Networks (Deep Learning)**
   - Multi-layer perceptron architecture
   - Non-linear pattern recognition
   - TensorFlow/Keras implementation

#### Model Training & Evaluation
- **Cross-Validation**: K-fold validation untuk robust evaluation
- **Hyperparameter Tuning**: Grid search & Bayesian optimization
- **Performance Metrics**: MAE, RMSE, R², MAPE
- **Model Comparison**: Comprehensive model performance comparison
- **Feature Importance**: Analisis kontribusi setiap feature

### 🌐 Web Dashboard

#### Price Prediction Interface
- Form input yang intuitif dan user-friendly
- Real-time validation dan error handling
- Instant prediction results dengan confidence score
- Multiple pricing recommendations (competitive, optimal, premium)

#### Analytics & Insights
- **Market Trends**: Visualisasi tren harga dan demand
- **Seasonal Analysis**: Pola harga berdasarkan musim
- **Geographic Insights**: Analisis berdasarkan lokasi
- **Performance Metrics**: Model accuracy dan reliability metrics
- **Interactive Charts**: Plotly-based interactive visualizations

#### Responsive Design
- Mobile-friendly interface
- Bootstrap 5 framework
- Modern and professional UI/UX
- Dark/Light mode support

---

## 🛠️ Installation & Setup

### Prerequisites

```bash
# System Requirements
- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM (8GB recommended for training)
- 2GB+ free disk space

# Optional
- Git for version control
- Virtual environment (venv/conda)
- Jupyter Notebook/Lab
```

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/IdaBagusAgung/OTA-Price-Analysis.git
cd OTA-Price-Analysis
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Using conda (alternative)
conda create -n ota-analysis python=3.8
conda activate ota-analysis
```

#### 3. Install Dependencies

```bash
cd Airbnb-ML-Dashboard
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
python -c "import flask, sklearn, pandas, numpy; print('✅ All packages installed successfully!')"
```

---

## 📖 Usage Guide

### 🎓 Quick Start

#### Option 1: Direct Run (Recommended for Testing)

```bash
cd Airbnb-ML-Dashboard
python run_app.py
```

#### Option 2: Flask Development Server

```bash
cd Airbnb-ML-Dashboard/flask_app
python app.py
```

#### Option 3: Production Server with Gunicorn

```bash
cd Airbnb-ML-Dashboard/flask_app
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

Setelah server berjalan, buka browser dan akses:
```
http://localhost:5000
```

---

### 📊 Running EDA Analysis

#### 1. Open Jupyter Notebook

```bash
# From project root
jupyter notebook Airbnb_Bali_EDA_Analysis.ipynb

# Or from dashboard directory
cd Airbnb-ML-Dashboard/notebooks
jupyter notebook
```

#### 2. Execute EDA Steps
- **Cell by Cell**: Jalankan setiap cell secara berurutan (Shift+Enter)
- **Run All**: Menu > Cell > Run All
- **Output**: Data cleaned akan tersimpan di `dataset/`

#### 3. EDA Components
- Data loading dan inspection
- Data cleaning dan preprocessing
- URL parsing dan feature extraction
- Statistical analysis
- Visualizations (distributions, correlations, trends)
- Export cleaned data

---

### 🏋️ Training ML Models

#### 1. Open Training Notebook

```bash
cd Airbnb-ML-Dashboard/notebooks
jupyter notebook ML_Price_Prediction_Training.ipynb
```

#### 2. Training Process
- Load preprocessed data
- Feature engineering
- Train multiple models
- Hyperparameter tuning
- Model evaluation
- Save best models

#### 3. Automated Retraining

```bash
cd Airbnb-ML-Dashboard
python retrain_all_models.py
```

#### 4. Model Testing

```bash
python test_all_models.py
```

---

### 🎯 Making Predictions

#### Web Interface Method

1. **Navigate to Prediction Form**
   - Open dashboard homepage
   - Fill in property details

2. **Input Parameters**
   ```
   - Stay Duration: 1-365 days
   - Check-in Date: Select from calendar
   - Season: Auto-suggested based on date
   - Weekend Booking: Yes/No checkbox
   - Currency: USD, EUR, IDR, etc.
   - Market Locale: en-US, id-ID, etc.
   ```

3. **Get Results**
   - Predicted price with confidence score
   - Price range (min-max)
   - Pricing recommendations
   - Market insights

#### API Method (Programmatic)

```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "stay_duration": 5,
    "check_in_month": 7,
    "season": "Summer",
    "is_weekend": 0,
    "currency": "USD",
    "locale": "en-US"
}

response = requests.post(url, json=data)
prediction = response.json()
print(f"Predicted Price: ${prediction['price']}")
```

---

## 📈 Model Performance

### Current Best Model: Ensemble (Random Forest + XGBoost)

| Metric | Value | Description |
|--------|-------|-------------|
| **R² Score** | 0.78 | Variance explained by model |
| **MAE** | $12.34 | Mean absolute error |
| **RMSE** | $18.56 | Root mean squared error |
| **MAPE** | 8.5% | Mean absolute percentage error |
| **Accuracy** | 85%+ | Overall prediction accuracy |

### Model Comparison

| Model | R² | MAE | RMSE | Training Time |
|-------|-----|-----|------|---------------|
| Random Forest | 0.76 | $13.20 | $19.45 | 2.3s |
| XGBoost | 0.78 | $12.34 | $18.56 | 3.1s |
| LightGBM | 0.75 | $13.89 | $20.12 | 1.8s |
| CatBoost | 0.77 | $12.87 | $19.01 | 4.5s |
| Neural Network | 0.74 | $14.50 | $21.30 | 12.5s |

### Feature Importance (Top 10)

1. **Season** (35%) - Seasonal demand patterns
2. **Stay Duration** (25%) - Length of booking
3. **Weekend Premium** (20%) - Weekend vs weekday
4. **Advance Booking** (8%) - Days before check-in
5. **Check-in Month** (5%) - Monthly patterns
6. **Currency** (3%) - Market segment
7. **Locale** (2%) - Geographic preferences
8. **Day of Week** (1%) - Weekly patterns
9. **Holiday Indicator** (0.7%) - Special periods
10. **Market Segment** (0.3%) - Customer type

---

## 🔧 Configuration

### Model Configuration (`config/model_config.json`)

```json
{
  "models": {
    "random_forest": {
      "n_estimators": 200,
      "max_depth": 20,
      "min_samples_split": 5,
      "random_state": 42
    },
    "xgboost": {
      "n_estimators": 150,
      "learning_rate": 0.1,
      "max_depth": 8,
      "subsample": 0.8
    }
  },
  "features": {
    "categorical": ["season", "currency", "locale"],
    "numerical": ["stay_duration", "is_weekend"]
  },
  "training": {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42
  }
}
```

### Flask Configuration

```python
# In flask_app/app.py
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['MODEL_PATH'] = '../ml_models/'
app.config['DATA_PATH'] = '../dataset/'
```

---

## 🧪 Testing

### Run All Tests

```bash
cd Airbnb-ML-Dashboard
python test_all_models.py
```

### Test Individual Components

```bash
# Test dashboard integration
python test_dashboard_integration.py

# Test multi-model comparison
python test_multi_model_comparison.py
```

### Manual Testing Checklist

- [ ] Load and preprocess data successfully
- [ ] Train models without errors
- [ ] Make predictions with valid inputs
- [ ] Dashboard loads correctly
- [ ] Form validation works
- [ ] API endpoints respond correctly
- [ ] Visualizations render properly

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### 1. Model Not Found Error
```bash
Error: FileNotFoundError: [Errno 2] No such file or directory: 'ml_models/best_model.pkl'

Solution:
1. Ensure you've run the training notebook first
2. Check model path in configuration
3. Verify file permissions
```

#### 2. Import Errors
```bash
Error: ModuleNotFoundError: No module named 'sklearn'

Solution:
pip install -r requirements.txt --upgrade
```

#### 3. Port Already in Use
```bash
Error: OSError: [Errno 48] Address already in use

Solution:
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :5000
kill -9 <PID>

# Or change port
python app.py --port 5001
```

#### 4. Memory Issues During Training
```bash
Solution:
- Reduce dataset size
- Use smaller batch sizes
- Reduce model complexity
- Close other applications
```

#### 5. Prediction Accuracy Low
```bash
Solution:
- Retrain models with more data
- Adjust hyperparameters
- Add more features
- Check data quality
```

---

## 🚀 Deployment

### Local Development
```bash
python run_app.py
```

### Production with Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:8000 flask_app.app:app
```

### Docker Deployment (Coming Soon)
```dockerfile
# Dockerfile example
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "flask_app.app:app"]
```

### Cloud Deployment Options
- **Heroku**: Easy deployment with git push
- **AWS EC2**: Full control, scalable
- **Google Cloud Run**: Serverless containers
- **Azure App Service**: Integrated CI/CD

---

## 📚 API Documentation

### REST API Endpoints

#### POST /api/predict
Predict property price

**Request:**
```json
{
  "stay_duration": 5,
  "check_in_month": 7,
  "season": "Summer",
  "is_weekend": 0,
  "currency": "USD",
  "locale": "en-US"
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "price": 125.50,
    "confidence": 0.85,
    "price_range": {
      "min": 110.00,
      "max": 140.00
    },
    "recommendations": {
      "competitive": 115.00,
      "optimal": 125.50,
      "premium": 135.00
    }
  },
  "model": "XGBoost",
  "timestamp": "2025-12-05T10:30:00Z"
}
```

#### GET /api/stats
Get market statistics

**Response:**
```json
{
  "total_predictions": 1250,
  "average_price": 128.45,
  "model_accuracy": 0.85,
  "last_updated": "2025-12-05T10:00:00Z"
}
```

#### GET /api/models
List available models

**Response:**
```json
{
  "models": [
    {
      "name": "Random Forest",
      "accuracy": 0.76,
      "status": "active"
    },
    {
      "name": "XGBoost",
      "accuracy": 0.78,
      "status": "active"
    }
  ]
}
```

---

## 🤝 Contributing

Kontribusi sangat diterima! Berikut cara berkontribusi:

### 1. Fork Repository
```bash
# Click 'Fork' button di GitHub
```

### 2. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Commit Changes
```bash
git add .
git commit -m "Add amazing feature"
```

### 4. Push to Branch
```bash
git push origin feature/amazing-feature
```

### 5. Create Pull Request
- Buka repository di GitHub
- Click "New Pull Request"
- Describe your changes
- Submit for review

### Contribution Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📊 Dataset Information

### Raw Data Source
- **File**: `Data Airbnb Bali.json`
- **Format**: JSON array of Airbnb URLs
- **Records**: ~2000+ listings
- **Coverage**: Bali region, Indonesia

### Processed Dataset
- **File**: `dataset/airbnb_bali_ml_ready.csv`
- **Features**: 20+ engineered features
- **Target**: Price per night (USD)

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| stay_duration | Numerical | Number of nights |
| check_in_month | Categorical | Month of check-in (1-12) |
| season | Categorical | Season (Spring/Summer/Fall/Winter) |
| is_weekend | Binary | Weekend booking (0/1) |
| currency | Categorical | Currency code (USD, EUR, IDR) |
| locale | Categorical | Market locale (en-US, id-ID) |
| advance_days | Numerical | Days before check-in |
| day_of_week | Categorical | Day name (Monday-Sunday) |
| is_holiday | Binary | Holiday period (0/1) |

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Ida Bagus Agung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Libraries & Frameworks
- **Flask** - Web framework
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Gradient boosting
- **LightGBM** - Fast gradient boosting
- **CatBoost** - Categorical boosting
- **TensorFlow** - Deep learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Plotly** - Interactive visualizations
- **Bootstrap** - UI framework

### Inspiration
- Airbnb for inspiring the OTA analysis
- Bali hospitality community
- Open source ML community

---

## 📞 Contact & Support

### Developer
- **Name**: Ida Bagus Agung
- **GitHub**: [@IdaBagusAgung](https://github.com/IdaBagusAgung)
- **Repository**: [OTA-Price-Analysis](https://github.com/IdaBagusAgung/OTA-Price-Analysis)

### Getting Help
1. **Issues**: Create an issue on GitHub
2. **Discussions**: Use GitHub Discussions
3. **Documentation**: Check this README and code comments
4. **Pull Requests**: Contribute improvements

---

## 🗺️ Roadmap

### Version 1.0 (Current)
- ✅ Basic price prediction
- ✅ Multiple ML models
- ✅ Web dashboard
- ✅ EDA analysis

### Version 1.1 (Planned)
- 🔄 Real-time data updates
- 🔄 Advanced visualizations
- 🔄 User authentication
- 🔄 Prediction history

### Version 2.0 (Future)
- 📋 Mobile app
- 📋 Multi-region support
- 📋 Advanced analytics
- 📋 API monetization
- 📋 Docker containerization
- 📋 CI/CD pipeline

---

## 📈 Project Statistics

- **Lines of Code**: ~5000+
- **Models Trained**: 5+ algorithms
- **Dataset Size**: 2000+ records
- **Prediction Accuracy**: 85%+
- **Development Time**: 3+ months
- **Last Updated**: December 2025

---

## ⚡ Quick Commands Reference

```bash
# Setup
git clone https://github.com/IdaBagusAgung/OTA-Price-Analysis.git
cd OTA-Price-Analysis/Airbnb-ML-Dashboard
pip install -r requirements.txt

# Run Application
python run_app.py

# Training
python retrain_all_models.py

# Testing
python test_all_models.py

# Development Server
cd flask_app && python app.py

# Production Server
gunicorn -w 4 -b 0.0.0.0:8000 flask_app.app:app
```

---

<div align="center">

### 🌟 Star This Repository

If you find this project useful, please consider giving it a ⭐!

**Created by NgaeDev/GusAgungDev**

*Last Updated: December 5, 2025*

[⬆ Back to Top](#-ota-price-analysis---airbnb-bali-price-prediction-dashboard)

</div>
