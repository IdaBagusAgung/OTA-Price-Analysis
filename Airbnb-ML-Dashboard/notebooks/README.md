# Airbnb ML Dashboard - Model Training Documentation

This folder contains comprehensive machine learning and deep learning model training notebooks for the Airbnb price prediction dashboard.

## 📚 Notebook Overview

### 1. `Comprehensive_ML_Model_Training.ipynb`
**Main training notebook with 15+ machine learning models**

**Traditional ML Models (6):**
- Linear Regression
- Ridge Regression  
- Lasso Regression
- Support Vector Regression (SVR)
- Decision Tree Regressor
- K-Nearest Neighbors (KNN)

**Ensemble Models (6):**
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Extra Trees

**Deep Learning Models (3):**
- Multi-layer Perceptron (MLP)
- Deep Neural Network with BatchNormalization
- CNN-inspired 1D model

**Key Features:**
- Automated model training and evaluation
- Organized model saving structure
- Performance comparison and visualization
- Feature importance analysis
- Model registry creation

### 2. `Advanced_Model_Optimization.ipynb`
**Hyperparameter tuning and ensemble optimization**

**Optimization Techniques:**
- Bayesian Optimization with scikit-optimize
- Grid Search and Random Search
- Cross-validation
- Advanced feature engineering

**Advanced Ensembles:**
- Stacking Regressor
- Voting Regressor
- Production-ready model packaging

**Key Features:**
- Hyperparameter spaces for top models
- Model stacking with meta-learner
- Production model package creation
- Inference code generation

### 3. `Advanced_Deep_Learning_Models.ipynb`
**State-of-the-art deep learning for tabular data**

**Advanced Architectures:**
- Wide & Deep Networks (Google)
- Attention-based Neural Networks
- TabNet (Google's tabular DL model)
- Autoencoder-based feature learning
- Advanced ensemble with ResNet/Highway networks

**Key Features:**
- Embedding layers for categorical features
- Multi-head attention mechanisms
- Feature learning with autoencoders
- Advanced regularization techniques
- Comprehensive model registry

## 📁 Model Organization Structure

```
ml_models/
├── traditional_ml/          # Basic ML models
│   ├── linear_regression_*.pkl
│   ├── ridge_regression_*.pkl
│   └── ...
├── ensemble_models/         # Tree-based & boosting models
│   ├── random_forest_*.pkl
│   ├── xgboost_*.pkl
│   ├── stacking_ensemble_*.pkl
│   └── ...
├── deep_learning/          # Neural network models
│   ├── mlp_*.h5
│   ├── wide_deep_model_*.h5
│   ├── attention_model_*.h5
│   ├── tabnet_model_*/
│   └── ...
└── model_artifacts/        # Preprocessing objects
    ├── scalers/
    ├── encoders/
    └── ...
```

## 🚀 Quick Start Guide

### 1. Run Basic Training
```python
# Open Comprehensive_ML_Model_Training.ipynb
# Run all cells to train 15+ models
```

### 2. Optimize Best Models
```python
# After basic training, open Advanced_Model_Optimization.ipynb
# This will hyperparameter tune the best performers
```

### 3. Train Advanced Deep Learning
```python
# Open Advanced_Deep_Learning_Models.ipynb
# Train state-of-the-art tabular deep learning models
```

## 📊 Model Performance Tracking

Each notebook generates:
- **Performance CSV**: Comparison of all models with metrics
- **Model Registry JSON**: Metadata and paths for all models
- **Feature Importance**: Analysis of most predictive features
- **Visualizations**: Performance comparison plots

## 🔧 Model Integration

### Loading Best Model
```python
from ml_models.model_loader import ModelLoader

loader = ModelLoader('path/to/ml_models')
best_model_info = loader.get_best_model()
print(f"Best model: {best_model_info['name']}")
```

### Using Production Model
```python
from ml_models.airbnb_price_predictor import AirbnbPricePredictor

predictor = AirbnbPricePredictor()
sample_input = {
    'stay_duration': 5,
    'check_in_month': 12,
    'season': 'Winter',
    'is_weekend': 0,
    'locale': 'en-US',
    'currency': 'USD'
}
prediction = predictor.predict(sample_input)
print(f"Predicted price: ${prediction['predicted_price']:.2f}")
```

## 📋 Dataset Requirements

**Input Dataset**: `dataset/airbnb_bali_ml_ready.csv`

**Required Features:**
- `stay_duration`: Length of stay in days
- `check_in_month`: Month of check-in (1-12)
- `check_in_day_of_week`: Day of week (0-6)
- `season`: Seasonal category (Spring/Summer/Fall/Winter)
- `is_weekend`: Weekend indicator (0/1)
- `locale`: Language/country locale
- `currency`: Pricing currency
- `price`: Target variable (nightly rate)

## 🔍 Model Evaluation Metrics

All models are evaluated using:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error

## 🛠️ Dependencies

**Core Libraries:**
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- xgboost, lightgbm, catboost

**Deep Learning:**
- tensorflow/keras
- pytorch-tabnet (for TabNet)

**Optimization:**
- scikit-optimize (Bayesian optimization)

**Install All Dependencies:**
```bash
pip install -r requirements.txt
```

## 📈 Expected Performance

**Traditional ML Models**: R² ~ 0.65-0.75
**Ensemble Models**: R² ~ 0.75-0.85
**Deep Learning Models**: R² ~ 0.80-0.90
**Optimized Models**: R² ~ 0.85-0.95

## 🚨 Important Notes

1. **Run notebooks in order** for best results
2. **Training time**: 30min - 2hrs depending on hardware
3. **Memory requirements**: 4GB+ RAM recommended
4. **GPU acceleration**: Optional but recommended for deep learning
5. **Model files**: Can be 100MB+ total

## 🔗 Integration with Dashboard

The trained models integrate seamlessly with the Flask dashboard:

1. Models are automatically saved in organized folders
2. Model registry provides metadata for dashboard
3. Production predictor class handles all preprocessing
4. Models can be hot-swapped without code changes

## 📞 Support

For questions or issues:
1. Check model registry JSON files for model metadata
2. Verify dataset format matches requirements
3. Ensure all dependencies are installed
4. Check console output for training progress and errors

---

**Ready to train world-class ML models for Airbnb price prediction! 🎯**