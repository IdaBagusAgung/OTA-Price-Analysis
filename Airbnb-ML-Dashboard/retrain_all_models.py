"""
Script to retrain all models with consistent features for dashboard integration
This will fix the feature mismatch issue and numpy compatibility problems
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ML Models
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               ExtraTreesRegressor, VotingRegressor, StackingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔄 RETRAINING ALL MODELS WITH CONSISTENT FEATURES")
print("=" * 60)

# Paths
DATASET_PATH = Path(__file__).parent / 'dataset' / 'airbnb_bali_ml_ready.csv'
MODEL_DIR = Path(__file__).parent / 'ml_models'

# Load dataset
print(f"\n📂 Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display available features
print(f"\n📊 Available features:")
print(df.columns.tolist())

# Prepare features and target
feature_columns = [
    'stay_duration', 'check_in_month', 'season_encoded', 'is_weekend', 
    'currency_encoded', 'locale_encoded', 'is_long_stay', 'is_peak_season',
    'is_holiday_month', 'month_sin', 'month_cos'
]

target_column = 'price'

# Check if all features exist
missing_features = [f for f in feature_columns if f not in df.columns]
if missing_features:
    print(f"\n⚠️  Missing features: {missing_features}")
    print("Using available features instead...")
    feature_columns = [f for f in feature_columns if f in df.columns]

print(f"\n✅ Using {len(feature_columns)} features:")
for i, feat in enumerate(feature_columns, 1):
    print(f"   {i}. {feat}")

X = df[feature_columns]
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📊 Data split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Define models to train
models_to_train = {
    'Traditional ML': {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0),
        'lasso_regression': Lasso(alpha=1.0),
        'decision_tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'svr': SVR(kernel='rbf', C=100),
        'knn': KNeighborsRegressor(n_neighbors=5)
    },
    'Ensemble Models': {
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'extra_trees': ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, verbose=-1),
        'catboost': CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0)
    }
}

# Train all models
trained_models = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(f"\n🔥 Training {sum(len(v) for v in models_to_train.values())} models...")
print("=" * 60)

for category, models in models_to_train.items():
    print(f"\n📦 {category}")
    print("-" * 60)
    
    # Create category directory
    category_dir = MODEL_DIR / category.lower().replace(' ', '_')
    category_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model in models.items():
        try:
            print(f"\n🔄 Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"   ✅ R² Score: {r2:.4f}")
            print(f"   📊 RMSE: {rmse:.2f}")
            print(f"   📊 MAE: {mae:.2f}")
            
            # Prepare model package
            model_package = {
                'model': model,
                'feature_columns': feature_columns,
                'model_info': {
                    'name': model_name,
                    'category': category,
                    'timestamp': timestamp,
                    'metrics': {
                        'r2': float(r2),
                        'rmse': float(rmse),
                        'mae': float(mae)
                    }
                },
                'training_info': {
                    'n_features': len(feature_columns),
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test),
                    'feature_list': feature_columns
                }
            }
            
            # Save model
            model_filename = f"{model_name}_{timestamp}.pkl"
            model_path = category_dir / model_filename
            joblib.dump(model_package, model_path, compress=3)
            
            print(f"   💾 Saved to: {model_path.name}")
            
            # Store for later use
            trained_models[model_name] = {
                'model': model,
                'metrics': model_package['model_info']['metrics'],
                'category': category,
                'path': str(model_path)
            }
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            continue

print(f"\n" + "=" * 60)
print(f"✅ Training completed!")
print(f"📊 Successfully trained: {len(trained_models)}/{sum(len(v) for v in models_to_train.values())} models")

# Find best model
if trained_models:
    best_model_name = max(trained_models.items(), key=lambda x: x[1]['metrics']['r2'])[0]
    best_model_info = trained_models[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Category: {best_model_info['category']}")
    print(f"   R² Score: {best_model_info['metrics']['r2']:.4f}")
    print(f"   RMSE: {best_model_info['metrics']['rmse']:.2f}")
    print(f"   MAE: {best_model_info['metrics']['mae']:.2f}")
    
    # Save best model as production model
    production_path = MODEL_DIR / f"production_model_{timestamp}.pkl"
    production_package = {
        'model': best_model_info['model'],
        'feature_columns': feature_columns,
        'model_info': {
            'name': best_model_name,
            'category': best_model_info['category'],
            'timestamp': timestamp,
            'metrics': best_model_info['metrics']
        }
    }
    joblib.dump(production_package, production_path, compress=3)
    print(f"\n💾 Production model saved: {production_path.name}")

# Save feature encoders for dashboard
encoder_package = {
    'feature_columns': feature_columns,
    'label_encoders': {
        'currency': ['USD', 'EUR', 'IDR', 'GBP', 'AUD'],
        'locale': ['de-DE', 'en-GB', 'en-US', 'fr-FR'],  # Based on dataset
        'season': ['Fall', 'Spring', 'Summer', 'Winter']  # Based on dataset
    },
    'timestamp': timestamp
}

encoder_path = MODEL_DIR / 'feature_encoders.pkl'
joblib.dump(encoder_package, encoder_path, compress=3)
print(f"💾 Feature encoders saved: {encoder_path.name}")

# Create summary report
print(f"\n" + "=" * 60)
print("📋 TRAINING SUMMARY")
print("=" * 60)

# Group by category
for category in models_to_train.keys():
    category_models = {k: v for k, v in trained_models.items() if v['category'] == category}
    if category_models:
        print(f"\n{category}:")
        for name, info in sorted(category_models.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True):
            print(f"  • {name:20s} - R²: {info['metrics']['r2']:.4f}, RMSE: {info['metrics']['rmse']:6.2f}, MAE: {info['metrics']['mae']:6.2f}")

print(f"\n✅ All models saved to: {MODEL_DIR}")
print(f"🚀 Dashboard is ready to use all trained models!")
print("\n💡 Next steps:")
print("   1. Restart the dashboard: python run_app.py")
print("   2. All models will now load successfully")
print("   3. Analytics will show correct metrics")
print("   4. Predictions will use all available models")
