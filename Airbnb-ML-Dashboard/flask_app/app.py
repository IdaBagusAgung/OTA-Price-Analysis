# 🏨 Airbnb Bali Price Prediction Dashboard
# Advanced Flask Web Application with Machine Learning Integration

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
from datetime import datetime, date
import logging
from werkzeug.exceptions import RequestEntityTooLarge

# Add parent directory to path for model imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model manager
try:
    from ml_models.model_manager import ModelManager, load_best_model
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False
    print("⚠️ ModelManager not available. Using fallback mode.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'airbnb-bali-ml-dashboard-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload

# Configuration
class Config:
    MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_models'))
    DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    STATIC_PATH = 'static'
    TEMPLATES_PATH = 'templates'
    
    # Currency exchange rates (as of November 2025)
    EXCHANGE_RATES = {
        'USD': 1.0,
        'IDR': 15750.0,  # 1 USD = 15,750 IDR
        'EUR': 0.92,
        'GBP': 0.79,
        'AUD': 1.52
    }

def convert_currency(amount_usd, target_currency='USD'):
    """Convert USD amount to target currency"""
    if target_currency not in Config.EXCHANGE_RATES:
        return amount_usd
    return amount_usd * Config.EXCHANGE_RATES[target_currency]

def format_currency(amount, currency='USD'):
    """Format currency with appropriate symbol and formatting"""
    if currency == 'IDR':
        return f"Rp {amount:,.0f}"
    elif currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
    elif currency == 'GBP':
        return f"£{amount:,.2f}"
    elif currency == 'AUD':
        return f"A${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

# Global variables for model and encoders
model_predictor = None
model_manager = None
feature_encoders = None
feature_columns = None
model_info = {}
available_models = {}

def clean_model_name(filename):
    """Clean model filename by removing timestamp and file extension"""
    import re
    # Remove .pkl extension
    name = filename.replace('.pkl', '')
    # Remove timestamp pattern like _20251113_144614
    name = re.sub(r'_\d{8}_\d{6}', '', name)
    # Replace underscores with spaces and title case
    name = name.replace('_', ' ').title()
    return name

def load_model_and_encoders():
    """Load the trained model using ModelManager"""
    global model_predictor, model_manager, feature_encoders, feature_columns, model_info, available_models
    
    try:
        if MODEL_MANAGER_AVAILABLE:
            # Use ModelManager to load best model
            logger.info("Loading model using ModelManager...")
            model_manager = ModelManager(Config.MODEL_PATH)
            
            # Get best model info
            best_model_info = model_manager.get_best_model_info()
            if best_model_info:
                # Clean the model name
                cleaned_name = clean_model_name(best_model_info['name'])
                logger.info(f"✅ Best model found: {cleaned_name}")
                logger.info(f"   R² Score: {best_model_info['metrics']['r2']:.4f}")
                
                # Load the predictor
                model_predictor = model_manager.load_model_for_prediction()
                
                if model_predictor:
                    model_info = {
                        'model_type': cleaned_name,
                        'r2_score': best_model_info['metrics'].get('r2', 0.85),
                        'rmse': best_model_info['metrics'].get('rmse', 25.0),
                        'mae': best_model_info['metrics'].get('mae', 15.0),
                        'timestamp': best_model_info.get('timestamp', 'N/A')
                    }
                    
                    # Get available models
                    available_models = model_manager.list_available_models()
                    total_models = sum(len(v) for v in available_models.values())
                    logger.info(f"✅ Total trained models available: {total_models}")
                    
                    # Load feature encoders from model artifacts
                    artifacts_path = os.path.join(Config.MODEL_PATH, 'model_artifacts')
                    encoder_files = [f for f in os.listdir(artifacts_path) 
                                   if 'preprocessing' in f and f.endswith('.pkl')]
                    
                    if encoder_files:
                        latest_artifact = sorted(encoder_files)[-1]
                        import joblib
                        artifacts = joblib.load(os.path.join(artifacts_path, latest_artifact))
                        feature_encoders = artifacts.get('label_encoders', {})
                        feature_columns = artifacts.get('feature_columns', [])
                        logger.info("✅ Feature encoders loaded from artifacts")
                    
                    return True
            
        # Fallback to old method if ModelManager not available
        logger.warning("⚠️ Using fallback model loading method...")
        return load_fallback_model()
        
    except Exception as e:
        logger.error(f"❌ Error loading model with ModelManager: {e}")
        logger.warning("⚠️ Attempting fallback model loading...")
        return load_fallback_model()

def load_fallback_model():
    """Fallback method to load model without ModelManager"""
    global model_predictor, feature_encoders, feature_columns, model_info
    
    try:
        # Try to load production model
        import joblib
        production_files = [f for f in os.listdir(Config.MODEL_PATH) 
                          if f.startswith('production_model_') and f.endswith('.pkl')]
        
        if production_files:
            latest_production = sorted(production_files)[-1]
            production_path = os.path.join(Config.MODEL_PATH, latest_production)
            
            production_package = joblib.load(production_path)
            model_predictor = production_package['model']
            feature_columns = production_package.get('feature_columns', [])
            info = production_package.get('model_info', {})
            
            # Ensure model_info has all required fields with defaults
            model_info = {
                'model_type': clean_model_name(info.get('name', 'Production Model')),
                'r2_score': info.get('r2_score', info.get('metrics', {}).get('r2', 0.85)),
                'rmse': info.get('rmse', info.get('metrics', {}).get('rmse', 25.0)),
                'mae': info.get('mae', info.get('metrics', {}).get('mae', 15.0)),
                'timestamp': info.get('timestamp', 'N/A')
            }
            
            logger.info(f"✅ Production model loaded: {model_info['model_type']}")
            
            # Load encoders from package
            feature_encoders = production_package.get('label_encoders', {})
            
            return True
        else:
            logger.warning("⚠️ No production model found. Using demo mode.")
            return create_demo_model()
            
    except Exception as e:
        logger.error(f"❌ Fallback loading failed: {e}")
        return create_demo_model()

def create_demo_model():
    """Create demo model for testing - with trained dummy model"""
    global model_predictor, feature_encoders, feature_columns, model_info
    
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Create and train a simple model with dummy data
    model_predictor = RandomForestRegressor(n_estimators=10, random_state=42)
    
    # Train with dummy data so it's fitted - using 11 features to match retrained models
    X_dummy = np.random.rand(100, 11)  # 100 samples, 11 features
    y_dummy = 50 + X_dummy[:, 0] * 20 + np.random.randn(100) * 5  # Simple linear relationship with noise
    model_predictor.fit(X_dummy, y_dummy)
    
    # Use same feature columns as retrained models
    feature_columns = [
        'stay_duration', 'check_in_month', 'season_encoded', 'is_weekend', 
        'currency_encoded', 'locale_encoded', 'is_long_stay', 'is_peak_season',
        'is_holiday_month', 'month_sin', 'month_cos'
    ]
    
    # Ensure model_info has all required fields
    model_info = {
        'model_type': 'RandomForest (Demo)', 
        'r2_score': 0.85,
        'rmse': 25.0,
        'mae': 15.0,
        'timestamp': 'Demo Model'
    }
    
    feature_encoders = {
        'currency': ['USD', 'EUR', 'IDR', 'GBP', 'AUD'],
        'locale': ['de-DE', 'en-GB', 'en-US', 'fr-FR'],
        'season': ['Fall', 'Spring', 'Summer', 'Winter']
    }
    
    logger.warning("⚠️ Running in DEMO mode with sample model")
    return True

def predict_price(input_data):
    """Make price prediction using the loaded model with enhanced features"""
    try:
        if model_predictor is None:
            return None, "Model not loaded"
        
        # Extract features from input_data
        stay_duration = float(input_data.get('stay_duration', 1))
        check_in_month = int(input_data.get('check_in_month', 1))
        is_weekend = int(input_data.get('is_weekend', 0))
        
        # Encode categorical features
        locale_encoded, currency_encoded, season_encoded = encode_categorical_features(
            input_data.get('locale', 'en-US'),
            input_data.get('currency', 'USD'),
            input_data.get('season', 'Spring')
        )
        
        # Derive additional features (matching training data)
        is_long_stay = 1 if stay_duration >= 7 else 0
        is_peak_season = 1 if check_in_month in [6, 7, 8, 12] else 0  # Summer & December
        is_holiday_month = 1 if check_in_month in [1, 7, 8, 12] else 0
        month_sin = np.sin(2 * np.pi * check_in_month / 12)
        month_cos = np.cos(2 * np.pi * check_in_month / 12)
        
        # Build feature array matching the 11 features the model was trained with
        # ['stay_duration', 'check_in_month', 'season_encoded', 'is_weekend', 
        #  'currency_encoded', 'locale_encoded', 'is_long_stay', 'is_peak_season',
        #  'is_holiday_month', 'month_sin', 'month_cos']
        feature_array = np.array([[
            stay_duration,
            check_in_month,
            season_encoded,
            is_weekend,
            currency_encoded,
            locale_encoded,
            is_long_stay,
            is_peak_season,
            is_holiday_month,
            month_sin,
            month_cos
        ]])
        
        # Make prediction
        if hasattr(model_predictor, 'predict'):
            prediction = model_predictor.predict(feature_array)[0]
        else:
            # Dummy prediction for demo
            prediction = 50 + stay_duration * 10 + np.random.normal(0, 5)
        
        return max(25, prediction), None  # Ensure minimum price of $25
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, str(e)

def generate_all_features(input_data):
    """Generate complete feature set (11 features) for model prediction - matches retrained models"""
    # Basic features
    stay_duration = input_data.get('stay_duration', 1)
    check_in_month = input_data.get('check_in_month', 6)
    is_weekend = input_data.get('is_weekend', 0)
    
    # Encode categorical features
    locale_encoded, currency_encoded, season_encoded = encode_categorical_features(
        input_data.get('locale', 'en-US'),
        input_data.get('currency', 'USD'),
        input_data.get('season', 'Summer')
    )
    
    # Derived features (matching training data)
    is_long_stay = 1 if stay_duration >= 7 else 0
    is_peak_season = 1 if check_in_month in [6, 7, 8, 12] else 0  # Summer & December
    is_holiday_month = 1 if check_in_month in [1, 7, 8, 12] else 0
    month_sin = np.sin(2 * np.pi * check_in_month / 12)
    month_cos = np.cos(2 * np.pi * check_in_month / 12)
    
    # Complete feature dictionary with exactly 11 features matching retrained models
    features = {
        'stay_duration': stay_duration,
        'check_in_month': check_in_month,
        'season_encoded': season_encoded,
        'is_weekend': is_weekend,
        'currency_encoded': currency_encoded,
        'locale_encoded': locale_encoded,
        'is_long_stay': is_long_stay,
        'is_peak_season': is_peak_season,
        'is_holiday_month': is_holiday_month,
        'month_sin': month_sin,
        'month_cos': month_cos
    }
    
    return features

def encode_categorical_features(locale, currency, season):
    """Encode categorical features using the loaded encoders"""
    try:
        # Handle sklearn LabelEncoder objects (has classes_ attribute)
        locale_encoder = feature_encoders.get('locale')
        currency_encoder = feature_encoders.get('currency')
        season_encoder = feature_encoders.get('season')
        
        # Check if it's a LabelEncoder object
        if hasattr(locale_encoder, 'classes_'):
            # sklearn LabelEncoder - use classes_ attribute
            try:
                locale_encoded = list(locale_encoder.classes_).index(locale)
            except (ValueError, AttributeError):
                locale_encoded = 0
                
            try:
                currency_encoded = list(currency_encoder.classes_).index(currency)
            except (ValueError, AttributeError):
                currency_encoded = 0
                
            try:
                season_encoded = list(season_encoder.classes_).index(season)
            except (ValueError, AttributeError):
                season_encoded = 0
                
        elif isinstance(locale_encoder, dict):
            # Dict format
            locale_encoded = locale_encoder.get('mapping', {}).get(locale, 0)
            currency_encoded = currency_encoder.get('mapping', {}).get(currency, 0)
            season_encoded = season_encoder.get('mapping', {}).get(season, 0)
        elif isinstance(locale_encoder, list):
            # List format
            locale_encoded = locale_encoder.index(locale) if locale in locale_encoder else 0
            currency_encoded = currency_encoder.index(currency) if currency in currency_encoder else 0
            season_encoded = season_encoder.index(season) if season in season_encoder else 0
        else:
            # Fallback
            locale_encoded, currency_encoded, season_encoded = 0, 0, 0
        
        return locale_encoded, currency_encoded, season_encoded
    except Exception as e:
        logger.error(f"❌ Encoding error: {e}")
        return 0, 0, 0

@app.route('/')
def index():
    """Home page with prediction form"""
    # Get available options from encoders
    # Handle sklearn LabelEncoder objects
    if hasattr(feature_encoders.get('locale'), 'classes_'):
        locales = list(feature_encoders['locale'].classes_)
        currencies = list(feature_encoders['currency'].classes_)
        seasons = list(feature_encoders['season'].classes_)
    elif isinstance(feature_encoders.get('locale'), dict):
        if 'classes_' in feature_encoders.get('locale', {}):
            locales = list(feature_encoders['locale']['classes_'])
            currencies = list(feature_encoders['currency']['classes_'])
            seasons = list(feature_encoders['season']['classes_'])
        else:
            locales = list(feature_encoders.get('locale', {}).get('classes', ['en-US', 'en-GB', 'de-DE']))
            currencies = list(feature_encoders.get('currency', {}).get('classes', ['USD', 'EUR', 'IDR']))
            seasons = list(feature_encoders.get('season', {}).get('classes', ['Spring', 'Summer', 'Fall', 'Winter']))
    else:
        locales = feature_encoders.get('locale', ['en-US', 'en-GB', 'de-DE', 'fr-FR', 'en-AU'])
        currencies = feature_encoders.get('currency', ['USD', 'EUR', 'IDR', 'GBP', 'AUD'])
        seasons = feature_encoders.get('season', ['Spring', 'Summer', 'Fall', 'Winter'])
    
    return render_template('index.html', 
                         locales=locales, 
                         currencies=currencies, 
                         seasons=seasons,
                         model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        stay_duration = float(request.form.get('stay_duration', 1))
        locale = request.form.get('locale', 'en-US')
        currency = request.form.get('currency', 'USD')
        display_currency = request.form.get('display_currency', 'USD')  # New: display currency option
        season = request.form.get('season', 'Summer')
        is_weekend = int(request.form.get('is_weekend', 0))
        check_in_month = int(request.form.get('check_in_month', 6))
        check_in_day = int(request.form.get('check_in_day_of_week', 1))
        
        # Validate inputs
        if stay_duration < 1 or stay_duration > 365:
            flash('Stay duration must be between 1 and 365 days', 'error')
            return redirect(url_for('index'))
        
        if check_in_month < 1 or check_in_month > 12:
            flash('Check-in month must be between 1 and 12', 'error')
            return redirect(url_for('index'))
        
        # Prepare input data dict
        input_data = {
            'stay_duration': stay_duration,
            'check_in_month': check_in_month,
            'check_in_day_of_week': check_in_day,
            'season': season,
            'is_weekend': is_weekend,
            'locale': locale,
            'currency': currency
        }
        
        # Make prediction with best model (function handles feature engineering internally)
        # Prediction is always in USD
        predicted_price_usd, error = predict_price(input_data)
        
        if error:
            flash(f'Prediction error: {error}', 'error')
            return redirect(url_for('index'))
        
        # Generate ALL features (22 features) for multi-model predictions
        all_features = generate_all_features(input_data)
        
        # Get predictions from all models
        all_model_predictions = []
        if MODEL_MANAGER_AVAILABLE and model_manager:
            try:
                all_models = model_manager.list_available_models()
                
                for category, models in all_models.items():
                    for model_file in models:
                        try:
                            # Skip .h5 files (Keras models) for now
                            if model_file.endswith('.h5'):
                                continue
                            
                            # Load each model
                            import joblib
                            model_path = os.path.join(Config.MODEL_PATH, category, model_file)
                            model_package = joblib.load(model_path)
                            
                            # Handle different model formats
                            if isinstance(model_package, dict):
                                model_obj = model_package.get('model')
                                raw_name = model_package.get('model_info', {}).get('name', model_file.replace('.pkl', ''))
                                metrics = model_package.get('model_info', {}).get('metrics', {})
                            else:
                                # Old format - direct model object
                                model_obj = model_package
                                raw_name = model_file.replace('.pkl', '')
                                metrics = {}
                            
                            # Clean the model name (remove timestamp)
                            model_name = clean_model_name(raw_name)
                            
                            # Make prediction with this model
                            if hasattr(model_obj, 'predict'):
                                # Direct prediction using complete feature generation
                                try:
                                    # Get feature columns for this specific model
                                    if isinstance(model_package, dict):
                                        feat_cols = model_package.get('feature_columns', [])
                                    else:
                                        feat_cols = feature_columns
                                    
                                    # Use the all_features dict we generated above (22 features)
                                    # Build feature array in the exact order the model expects
                                    if feat_cols:
                                        feature_array = np.array([[all_features.get(col, 0) for col in feat_cols]])
                                    else:
                                        # Fallback: use first 7 basic features for old models
                                        feature_array = np.array([[
                                            all_features['stay_duration'],
                                            all_features['check_in_month'],
                                            all_features['check_in_day_of_week'],
                                            all_features['season'],
                                            all_features['is_weekend'],
                                            all_features['locale'],
                                            all_features['currency']
                                        ]])
                                    
                                    pred = model_obj.predict(feature_array)[0]
                                    pred = max(25, pred)  # Minimum price
                                    
                                    all_model_predictions.append({
                                        'model_name': model_name,
                                        'category': category,
                                        'prediction_usd': round(pred, 2),
                                        'prediction': round(convert_currency(pred, display_currency), 2),
                                        'formatted_price': format_currency(convert_currency(pred, display_currency), display_currency),
                                        'r2_score': metrics.get('r2', 0) if metrics else 0,
                                        'rmse': metrics.get('rmse', 0) if metrics else 0,
                                        'mae': metrics.get('mae', 0) if metrics else 0
                                    })
                                except Exception as pred_error:
                                    logger.warning(f"Could not predict with {model_name}: {pred_error}")
                                    continue
                        except Exception as model_error:
                            logger.warning(f"Could not load model {model_file}: {model_error}")
                            continue
                
                # Sort by R² score (best first)
                all_model_predictions.sort(key=lambda x: x['r2_score'], reverse=True)
                
            except Exception as e:
                logger.error(f"Error getting all model predictions: {e}")
        
        # Convert to display currency
        predicted_price = convert_currency(predicted_price_usd, display_currency)
        total_price = predicted_price * stay_duration
        price_per_person = predicted_price  # Assuming single occupancy
        
        # Calculate ensemble statistics
        ensemble_stats = {}
        if all_model_predictions:
            predictions = [p['prediction'] for p in all_model_predictions]
            avg_val = round(np.mean(predictions), 2)
            median_val = round(np.median(predictions), 2)
            min_val = round(np.min(predictions), 2)
            max_val = round(np.max(predictions), 2)
            std_val = round(np.std(predictions), 2)
            
            ensemble_stats = {
                'average': format_currency(avg_val, display_currency),
                'median': format_currency(median_val, display_currency),
                'min': format_currency(min_val, display_currency),
                'max': format_currency(max_val, display_currency),
                'std': format_currency(std_val, display_currency),
                'count': len(predictions),
                # Raw values for calculations
                'average_raw': avg_val,
                'median_raw': median_val,
                'min_raw': min_val,
                'max_raw': max_val,
                'std_raw': std_val
            }
        
        # Prepare result data
        result = {
            'predicted_price': round(predicted_price, 2),
            'predicted_price_usd': round(predicted_price_usd, 2),  # Keep USD price for reference
            'total_price': round(total_price, 2),
            'price_per_person': round(price_per_person, 2),
            'stay_duration': int(stay_duration),
            'locale': locale,
            'currency': currency,
            'display_currency': display_currency,
            'formatted_price': format_currency(predicted_price, display_currency),
            'formatted_total': format_currency(total_price, display_currency),
            'exchange_rate': Config.EXCHANGE_RATES.get(display_currency, 1.0),
            'season': season,
            'is_weekend': bool(is_weekend),
            'check_in_month': check_in_month,
            'model_info': model_info,
            'all_model_predictions': all_model_predictions,
            'ensemble_stats': ensemble_stats
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        flash(f'An error occurred: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        stay_duration = float(data.get('stay_duration', 1))
        locale = data.get('locale', 'en-US')
        currency = data.get('currency', 'USD')
        season = data.get('season', 'Summer')
        is_weekend = int(data.get('is_weekend', 0))
        check_in_month = int(data.get('check_in_month', 6))
        check_in_day = int(data.get('check_in_day_of_week', 1))
        
        # Prepare input data
        input_data = {
            'stay_duration': stay_duration,
            'check_in_month': check_in_month,
            'check_in_day_of_week': check_in_day,
            'season': season,
            'is_weekend': is_weekend,
            'locale': locale,
            'currency': currency
        }
        
        # Make prediction
        predicted_price, error = predict_price(input_data)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Get available models info
        models_summary = {}
        if MODEL_MANAGER_AVAILABLE and model_manager:
            try:
                models_summary = model_manager.get_model_performance_summary()
            except:
                pass
        
        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'total_price': round(predicted_price * stay_duration, 2),
            'currency': currency,
            'model_info': model_info,
            'available_models': models_summary
        })
        
    except Exception as e:
        logger.error(f"❌ API prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics')
def analytics():
    """Analytics dashboard with insights"""
    try:
        # Get comprehensive model analytics
        all_models_info = []
        
        if MODEL_MANAGER_AVAILABLE and model_manager:
            try:
                all_models = model_manager.list_available_models()
                
                for category, models in all_models.items():
                    for model_file in models:
                        try:
                            # Skip .h5 files (Keras models) due to compatibility issues
                            if model_file.endswith('.h5'):
                                continue
                            
                            # Skip files with known compatibility issues
                            if 'catboost' in model_file.lower() and 'numpy' in str(model_file):
                                logger.warning(f"Skipping {model_file} due to numpy version compatibility")
                                continue
                            
                            import joblib
                            model_path = os.path.join(Config.MODEL_PATH, category, model_file)
                            
                            # Try to load the model with error handling
                            try:
                                model_package = joblib.load(model_path)
                            except Exception as load_error:
                                logger.warning(f"Could not load model {model_file}: {load_error}")
                                continue
                            
                            # Handle different formats
                            if isinstance(model_package, dict):
                                info = model_package.get('model_info', {})
                                if isinstance(info, dict):
                                    metrics = info.get('metrics', {})
                                    raw_name = info.get('name', model_file)
                                else:
                                    metrics = {}
                                    raw_name = model_file
                            else:
                                # Old format
                                info = {}
                                metrics = {}
                                raw_name = model_file
                            
                            # Clean the model name (remove timestamp)
                            model_name = clean_model_name(raw_name)
                            
                            # Safely extract metrics with defaults
                            r2_score = 0.0
                            rmse = 0.0
                            mae = 0.0
                            training_time = 0.0
                            timestamp = 'N/A'
                            
                            if isinstance(metrics, dict):
                                r2_score = metrics.get('r2', 0.0)
                                rmse = metrics.get('rmse', 0.0)
                                mae = metrics.get('mae', 0.0)
                            
                            if isinstance(info, dict):
                                training_time = info.get('training_time', 0.0)
                                timestamp = info.get('timestamp', 'N/A')
                            
                            all_models_info.append({
                                'name': model_name,
                                'category': category,
                                'r2_score': r2_score,
                                'rmse': rmse,
                                'mae': mae,
                                'training_time': training_time,
                                'timestamp': timestamp
                            })
                        except Exception as e:
                            logger.warning(f"Could not load model info {model_file}: {e}")
                            continue
                
                # Sort by R² score
                all_models_info.sort(key=lambda x: x['r2_score'], reverse=True)
                
            except Exception as e:
                logger.error(f"Error loading model analytics: {e}")
        
        # Calculate summary statistics
        if all_models_info:
            r2_scores = [m['r2_score'] for m in all_models_info]
            rmse_scores = [m['rmse'] for m in all_models_info]
            mae_scores = [m['mae'] for m in all_models_info]
            
            summary_stats = {
                'total_models': len(all_models_info),
                'best_r2': max(r2_scores) if r2_scores else 0,
                'avg_r2': np.mean(r2_scores) if r2_scores else 0,
                'best_rmse': min(rmse_scores) if rmse_scores else 0,
                'avg_rmse': np.mean(rmse_scores) if rmse_scores else 0,
                'best_mae': min(mae_scores) if mae_scores else 0,
                'avg_mae': np.mean(mae_scores) if mae_scores else 0,
                'best_model': all_models_info[0]['name'] if all_models_info else 'N/A'
            }
        else:
            summary_stats = {
                'total_models': 0,
                'best_r2': 0,
                'avg_r2': 0,
                'best_rmse': 0,
                'avg_rmse': 0,
                'best_mae': 0,
                'avg_mae': 0,
                'best_model': 'N/A'
            }
        
        # Category breakdown
        category_stats = {}
        for model in all_models_info:
            cat = model['category']
            if cat not in category_stats:
                category_stats[cat] = {
                    'count': 0,
                    'avg_r2': 0,
                    'best_r2': 0,
                    'models': []
                }
            category_stats[cat]['count'] += 1
            category_stats[cat]['models'].append(model)
        
        # Calculate category averages
        for cat, data in category_stats.items():
            r2_scores = [m['r2_score'] for m in data['models']]
            data['avg_r2'] = np.mean(r2_scores) if r2_scores else 0
            data['best_r2'] = max(r2_scores) if r2_scores else 0
        
        # Ensure current_model has all required fields with safe defaults
        current_model_safe = {
            'model_type': model_info.get('model_type', 'Unknown Model'),
            'r2_score': model_info.get('r2_score', 0.85),
            'rmse': model_info.get('rmse', 25.0),
            'mae': model_info.get('mae', 15.0),
            'timestamp': model_info.get('timestamp', 'N/A')
        }
        
        analytics_data = {
            'all_models': all_models_info,
            'summary_stats': summary_stats,
            'category_stats': category_stats,
            'current_model': current_model_safe
        }
        
        return render_template('analytics.html', analytics=analytics_data)
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        flash('Error loading analytics', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page with model information"""
    return render_template('about.html', model_info=model_info)

@app.route('/api/models')
def api_models():
    """API endpoint to get available models information"""
    try:
        if MODEL_MANAGER_AVAILABLE and model_manager:
            models_list = model_manager.list_available_models()
            performance_summary = model_manager.get_model_performance_summary()
            best_model = model_manager.get_best_model_info()
            
            return jsonify({
                'status': 'success',
                'best_model': best_model,
                'available_models': models_list,
                'performance_summary': performance_summary,
                'total_models': sum(len(v) for v in models_list.values())
            })
        else:
            return jsonify({
                'status': 'limited',
                'message': 'ModelManager not available. Using fallback model.',
                'current_model': model_info
            })
    except Exception as e:
        logger.error(f"❌ Models API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    flash('File too large', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    print("🚀 STARTING AIRBNB BALI PRICE PREDICTION DASHBOARD")
    print("=" * 60)
    
    # Load model and encoders
    print("\n📦 Loading ML Models...")
    if load_model_and_encoders():
        print("✅ Model and encoders loaded successfully")
        
        if MODEL_MANAGER_AVAILABLE and model_manager:
            print("\n📊 Model Information:")
            print(f"   Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"   R² Score: {model_info.get('r2_score', 0):.4f}")
            print(f"   RMSE: {model_info.get('rmse', 0):.2f}")
            print(f"   MAE: {model_info.get('mae', 0):.2f}")
            
            # Display available models count
            if available_models:
                total = sum(len(v) for v in available_models.values())
                print(f"\n✨ Total trained models available: {total}")
                for category, models in available_models.items():
                    if models:
                        print(f"   - {category}: {len(models)} models")
    else:
        print("⚠️ Running with demo/fallback model")
    
    print("\n🌐 Dashboard starting...")
    print("📊 Available at: http://localhost:5000")
    print("🎯 API endpoint: http://localhost:5000/api/predict")
    print("📈 Analytics: http://localhost:5000/analytics")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)