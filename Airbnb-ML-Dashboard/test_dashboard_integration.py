"""
Quick Test Script for Dashboard Integration
Tests if the model loading and prediction works correctly
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("=" * 80)
print("🧪 TESTING DASHBOARD INTEGRATION")
print("=" * 80)
print()

# Test 1: Import model_manager
print("Test 1: Importing ModelManager...")
try:
    from ml_models.model_manager import ModelManager, load_best_model
    print("✅ ModelManager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ModelManager: {e}")
    sys.exit(1)
print()

# Test 2: Initialize ModelManager
print("Test 2: Initializing ModelManager...")
try:
    model_path = os.path.join(os.path.dirname(__file__), 'ml_models')
    model_path = os.path.abspath(model_path)
    print(f"   Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist!")
        sys.exit(1)
    
    model_manager = ModelManager(model_path)
    print("✅ ModelManager initialized")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 3: List available models
print("Test 3: Listing available models...")
try:
    available_models = model_manager.list_available_models()
    total = sum(len(v) for v in available_models.values())
    print(f"✅ Found {total} total models:")
    for category, models in available_models.items():
        if models:
            print(f"   - {category}: {len(models)} models")
except Exception as e:
    print(f"❌ Failed to list models: {e}")
    sys.exit(1)
print()

# Test 4: Get best model
print("Test 4: Getting best model...")
try:
    best_model_info = model_manager.get_best_model_info()
    if best_model_info:
        print(f"✅ Best model: {best_model_info['name']}")
        print(f"   R² Score: {best_model_info['metrics']['r2']:.4f}")
        print(f"   RMSE: {best_model_info['metrics']['rmse']:.2f}")
        print(f"   MAE: {best_model_info['metrics']['mae']:.2f}")
    else:
        print("⚠️ No best model found")
except Exception as e:
    print(f"❌ Failed to get best model: {e}")
    sys.exit(1)
print()

# Test 5: Load model for prediction
print("Test 5: Loading model predictor...")
try:
    predictor = model_manager.load_model_for_prediction()
    if predictor:
        print("✅ Model predictor loaded successfully")
    else:
        print("❌ Failed to load predictor")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 6: Make a test prediction
print("Test 6: Making test prediction...")
try:
    test_input = {
        'stay_duration': 3,
        'check_in_month': 7,
        'check_in_day_of_week': 5,
        'season': 'Summer',
        'is_weekend': 1,
        'locale': 'en-US',
        'currency': 'USD'
    }
    
    print(f"   Input: {test_input}")
    prediction = predictor.predict(test_input)
    print(f"✅ Prediction successful: ${prediction:.2f} per night")
    print(f"   Total for {test_input['stay_duration']} nights: ${prediction * test_input['stay_duration']:.2f}")
except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 7: Test Flask app imports
print("Test 7: Testing Flask app imports...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))
    from flask import Flask
    print("✅ Flask imports successful")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")
    print("   Run: pip install flask")
    sys.exit(1)
print()

# Summary
print("=" * 80)
print("✨ ALL TESTS PASSED! Dashboard is ready to run.")
print("=" * 80)
print()
print("🚀 To start the dashboard:")
print("   PowerShell: .\\START_DASHBOARD.ps1")
print("   Or directly: cd flask_app && python app.py")
print()
print("🌐 Dashboard will be available at: http://localhost:5000")
print()
