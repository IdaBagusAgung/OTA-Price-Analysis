# Test Model Loading & Predictions
# Script untuk menguji loading dan prediksi dari semua model

import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 80)
print("🧪 TESTING ALL MODELS - Model Loading & Prediction Test")
print("=" * 80)
print()

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_models')

# Test input data
test_input = {
    'stay_duration': 3,
    'check_in_month': 7,
    'check_in_day_of_week': 5,
    'season': 'Summer',
    'is_weekend': 1,
    'locale': 'en-US',
    'currency': 'USD'
}

print(f"📋 Test Input Data:")
for key, value in test_input.items():
    print(f"   {key}: {value}")
print()

# Get all model categories
categories = ['traditional_ml', 'ensemble_models', 'deep_learning']
all_models = {}

for category in categories:
    category_path = os.path.join(MODEL_PATH, category)
    if os.path.exists(category_path):
        models = [f for f in os.listdir(category_path) if f.endswith('.pkl')]
        all_models[category] = models
        print(f"📁 {category}: {len(models)} models found")

print()
print("=" * 80)
print("🔬 TESTING EACH MODEL")
print("=" * 80)
print()

# Test results
test_results = {
    'success': [],
    'failed': [],
    'skipped': []
}

model_count = 0
for category, models in all_models.items():
    print(f"\n{'='*80}")
    print(f"📂 Category: {category.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    for model_file in models:
        model_count += 1
        model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
        
        # Skip .h5 files
        if model_file.endswith('.h5'):
            print(f"⏭️  [{model_count}] {model_name}")
            print(f"   Status: SKIPPED (Keras model)")
            test_results['skipped'].append({
                'name': model_name,
                'file': model_file,
                'category': category,
                'reason': 'Keras model (.h5)'
            })
            print()
            continue
        
        print(f"🔍 [{model_count}] Testing: {model_name}")
        print(f"   File: {model_file}")
        
        try:
            # Load model
            model_path = os.path.join(MODEL_PATH, category, model_file)
            model_package = joblib.load(model_path)
            
            # Check model format
            if isinstance(model_package, dict):
                print(f"   Format: ✅ Dictionary (New Format)")
                model_obj = model_package.get('model')
                model_info = model_package.get('model_info', {})
                metrics = model_info.get('metrics', {})
                encoders = model_package.get('label_encoders', {})
                feature_cols = model_package.get('feature_columns', [])
                
                print(f"   Name: {model_info.get('name', 'N/A')}")
                print(f"   R² Score: {metrics.get('r2', 0):.4f}")
                print(f"   RMSE: {metrics.get('rmse', 0):.2f}")
                print(f"   MAE: {metrics.get('mae', 0):.2f}")
                print(f"   Features: {len(feature_cols)} columns")
                
            else:
                print(f"   Format: ⚠️  Direct Model (Old Format)")
                model_obj = model_package
                metrics = {}
                print(f"   Warning: No metadata available")
            
            # Test prediction
            if hasattr(model_obj, 'predict'):
                # Create COMPLETE feature set (22 features)
                stay_dur = test_input['stay_duration']
                month = test_input['check_in_month']
                day = test_input['check_in_day_of_week']
                weekend = test_input['is_weekend']
                
                features = {
                    # Basic (7)
                    'stay_duration': stay_dur,
                    'check_in_month': month,
                    'check_in_day_of_week': day,
                    'season': 2,  # Summer
                    'is_weekend': weekend,
                    'locale': 0,  # en-US
                    'currency': 0,  # USD
                    # Additional base (2)
                    'check_in_year': 2025,
                    'check_in_week_of_year': (month - 1) * 4 + (day // 7),
                    # Interactions (3)
                    'stay_duration_weekend': stay_dur * weekend,
                    'month_weekend': month * weekend,
                    'season_weekend': 2 * weekend,
                    # Polynomials (5)
                    'stay_duration_squared': stay_dur ** 2,
                    'stay_duration_sqrt': np.sqrt(stay_dur),
                    'check_in_month_squared': month ** 2,
                    'check_in_month_sqrt': np.sqrt(month),
                    'season_squared': 2 ** 2,
                    # Cyclical (4)
                    'month_sin': np.sin(2 * np.pi * month / 12),
                    'month_cos': np.cos(2 * np.pi * month / 12),
                    'day_sin': np.sin(2 * np.pi * day / 7),
                    'day_cos': np.cos(2 * np.pi * day / 7)
                }
                
                # Try prediction
                try:
                    if isinstance(model_package, dict) and feature_cols:
                        # Use proper feature columns
                        feature_array = np.array([[features.get(col, 0) for col in feature_cols]])
                    else:
                        # Fallback: first 7 features
                        feature_array = np.array([[
                            features['stay_duration'], features['check_in_month'],
                            features['check_in_day_of_week'], features['season'],
                            features['is_weekend'], features['locale'], features['currency']
                        ]])
                    
                    prediction = model_obj.predict(feature_array)[0]
                    prediction = max(25, prediction)  # Min price
                    
                    print(f"   Prediction: ${prediction:.2f}")
                    print(f"   Status: ✅ SUCCESS")
                    
                    test_results['success'].append({
                        'name': model_name,
                        'file': model_file,
                        'category': category,
                        'prediction': round(prediction, 2),
                        'r2_score': metrics.get('r2', 0) if metrics else 0,
                        'rmse': metrics.get('rmse', 0) if metrics else 0
                    })
                    
                except Exception as pred_error:
                    print(f"   Prediction: ❌ FAILED")
                    print(f"   Error: {str(pred_error)}")
                    print(f"   Status: ⚠️  LOADED BUT PREDICTION FAILED")
                    
                    test_results['failed'].append({
                        'name': model_name,
                        'file': model_file,
                        'category': category,
                        'error': str(pred_error),
                        'stage': 'prediction'
                    })
            else:
                print(f"   Status: ❌ NO PREDICT METHOD")
                test_results['failed'].append({
                    'name': model_name,
                    'file': model_file,
                    'category': category,
                    'error': 'No predict method',
                    'stage': 'validation'
                })
                
        except Exception as e:
            print(f"   Loading: ❌ FAILED")
            print(f"   Error: {str(e)}")
            print(f"   Status: ❌ LOAD FAILED")
            
            test_results['failed'].append({
                'name': model_name,
                'file': model_file,
                'category': category,
                'error': str(e),
                'stage': 'loading'
            })
        
        print()

# Print summary
print("\n" + "=" * 80)
print("📊 TEST SUMMARY")
print("=" * 80)
print()

total_tested = len(test_results['success']) + len(test_results['failed'])
total_skipped = len(test_results['skipped'])
total_models = total_tested + total_skipped

print(f"Total Models: {total_models}")
print(f"   ✅ Success: {len(test_results['success'])} ({len(test_results['success'])/total_models*100:.1f}%)")
print(f"   ❌ Failed:  {len(test_results['failed'])} ({len(test_results['failed'])/total_models*100:.1f}%)")
print(f"   ⏭️  Skipped: {len(test_results['skipped'])} ({len(test_results['skipped'])/total_models*100:.1f}%)")
print()

if test_results['success']:
    print("=" * 80)
    print("✅ SUCCESSFUL MODELS (Working)")
    print("=" * 80)
    print()
    
    # Sort by R² score
    test_results['success'].sort(key=lambda x: x['r2_score'], reverse=True)
    
    for i, model in enumerate(test_results['success'], 1):
        print(f"{i}. {model['name']}")
        print(f"   Category: {model['category']}")
        print(f"   Prediction: ${model['prediction']:.2f}")
        print(f"   R² Score: {model['r2_score']:.4f}")
        print(f"   RMSE: {model['rmse']:.2f}")
        print()

if test_results['failed']:
    print("=" * 80)
    print("❌ FAILED MODELS (Need Fixing)")
    print("=" * 80)
    print()
    
    for i, model in enumerate(test_results['failed'], 1):
        print(f"{i}. {model['name']}")
        print(f"   File: {model['file']}")
        print(f"   Category: {model['category']}")
        print(f"   Stage: {model['stage']}")
        print(f"   Error: {model['error'][:100]}")
        print()

if test_results['skipped']:
    print("=" * 80)
    print("⏭️  SKIPPED MODELS (Keras/Special Format)")
    print("=" * 80)
    print()
    
    for i, model in enumerate(test_results['skipped'], 1):
        print(f"{i}. {model['name']}")
        print(f"   Reason: {model['reason']}")
        print()

# Recommendations
print("=" * 80)
print("💡 RECOMMENDATIONS")
print("=" * 80)
print()

if len(test_results['failed']) > 0:
    print("⚠️  Some models failed to load or predict:")
    print("   1. Check if models were trained with compatible scikit-learn versions")
    print("   2. Re-train failed models with current environment")
    print("   3. Or exclude them from multi-model predictions")
    print()

if len(test_results['success']) > 0:
    print(f"✅ {len(test_results['success'])} models working correctly!")
    print("   These models can be used for multi-model predictions")
    print()

if len(test_results['skipped']) > 0:
    print(f"📝 {len(test_results['skipped'])} Keras models skipped")
    print("   Consider adding separate Keras model loading if needed")
    print()

print("=" * 80)
print("🏁 TEST COMPLETE")
print("=" * 80)

# Save results to JSON
import json
results_file = os.path.join(os.path.dirname(__file__), 'model_test_results.json')
with open(results_file, 'w') as f:
    json.dump({
        'test_date': datetime.now().isoformat(),
        'test_input': test_input,
        'summary': {
            'total': total_models,
            'success': len(test_results['success']),
            'failed': len(test_results['failed']),
            'skipped': len(test_results['skipped'])
        },
        'results': test_results
    }, f, indent=2)

print(f"\n📄 Results saved to: {results_file}")
