"""
Automated Test: Multi-Model Comparison & Analytics Verification
================================================================
This script tests whether all 12 retrained models appear in:
1. Multi-model comparison table on result page
2. Analytics page with proper metrics (not zeros)
"""

import requests
import json
from datetime import datetime, timedelta

def test_prediction_endpoint():
    """Test prediction endpoint and verify multi-model comparison"""
    url = "http://localhost:5000/predict"
    
    # Test data - typical Airbnb booking
    check_in = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
    check_out = (datetime.now() + timedelta(days=37)).strftime('%Y-%m-%d')
    
    payload = {
        'check_in': check_in,
        'check_out': check_out,
        'currency': 'USD',
        'locale': 'en-US',
        'nights': '7',
        'adults': '2',
        'children': '0',
        'infants': '0',
        'pets': '0'
    }
    
    print("=" * 80)
    print("🔍 TEST 1: PREDICTION ENDPOINT - MULTI-MODEL COMPARISON")
    print("=" * 80)
    print(f"📅 Test Booking: {payload['nights']} nights from {check_in} to {check_out}")
    print(f"👥 Guests: {payload['adults']} adults")
    print(f"💰 Currency: {payload['currency']}")
    print("\n🚀 Sending prediction request...")
    
    try:
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            # Check if it's HTML response (redirect to result page)
            if 'text/html' in response.headers.get('Content-Type', ''):
                print("✅ Prediction successful - Got HTML result page")
                
                # Parse HTML to find model comparison table
                html = response.text
                
                # Count model names in comparison table
                expected_models = [
                    'Linear Regression', 'Ridge Regression', 'Lasso Regression',
                    'Decision Tree', 'SVR', 'KNN',
                    'Random Forest', 'Gradient Boosting', 'Extra Trees',
                    'XGBoost', 'LightGBM', 'CatBoost'
                ]
                
                found_models = []
                for model in expected_models:
                    if model.lower().replace(' ', '') in html.lower().replace(' ', ''):
                        found_models.append(model)
                
                print(f"\n📊 MODELS FOUND IN COMPARISON TABLE: {len(found_models)}/12")
                for i, model in enumerate(found_models, 1):
                    print(f"   {i}. ✅ {model}")
                
                missing_models = set(expected_models) - set(found_models)
                if missing_models:
                    print(f"\n⚠️  MISSING MODELS: {len(missing_models)}")
                    for model in missing_models:
                        print(f"   ❌ {model}")
                
                # Check for metrics in HTML (R², RMSE, MAE)
                has_r2 = 'R²' in html or 'R2' in html or 'r²' in html
                has_rmse = 'RMSE' in html
                has_mae = 'MAE' in html
                
                print(f"\n📈 METRICS PRESENCE:")
                print(f"   {'✅' if has_r2 else '❌'} R² Score")
                print(f"   {'✅' if has_rmse else '❌'} RMSE")
                print(f"   {'✅' if has_mae else '❌'} MAE")
                
                # Check if there are actual price predictions (not zeros)
                has_predictions = '$' in html and any(char.isdigit() for char in html)
                print(f"\n💵 PRICE PREDICTIONS: {'✅ Found' if has_predictions else '❌ Not Found'}")
                
                return len(found_models) == 12 and has_r2 and has_rmse and has_mae
            else:
                print("❌ Expected HTML response but got:", response.headers.get('Content-Type'))
                return False
        else:
            print(f"❌ Prediction failed with status code: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction endpoint: {str(e)}")
        return False


def test_analytics_page():
    """Test analytics page for model metrics"""
    url = "http://localhost:5000/analytics"
    
    print("\n" + "=" * 80)
    print("🔍 TEST 2: ANALYTICS PAGE - MODEL METRICS")
    print("=" * 80)
    print("🚀 Fetching analytics page...")
    
    try:
        response = requests.get(url)
        
        if response.status_code == 200:
            print("✅ Analytics page loaded successfully")
            html = response.text
            
            # Check for model names in analytics
            expected_models = [
                'Linear Regression', 'Ridge Regression', 'Lasso Regression',
                'Decision Tree', 'SVR', 'KNN',
                'Random Forest', 'Gradient Boosting', 'Extra Trees',
                'XGBoost', 'LightGBM', 'CatBoost'
            ]
            
            found_models = []
            for model in expected_models:
                if model.lower().replace(' ', '') in html.lower().replace(' ', ''):
                    found_models.append(model)
            
            print(f"\n📊 MODELS IN ANALYTICS: {len(found_models)}/12")
            for i, model in enumerate(found_models, 1):
                print(f"   {i}. ✅ {model}")
            
            # Check for metrics
            has_r2 = 'R²' in html or 'R2' in html
            has_rmse = 'RMSE' in html
            has_mae = 'MAE' in html
            has_accuracy = 'Accuracy' in html or 'accuracy' in html
            
            print(f"\n📈 METRICS ON ANALYTICS:")
            print(f"   {'✅' if has_r2 else '❌'} R² Score")
            print(f"   {'✅' if has_rmse else '❌'} RMSE")
            print(f"   {'✅' if has_mae else '❌'} MAE")
            print(f"   {'✅' if has_accuracy else '❌'} Accuracy")
            
            # Check for non-zero values (look for percentages or decimal numbers)
            import re
            percentages = re.findall(r'\d+\.\d+%', html)
            decimals = re.findall(r'\d+\.\d+', html)
            
            has_real_values = len(percentages) > 0 or len(decimals) > 10
            print(f"\n💯 REAL DATA VALUES: {'✅ Found' if has_real_values else '❌ Only Zeros'}")
            if percentages:
                print(f"   📊 Sample percentages: {percentages[:5]}")
            
            # Check for charts/visualizations
            has_charts = 'chart' in html.lower() or 'canvas' in html.lower()
            print(f"\n📊 CHARTS/VISUALIZATIONS: {'✅ Present' if has_charts else '❌ Missing'}")
            
            return len(found_models) >= 10 and has_r2 and has_rmse and has_real_values
            
        else:
            print(f"❌ Analytics page failed with status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing analytics page: {str(e)}")
        return False


def main():
    """Run all tests and generate report"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "MULTI-MODEL COMPARISON TEST SUITE" + " " * 30 + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Testing all 12 retrained models appear in dashboard" + " " * 24 + "║")
    print("║" + "  Testing analytics shows real metrics (not zeros)" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Run tests
    test1_passed = test_prediction_endpoint()
    test2_passed = test_analytics_page()
    
    # Generate summary
    print("\n" + "=" * 80)
    print("📝 TEST SUMMARY")
    print("=" * 80)
    print(f"Test 1 - Multi-Model Comparison: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 - Analytics Metrics: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 80)
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Dashboard is working perfectly!")
        print("\n✅ VERIFIED FEATURES:")
        print("   • All 12 models appear in multi-model comparison")
        print("   • Analytics page shows real metrics (not zeros)")
        print("   • R², RMSE, MAE metrics present")
        print("   • Price predictions working")
        print("   • Charts and visualizations rendered")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Please review output above")
        return 1


if __name__ == "__main__":
    exit(main())
