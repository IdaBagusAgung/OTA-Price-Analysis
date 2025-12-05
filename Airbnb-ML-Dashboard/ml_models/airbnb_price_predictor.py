import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

class AirbnbPricePredictor:
    def __init__(self, model_path=None):
        """Initialize the price predictor with the trained model"""
        if model_path is None:
            # Use relative path from this file's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "production_model_20251113_145825.pkl")
        
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.feature_columns = self.model_package['feature_columns']
        self.label_encoders = self.model_package['label_encoders']
        self.model_info = self.model_package['model_info']
        
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Create DataFrame from input
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Apply label encoding for categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str))
        
        # Feature engineering
        # Interaction features
        df['stay_duration_weekend'] = df['stay_duration'] * df['is_weekend']
        df['month_weekend'] = df['check_in_month'] * df['is_weekend']
        df['season_weekend'] = df['season'] * df['is_weekend']
        
        # Polynomial features
        numerical_for_poly = ['stay_duration', 'check_in_month', 'check_in_day_of_week']
        for col in numerical_for_poly:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(df[col] + 1)
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['check_in_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['check_in_month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['check_in_day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['check_in_day_of_week'] / 7)
        
        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select and order columns as expected by the model
        df = df[self.feature_columns]
        
        return df
    
    def predict(self, input_data):
        """Make price prediction"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)
            
            return {
                'predicted_price': float(prediction[0]),
                'model_name': self.model_info['name'],
                'model_r2_score': self.model_info['r2_score'],
                'prediction_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'predicted_price': None
            }
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info
    
    def batch_predict(self, input_data_list):
        """Make predictions for multiple inputs"""
        results = []
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        return results

# Example usage:
# predictor = AirbnbPricePredictor()
# sample_input = {
#     'stay_duration': 5,
#     'check_in_month': 12,
#     'check_in_day_of_week': 1,
#     'season': 'Winter',
#     'is_weekend': 0,
#     'locale': 'en-US',
#     'currency': 'USD'
# }
# prediction = predictor.predict(sample_input)
# print(prediction)
