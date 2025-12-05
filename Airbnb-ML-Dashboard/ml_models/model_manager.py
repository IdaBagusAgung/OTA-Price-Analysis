"""
Model Manager for Airbnb ML Dashboard

This module provides utilities for managing trained models,
loading them for predictions, and integrating with the Flask dashboard.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """
    Comprehensive model management system for the Airbnb ML Dashboard
    """
    
    def __init__(self, model_base_path: str):
        """
        Initialize ModelManager
        
        Args:
            model_base_path (str): Base path to the ml_models directory
        """
        self.model_base_path = model_base_path
        self.model_registry = None
        self.advanced_dl_registry = None
        self.load_registries()
    
    def load_registries(self):
        """Load model registries if they exist"""
        try:
            registry_path = os.path.join(self.model_base_path, "model_registry.json")
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
            
            advanced_registry_path = os.path.join(self.model_base_path, "advanced_dl_model_registry.json")
            if os.path.exists(advanced_registry_path):
                with open(advanced_registry_path, 'r') as f:
                    self.advanced_dl_registry = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load model registries: {e}")
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List all available trained models
        
        Returns:
            Dict containing model categories and their models
        """
        models = {
            'traditional_ml': [],
            'ensemble_models': [],
            'deep_learning': [],
            'optimized_models': []
        }
        
        for category in models.keys():
            category_path = os.path.join(self.model_base_path, category)
            if os.path.exists(category_path):
                files = os.listdir(category_path)
                model_files = [f for f in files if f.endswith(('.pkl', '.h5')) or os.path.isdir(os.path.join(category_path, f))]
                models[category] = model_files
        
        return models
    
    def get_best_model_info(self) -> Optional[Dict]:
        """
        Get information about the best performing model
        
        Returns:
            Dictionary with best model information
        """
        if self.model_registry and 'best_model' in self.model_registry:
            return self.model_registry['best_model']
        elif self.advanced_dl_registry and 'best_advanced_dl_model' in self.advanced_dl_registry:
            return self.advanced_dl_registry['best_advanced_dl_model']
        else:
            return None
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary of all trained models
        
        Returns:
            DataFrame with model performance metrics
        """
        all_results = []
        
        # Add traditional/ensemble models
        if self.model_registry and 'all_models' in self.model_registry:
            all_results.extend(self.model_registry['all_models'])
        
        # Add advanced deep learning models
        if self.advanced_dl_registry and 'all_advanced_dl_models' in self.advanced_dl_registry:
            advanced_models = self.advanced_dl_registry['all_advanced_dl_models']
            for model in advanced_models:
                model['model_name'] = f"{model['model_name']} (Advanced DL)"
            all_results.extend(advanced_models)
        
        if all_results:
            df = pd.DataFrame(all_results)
            return df.sort_values('r2', ascending=False)
        else:
            return pd.DataFrame()
    
    def load_model_for_prediction(self, model_name: Optional[str] = None) -> Optional['ModelPredictor']:
        """
        Load a specific model for making predictions
        
        Args:
            model_name (str, optional): Name of the model to load. If None, loads best model.
            
        Returns:
            ModelPredictor instance or None if model not found
        """
        try:
            # First check for production model (prioritize this over registry)
            production_files = [f for f in os.listdir(self.model_base_path) 
                              if f.startswith('production_model_') and f.endswith('.pkl')]
            
            if production_files:
                # Use the latest production model
                latest_prod = sorted(production_files)[-1]
                prod_path = os.path.join(self.model_base_path, latest_prod)
                print(f"Loading production model: {latest_prod}")
                
                prod_package = joblib.load(prod_path)
                return SklearnModelPredictor(
                    prod_package['model'], 
                    self.model_base_path,
                    preprocessing_info=prod_package.get('preprocessing_info'),
                    label_encoders=prod_package.get('label_encoders'),
                    feature_columns=prod_package.get('feature_columns')
                )
            
            # Fallback to model_name lookup
            if model_name is None:
                # Load best model from registry
                best_model_info = self.get_best_model_info()
                if not best_model_info:
                    print("No best model information or production model found")
                    return None
                model_name = best_model_info['name']
            
            # Try to find and load the model by name
            return self._load_specific_model(model_name)
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_specific_model(self, model_name: str) -> Optional['ModelPredictor']:
        """Load a specific model based on its name"""
        
        # Search for the model in all categories
        for category in ['traditional_ml', 'ensemble_models', 'deep_learning']:
            category_path = os.path.join(self.model_base_path, category)
            if not os.path.exists(category_path):
                continue
                
            files = os.listdir(category_path)
            
            # Look for model files containing the model name
            for file in files:
                if model_name.lower().replace(' ', '_') in file.lower():
                    file_path = os.path.join(category_path, file)
                    
                    try:
                        if file.endswith('.pkl'):
                            model = joblib.load(file_path)
                            return SklearnModelPredictor(model, self.model_base_path)
                        elif file.endswith('.h5'):
                            import tensorflow as tf
                            model = tf.keras.models.load_model(file_path)
                            return KerasModelPredictor(model, self.model_base_path)
                        elif os.path.isdir(file_path):
                            # TabNet or similar directory-based models
                            return TabNetModelPredictor(file_path, self.model_base_path)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        continue
        
        print(f"Model {model_name} not found")
        return None
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str, optional): Name of the model. If None, uses best model.
            
        Returns:
            DataFrame with feature importance or None
        """
        try:
            predictor = self.load_model_for_prediction(model_name)
            if predictor and hasattr(predictor, 'get_feature_importance'):
                return predictor.get_feature_importance()
        except Exception as e:
            print(f"Error getting feature importance: {e}")
        
        return None
    
    def compare_models(self, metric: str = 'r2') -> pd.DataFrame:
        """
        Compare models by a specific metric
        
        Args:
            metric (str): Metric to compare by ('r2', 'rmse', 'mae')
            
        Returns:
            DataFrame sorted by the specified metric
        """
        df = self.get_model_performance_summary()
        if not df.empty and metric in df.columns:
            ascending = False if metric == 'r2' else True
            return df.sort_values(metric, ascending=ascending)
        return df


class ModelPredictor:
    """Base class for model predictors"""
    
    def __init__(self, model, model_base_path: str, preprocessing_info=None, label_encoders=None, feature_columns=None):
        self.model = model
        self.model_base_path = model_base_path
        self.preprocessing_info = preprocessing_info
        self.label_encoders = label_encoders or {}
        self.feature_columns = feature_columns or []
        
        # Load additional artifacts if not provided
        if not self.preprocessing_info:
            self.preprocessing_artifacts = self._load_preprocessing_artifacts()
        else:
            self.preprocessing_artifacts = preprocessing_info
    
    def _load_preprocessing_artifacts(self) -> Optional[Dict]:
        """Load preprocessing artifacts (scalers, encoders, etc.)"""
        artifacts_dir = os.path.join(self.model_base_path, "model_artifacts")
        
        # Try to find the most recent preprocessing artifacts
        try:
            for file in os.listdir(artifacts_dir):
                if file.startswith("advanced_dl_preprocessing") and file.endswith(".pkl"):
                    artifacts_path = os.path.join(artifacts_dir, file)
                    return joblib.load(artifacts_path)
        except:
            pass
        
        return None
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """
        Make prediction on input data
        
        Args:
            input_data: Input data for prediction
            
        Returns:
            Dictionary with prediction results
        """
        raise NotImplementedError("Subclasses must implement predict method")


class SklearnModelPredictor(ModelPredictor):
    """Predictor for scikit-learn based models"""
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> float:
        """Make prediction using sklearn model - returns just the price"""
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            # Apply preprocessing (feature engineering + encoding)
            df_processed = self._preprocess_input(df)
            
            # Make prediction
            prediction = self.model.predict(df_processed)
            
            return float(prediction[0])
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _preprocess_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full preprocessing pipeline including feature engineering"""
        df_processed = df.copy()
        
        # Step 1: Label encode categorical features
        if self.label_encoders:
            for col in ['locale', 'currency', 'season']:
                if col in df_processed.columns:
                    # Handle both sklearn LabelEncoder and simple list format
                    encoder = self.label_encoders.get(col)
                    if hasattr(encoder, 'transform'):
                        # sklearn LabelEncoder
                        df_processed[col] = encoder.transform(df_processed[col].astype(str))
                    elif isinstance(encoder, dict) and 'classes_' in encoder:
                        # Saved encoder format
                        classes = encoder['classes_']
                        df_processed[col] = df_processed[col].map(
                            {c: i for i, c in enumerate(classes)}
                        ).fillna(0).astype(int)
        elif self.preprocessing_artifacts and 'label_encoders' in self.preprocessing_artifacts:
            for col, encoder in self.preprocessing_artifacts['label_encoders'].items():
                if col in df_processed.columns:
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        # Step 2: Create interaction features
        if 'stay_duration' in df_processed.columns and 'is_weekend' in df_processed.columns:
            df_processed['stay_duration_weekend'] = df_processed['stay_duration'] * df_processed['is_weekend']
            df_processed['month_weekend'] = df_processed.get('check_in_month', 1) * df_processed['is_weekend']
            df_processed['season_weekend'] = df_processed.get('season', 0) * df_processed['is_weekend']
        
        # Step 3: Create polynomial features
        if 'stay_duration' in df_processed.columns:
            df_processed['stay_duration_squared'] = df_processed['stay_duration'] ** 2
            df_processed['stay_duration_sqrt'] = np.sqrt(df_processed['stay_duration'])
        
        # Step 4: Create cyclical encoding
        if 'check_in_month' in df_processed.columns:
            df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['check_in_month'] / 12)
            df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['check_in_month'] / 12)
        
        if 'check_in_day_of_week' in df_processed.columns:
            df_processed['day_sin'] = np.sin(2 * np.pi * df_processed['check_in_day_of_week'] / 7)
            df_processed['day_cos'] = np.cos(2 * np.pi * df_processed['check_in_day_of_week'] / 7)
        
        # Step 5: Ensure all expected features are present in correct order
        if self.feature_columns:
            # Add any missing columns with default values
            for col in self.feature_columns:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            
            # Select and reorder columns to match training
            df_processed = df_processed[self.feature_columns]
        
        return df_processed
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance for tree-based models"""
        if hasattr(self.model, 'feature_importances_'):
            feature_cols = self.preprocessing_artifacts.get('feature_columns', [])
            if feature_cols:
                return pd.DataFrame({
                    'feature': feature_cols,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        return None


class KerasModelPredictor(ModelPredictor):
    """Predictor for Keras/TensorFlow models"""
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction using Keras model"""
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            # Apply preprocessing
            if self.preprocessing_artifacts:
                df_processed = self._preprocess_input(df)
            else:
                df_processed = df
            
            # Make prediction
            prediction_scaled = self.model.predict(df_processed, verbose=0)
            
            # Inverse transform if target scaler is available
            if self.preprocessing_artifacts and 'target_scaler' in self.preprocessing_artifacts:
                prediction = self.preprocessing_artifacts['target_scaler'].inverse_transform(prediction_scaled)
            else:
                prediction = prediction_scaled
            
            return {
                'predicted_price': float(prediction[0][0]),
                'model_type': 'keras',
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'predicted_price': None
            }
    
    def _preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing to input data for Keras models"""
        df_processed = df.copy()
        
        # Apply label encoding
        if 'label_encoders' in self.preprocessing_artifacts:
            for col, encoder in self.preprocessing_artifacts['label_encoders'].items():
                if col in df_processed.columns:
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        # Apply scaling
        if 'numerical_scaler' in self.preprocessing_artifacts:
            return self.preprocessing_artifacts['numerical_scaler'].transform(df_processed)
        
        return df_processed.values


class TabNetModelPredictor(ModelPredictor):
    """Predictor for TabNet models"""
    
    def __init__(self, model_path: str, model_base_path: str):
        self.model_path = model_path
        self.model_base_path = model_base_path
        self.model = None
        self.preprocessing_artifacts = self._load_preprocessing_artifacts()
        self._load_tabnet_model()
    
    def _load_tabnet_model(self):
        """Load TabNet model"""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            self.model = TabNetRegressor()
            self.model.load_model(self.model_path)
        except ImportError:
            print("TabNet not available. Install pytorch-tabnet to use TabNet models.")
        except Exception as e:
            print(f"Error loading TabNet model: {e}")
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction using TabNet model"""
        if self.model is None:
            return {'error': 'TabNet model not loaded', 'predicted_price': None}
        
        try:
            # Convert input to DataFrame if it's a dictionary
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()
            
            # Apply preprocessing
            if self.preprocessing_artifacts:
                df_processed = self._preprocess_input(df)
            else:
                df_processed = df.values.astype(np.float32)
            
            # Make prediction
            prediction = self.model.predict(df_processed)
            
            return {
                'predicted_price': float(prediction[0]),
                'model_type': 'tabnet',
                'prediction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'predicted_price': None
            }
    
    def _preprocess_input(self, df: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing to input data for TabNet"""
        df_processed = df.copy()
        
        # Apply label encoding
        if 'label_encoders' in self.preprocessing_artifacts:
            for col, encoder in self.preprocessing_artifacts['label_encoders'].items():
                if col in df_processed.columns:
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        return df_processed.values.astype(np.float32)


# Convenience function for easy model loading
def load_best_model(model_base_path: str) -> Optional[ModelPredictor]:
    """
    Load the best performing model for predictions
    
    Args:
        model_base_path (str): Path to the ml_models directory
        
    Returns:
        ModelPredictor instance or None
    """
    manager = ModelManager(model_base_path)
    return manager.load_model_for_prediction()


# Example usage and testing functions
def test_model_manager(model_base_path: str):
    """Test the ModelManager functionality"""
    manager = ModelManager(model_base_path)
    
    print("Available models:")
    models = manager.list_available_models()
    for category, model_list in models.items():
        print(f"  {category}: {len(model_list)} models")
    
    print("\nBest model info:")
    best_model = manager.get_best_model_info()
    if best_model:
        print(f"  Name: {best_model['name']}")
        print(f"  R² Score: {best_model['metrics']['r2']:.4f}")
    
    print("\nModel performance summary:")
    summary = manager.get_model_performance_summary()
    if not summary.empty:
        print(summary.head())
    
    print("\nTesting model prediction:")
    predictor = manager.load_model_for_prediction()
    if predictor:
        sample_input = {
            'stay_duration': 5,
            'check_in_month': 12,
            'check_in_day_of_week': 1,
            'season': 'Winter',
            'is_weekend': 0,
            'locale': 'en-US',
            'currency': 'USD'
        }
        result = predictor.predict(sample_input)
        print(f"Prediction result: {result}")


if __name__ == "__main__":
    # Test the model manager
    model_base_path = r"c:\Users\proda\OneDrive\Documents\Gus Agung\PROJECT ISENG\PROJECT AFTER LULUS\GYE-Project\GYE-OTA-ANALYSIS\Airbnb-ML-Dashboard\ml_models"
    test_model_manager(model_base_path)