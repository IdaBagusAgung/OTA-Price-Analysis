import joblib
import json
import os
import numpy as np

class ModelLoader:
    def __init__(self, model_base_path):
        self.model_base_path = model_base_path
        self.registry_path = os.path.join(model_base_path, "model_registry.json")
        self.load_registry()
    
    def load_registry(self):
        with open(self.registry_path, 'r') as f:
            self.registry = json.load(f)
    
    def get_best_model(self):
        best_model_info = self.registry['best_model']
        return best_model_info
    
    def load_model_by_name(self, model_name):
        # Implementation for loading specific models
        # This would be expanded based on the specific model type
        pass
    
    def get_feature_info(self):
        return {
            'features': self.registry['feature_columns'],
            'categorical': self.registry['categorical_features'],
            'numerical': self.registry['numerical_features']
        }
