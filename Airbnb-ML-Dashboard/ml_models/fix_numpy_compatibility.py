"""
Quick fix script to resolve numpy._core compatibility issue
This script reloads and resaves all models with the current numpy version
"""
import os
import sys
import joblib
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def fix_model_numpy_compatibility():
    """Reload and resave all models to fix numpy compatibility"""
    
    print("🔧 NUMPY COMPATIBILITY FIX")
    print("=" * 50)
    print(f"Current numpy version: {np.__version__}")
    print()
    
    # Get models directory
    models_dir = Path(__file__).parent
    
    # List all .pkl files (including subdirectories)
    model_files = list(models_dir.rglob("*.pkl"))
    
    # Exclude backup files
    model_files = [f for f in model_files if '.backup' not in str(f)]
    
    if not model_files:
        print("❌ No model files found")
        return
    
    print(f"Found {len(model_files)} model files")
    print()
    
    fixed_count = 0
    failed_count = 0
    
    for model_file in model_files:
        try:
            print(f"Processing: {model_file.name}...")
            
            # Try to load the model
            model_package = joblib.load(str(model_file))
            
            # Create backup
            backup_file = model_file.with_suffix('.pkl.backup')
            if not backup_file.exists():
                os.rename(str(model_file), str(backup_file))
                print(f"  ✅ Backed up to: {backup_file.name}")
            
            # Resave with current numpy version
            joblib.dump(model_package, str(model_file), compress=3)
            print(f"  ✅ Resaved with numpy {np.__version__}")
            
            fixed_count += 1
            
        except ModuleNotFoundError as e:
            if 'numpy._core' in str(e):
                print(f"  ⚠️  Skipped (numpy._core issue): {model_file.name}")
                failed_count += 1
            else:
                print(f"  ❌ Error: {e}")
                failed_count += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1
    
    print()
    print("=" * 50)
    print(f"✅ Successfully fixed: {fixed_count} models")
    print(f"❌ Failed to fix: {failed_count} models")
    print()
    
    if failed_count > 0:
        print("⚠️  Some models couldn't be fixed due to incompatible pickle format")
        print("💡 Solution: Retrain these models with the current environment")
        print()

if __name__ == "__main__":
    fix_model_numpy_compatibility()
