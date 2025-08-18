#!/usr/bin/env python3
"""
Test script to verify the stress level prediction project setup.
"""

import sys
import importlib.util
from pathlib import Path

def test_imports():
    """Test if all required packages are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 
        'plotly', 'jupyter', 'xgboost', 
        'lightgbm', 'imblearn', 'yellowbrick', 'tqdm', 
        'joblib', 'yaml', 'statsmodels'
    ]
    
    # Optional packages (may not be available on all Python versions)
    optional_packages = ['ydata_profiling', 'pandas_profiling']
    
    print("Testing package imports...")
    failed_imports = []
    optional_failed = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            elif package == 'imblearn':
                __import__('imblearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    # Test optional packages
    profiling_available = False
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
            profiling_available = True
            break
        except ImportError:
            optional_failed.append(package)
    
    if not profiling_available:
        print("⚠ Data profiling packages not available (ydata_profiling/pandas_profiling)")
        print("  You can still run all analyses, but automated profiling will be skipped")
    
    return failed_imports

def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    
    # Add src to path
    src_path = Path(__file__).parent / 'src'
    sys.path.insert(0, str(src_path))
    
    modules_to_test = [
        'utils.config',
        'data.data_loader',
        'data.data_preprocessor',
        'features.feature_selector',
        'features.feature_engineering',
        'models.model_trainer',
        'models.model_evaluator'
    ]
    
    failed_modules = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_modules.append(module)
    
    return failed_modules

def test_directories():
    """Test if all required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/external',
        'notebooks',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'models',
        'reports/figures',
        'reports/results'
    ]
    
    missing_dirs = []
    project_root = Path(__file__).parent
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path}")
            missing_dirs.append(dir_path)
    
    return missing_dirs

def main():
    """Run all tests."""
    print("STRESS LEVEL PREDICTION PROJECT SETUP TEST")
    print("=" * 50)
    
    # Test package imports
    failed_imports = test_imports()
    
    # Test custom modules
    failed_modules = test_custom_modules()
    
    # Test directories
    missing_dirs = test_directories()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if not failed_imports and not failed_modules and not missing_dirs:
        print("🎉 ALL TESTS PASSED!")
        print("Your project setup is complete and ready to use.")
        print("\nNext steps:")
        print("1. Add your dataset to data/raw/")
        print("2. Run the notebooks in order (01, 02, 03, 04, 05)")
        print("3. Start with 01_data_exploration.ipynb")
    else:
        print("❌ SOME TESTS FAILED:")
        
        if failed_imports:
            print(f"\nFailed package imports: {failed_imports}")
            print("Try: pip install -r requirements.txt")
        
        if failed_modules:
            print(f"\nFailed custom modules: {failed_modules}")
            print("Check Python path and module structure")
        
        if missing_dirs:
            print(f"\nMissing directories: {missing_dirs}")
            print("Check directory structure")
    
    print(f"\nProject location: {Path(__file__).parent}")

if __name__ == "__main__":
    main()
