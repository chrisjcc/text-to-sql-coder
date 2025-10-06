#!/usr/bin/env python3
"""
Test script to identify import and dependency issues in the codebase.
This script will help identify what breaks without actually installing all dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_basic_imports():
    """Test basic Python imports that should work."""
    try:
        import sqlite3
        print("✓ sqlite3: OK")
    except ImportError as e:
        print(f"✗ sqlite3: {e}")
    
    try:
        import hashlib
        print("✓ hashlib: OK") 
    except ImportError as e:
        print(f"✗ hashlib: {e}")
        
    try:
        import signal
        print("✓ signal: OK")
        # Test Unix-specific signal
        if hasattr(signal, 'SIGALRM'):
            print("✓ signal.SIGALRM: Available (Unix)")
        else:
            print("⚠ signal.SIGALRM: NOT available (Windows)")
    except ImportError as e:
        print(f"✗ signal: {e}")

def test_config_imports():
    """Test configuration module imports."""
    try:
        from config.training_config import Config
        print("✓ Config import: OK")
        
        config = Config()
        config.validate()
        print("✓ Config validation: OK")
        
    except Exception as e:
        print(f"✗ Config: {e}")

def test_external_dependencies():
    """Test external dependencies that need installation."""
    dependencies = [
        'torch',
        'transformers', 
        'datasets',
        'peft',
        'trl',
        'verifiers',
        'sqlparse',
        'huggingface_hub',
        'wandb',
        'python-dotenv'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep}: OK")
        except ImportError as e:
            print(f"✗ {dep}: NOT installed - {e}")

def test_hub_utils_import_issue():
    """Test the deprecated huggingface_hub import."""
    try:
        # This is the problematic import in hub_utils.py
        from huggingface_hub import repo_exists
        print("✓ repo_exists direct import: OK (but deprecated)")
    except ImportError:
        print("⚠ repo_exists direct import: DEPRECATED/FAILED")
        
        try:
            # This is the correct way
            from huggingface_hub import HfApi
            api = HfApi()
            print("✓ HfApi.repo_exists: OK (correct approach)")
        except ImportError as e:
            print(f"✗ HfApi: {e}")

if __name__ == "__main__":
    print("="*60)
    print("IMPORT ANALYSIS FOR GRPO TRAINING PIPELINE")
    print("="*60)
    
    print("\n1. Basic Python imports:")
    test_basic_imports()
    
    print("\n2. Config module:")
    test_config_imports()
    
    print("\n3. External dependencies:")
    test_external_dependencies()
    
    print("\n4. Hub utils import issue:")
    test_hub_utils_import_issue()
    
    print("\n" + "="*60)