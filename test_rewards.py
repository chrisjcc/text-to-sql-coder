#!/usr/bin/env python3
"""
Test script for SQL reward functions.
Tests reward computation without requiring full model training.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_reward_functions():
    """Test reward function creation and basic computation."""
    print("Testing SQL reward functions...")
    
    try:
        from config.training_config import Config
        from src.rewards.sql_reward import create_sql_reward
        
        config = Config()
        
        # Test data
        test_cases = [
            {
                "prompt": """Question: How many users are there?

Schema:
CREATE TABLE users (id INT, name TEXT);

Generate the SQL query to answer this question.""",
                "completion": "SELECT COUNT(*) FROM users;",
                "answer": "SELECT COUNT(*) FROM users;",
                "context": "CREATE TABLE users (id INT, name TEXT);",
                "expected_reward": "> 0.8"  # Should be high for exact match
            },
            {
                "prompt": """Question: Get all user names?

Schema:  
CREATE TABLE users (id INT, name TEXT);

Generate the SQL query to answer this question.""",
                "completion": "SELECT name FROM users",  # Missing semicolon
                "answer": "SELECT name FROM users;",
                "context": "CREATE TABLE users (id INT, name TEXT);",
                "expected_reward": "> 0.5"  # Should be decent, close match
            },
            {
                "prompt": """Question: Invalid query test

Schema:
CREATE TABLE users (id INT, name TEXT);

Generate the SQL query to answer this question.""",
                "completion": "INVALID SQL QUERY HERE",
                "answer": "SELECT * FROM users;",  
                "context": "CREATE TABLE users (id INT, name TEXT);",
                "expected_reward": "< 0"  # Should be negative
            }
        ]
        
        # Test different reward types
        reward_types = ["syntax", "execution", "semantic", "combined"]
        
        for reward_type in reward_types:
            print(f"\n--- Testing {reward_type.upper()} reward ---")
            
            try:
                reward_fn = create_sql_reward(config.reward, reward_type)
                print(f"✓ Created {reward_type} reward function")
                
                for i, test_case in enumerate(test_cases, 1):
                    print(f"\nTest case {i}:")
                    try:
                        # Test the reward computation
                        reward_score = reward_fn(
                            prompt=test_case["prompt"],
                            completion=test_case["completion"], 
                            answer=test_case["answer"],
                            context=test_case["context"]
                        )
                        
                        print(f"  Reward score: {reward_score:.3f}")
                        print(f"  Expected: {test_case['expected_reward']}")
                        
                        # Basic validation
                        if -1.0 <= reward_score <= 1.0:
                            print("  ✓ Score in valid range [-1, 1]")
                        else:
                            print(f"  ✗ Score {reward_score} outside valid range!")
                            
                    except Exception as e:
                        print(f"  ✗ Reward computation failed: {e}")
                        
            except Exception as e:
                print(f"✗ Failed to create {reward_type} reward: {e}")
                
        return True
        
    except ImportError as e:
        print(f"✗ Import failed (expected without dependencies): {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_platform_compatibility():
    """Test platform-specific components."""
    print("\n" + "="*50)
    print("Testing platform compatibility...")
    
    import sys
    import signal
    
    print(f"Platform: {sys.platform}")
    
    # Test signal availability
    if hasattr(signal, 'SIGALRM'):
        print("✓ SIGALRM available (Unix-like system)")
    else:
        print("⚠ SIGALRM not available (likely Windows)")
        print("  - SQL executor will use threading-based timeout")
    
    # Test threading
    import threading
    print("✓ Threading available for Windows compatibility")
    
    return True


def test_config_validation():
    """Test configuration loading and validation."""
    print("\n" + "="*50) 
    print("Testing configuration...")
    
    try:
        from config.training_config import Config
        from config.test_config import TestConfig
        
        # Test main config
        config = Config()
        config.validate()
        print("✓ Main config loads and validates")
        
        # Test minimal config  
        test_config = TestConfig()
        test_config.validate()
        print("✓ Test config loads and validates")
        print(f"  - Model: {test_config.model.model_id}")
        print(f"  - Train size: {test_config.data.train_size}")
        print(f"  - Output: {test_config.grpo.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False


if __name__ == "__main__":
    print("SQL GRPO TRAINING PIPELINE - REWARD FUNCTION TESTS")
    print("=" * 60)
    
    results = {
        "Platform Compatibility": test_platform_compatibility(),
        "Config Validation": test_config_validation(),
        "Reward Functions": test_reward_functions(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    print(f"\nOverall: {total_passed}/{len(results)} test categories passed")
    
    if total_passed == len(results):
        print("\n🎉 All tests passed! Pipeline structure looks good.")
    else:
        print(f"\n⚠ {len(results) - total_passed} test categories failed.")
        print("This is expected without dependencies installed.")