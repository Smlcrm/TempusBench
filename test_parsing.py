#!/usr/bin/env python3

import sys
sys.path.append('/Users/gandresr/Documents/GitHub/benchmark')

from benchmarking_pipeline.pipeline.preprocessor import Preprocessor

def test_parsing():
    """Test the refactored parsing logic."""
    
    config = {
        'task': {
            'dataset': {
                'normalize': False,
                'handle_missing': 'interpolate'
            }
        }
    }
    
    preprocessor = Preprocessor(config)
    
    # Test cases with different empty value patterns
    test_cases = [
        ('[[3.0, None, None, None, None, None, 2.0]]', "Already has None"),
        ('[[3.0, , , , , , 2.0]]', "Empty values"),
        ('[[3.0, "", "", "", "", "", 2.0]]', "Empty strings"),
        ('[[3.0, "None", "None", "None", "None", "None", 2.0]]', "String None"),
        ('[[, 1, 2, 3]]', "Leading empty"),
        ('[[1, 2, 3, ]]', "Trailing empty"),
        ('[[1, , 3, , 5]]', "Mixed empty values"),
        ('[[1.0, 2.0, 3.0, 4.0, 5.0]]', "No empty values"),
    ]
    
    print("=== Testing Refactored Parsing Logic ===")
    
    for i, (test_case, description) in enumerate(test_cases):
        print(f"\nTest case {i+1}: {description}")
        print(f"Input: {test_case}")
        
        try:
            timestamps, time_start, freq, target = preprocessor.clean(
                time_start="2001-01-01T00",
                freq="1H", 
                target_raw=test_case
            )
            
            print(f"Success! Shape: {target.shape}, Dtype: {target.dtype}")
            print(f"Has NaN: {target.dtype == 'float64' and hasattr(target, 'isnan') and target.dtype == 'float64'}")
            print(f"Result: {target}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_parsing()
