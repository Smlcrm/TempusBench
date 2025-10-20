#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import ast
import re

# Add the benchmarking_pipeline to the path
sys.path.append('/Users/gandresr/Documents/GitHub/benchmark')

def test_improved_parsing():
    """Test the improved parsing logic for empty values."""
    
    print("=== Testing Improved Parsing Logic ===")
    
    # Test cases with different empty value patterns
    test_cases = [
        '[[3.0, None, None, None, None, None, 2.0]]',  # Already has None
        '[[3.0, , , , , , 2.0]]',  # Empty values
        '[[3.0, "", "", "", "", "", 2.0]]',  # Empty strings
        '[[3.0, "None", "None", "None", "None", "None", 2.0]]',  # String "None"
        '[[, 1, 2, 3]]',  # Leading empty
        '[[1, 2, 3, ]]',  # Trailing empty
        '[[1, , 3, , 5]]',  # Mixed empty values
        '[[1.0, 2.0, 3.0, 4.0, 5.0]]',  # No empty values
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        
        # Apply the improved parsing logic
        target_raw = test_case
        
        # Clean the target string to handle empty values before parsing
        target_cleaned_str = re.sub(r',\s*,', ', None,', target_raw)
        target_cleaned_str = re.sub(r'\[\s*,', '[None,', target_cleaned_str)
        target_cleaned_str = re.sub(r',\s*\]', ', None]', target_cleaned_str)
        
        print(f"  After regex cleaning: {target_cleaned_str}")
        
        try:
            target_parsed = ast.literal_eval(target_cleaned_str)
            print(f"  Parsed successfully: {target_parsed}")
        except SyntaxError:
            # If still fails, try more aggressive cleaning
            target_cleaned_str = target_cleaned_str.replace('""', 'None').replace("''", 'None')
            print(f"  After string replacement: {target_cleaned_str}")
            try:
                target_parsed = ast.literal_eval(target_cleaned_str)
                print(f"  Parsed successfully after string replacement: {target_parsed}")
            except SyntaxError as e:
                print(f"  ERROR: Still failed to parse: {e}")
                continue
        
        # Convert to numpy array with NaN handling
        target_cleaned = []
        for row in target_parsed:
            cleaned_row = []
            for val in row:
                if val is None or val == "" or val == "None":
                    cleaned_row.append(np.nan)
                else:
                    try:
                        cleaned_row.append(float(val))
                    except (ValueError, TypeError):
                        cleaned_row.append(np.nan)
            target_cleaned.append(cleaned_row)
        
        target = np.array(target_cleaned)
        print(f"  Final result shape: {target.shape}")
        print(f"  Final result dtype: {target.dtype}")
        print(f"  Final result: {target}")
        print(f"  Has NaN: {np.isnan(target).any()}")
        print(f"  NaN count: {np.isnan(target).sum()}")

if __name__ == "__main__":
    test_improved_parsing()
