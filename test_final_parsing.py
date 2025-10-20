#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import ast
import re

def test_final_parsing():
    """Test the final improved parsing logic."""
    
    print("=== Testing Final Improved Parsing Logic ===")
    
    # Test the problematic case
    test_case = '[[3.0, , , , , , 2.0]]'
    print(f"Test case: {test_case}")
    
    # Apply the improved parsing logic
    target_raw = test_case
    
    # Clean the target string to handle empty values before parsing
    target_cleaned_str = target_raw
    while ', ,' in target_cleaned_str or ',,' in target_cleaned_str:
        target_cleaned_str = re.sub(r',\s*,', ', None,', target_cleaned_str)
        print(f"  After iteration: {target_cleaned_str}")
    
    # Handle leading empty values like "[, 1, 2]" -> "[None, 1, 2]"
    target_cleaned_str = re.sub(r'\[\s*,', '[None,', target_cleaned_str)
    # Handle trailing empty values like "[1, 2, ]" -> "[1, 2, None]"
    target_cleaned_str = re.sub(r',\s*\]', ', None]', target_cleaned_str)
    
    print(f"  Final cleaned: {target_cleaned_str}")
    
    try:
        target_parsed = ast.literal_eval(target_cleaned_str)
        print(f"  Parsed successfully: {target_parsed}")
        
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
        
    except SyntaxError as e:
        print(f"  ERROR: Failed to parse: {e}")

if __name__ == "__main__":
    test_final_parsing()
