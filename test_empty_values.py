#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import ast

# Add the benchmarking_pipeline to the path
sys.path.append('/Users/gandresr/Documents/GitHub/benchmark')

def test_empty_values():
    """Test what happens with CSV files containing empty values."""
    
    print("=== Testing CSV with Empty Values ===")
    
    # Create a test CSV with empty values
    test_csv_content = """item_id,start,freq,target
0,2001-01-01T00,1H,"[[3.0, None, None, None, None, None, 2.0], [1.0, , , , , , 5.0]]"
"""
    
    # Write test CSV
    test_csv_path = "/tmp/test_empty_values.csv"
    with open(test_csv_path, 'w') as f:
        f.write(test_csv_content)
    
    print(f"Test CSV content:")
    print(test_csv_content)
    
    # Test pandas read_csv behavior
    print(f"\n=== Pandas read_csv behavior ===")
    try:
        file_data = pd.read_csv(test_csv_path)
        print(f"CSV shape: {file_data.shape}")
        print(f"Columns: {file_data.columns.tolist()}")
        print(f"Data types: {file_data.dtypes}")
        print(f"First row target: {file_data['target'].iloc[0]}")
        print(f"Target type: {type(file_data['target'].iloc[0])}")
    except Exception as e:
        print(f"ERROR reading CSV: {e}")
        return
    
    # Test ast.literal_eval behavior
    print(f"\n=== ast.literal_eval behavior ===")
    try:
        target_raw = file_data["target"].iloc[0]
        print(f"Raw target: {target_raw}")
        target_parsed = ast.literal_eval(target_raw)
        print(f"Parsed target: {target_parsed}")
        print(f"Parsed type: {type(target_parsed)}")
        print(f"First row: {target_parsed[0]}")
        print(f"Second row: {target_parsed[1]}")
    except Exception as e:
        print(f"ERROR parsing target: {e}")
        return
    
    # Test current cleaning logic
    print(f"\n=== Current cleaning logic ===")
    try:
        target_cleaned = []
        for row in target_parsed:
            print(f"Processing row: {row}")
            cleaned_row = [float(val) if val is not None else np.nan for val in row]
            print(f"Cleaned row: {cleaned_row}")
            target_cleaned.append(cleaned_row)
        target = np.array(target_cleaned)
        print(f"Final target shape: {target.shape}")
        print(f"Final target dtype: {target.dtype}")
        print(f"Final target: {target}")
    except Exception as e:
        print(f"ERROR in cleaning: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with different empty value formats
    print(f"\n=== Testing different empty value formats ===")
    
    test_cases = [
        '[[3.0, None, None, None, None, None, 2.0]]',
        '[[3.0, , , , , , 2.0]]',  # This will likely fail
        '[[3.0, "", "", "", "", "", 2.0]]',  # Empty strings
        '[[3.0, "None", "None", "None", "None", "None", 2.0]]',  # String "None"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}: {test_case}")
        try:
            parsed = ast.literal_eval(test_case)
            print(f"  Parsed: {parsed}")
            cleaned = [float(val) if val is not None and val != "" else np.nan for val in parsed[0]]
            print(f"  Cleaned: {cleaned}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Clean up
    os.remove(test_csv_path)

if __name__ == "__main__":
    test_empty_values()
