#!/usr/bin/env python3
"""
Generate LaTeX tables from pivot CSV files in main_results directory.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

def format_number(value):
    """Format a number for LaTeX display."""
    if value == '' or value is None:
        return '---'
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        return '---'
    
    # Handle very small numbers with scientific notation (like 1.129e-5)
    if abs(num) < 1e-3 and num != 0:
        # Format as 1.129e-5 (3 digits after decimal, then e notation)
        return f"{num:.3e}"
    # Handle very large numbers with scientific notation
    elif abs(num) > 1e6:
        return f"{num:.3e}"
    # Format regular numbers - match reference style
    elif abs(num) < 1:
        # For numbers < 1, use up to 5 significant digits
        return f"{num:.5g}"
    elif abs(num) < 10:
        # 4 decimal places for numbers 1-10
        return f"{num:.4f}"
    elif abs(num) < 100:
        # 4 decimal places for numbers 10-100
        return f"{num:.4f}"
    elif abs(num) < 1000:
        # 4 decimal places for numbers 100-1000
        return f"{num:.4f}"
    else:
        # For larger numbers, use 4 decimal places
        return f"{num:.4f}"

def read_pivot_csv(filepath):
    """Read a pivot CSV file and return data as dict."""
    data = {}
    datasets = []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        datasets = [col for col in reader.fieldnames if col != 'model_name']
        
        for row in reader:
            model = row['model_name']
            data[model] = {}
            for dataset in datasets:
                value = row.get(dataset, '').strip()
                data[model][dataset] = value
    
    return data, datasets

def find_best_values(data, datasets):
    """Find the minimum value for each dataset (column)."""
    best = {}
    for dataset in datasets:
        best_value = None
        for model, values in data.items():
            val = values.get(dataset, '').strip()
            if val and val != '---':
                try:
                    num = float(val)
                    if best_value is None or num < best_value:
                        best_value = num
                except (ValueError, TypeError):
                    pass
        best[dataset] = best_value
    return best

def get_short_dataset_name(dataset):
    """Convert long dataset name to short name for table header."""
    # Remove common suffixes
    name = dataset.replace('_univariate_', '_').replace('_univariate', '')
    parts = name.split('_')
    
    # Extract meaningful parts (usually first 2-3 words)
    if len(parts) >= 2:
        # Take first part (task name) and capitalize properly
        short = parts[0].replace('_', ' ').title()
        # If it's very short, add the second part
        if len(short) < 8 and len(parts) > 1:
            short += ' ' + parts[1].title()
    else:
        short = name.replace('_', ' ').title()
    
    # Truncate if too long (max ~15 chars for readability)
    if len(short) > 15:
        short = short[:12] + '...'
    
    return short

def generate_latex_table(metric_name, data, datasets, best_values):
    """Generate LaTeX table for a metric."""
    # Get all models, sorted
    models = sorted(data.keys())
    
    # Escape underscores in model names for LaTeX
    def escape_model(name):
        return name.replace('_', '\\_')
    
    # Generate table header
    latex = f"""\\begin{{sidewaystable}}[!p]

\\centering

\\caption{{{metric_name.upper()} Results}}

\\label{{tab:{metric_name.lower()}}}

\\begin{{adjustbox}}{{width=0.95\\textheight,center}}

\\small

\\begin{{tabular}}{{l{'c' * len(datasets)}}}

\\toprule

Model"""
    
    # Add dataset column headers (shortened names)
    for dataset in datasets:
        short_name = get_short_dataset_name(dataset)
        latex += f" & {short_name}"
    
    latex += " \\\\\n\\midrule\n"
    
    # Add data rows
    for model in models:
        latex += escape_model(model)
        for dataset in datasets:
            value = data[model].get(dataset, '').strip()
            formatted = format_number(value)
            
            # Check if this is the best value
            if formatted != '---' and best_values.get(dataset) is not None:
                try:
                    num = float(value)
                    if abs(num - best_values[dataset]) < 1e-10:
                        formatted = f"\\textbf{{{formatted}}}"
                except (ValueError, TypeError):
                    pass
            
            latex += f" & {formatted}"
        latex += " \\\\\n"
    
    latex += """\\bottomrule
\\end{tabular}
\\end{adjustbox}
\\end{sidewaystable}

"""
    
    return latex

def main():
    """Main function to generate LaTeX file."""
    main_results_dir = Path('main_results')
    
    # Metric names mapping
    metric_names = {
        'mae': 'MAE',
        'mape': 'MAPE',
        'mase': 'MASE',
        'rmse': 'RMSE',
        'quantile_score': 'Quantile Score',
        'weighted_interval_score': 'Weighted Interval Score',
        'crps': 'CRPS'
    }
    
    # Find all pivot CSV files
    pivot_files = {}
    for csv_file in main_results_dir.glob('*_pivot.csv'):
        metric = csv_file.stem.replace('_pivot', '')
        if metric in metric_names:
            pivot_files[metric] = csv_file
    
    # Generate LaTeX content
    latex_content = """% REQUIRED PACKAGES - Add these to your main LaTeX document preamble:
% \\usepackage{rotating}    % for sidewaystable (landscape tables)
% \\usepackage{booktabs}    % for professional table rules
% \\usepackage{adjustbox}   % for table resizing (optional - can remove if not available)

"""
    
    # Process each metric in order
    metric_order = ['mae', 'mape', 'mase', 'rmse', 'quantile_score', 'weighted_interval_score', 'crps']
    
    for metric in metric_order:
        if metric not in pivot_files:
            continue
        
        csv_file = pivot_files[metric]
        print(f"Processing {csv_file.name}...")
        
        data, datasets = read_pivot_csv(csv_file)
        best_values = find_best_values(data, datasets)
        
        table_latex = generate_latex_table(metric_names[metric], data, datasets, best_values)
        latex_content += table_latex
    
    # Write output file
    output_file = main_results_dir / 'pivot_tables.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"\nGenerated LaTeX file: {output_file}")

if __name__ == '__main__':
    main()

