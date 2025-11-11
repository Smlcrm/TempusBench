#!/usr/bin/env python3
"""
Generate LaTeX winrate table from winrate CSV files in main_results directory.
"""

import csv
from pathlib import Path
from collections import defaultdict

def read_winrate_csv(filepath):
    """Read a winrate CSV file and return data as dict."""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row['model_name']
            winrate = row.get('average_win_rate', '').strip()
            data[model] = winrate
    return data

def format_winrate(value):
    """Format winrate value for LaTeX display."""
    if value == '' or value is None:
        return '---'
    
    try:
        num = float(value)
        # Format as 4 decimal places
        return f"{num:.4f}"
    except (ValueError, TypeError):
        return '---'

def get_model_display_name(model):
    """Convert model name to display name."""
    name_mapping = {
        'arima': 'ARIMA',
        'chronos': 'Chronos',
        'croston_classic': 'Croston',
        'exponential_smoothing': 'Exp. Smooth',
        'lafn': 'LAFN',
        'lagllama': 'LagLlama',
        'lstm': 'LSTM',
        'moirai': 'Moirai',
        'moirai_moe': 'Moirai MOE',
        'moment': 'Moment',
        'prophet': 'Prophet',
        'random_forest': 'RF',
        'seasonal_naive': 'Seas. Naive',
        'svr': 'SVR',
        'tabpfn': 'TabPFN',
        'theta': 'Theta',
        'timesfm': 'TimesFM',
        'tiny_time_mixer': 'TTM',
        'toto': 'Toto',
        'xgboost': 'XGBoost'
    }
    return name_mapping.get(model, model.replace('_', ' ').title())

def main():
    """Main function to generate LaTeX winrate table."""
    main_results_dir = Path('main_results')
    
    # Metric names and their display names
    metrics = {
        'mae': 'MAE',
        'mape': 'MAPE',
        'mase': 'MASE',
        'rmse': 'RMSE',
        'quantile_score': 'Quantile Score',
        'weighted_interval_score': 'Weighted Interval Score',
        'crps': 'CRPS'
    }
    
    # Read all winrate files
    all_data = {}
    all_models = set()
    
    for metric, display_name in metrics.items():
        csv_file = main_results_dir / f'{metric}_winrate.csv'
        if csv_file.exists():
            data = read_winrate_csv(csv_file)
            all_data[metric] = data
            all_models.update(data.keys())
    
    # Sort models (you can customize this ordering)
    # Get all models and sort them
    models = sorted(all_models)
    
    # Generate LaTeX content
    latex_content = """\\begin{table}[h]

\\caption{Average Win Rates for deterministic and probabilistic forecasting models.}

\\centering

"""
    
    # Group metrics into pairs for side-by-side subtables
    metric_list = list(metrics.items())
    
    # Create subtables in pairs
    for i in range(0, len(metric_list), 2):
        metric1 = metric_list[i]
        metric2 = metric_list[i + 1] if i + 1 < len(metric_list) else None
        
        # First subtable
        latex_content += f"""\\begin{{subtable}}{{0.48\\textwidth}}

\\centering

\\caption{{Average Win Rate for {metric1[1]} Metric.}}

\\label{{tab:{metric1[0]}_winrate}}

\\begin{{tabular}}{{lc}}

\\toprule

\\textbf{{Model}} & \\textbf{{Win Rate}} \\

\\midrule

"""
        # Add rows for first metric
        data1 = all_data.get(metric1[0], {})
        for model in models:
            if model in data1:
                display_name = get_model_display_name(model)
                winrate = format_winrate(data1[model])
                latex_content += f"{display_name} & {winrate} \\\\\n"
        
        latex_content += """\\bottomrule

\\end{tabular}

\\end{subtable}

"""
        
        # Second subtable if exists
        if metric2:
            latex_content += """\\hfill

"""
            latex_content += f"""\\begin{{subtable}}{{0.48\\textwidth}}

\\centering

\\caption{{Average Win Rate for {metric2[1]} Metric}}

\\label{{tab:{metric2[0]}_winrate}}

\\begin{{tabular}}{{lc}}

\\toprule

\\textbf{{Model}} & \\textbf{{Win Rate}} \\

\\midrule

"""
            # Add rows for second metric
            data2 = all_data.get(metric2[0], {})
            for model in models:
                if model in data2:
                    display_name = get_model_display_name(model)
                    winrate = format_winrate(data2[model])
                    latex_content += f"{display_name} & {winrate} \\\\\n"
            
            latex_content += """\\bottomrule

\\end{tabular}

\\end{subtable}

"""
        
        # Add spacing between pairs
        if i + 2 < len(metric_list):
            latex_content += """\\vspace{0.3cm}

"""
    
    latex_content += """\\vspace{-0.4cm}

\\end{table}

"""
    
    # Write output file
    output_file = main_results_dir / 'winrate_table_generated.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"Generated LaTeX file: {output_file}")

if __name__ == '__main__':
    main()





