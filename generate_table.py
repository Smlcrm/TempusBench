#!/usr/bin/env python3
"""
Script to parse all task.yaml files and generate the LaTeX table
"""
import os
import yaml
import re
from pathlib import Path
from collections import defaultdict

# Mapping from folder names to (benchmark_category, benchmark_type, task_description, citation)
BENCHMARK_MAPPING = {
    # Univariate tasks
    'swjobpostings_software_univariate': ('Domain', 'Software', 'Software Development Job Postings', r'\citep{fred2025indeed}'),
    'software_nonstationary_univariate': ('Trend', 'Non-stationary', 'Software Development Job Postings', r'\citep{fred2025indeed}'),
    'synthetic_additive2_univariate': ('Decomposition', 'Additive', 'Synthetically Generated Additive', r'(\Cref{ssec:cyclic_additive})'),
    'synthetic_multiplticative_univariate': ('Decomposition', 'Multiplicative', 'Synthetically Generated Multiplicative', r'(\Cref{ssec:cyclic_multiplicative})'),
    'coinbase_days_univariate': ('Frequency', 'Days', 'Coinbase Litecoin', r'\citep{fred2025cbltcusd}'),
    'federalfuns_weeks_univariate': ('Frequency', 'Weeks', 'Federal Funds Effective Rate', r'\citep{fred2025ff}'),
    'inventories_months_univariate': ('Frequency', 'Months', 'Inventories to Sales Ratio', r'\citep{fred2025mnfctrirsa}'),
    'german_quaterly_univariate': ('Frequency', 'Quarters', 'German House Prices', r'\citep{fred2025qder628bis}'),
    'pconsumption_years_univariate': ('Frequency', 'Years', 'Personal Consumption Expenditures', r'\citep{fred2025pce}'),
    'synthetic_cyclic_univariate': ('Seasonality', 'Periodic', 'Synthetic Cyclic', ''),
    'synthetic_nonstationary_univariate': ('Seasonality', 'Quasiperiodic', 'Synthetic Non-stationary', ''),
    'electricity_energy_univariate': ('Domain', 'Energy', 'Room SplitSmart', r'\citep{bitspilani2024splitsmart}'),
    'madrid_transport_univariate': ('Domain', 'Transport', 'Madrid BEN pollution', r'\citep{banuelos-gimeno2024initial}'),
    'delhi_climate_univariate': ('Domain', 'Climate', 'Delhi Climate', r'\citep{sumanthvrao2021dailyclimate}'),
    'wtraffic_web_univariate': ('Domain', 'Web', 'Web Traffic', r'\citep{raminhuseyn2024webtraffic}'),
    'germanhouses_sales_univariate': ('Domain', 'Sales', 'German House Prices', r'\citep{fred2025qder628bis}'),
    'soil_nature_univariate': ('Domain', 'Nature', 'Soil Monitoring', r'\citep{noeyislearning2024soil}'),
    'coinbase_economics_univariate': ('Domain', 'Economics/Finance', 'Coinbase Litecoin', r'\citep{fred2025cbltcusd}'),
    'employees_healthcare_univariate': ('Domain', 'Healthcare', 'Employees Health Care', r'\citep{fred2025ces6562}'),
    'invetories_manufacturing_univariate': ('Domain', 'Manufacturing', 'Inventories to Sales Ratio', r'\citep{fred2025mnfctrirsa}'),
    'patient_sparse_univariate': ('Data sparsity', 'Sparse', 'Patient Chart', r'\citep{johnson2019mimic}'),
    'chickenpox_dense_univariate': ('Data sparsity', 'Dense', 'Chicken Pox', r'\citep{uci2021chickenpox}'),
    'forestfires_continuous_univariate': ('Value type', 'Continuous', 'Forest Fires', r'\citep{cortez2007forest}'),
    'occupancy_count_univariate': ('Value type', 'Count', 'Occupancy', r'\citep{singh2018room}'),
    'absent_binary_univariate': ('Value type', 'Binary', 'Absenteeism at Work', r'\citep{martiniano2012absenteeism}'),
    'retail_categorical_univariate': ('Value type', 'Categorical', 'Online Retail', r'\citep{chen2015online}'),
    
    # Multivariate tasks
    'utah_manufacturing_multivariate': ('Frequency', 'Seconds', 'Utah Drilling', r'\citep{egi2025utahforge}'),
    'ltstock_minutes_multivariate': ('Frequency', 'Minutes', 'Historical Stock Data (2003-2024)', r'\citep{deltatrup2024lt}'),
    'ltstock_longest_multivariate': ('Frequency', 'Minutes', 'Historical Stock Data (2003-2024, Longest)', r'\citep{deltatrup2024lt}'),
    # Note: utah_manufacturing also appears in Domain/Manufacturing in original table, but we'll use Frequency:Seconds as primary
    'madrid_hours_multivariate': ('Frequency', 'Hours', 'Madrid Transport Pollution', r'\citep{ignacioqg2022pollution}'),
    'indiagold_days_multivariate': ('Frequency', 'Days', 'Gold Price in India', r'\citep{chodavadiya2025goldprice}'),
    'baggage_months_multivariate': ('Frequency', 'Months', 'Airlines Baggage Complains', r'\citep{gabrielsantello2023airline}'),
    'splitsmart_energy_multivariate': ('Domain', 'Energy', 'Room SplitSmart', r'\citep{bitspilani2024splitsmart}'),
    'madrid_transport_multivariate': ('Domain', 'Transport', 'Madrid BEN pollution', r'\citep{banuelos-gimeno2024initial}'),
    'madrid_noisy_multivariate': ('Domain', 'Transport', 'Madrid BEN pollution (Noisy)', r'\citep{ignacioqg2022pollution}'),
    'batadal_software_multivariate': ('Domain', 'Software', r'\makecell[c]{Cyber Attacks on \\Water Distribution Networks}', r'\citep{taormina2018battle}'),
    'baggage_sales_multivariate': ('Domain', 'Sales', 'Airlines Baggage Complains', r'\citep{gabrielsantello2023airline}'),
    'soil_nature_multivariate': ('Domain', 'Nature', 'Soil Monitoring', r'\citep{noeyislearning2024soil}'),
    'soil_500_multivariate': ('Domain', 'Nature', 'Soil Monitoring (500)', r'\citep{noeyislearning2024soil}'),
    'goldindia_economics_multivariate': ('Domain', 'Economics/Finance', 'Gold Price in India', r'\citep{chodavadiya2025goldprice}'),
    'goldindia_real_multivariate': ('Domain', 'Economics/Finance', 'Gold Price in India (Real)', r'\citep{chodavadiya2025goldprice}'),
    'nyccovid_healthcare_multivariate': ('Domain', 'Healthcare', 'NYC Covid Cases', r'\citep{nyc2025covid}'),
    'goldindia_dense_multivariate': ('Data sparsity', 'Dense', 'Gold Price in India', r'\citep{chodavadiya2025goldprice}'),
    'goldindia_continuous_multivariate': ('Value type', 'Continuous', 'Gold Price in India', r'\citep{chodavadiya2025goldprice}'),
    'madrid_count_multivariate': ('Value type', 'Count', 'Madrid BEN pollution', r'\citep{banuelos-gimeno2024initial}'),
    'batadal_nonstationary_multivariate': ('Trend', 'Non-stationary', 'Electricity Consumption', r'\citep{10.1145/3209978.3210006}'),
    'madrid_cyclical_multivariate': ('Seasonality', 'Periodic', 'Madrid Transport (Cyclical)', r'\citep{ignacioqg2022pollution}'),
    'baggage_100_multivariate': ('Domain', 'Transport', 'Airlines Baggage Complains (100)', r'\citep{gabrielsantello2023airline}'),
    # Exclude test tasks
    # 'multivariate_test': ('Domain', 'Test', 'Multivariate Test', ''),
}

def parse_task_yaml(task_path):
    """Parse a task.yaml file and extract relevant information"""
    with open(task_path, 'r') as f:
        content = f.read()
    
    # Parse YAML
    task_data = yaml.safe_load(content)
    
    # Extract from comments
    variates_match = re.search(r'number of variates\s*:\s*(\d+)', content)
    steps_match = re.search(r'number of time-steps\s*:\s*(\d+)', content)
    
    return {
        'task_name': task_data['task']['task_name'],
        'context_window': task_data['task']['context_window'],
        'forecast_horizon': task_data['task']['forecast_horizon'],
        'num_variates': int(variates_match.group(1)) if variates_match else None,
        'num_steps': int(steps_match.group(1)) if steps_match else None,
    }

def get_benchmark_info(folder_name):
    """Get benchmark category and task description from folder name"""
    if folder_name in BENCHMARK_MAPPING:
        return BENCHMARK_MAPPING[folder_name]
    
    # Default fallback
    tasktype = 'Univariate' if 'univariate' in folder_name else 'Multivariate'
    return (None, None, folder_name.replace('_', ' ').title(), '')

def generate_latex_table(all_tasks):
    """Generate LaTeX table from task data"""
    
    # Group by benchmark category and type
    grouped = defaultdict(list)
    for task in all_tasks:
        key = (task['benchmark_cat'], task['benchmark_type'])
        grouped[key].append(task)
    
    # Sort by benchmark category
    sorted_groups = sorted(grouped.items(), key=lambda x: (
        ['Trend', 'Decomposition', 'Frequency', 'Seasonality', 'Domain', 'Data sparsity', 'Value type'].index(x[0][0]) if x[0][0] in ['Trend', 'Decomposition', 'Frequency', 'Seasonality', 'Domain', 'Data sparsity', 'Value type'] else 99,
        x[0][1]
    ))
    
    lines = []
    lines.append(r'\begin{table*}[!htbp]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{5pt}')
    lines.append(r'\renewcommand{\arraystretch}{1.1}')
    lines.append(r'\caption{Summary of datasets used for benchmark tasks.}')
    lines.append(r'\label{tab:benchmark_summary}')
    lines.append(r'\begin{tabular}{l l c c c c}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Benchmark} & \textbf{Task} & \textbf{l} & \textbf{h} & \textbf{n} & \textbf{m} \\')
    lines.append(r'\midrule')
    
    current_category = None
    for (benchmark_cat, benchmark_type), tasks in sorted_groups:
        if benchmark_cat != current_category:
            if current_category is not None:
                lines.append(r'\hline')
            current_category = benchmark_cat
            lines.append(rf'\multicolumn{{6}}{{c}}{{\textbf{{{benchmark_cat}}}}} \\')
            lines.append(r'\hline')
        
        # Sort tasks within group by tasktype, then task name
        tasks.sort(key=lambda x: (x['tasktype'], x['task']))
        
        for task in tasks:
            task_desc = task['task']
            if task['citation']:
                task_desc += f" {task['citation']}"
            
            # Format benchmark column as "Task type (Benchmark type)"
            benchmark_col = f"{task['tasktype']} ({benchmark_type})"
            
            lines.append(
                rf"{benchmark_col} & {task_desc} & "
                rf"{task['l']} & {task['h']} & {task['n']} & {task['m']} \\"
            )
    
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\vspace{-2mm}')
    lines.append(r'\end{table*}')
    
    return '\n'.join(lines)

def main():
    tasks_dir = Path('tempus_bench/tasks')
    all_tasks = []
    seen_tasks = set()  # Track to avoid duplicates
    
    # Process all task.yaml files
    for task_dir in tasks_dir.rglob('task.yaml'):
        folder_name = task_dir.parent.name
        
        # Skip test tasks
        if 'test' in folder_name.lower():
            continue
        
        tasktype = 'Univariate' if 'univariate' in str(task_dir.parent) else 'Multivariate'
        
        try:
            task_info = parse_task_yaml(task_dir)
            benchmark_cat, benchmark_type, task_desc, citation = get_benchmark_info(folder_name)
            
            # Create unique key to avoid duplicates (using folder name to handle same description in different contexts)
            task_key = folder_name
            if task_key in seen_tasks:
                continue
            seen_tasks.add(task_key)
            
            all_tasks.append({
                'folder': folder_name,
                'tasktype': tasktype,
                'benchmark_cat': benchmark_cat or 'Unknown',
                'benchmark_type': benchmark_type or 'Unknown',
                'task': task_desc,
                'citation': citation,
                'l': task_info['context_window'],
                'h': task_info['forecast_horizon'],
                'n': task_info['num_steps'],
                'm': task_info['num_variates'],
            })
        except Exception as e:
            print(f"Error processing {task_dir}: {e}")
            continue
    
    # Generate LaTeX
    latex_table = generate_latex_table(all_tasks)
    print(latex_table)
    
    return all_tasks, latex_table

if __name__ == '__main__':
    tasks, table = main()
    with open('tasks_tables_generated.tex', 'w') as f:
        f.write(table)
