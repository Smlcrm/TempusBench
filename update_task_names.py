#!/usr/bin/env python3
"""
Script to replace dataset.path with task.name and remove dataset.path from all task.yaml files.
"""

import os
import glob
from pathlib import Path

def update_task_names():
    """Replace dataset.path with task.name and remove dataset.path."""
    
    # Find all task.yaml files
    tasks_dir = Path("tempus_bench/tasks")
    task_files = list(tasks_dir.rglob("task.yaml"))
    
    print(f"Found {len(task_files)} task.yaml files")
    
    modified_count = 0
    
    for task_file in task_files:
        try:
            # Extract task name from the directory name
            task_dir = task_file.parent
            task_name = task_dir.name
            
            # Read the current task.yaml file
            with open(task_file, 'r') as f:
                content = f.read()
            
            # Parse and modify the content
            lines = content.split('\n')
            modified_lines = []
            
            # Track if we've added task.name and if we're in dataset section
            task_name_added = False
            in_dataset_section = False
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # Check if we're entering the dataset section
                if stripped == 'dataset:':
                    in_dataset_section = True
                    modified_lines.append(line)
                    continue
                
                # Check if we're leaving the dataset section (next top-level key)
                if in_dataset_section and stripped and not line.startswith(' ') and not line.startswith('\t'):
                    in_dataset_section = False
                
                # If we're in dataset section and find path line, skip it
                if in_dataset_section and 'path:' in line:
                    print(f"Removing path line from {task_file}: {line.strip()}")
                    continue
                
                # Add task.name after the task: line
                if stripped == 'task:' and not task_name_added:
                    modified_lines.append(line)
                    # Add task.name with proper indentation
                    indent = len(line) - len(line.lstrip())
                    modified_lines.append(' ' * (indent + 2) + f'name: {task_name}')
                    task_name_added = True
                    print(f"Added task.name to {task_file}: name: {task_name}")
                    continue
                
                modified_lines.append(line)
            
            # Write back the modified content
            new_content = '\n'.join(modified_lines)
            if new_content != content:
                with open(task_file, 'w') as f:
                    f.write(new_content)
                
                print(f"Modified: {task_file}")
                modified_count += 1
            else:
                print(f"No changes needed: {task_file}")
                
        except Exception as e:
            print(f"Error processing {task_file}: {e}")
    
    print(f"\nTotal files modified: {modified_count}")

if __name__ == "__main__":
    update_task_names()
