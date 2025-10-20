#!/bin/bash

# Centralized benchmark runner for all dataset configurations
# This script runs benchmarks for all generated config files

CONFIGS_DIR="benchmarking_pipeline/configs/datasets"
DATASETS_DIR="benchmarking_pipeline/datasets"
LOG_FILE="benchmark_runs.log"
ERROR_LOG="benchmark_errors.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Starting benchmark runs at $(date)"
echo "==========================================" | tee -a $LOG_FILE

# Initialize counters
TOTAL=0
SUCCESS=0
FAILED=0

# Function to run a single benchmark
run_benchmark() {
    local config_file=$1
    local dataset_name=$2
    local dataset_type=$3
    local dataset_dir=$4
    
    echo -e "${YELLOW}Running $dataset_name ($dataset_type)...${NC}" | tee -a $LOG_FILE
    
    # Activate conda environment and run benchmark
    conda activate sim.benchmarks
    python benchmarking_pipeline/run_benchmark.py \
        --config "$config_file" \
        --datasets-dir "$dataset_dir" \
        >> $LOG_FILE 2>> $ERROR_LOG
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}SUCCESS: $dataset_name${NC}" | tee -a $LOG_FILE
        ((SUCCESS++))
    else
        echo -e "${RED}FAILED: $dataset_name (exit code: $exit_code)${NC}" | tee -a $LOG_FILE
        ((FAILED++))
    fi
    
    ((TOTAL++))
}

# Run univariate datasets
echo "=========================================="
echo "Running UNIVARIATE datasets..."
echo "==========================================" | tee -a $LOG_FILE

for config in $CONFIGS_DIR/univariate/*.yaml; do
    if [ -f "$config" ]; then
        dataset_name=$(basename $config .yaml)
        dataset_dir="$DATASETS_DIR/univariate/$dataset_name"
        
        if [ -d "$dataset_dir" ]; then
            run_benchmark "$config" "$dataset_name" "univariate" "$dataset_dir"
        else
            echo -e "${RED}SKIP: $dataset_name (dataset directory not found: $dataset_dir)${NC}" | tee -a $LOG_FILE
            ((FAILED++))
            ((TOTAL++))
        fi
    fi
done

# Run multivariate datasets
echo "=========================================="
echo "Running MULTIVARIATE datasets..."
echo "==========================================" | tee -a $LOG_FILE

for config in $CONFIGS_DIR/multivariate/*.yaml; do
    if [ -f "$config" ]; then
        dataset_name=$(basename $config .yaml)
        dataset_dir="$DATASETS_DIR/multivariate/$dataset_name"
        
        if [ -d "$dataset_dir" ]; then
            run_benchmark "$config" "$dataset_name" "multivariate" "$dataset_dir"
        else
            echo -e "${RED}SKIP: $dataset_name (dataset directory not found: $dataset_dir)${NC}" | tee -a $LOG_FILE
            ((FAILED++))
            ((TOTAL++))
        fi
    fi
done

# Final summary
echo "=========================================="
echo "Benchmark runs completed at $(date)"
echo "==========================================" | tee -a $LOG_FILE
echo -e "Total datasets: ${TOTAL}" | tee -a $LOG_FILE
echo -e "${GREEN}Successful: ${SUCCESS}${NC}" | tee -a $LOG_FILE
echo -e "${RED}Failed: ${FAILED}${NC}" | tee -a $LOG_FILE

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Check $ERROR_LOG for error details${NC}" | tee -a $LOG_FILE
fi

echo "Detailed logs saved to: $LOG_FILE"
echo "Error logs saved to: $ERROR_LOG"

# Exit with error code if any benchmarks failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
