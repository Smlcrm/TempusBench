#!/bin/bash

# Test script to validate preprocessor fixes and config generation on a subset of datasets
# This runs a few representative datasets to ensure everything works before running the full suite

CONFIGS_DIR="benchmarking_pipeline/configs/datasets"
DATASETS_DIR="benchmarking_pipeline/datasets"
LOG_FILE="test_benchmark_runs.log"
ERROR_LOG="test_benchmark_errors.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Starting TEST benchmark runs at $(date)"
echo "==========================================" | tee -a $LOG_FILE

# Test datasets (mix of univariate and multivariate)
TEST_DATASETS=(
    "univariate/chickenpox_dense_univariate"
    "univariate/patient_sparse_univariate"
    "multivariate/madrid_count_multivariate"
    "multivariate/soil_500_multivariate"
)

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
    
    echo -e "${YELLOW}Testing $dataset_name ($dataset_type)...${NC}" | tee -a $LOG_FILE
    
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

# Run test datasets
echo "=========================================="
echo "Running TEST datasets..."
echo "==========================================" | tee -a $LOG_FILE

for dataset_path in "${TEST_DATASETS[@]}"; do
    dataset_type=$(echo $dataset_path | cut -d'/' -f1)
    dataset_name=$(echo $dataset_path | cut -d'/' -f2)
    
    config_file="$CONFIGS_DIR/$dataset_path.yaml"
    dataset_dir="$DATASETS_DIR/$dataset_path"
    
    if [ -f "$config_file" ] && [ -d "$dataset_dir" ]; then
        run_benchmark "$config_file" "$dataset_name" "$dataset_type" "$dataset_dir"
    else
        echo -e "${RED}SKIP: $dataset_name (config or dataset directory not found)${NC}" | tee -a $LOG_FILE
        echo "  Config: $config_file" | tee -a $LOG_FILE
        echo "  Dataset: $dataset_dir" | tee -a $LOG_FILE
        ((FAILED++))
        ((TOTAL++))
    fi
done

# Final summary
echo "=========================================="
echo "Test benchmark runs completed at $(date)"
echo "==========================================" | tee -a $LOG_FILE
echo -e "Total test datasets: ${TOTAL}" | tee -a $LOG_FILE
echo -e "${GREEN}Successful: ${SUCCESS}${NC}" | tee -a $LOG_FILE
echo -e "${RED}Failed: ${FAILED}${NC}" | tee -a $LOG_FILE

if [ $FAILED -gt 0 ]; then
    echo -e "${YELLOW}Check $ERROR_LOG for error details${NC}" | tee -a $LOG_FILE
    echo "Last 20 lines of error log:" | tee -a $LOG_FILE
    tail -20 $ERROR_LOG | tee -a $LOG_FILE
else
    echo -e "${GREEN}All test datasets passed! Ready to run full benchmark suite.${NC}" | tee -a $LOG_FILE
fi

echo "Detailed logs saved to: $LOG_FILE"
echo "Error logs saved to: $ERROR_LOG"

# Exit with error code if any benchmarks failed
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi
