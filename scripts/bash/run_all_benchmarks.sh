#!/bin/bash

# Centralized benchmark runner for all task configurations
# This script runs benchmarks for all task files in tempus_bench/tasks/
# Usage: ./run_all_benchmarks.sh [univariate|multivariate|all]

TASKS_DIR="tempus_bench/tasks"
DATASETS_DIR="tempus_bench/datasets"
LOG_FILE="benchmark_runs.log"
ERROR_LOG="benchmark_errors.log"

# Parse command line argument
TASK_TYPE=${1:-"all"}

# Validate task type
if [ "$TASK_TYPE" != "univariate" ] && [ "$TASK_TYPE" != "multivariate" ] && [ "$TASK_TYPE" != "all" ]; then
    echo "Usage: $0 [univariate|multivariate|all]"
    echo "  univariate   - Run only univariate tasks"
    echo "  multivariate - Run only multivariate tasks"
    echo "  all          - Run all tasks (default)"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Starting benchmark runs at $(date)"
echo "==========================================" | tee -a $LOG_FILE

# Count total tasks to be run
TOTAL_TASKS=0
if [ "$TASK_TYPE" = "univariate" ] || [ "$TASK_TYPE" = "all" ]; then
    UNIVARIATE_COUNT=$(ls $TASKS_DIR/univariate/*.yaml 2>/dev/null | wc -l)
    TOTAL_TASKS=$((TOTAL_TASKS + UNIVARIATE_COUNT))
fi
if [ "$TASK_TYPE" = "multivariate" ] || [ "$TASK_TYPE" = "all" ]; then
    MULTIVARIATE_COUNT=$(ls $TASKS_DIR/multivariate/*.yaml 2>/dev/null | wc -l)
    TOTAL_TASKS=$((TOTAL_TASKS + MULTIVARIATE_COUNT))
fi

echo "Total tasks to run: $TOTAL_TASKS" | tee -a $LOG_FILE
echo "Task type: $TASK_TYPE" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE

# Initialize counters
TOTAL=0
SUCCESS=0
FAILED=0

# Function to run a single benchmark
run_benchmark() {
    local task_file=$1
    local task_name=$2
    local task_type=$3
    
    echo -e "${YELLOW}Running $task_name ($task_type)...${NC}" | tee -a $LOG_FILE
    
    # Activate conda environment and run benchmark
    conda activate sim.benchmarks
    python tempus_bench/run_benchmark.py \
        --config "$task_file" \
        --datasets-dir "$DATASETS_DIR" \
        >> $LOG_FILE 2>> $ERROR_LOG
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}SUCCESS: $task_name${NC}" | tee -a $LOG_FILE
        ((SUCCESS++))
    else
        echo -e "${RED}FAILED: $task_name (exit code: $exit_code)${NC}" | tee -a $LOG_FILE
        ((FAILED++))
    fi
    
    ((TOTAL++))
}

# Run tasks based on command line argument
if [ "$TASK_TYPE" = "univariate" ] || [ "$TASK_TYPE" = "all" ]; then
    echo "=========================================="
    echo "Running UNIVARIATE tasks..."
    echo "==========================================" | tee -a $LOG_FILE

    for task in $TASKS_DIR/univariate/*.yaml; do
        if [ -f "$task" ]; then
            task_name=$(basename $task .yaml)
            run_benchmark "$task" "$task_name" "univariate"
        fi
    done
fi

if [ "$TASK_TYPE" = "multivariate" ] || [ "$TASK_TYPE" = "all" ]; then
    echo "=========================================="
    echo "Running MULTIVARIATE tasks..."
    echo "==========================================" | tee -a $LOG_FILE

    for task in $TASKS_DIR/multivariate/*.yaml; do
        if [ -f "$task" ]; then
            task_name=$(basename $task .yaml)
            run_benchmark "$task" "$task_name" "multivariate"
        fi
    done
fi

# Final summary
echo "=========================================="
echo "Benchmark runs completed at $(date)"
echo "==========================================" | tee -a $LOG_FILE
echo -e "Total tasks: ${TOTAL}" | tee -a $LOG_FILE
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
