#!/bin/bash
# Script to execute Cloud Run jobs for all config files in GCS

set -e

PROJECT="sim-tempus-bench"
REGION="us-central1"
JOB_NAME="tempus-bench-job"
BUCKET="gs://tempus_bench_results/configs/"

echo "Fetching config files from GCS..."
CONFIGS=$(gcloud storage ls "$BUCKET" 2>&1 | grep "\.yaml$" | sed 's|gs://tempus_bench_results/configs/||' | sort -V)

TOTAL=$(echo "$CONFIGS" | grep -c . || echo "0")
echo "Found $TOTAL config files"

if [ "$TOTAL" -eq 0 ]; then
    echo "No config files found!"
    exit 1
fi

count=0
for config in $CONFIGS; do
    ((count++))
    echo "[$count/$TOTAL] Submitting job for: $config"
    
    gcloud run jobs execute "$JOB_NAME" \
        --region "$REGION" \
        --args "gs://tempus_bench_results/configs/$config" \
        --quiet 2>&1 | grep -E "(Execution|execution)" || echo "  ✓ Submitted"
    
    # Small delay to avoid rate limiting
    sleep 0.5
done

echo ""
echo "All $TOTAL jobs have been submitted!"
echo "Monitor executions with: gcloud run jobs executions list --job $JOB_NAME --region $REGION"



