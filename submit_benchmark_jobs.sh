#!/bin/bash
# Submit all benchmark_gcp jobs to Cloud Run

set -e

REGION="us-central1"
JOB_NAME="tempus-bench-job"
BUCKET="gs://tempus_bench_results/configs/"

echo "Fetching benchmark_gcp config files..."
CONFIGS=$(gcloud storage ls "$BUCKET" 2>&1 | grep "benchmark_gcp.*\.yaml$" | sed 's|gs://tempus_bench_results/configs/||' | sort -V)

TOTAL=$(echo "$CONFIGS" | wc -l | tr -d ' ')
echo "Found $TOTAL config files to submit"
echo ""

count=0
for config in $CONFIGS; do
    ((count++))
    printf "[%2d/%2d] Submitting: %-30s " "$count" "$TOTAL" "$config"
    
    if gcloud run jobs execute "$JOB_NAME" \
        --region "$REGION" \
        --args "$BUCKET$config" \
        --quiet > /dev/null 2>&1; then
        echo "✓"
    else
        echo "✗ FAILED"
    fi
    
    # Small delay to avoid rate limiting
    sleep 0.2
done

echo ""
echo "Submitted $count jobs!"
echo "Monitor with: gcloud run jobs executions list --job $JOB_NAME --region $REGION"



