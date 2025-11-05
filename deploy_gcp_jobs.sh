#!/bin/bash
set -e

# Script to deploy all benchmark_gcp_*.yaml jobs to Cloud Run Jobs
# Project: sim-tempus-bench

PROJECT_ID="sim-tempus-bench"
REGION="us-central1"
GCS_BUCKET="tempus_bench_results"
IMAGE="gcr.io/sim-tempus-bench/tempus-bench:latest"
SERVICE_ACCOUNT="tempus-bench-sa@sim-tempus-bench.iam.gserviceaccount.com"
CONFIG_DIR="tempus_bench/config"
GCS_CONFIG_PREFIX="gs://${GCS_BUCKET}/configs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Deploying Benchmark Jobs to Cloud Run"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image: ${IMAGE}"
echo ""

# Set the project
echo -e "${YELLOW}Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

# Skip upload - configs are already in GCS
echo -e "${YELLOW}Skipping upload - assuming configs are already in GCS${NC}"
echo ""

# Deploy Cloud Run Jobs
echo -e "${YELLOW}Creating Cloud Run Jobs...${NC}"
JOB_COUNT=0
FAILED=0

for config_file in ${CONFIG_DIR}/benchmark_gcp_*.yaml; do
    if [ -f "$config_file" ]; then
        config_name=$(basename "$config_file" .yaml)
        job_name=$(echo "${config_name}" | tr '_' '-')  # e.g., benchmark-gcp-1
        gcs_config_path="${GCS_CONFIG_PREFIX}/$(basename "$config_file")"
        
        echo "  Deploying job: ${job_name}..."
        
        # Create or update Cloud Run Job
        if gcloud run jobs describe ${job_name} --region=${REGION} --project=${PROJECT_ID} >/dev/null 2>&1; then
            echo "    Job ${job_name} exists, updating..."
            gcloud run jobs update ${job_name} \
                --region=${REGION} \
                --image=${IMAGE} \
                --service-account=${SERVICE_ACCOUNT} \
                --set-env-vars CLOUD_RUN=true,GCS_BUCKET=${GCS_BUCKET},GCP_PROJECT=${PROJECT_ID} \
                --memory=32Gi \
                --cpu=8 \
                --task-timeout=86400 \
                --max-retries=1 \
                --args="${gcs_config_path}" \
                --project=${PROJECT_ID} \
                --quiet || {
                echo -e "${RED}    Failed to update job ${job_name}${NC}"
                ((FAILED++))
                continue
            }
        else
            echo "    Creating new job ${job_name}..."
            gcloud run jobs create ${job_name} \
                --region=${REGION} \
                --image=${IMAGE} \
                --service-account=${SERVICE_ACCOUNT} \
                --set-env-vars CLOUD_RUN=true,GCS_BUCKET=${GCS_BUCKET},GCP_PROJECT=${PROJECT_ID} \
                --memory=32Gi \
                --cpu=8 \
                --task-timeout=86400 \
                --max-retries=1 \
                --args="${gcs_config_path}" \
                --project=${PROJECT_ID} \
                --quiet || {
                echo -e "${RED}    Failed to create job ${job_name}${NC}"
                ((FAILED++))
                continue
            }
        fi
        
        ((JOB_COUNT++))
        echo -e "${GREEN}    ✓ Job ${job_name} deployed${NC}"
    fi
done

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo -e "${GREEN}Cloud Run Jobs created/updated: ${JOB_COUNT}${NC}"
if [ ${FAILED} -gt 0 ]; then
    echo -e "${RED}Failed deployments: ${FAILED}${NC}"
fi
echo ""
echo "To execute a job, run:"
echo "  gcloud run jobs execute <job_name> --region=${REGION}"
echo ""
echo "To list all jobs:"
echo "  gcloud run jobs list --region=${REGION}"
echo ""

