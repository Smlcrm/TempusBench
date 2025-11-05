# Building and Deploying Docker Container

This document describes how to build the Docker container using Google Cloud Build and deploy it to Cloud Run.

## Prerequisites

1. Google Cloud Project: `sim-tempus-bench`
2. Google Cloud SDK installed and configured
3. Permissions to use Cloud Build and Container Registry
4. GCS bucket `tempus_bench_results` created with appropriate permissions

## Building with Google Cloud Build

### Option 1: Using Cloud Build (Recommended)

Build and push the Docker image to Google Container Registry:

```bash
# Set the project
gcloud config set project sim-tempus-bench

# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

This will:
1. Build the Docker image
2. Tag it with `latest`, `$SHORT_SHA`, and `$BUILD_ID`
3. Push all tags to `gcr.io/sim-tempus-bench/tempus-bench`

### Option 2: Local Build (for testing)

Build locally for testing before pushing:

```bash
# Build locally
docker build -t tempus-bench:local .

# Test locally (see test_local.sh)
./test_local.sh
```

## Deploying to Cloud Run

### Prerequisites

1. Ensure the GCS bucket `tempus_bench_results` exists:
   ```bash
   gsutil mb gs://tempus_bench_results
   ```

2. Create or use a service account with Storage Object Admin permissions:
   ```bash
   # Create service account (if needed)
   gcloud iam service-accounts create tempus-bench-sa \
     --display-name="Tempus Bench Cloud Run Service Account"

   # Grant Storage permissions
   gcloud projects add-iam-policy-binding sim-tempus-bench \
     --member="serviceAccount:tempus-bench-sa@sim-tempus-bench.iam.gserviceaccount.com" \
     --role="roles/storage.objectAdmin"
   ```

### Deploy to Cloud Run

```bash
# Deploy the service
gcloud run deploy tempus-bench \
  --image gcr.io/sim-tempus-bench/tempus-bench:latest \
  --platform managed \
  --region us-central1 \
  --service-account tempus-bench-sa@sim-tempus-bench.iam.gserviceaccount.com \
  --set-env-vars CLOUD_RUN=true,GCS_BUCKET=tempus_bench_results,GCP_PROJECT=sim-tempus-bench \
  --memory 32Gi \
  --cpu 8 \
  --timeout 86400 \
  --max-instances 1 \
  --args gs://tempus_bench_results/configs/benchmark.yaml
```

**Note:** Adjust memory, CPU, and timeout based on your benchmark requirements. Cloud Run supports up to:
- Memory: 32Gi
- CPU: 8 vCPU
- Timeout: 86400 seconds (24 hours)

### Running a Job with Custom Config

To run with a different config file stored in GCS:

```bash
gcloud run jobs execute tempus-bench-job \
  --region us-central1 \
  --args gs://tempus_bench_results/configs/my_custom_config.yaml
```

Or if using Cloud Run service:

```bash
gcloud run services update tempus-bench \
  --region us-central1 \
  --args gs://tempus_bench_results/configs/my_custom_config.yaml
```

## Config File Setup

Upload your benchmark configuration file to GCS:

```bash
# Upload config file to GCS
gsutil cp tempus_bench/config/benchmark.yaml \
  gs://tempus_bench_results/configs/benchmark.yaml

# Or upload a custom config
gsutil cp my_custom_config.yaml \
  gs://tempus_bench_results/configs/my_custom_config.yaml
```

## Viewing Results

Results are automatically uploaded to:
```
gs://tempus_bench_results/runs/run_<timestamp>/
```

To download results:
```bash
# List runs
gsutil ls gs://tempus_bench_results/runs/

# Download a specific run
gsutil -m cp -r gs://tempus_bench_results/runs/run_<timestamp> ./local_runs/
```

## Monitoring

View Cloud Run logs:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=tempus-bench" \
  --limit 50 \
  --format json
```

## Troubleshooting

1. **Build fails**: Check Cloud Build logs in the GCP Console
2. **Upload fails**: Verify service account has Storage Object Admin role
3. **Config download fails**: Ensure config file exists in GCS and path is correct
4. **Timeout**: Increase timeout value or break down benchmarks into smaller jobs



