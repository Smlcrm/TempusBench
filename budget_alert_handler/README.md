# Budget Alert Handler

This directory contains a Cloud Function that automatically stops compute resources when a budget alert indicates 100% budget usage has been reached for the `sim-tempus-bench` project.

## Overview

When the GCP budget reaches 100%, a Pub/Sub message is sent to the `budget-alerts` topic, which triggers this Cloud Function. The function will:

1. **Stop Cloud Run services**: Scales all Cloud Run services (e.g., `tempus-bench`) to 0 instances
2. **Cancel Cloud Build builds**: Cancels any running Cloud Build jobs
3. **Stop Compute Engine instances**: Stops any running VM instances (if any exist)

## Prerequisites

- Google Cloud SDK installed and configured
- Access to the `sim-tempus-bench` project
- Billing account linked to the project
- Budget configured in GCP Console

## Deployment

### Step 1: Deploy the Cloud Function

```bash
cd budget_alert_handler
chmod +x deploy.sh
./deploy.sh
```

This will:
- Create the Pub/Sub topic `budget-alerts`
- Deploy the Cloud Function `budget-alert-handler`
- Grant necessary IAM permissions

### Step 2: Configure Budget Alert

#### Option A: Using GCP Console (Recommended)

1. Go to [GCP Budgets](https://console.cloud.google.com/billing/budgets?project=sim-tempus-bench)
2. Click **Create Budget** or edit an existing budget
3. Configure your budget amount and period
4. Under **Manage notifications**, click **Add Notification**
5. Set threshold to **100% (1.0)**
6. Select **Pub/Sub** as the notification channel
7. Choose topic: `projects/sim-tempus-bench/topics/budget-alerts`
8. Save the budget

#### Option B: Using gcloud CLI

```bash
# Get billing account ID
BILLING_ACCOUNT=$(gcloud billing projects describe sim-tempus-bench --format='value(billingAccountName)' | cut -d'/' -f2)

# Create budget with Pub/Sub alert at 100%
gcloud billing budgets create \
  --billing-account=${BILLING_ACCOUNT} \
  --display-name="Tempus Bench Budget" \
  --budget-amount=100USD \
  --threshold-rule=percent=100 \
  --pubsub-topic=projects/sim-tempus-bench/topics/budget-alerts
```

### Step 3: Verify Setup

```bash
# Check Cloud Function is deployed
gcloud functions describe budget-alert-handler --region=us-central1 --gen2

# Check Pub/Sub topic exists
gcloud pubsub topics describe budget-alerts

# View Cloud Function logs
gcloud functions logs read budget-alert-handler --region=us-central1 --gen2 --limit=50
```

## Testing

After deployment, run the test script to verify the function works correctly:

```bash
cd budget_alert_handler
./test.sh
```

This will:
1. Verify the Cloud Function is deployed
2. Verify the Pub/Sub topic exists
3. Publish a test budget alert message (100% threshold)
4. Wait for the function to process
5. Show recent logs

**⚠️ WARNING**: The test script will trigger the actual shutdown behavior. The function will attempt to stop Cloud Run services, cancel Cloud Build jobs, and stop Compute Engine instances. Use only in a test environment or when you're prepared for resources to be stopped.

For a safer dry-run test that won't trigger resource shutdown:

```bash
./test_dry_run.sh
```

This sends a message with threshold < 100% to verify message parsing without stopping resources.

### Manual Testing

You can also manually publish test messages:

```bash
# Test with 100% threshold (will stop resources)
gcloud pubsub topics publish budget-alerts \
  --message='{"alertThresholdExceeded": 100, "budgetDisplayName": "Test Budget", "budgetAmount": 1000, "costAmount": 1000}' \
  --project=sim-tempus-bench

# Test with 50% threshold (won't stop resources)
gcloud pubsub topics publish budget-alerts \
  --message='{"alertThresholdExceeded": 50, "budgetDisplayName": "Test Budget", "budgetAmount": 1000, "costAmount": 500}' \
  --project=sim-tempus-bench
```

Then check the Cloud Function logs:

```bash
gcloud functions logs read budget-alert-handler --region=us-central1 --gen2 --limit=50
```

## Environment Variables

The Cloud Function uses these environment variables (configured during deployment):

- `GCP_PROJECT`: Project ID (default: `sim-tempus-bench`)
- `GCP_REGION`: GCP region (default: `us-central1`)
- `CLOUD_RUN_SERVICE`: Cloud Run service name to stop (default: `tempus-bench`)

## Permissions

The Cloud Function requires these IAM roles:

- `roles/run.admin` - To stop Cloud Run services
- `roles/cloudbuild.builds.editor` - To cancel Cloud Build jobs
- `roles/compute.instanceAdmin.v1` - To stop Compute Engine instances
- `roles/pubsub.subscriber` - To read Pub/Sub messages

These are automatically granted during deployment via `deploy.sh`.

## Monitoring

View Cloud Function logs:

```bash
gcloud functions logs read budget-alert-handler --region=us-central1 --gen2 --limit=50
```

Or in the GCP Console:
- Cloud Functions → budget-alert-handler → Logs

## Troubleshooting

### Function not triggering

1. Verify budget alert is configured to send to the correct Pub/Sub topic
2. Check Pub/Sub topic exists: `gcloud pubsub topics describe budget-alerts`
3. Verify Cloud Function is subscribed to the topic
4. Check Cloud Function logs for errors

### Resources not stopping

1. Verify Cloud Function has necessary IAM permissions
2. Check Cloud Function logs for permission errors
3. Ensure service names match (e.g., `tempus-bench` for Cloud Run)

### Budget alert not sending

1. Verify budget is configured with threshold at 100%
2. Check that budget amount has actually been reached
3. Verify Pub/Sub topic name matches in budget configuration

## Manual Resource Stopping

If needed, you can manually stop resources:

```bash
# Stop Cloud Run service
gcloud run services update tempus-bench \
  --region=us-central1 \
  --min-instances=0 \
  --max-instances=0

# Cancel running Cloud Build jobs
gcloud builds list --ongoing
gcloud builds cancel <BUILD_ID>

# Stop Compute Engine instances
gcloud compute instances list
gcloud compute instances stop <INSTANCE_NAME> --zone=<ZONE>
```

## Security Considerations

- The Cloud Function has broad permissions to stop compute resources
- Consider adding additional safeguards (e.g., requiring multiple alerts, confirmation)
- Monitor Cloud Function execution logs regularly
- Consider adding alert notifications when resources are stopped

