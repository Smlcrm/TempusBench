#!/bin/bash
set -e

# Script to deploy the budget alert handler Cloud Function
# Project: sim-tempus-bench

PROJECT_ID="sim-tempus-bench"
REGION="us-central1"
FUNCTION_NAME="budget-alert-handler"
TOPIC_NAME="budget-alerts"

echo "Deploying budget alert handler Cloud Function..."

# Set the project
gcloud config set project ${PROJECT_ID}

# Create Pub/Sub topic if it doesn't exist
echo "Creating Pub/Sub topic ${TOPIC_NAME}..."
gcloud pubsub topics create ${TOPIC_NAME} --project=${PROJECT_ID} || echo "Topic already exists"

# Create Pub/Sub subscription for testing (optional)
echo "Creating Pub/Sub subscription for testing..."
gcloud pubsub subscriptions create ${TOPIC_NAME}-subscription \
    --topic=${TOPIC_NAME} \
    --project=${PROJECT_ID} || echo "Subscription already exists"

# Deploy Cloud Function
echo "Deploying Cloud Function ${FUNCTION_NAME}..."
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=python311 \
    --region=${REGION} \
    --source=. \
    --entry-point=budget_alert_handler \
    --trigger-topic=${TOPIC_NAME} \
    --set-env-vars GCP_PROJECT=${PROJECT_ID},GCP_REGION=${REGION},CLOUD_RUN_SERVICE=tempus-bench \
    --memory=512MB \
    --timeout=540s \
    --max-instances=1 \
    --service-account=${PROJECT_ID}@appspot.gserviceaccount.com \
    --project=${PROJECT_ID}

# Grant necessary permissions to the Cloud Function service account
echo "Granting permissions to Cloud Function service account..."

# Get the service account email
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

# Grant Cloud Run Admin to stop Cloud Run services
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.admin" \
    --condition=None

# Grant Cloud Build Editor to cancel builds
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/cloudbuild.builds.editor" \
    --condition=None

# Grant Compute Instance Admin to stop VM instances
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/compute.instanceAdmin.v1" \
    --condition=None

# Grant Pub/Sub Subscriber to read messages
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/pubsub.subscriber" \
    --condition=None

# Grant Pub/Sub service account permission to invoke the function
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
PUBSUB_SA="service-${PROJECT_NUMBER}@gcp-sa-pubsub.iam.gserviceaccount.com"

echo "Granting Pub/Sub service account permission to invoke the function..."
gcloud run services add-iam-policy-binding ${FUNCTION_NAME} \
    --region=${REGION} \
    --member="serviceAccount:${PUBSUB_SA}" \
    --role="roles/run.invoker" \
    --project=${PROJECT_ID} || echo "Note: Pub/Sub permissions may already be configured"

echo "Budget alert handler deployed successfully!"
echo ""
echo "Next steps:"
echo "1. Test the deployment:"
echo "   ./test_dry_run.sh  # Safe test (won't stop resources)"
echo "   ./test.sh          # Full test (will attempt to stop resources)"
echo ""
echo "2. Configure budget alert to send notifications to Pub/Sub topic: ${TOPIC_NAME}"
echo "   In GCP Console: Billing > Budgets & alerts > [Your Budget] > Configure notifications"
echo "   Set threshold to 100% and select Pub/Sub topic: projects/${PROJECT_ID}/topics/${TOPIC_NAME}"
echo ""
echo "3. Monitor Cloud Function logs:"
echo "   gcloud functions logs read ${FUNCTION_NAME} --region=${REGION} --gen2 --limit=50"

