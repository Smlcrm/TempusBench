#!/bin/bash
set -e

# Script to test the budget alert handler after deployment
# This sends a test message to the Pub/Sub topic and verifies the Cloud Function is triggered

PROJECT_ID="sim-tempus-bench"
REGION="us-central1"
FUNCTION_NAME="budget-alert-handler"
TOPIC_NAME="budget-alerts"

echo "Testing budget alert handler..."
echo ""

# Set the project
gcloud config set project ${PROJECT_ID}

# Check if Cloud Function exists
echo "1. Checking if Cloud Function is deployed..."
if gcloud functions describe ${FUNCTION_NAME} --region=${REGION} --gen2 &>/dev/null; then
    echo "✓ Cloud Function ${FUNCTION_NAME} exists"
else
    echo "✗ Cloud Function ${FUNCTION_NAME} not found"
    echo "  Run deploy.sh first to deploy the function"
    exit 1
fi

# Check if Pub/Sub topic exists
echo ""
echo "2. Checking if Pub/Sub topic exists..."
if gcloud pubsub topics describe ${TOPIC_NAME} --project=${PROJECT_ID} &>/dev/null; then
    echo "✓ Pub/Sub topic ${TOPIC_NAME} exists"
else
    echo "✗ Pub/Sub topic ${TOPIC_NAME} not found"
    echo "  Run deploy.sh first to create the topic"
    exit 1
fi

# Create a test budget alert message
echo ""
echo "3. Creating test budget alert message..."
# Create message as single-line JSON for gcloud command
TEST_MESSAGE='{"budgetDisplayName":"Tempus Bench Budget (Test)","alertThresholdExceeded":100.0,"budgetAmount":1000.0,"costAmount":1000.0,"currencyCode":"USD","schemaVersion":"1.0"}'

echo "Test message:"
if command -v jq &> /dev/null; then
    echo "${TEST_MESSAGE}" | jq .
else
    echo "${TEST_MESSAGE}"
fi

echo ""
echo "4. Publishing test message to Pub/Sub topic..."
# gcloud will automatically base64 encode the message
gcloud pubsub topics publish ${TOPIC_NAME} \
    --message="${TEST_MESSAGE}" \
    --project=${PROJECT_ID}

echo "✓ Message published successfully"
echo ""
echo "5. Waiting 10 seconds for Cloud Function to process..."
sleep 10

# Check Cloud Function logs
echo ""
echo "6. Checking Cloud Function logs (last 20 lines)..."
echo "   (This may take a moment to appear)"
echo ""

gcloud functions logs read ${FUNCTION_NAME} \
    --region=${REGION} \
    --gen2 \
    --limit=20 \
    --project=${PROJECT_ID} || echo "No logs found yet (this is normal if the function just started)"

echo ""
echo "7. Testing complete!"
echo ""
echo "To view detailed logs:"
echo "  gcloud functions logs read ${FUNCTION_NAME} --region=${REGION} --gen2 --limit=50"
echo ""
echo "To view logs in GCP Console:"
echo "  https://console.cloud.google.com/functions/details/${REGION}/${FUNCTION_NAME}?project=${PROJECT_ID}&tab=logs"
echo ""
echo "⚠️  WARNING: If the test was successful, the function should have attempted to:"
echo "   - Scale Cloud Run services to 0 instances"
echo "   - Cancel running Cloud Build builds"
echo "   - Stop Compute Engine instances"
echo ""
echo "   Check the logs above to verify the function executed correctly."
echo "   If this was a real budget alert, resources would have been stopped!"

