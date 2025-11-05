#!/bin/bash
set -e

# Script to test the budget alert handler in dry-run mode
# This verifies the Cloud Function can parse messages correctly without stopping resources

PROJECT_ID="sim-tempus-bench"
REGION="us-central1"
FUNCTION_NAME="budget-alert-handler"
TOPIC_NAME="budget-alerts"

echo "Dry-run test for budget alert handler..."
echo "This test verifies message parsing without stopping resources."
echo ""

# Set the project
gcloud config set project ${PROJECT_ID}

# Create a test message that won't trigger resource stopping
echo "Creating test message with threshold < 100%..."
# Create message as single-line JSON for gcloud command
TEST_MESSAGE='{"budgetDisplayName":"Tempus Bench Budget (Test - Below Threshold)","alertThresholdExceeded":50.0,"budgetAmount":1000.0,"costAmount":500.0,"currencyCode":"USD","schemaVersion":"1.0"}'

echo "Test message:"
if command -v jq &> /dev/null; then
    echo "${TEST_MESSAGE}" | jq .
else
    echo "${TEST_MESSAGE}"
fi

echo ""
echo "Publishing test message to Pub/Sub topic..."
# gcloud will automatically base64 encode the message
gcloud pubsub topics publish ${TOPIC_NAME} \
    --message="${TEST_MESSAGE}" \
    --project=${PROJECT_ID}

echo "✓ Message published successfully"
echo ""
echo "Waiting 10 seconds for Cloud Function to process..."
sleep 10

# Check Cloud Function logs
echo ""
echo "Checking Cloud Function logs..."
gcloud functions logs read ${FUNCTION_NAME} \
    --region=${REGION} \
    --gen2 \
    --limit=10 \
    --project=${PROJECT_ID} || echo "No logs found"

echo ""
echo "Dry-run test complete!"
echo "The function should have received the message and determined threshold not reached."
echo "Check logs to verify it processed the message correctly."

