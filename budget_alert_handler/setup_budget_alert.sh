#!/bin/bash
set -e

# Script to set up budget alert that sends to Pub/Sub
# This requires manual steps in the GCP Console, but this script helps with setup

PROJECT_ID="sim-tempus-bench"
TOPIC_NAME="budget-alerts"
BUDGET_ID=""  # Set this to your budget ID if you know it

echo "Setting up budget alert for Pub/Sub..."
echo ""
echo "This script will help you set up the budget alert."
echo "You'll need to complete some steps in the GCP Console."
echo ""

# Check if topic exists
if gcloud pubsub topics describe ${TOPIC_NAME} --project=${PROJECT_ID} &>/dev/null; then
    echo "✓ Pub/Sub topic ${TOPIC_NAME} exists"
else
    echo "✗ Pub/Sub topic ${TOPIC_NAME} does not exist"
    echo "  Run deploy.sh first to create the topic and deploy the function"
    exit 1
fi

# Get billing account
echo ""
echo "Fetching billing account..."
BILLING_ACCOUNT=$(gcloud billing projects describe ${PROJECT_ID} --format="value(billingAccountName)" 2>/dev/null || echo "")

if [ -z "${BILLING_ACCOUNT}" ]; then
    echo "✗ No billing account found for project ${PROJECT_ID}"
    echo "  Please set up billing first"
    exit 1
fi

echo "✓ Billing account: ${BILLING_ACCOUNT}"
echo ""
echo "To configure the budget alert:"
echo "1. Go to: https://console.cloud.google.com/billing/budgets?project=${PROJECT_ID}"
echo "2. Click 'Create Budget' or edit existing budget"
echo "3. Set budget amount and period"
echo "4. Under 'Manage notifications', click 'Add Notification'"
echo "5. Set threshold to 100% (1.0)"
echo "6. Select 'Pub/Sub' as the notification channel"
echo "7. Choose topic: projects/${PROJECT_ID}/topics/${TOPIC_NAME}"
echo "8. Save the budget"
echo ""
echo "Alternatively, use gcloud CLI:"
echo ""
echo "gcloud billing budgets create \\"
echo "  --billing-account=\$(gcloud billing projects describe ${PROJECT_ID} --format='value(billingAccountName)' | cut -d'/' -f2) \\"
echo "  --display-name='Tempus Bench Budget' \\"
echo "  --budget-amount=\$AMOUNT \\"
echo "  --threshold-rule=percent=100 \\"
echo "  --pubsub-topic=projects/${PROJECT_ID}/topics/${TOPIC_NAME}"
echo ""



