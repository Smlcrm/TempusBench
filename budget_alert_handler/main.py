"""
Cloud Function to handle budget alerts and kill compute resources.

This function is triggered by Pub/Sub messages from budget alerts.
When a budget alert indicates 100% budget usage, it will:
1. Stop all Cloud Run services
2. Cancel any running Cloud Build builds
3. Stop any Compute Engine instances (if any exist)
"""

import base64
import json
import os
from datetime import datetime
from typing import Any, Dict

from google.api_core import exceptions
from google.cloud import compute_v1
from google.cloud import run_v2
from google.cloud.devtools import cloudbuild_v1
from google.protobuf import field_mask_pb2

PROJECT_ID = os.environ.get("GCP_PROJECT", "sim-tempus-bench")
CLOUD_RUN_SERVICE_NAME = os.environ.get("CLOUD_RUN_SERVICE", "tempus-bench")
REGION = os.environ.get("GCP_REGION", "us-central1")


def stop_cloud_run_service() -> Dict[str, Any]:
    """Stop all revisions of the Cloud Run service."""
    results = {"stopped_revisions": [], "errors": []}
    
    try:
        client = run_v2.ServicesClient()
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        
        # Get the service
        service_path = f"{parent}/services/{CLOUD_RUN_SERVICE_NAME}"
        
        try:
            service = client.get_service(name=service_path)
            print(f"Found Cloud Run service: {CLOUD_RUN_SERVICE_NAME}")
            
            # Scale service to 0 instances by updating the service
            service.template.scaling.min_instance_count = 0
            service.template.scaling.max_instance_count = 0
            
            # Update the service with field mask
            update_mask = field_mask_pb2.FieldMask()
            update_mask.paths.append("template.scaling.min_instance_count")
            update_mask.paths.append("template.scaling.max_instance_count")
            
            operation = client.update_service(
                service=service,
                update_mask=update_mask
            )
            
            results["stopped_revisions"].append(f"Scaled {CLOUD_RUN_SERVICE_NAME} to 0 instances")
            print(f"Scaled Cloud Run service {CLOUD_RUN_SERVICE_NAME} to 0 instances")
            
        except exceptions.NotFound:
            results["errors"].append(f"Cloud Run service {CLOUD_RUN_SERVICE_NAME} not found")
            print(f"Cloud Run service {CLOUD_RUN_SERVICE_NAME} not found")
        
    except Exception as e:
        error_msg = f"Error stopping Cloud Run service: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    return results


def cancel_cloud_builds() -> Dict[str, Any]:
    """Cancel all running Cloud Build builds."""
    results = {"cancelled_builds": [], "errors": []}
    
    try:
        client = cloudbuild_v1.CloudBuildClient()
        project_id = PROJECT_ID
        
        # List builds with status WORKING
        filter_str = f"status='WORKING'"
        
        try:
            builds = client.list_builds(
                project_id=project_id,
                filter=filter_str
            )
            
            for build in builds:
                try:
                    if build.status == cloudbuild_v1.Build.Status.WORKING:
                        client.cancel_build(
                            project_id=project_id,
                            id=build.id
                        )
                        results["cancelled_builds"].append(build.id)
                        print(f"Cancelled Cloud Build: {build.id}")
                except Exception as e:
                    error_msg = f"Error cancelling build {build.id}: {str(e)}"
                    results["errors"].append(error_msg)
                    print(error_msg)
            
            if not results["cancelled_builds"]:
                print("No running Cloud Build builds found")
                
        except Exception as e:
            error_msg = f"Error listing Cloud Build builds: {str(e)}"
            results["errors"].append(error_msg)
            print(error_msg)
    
    except Exception as e:
        error_msg = f"Error cancelling Cloud Builds: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    return results


def stop_compute_engine_instances() -> Dict[str, Any]:
    """Stop all running Compute Engine instances."""
    results = {"stopped_instances": [], "errors": []}
    
    try:
        instance_client = compute_v1.InstancesClient()
        project = PROJECT_ID
        
        # List all zones
        zones_client = compute_v1.ZonesClient()
        zones = zones_client.list(project=project)
        
        for zone in zones:
            zone_name = zone.name
            
            # List instances in this zone
            instances = instance_client.list(
                project=project,
                zone=zone_name
            )
            
            for instance in instances:
                if instance.status == "RUNNING":
                    try:
                        operation = instance_client.stop(
                            project=project,
                            zone=zone_name,
                            instance=instance.name
                        )
                        results["stopped_instances"].append(
                            f"{instance.name} in zone {zone_name}"
                        )
                        print(f"Stopped instance: {instance.name} in zone {zone_name}")
                    except Exception as e:
                        error_msg = f"Error stopping instance {instance.name}: {str(e)}"
                        results["errors"].append(error_msg)
                        print(error_msg)
        
        if not results["stopped_instances"]:
            print("No running Compute Engine instances found")
    
    except Exception as e:
        error_msg = f"Error stopping Compute Engine instances: {str(e)}"
        results["errors"].append(error_msg)
        print(error_msg)
    
    return results


def parse_budget_alert(message_data: str) -> Dict[str, Any]:
    """Parse the budget alert Pub/Sub message."""
    try:
        # Decode base64 message
        decoded = base64.b64decode(message_data).decode("utf-8")
        alert_data = json.loads(decoded)
        return alert_data
    except Exception as e:
        print(f"Error parsing budget alert: {str(e)}")
        return {}


def check_budget_threshold(alert_data: Dict[str, Any]) -> bool:
    """Check if the budget alert indicates 100% usage."""
    try:
        # Budget alert structure may vary, but typically includes:
        # - budgetDisplayName
        # - alertThresholdExceeded (percentage)
        # - costAmount or budgetAmount
        
        # Check for threshold exceeded field
        if "alertThresholdExceeded" in alert_data:
            threshold = float(alert_data["alertThresholdExceeded"])
            if threshold >= 100.0:
                print(f"Budget threshold exceeded: {threshold}%")
                return True
            else:
                print(f"Budget threshold not reached: {threshold}% < 100%")
                return False
        
        # Check for cost vs budget ratio
        if "costAmount" in alert_data and "budgetAmount" in alert_data:
            cost = float(alert_data["costAmount"])
            budget = float(alert_data["budgetAmount"])
            if budget > 0:
                ratio = cost / budget
                if ratio >= 1.0:
                    print(f"Budget exceeded: cost ({cost}) >= budget ({budget})")
                    return True
                else:
                    print(f"Budget not exceeded: cost ({cost}) < budget ({budget})")
                    return False
        
        # If we can't determine threshold, be cautious and assume it's 100%
        # This happens when budget alert format is unexpected
        print("Budget alert format unclear - assuming 100% threshold to be safe")
        return True
        
    except Exception as e:
        print(f"Error checking budget threshold: {str(e)} - assuming 100% threshold to be safe")
        # If we can't parse, err on the side of caution and stop resources
        return True


def budget_alert_handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    Cloud Function entry point for budget alerts.
    
    Supports both Cloud Functions Gen1 and Gen2 event formats.
    
    Args:
        event: Pub/Sub event data (CloudEvent format for Gen2, dict for Gen1)
        context: Cloud Functions context (optional for Gen2)
    
    Returns:
        Dictionary with results of stopping resources
    """
    timestamp = datetime.utcnow().isoformat()
    if context and hasattr(context, "timestamp"):
        timestamp = context.timestamp
    
    print(f"Budget alert received at {timestamp}")
    
    # Handle CloudEvent format (Gen2) or direct Pub/Sub format (Gen1)
    message_data = ""
    
    # Gen2 CloudEvent format
    if "message" in event:
        message_data = event["message"].get("data", "")
    # Gen1 Pub/Sub format
    elif "data" in event:
        message_data = event.get("data", "")
    # Direct base64 string
    elif isinstance(event, str):
        message_data = event
    # Try to extract from message attribute
    elif "attributes" in event and "data" in event:
        message_data = event.get("data", "")
    
    if not message_data:
        print(f"No message data found in event. Event keys: {list(event.keys()) if isinstance(event, dict) else 'not a dict'}")
        return {"error": "No message data found", "timestamp": timestamp}
    
    # Parse budget alert
    alert_data = parse_budget_alert(message_data)
    print(f"Parsed alert data: {json.dumps(alert_data, indent=2)}")
    
    # Check if we've reached 100% budget
    if not check_budget_threshold(alert_data):
        print("Budget threshold not reached - no action taken")
        return {"status": "threshold_not_reached"}
    
    print("Budget threshold reached (100%) - stopping compute resources")
    
    # Stop all compute resources
    results = {
        "cloud_run": stop_cloud_run_service(),
        "cloud_build": cancel_cloud_builds(),
        "compute_engine": stop_compute_engine_instances(),
    }
    
    summary = {
        "status": "resources_stopped",
        "timestamp": timestamp,
        "details": results,
    }
    
    print(f"Budget alert handling complete: {json.dumps(summary, indent=2)}")
    
    return summary
