#!/bin/bash
set -e

# Function to download config from GCS
download_config_from_gcs() {
    local gcs_path=$1
    local local_path=$2
    
    echo "Downloading config from ${gcs_path}..."
    
    # Extract bucket and object path from gs://bucket/path format
    if [[ $gcs_path =~ ^gs://([^/]+)/(.+)$ ]]; then
        local bucket="${BASH_REMATCH[1]}"
        local object_path="${BASH_REMATCH[2]}"
        
        # Use gsutil to download
        gsutil cp "${gcs_path}" "${local_path}" || {
            echo "Error: Failed to download config from ${gcs_path}"
            exit 1
        }
        
        echo "Config downloaded successfully to ${local_path}"
    else
        echo "Error: Invalid GCS path format. Expected gs://bucket/path"
        exit 1
    fi
}

# Main execution
main() {
    # Check if config path is provided
    if [ -z "$1" ]; then
        echo "Error: Config file path is required"
        echo "Usage: docker-entrypoint.sh <config_path>"
        echo "  Config path can be:"
        echo "    - Local path: /path/to/config.yaml"
        echo "    - GCS path: gs://bucket/path/config.yaml"
        exit 1
    fi
    
    CONFIG_PATH="$1"
    TEMP_CONFIG_PATH="/tmp/config.yaml"
    
    # Set Cloud Run environment variable
    export CLOUD_RUN=true
    
    # Determine if config is from GCS or local
    if [[ "$CONFIG_PATH" =~ ^gs:// ]]; then
        # Download from GCS
        download_config_from_gcs "$CONFIG_PATH" "$TEMP_CONFIG_PATH"
        ACTUAL_CONFIG_PATH="$TEMP_CONFIG_PATH"
    else
        # Use local path directly
        if [ ! -f "$CONFIG_PATH" ]; then
            echo "Error: Config file not found at ${CONFIG_PATH}"
            exit 1
        fi
        ACTUAL_CONFIG_PATH="$CONFIG_PATH"
    fi
    
    # Execute benchmark
    echo "Starting benchmark with config: ${ACTUAL_CONFIG_PATH}"
    python -m tempus_bench.run_benchmark --config "$ACTUAL_CONFIG_PATH"
    
    EXIT_CODE=$?
    
    # Clean up temporary config file if downloaded from GCS
    if [[ "$CONFIG_PATH" =~ ^gs:// ]] && [ -f "$TEMP_CONFIG_PATH" ]; then
        rm -f "$TEMP_CONFIG_PATH"
    fi
    
    exit $EXIT_CODE
}

# Run main function
main "$@"



