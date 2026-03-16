#!/bin/bash

# Simple script to clean up all conda environments for benchmarking models
# This will delete all environments that start with "benchmark."

echo "=========================================="
echo "Cleaning up conda environments for models"
echo "=========================================="

# Get list of conda environments and filter for benchmark.*
benchmark_envs=$(conda env list | grep "benchmark\." | awk '{print $1}')

if [ -z "$benchmark_envs" ]; then
    echo "No benchmark conda environments found."
    echo "=========================================="
    exit 0
fi

echo "Found benchmark conda environments:"
echo "$benchmark_envs"
echo ""

# Delete each environment
for env in $benchmark_envs; do
    echo "Deleting environment: $env"
    conda env remove -n "$env" -y
done

echo "=========================================="
echo "Cleanup completed!"
echo "=========================================="

# Verify cleanup
remaining=$(conda env list | grep -c "benchmark\." 2>/dev/null || echo "0")
echo "Remaining benchmark environments: $remaining"

if [ "$remaining" -eq 0 ]; then
    echo "✅ All benchmark conda environments have been successfully removed!"
else
    echo "⚠️  Some benchmark environments may still exist"
fi
