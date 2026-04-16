#!/bin/bash
# One-command installation script for SMLCRM Benchmark Pipeline

echo "🚀 Installing SMLCRM Benchmark Pipeline..."

echo "🔧 Creating conda environment tempus_bench..."
conda create -y -n tempus_bench python=3.11.13


echo "🔧 Activating conda environment tempus_bench..."
conda init bash
conda activate tempus_bench


# Install the package in development mode (dependencies from pyproject.toml)
echo "📦 Installing package and dependencies..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -e "$REPO_ROOT"

echo "✅ Installation complete!"
echo ""
echo "📖 Library README: README.md"
echo ""
echo "🎯 Run benchmarks (from repo root):"
echo "   python -m tempus_bench.run_benchmark --config tempus_bench/config/benchmark.yaml"
