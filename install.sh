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
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎯 You can now run benchmarks with:"
echo "   python tempus_bench/run_benchmark.py --config tempus_bench/config/benchmark.yaml"
