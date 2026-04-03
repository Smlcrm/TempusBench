#!/usr/bin/env bash
# Run matrix + import-boundary integration tests one model package at a time.
#
# Usage (from TempusBench repo root):
#   ./tests/integration/run_model_matrix_one_by_one.sh
#   MODEL_MATRIX_FAST=1 ./tests/integration/run_model_matrix_one_by_one.sh
#   ./tests/integration/run_model_matrix_one_by_one.sh -q --tb=line   # extra pytest args
#
# Requires RUN_MODEL_MATRIX_TEST=1 (set below by default). Foundation models still need
# MODEL_WEIGHTS_PATH when their tests are not skipped.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export RUN_MODEL_MATRIX_TEST="${RUN_MODEL_MATRIX_TEST:-1}"

read -r -a PKGS <<< "$(python3 -c "from tempus_bench.utils.paths import get_available_models; print(' '.join(sorted(get_available_models())))")"

for pkg in "${PKGS[@]}"; do
  echo "========== MODEL_MATRIX_ONLY=${pkg} =========="
  MODEL_MATRIX_ONLY="${pkg}" python3 -m pytest \
    tests/integration/test_all_models_import_boundary.py \
    tests/integration/test_all_models_inference_matrix.py \
    tests/integration/test_all_models_context_horizon_edges.py \
    -m "not slow" \
    --tb=short \
    "$@"
done

echo "All packages finished."
