"""Channel-independent covariate stacking for NBEATS, iTransformer, and PatchTSMixer.

Covariates (x_context / x_target) are stacked as additional columns in
y_context / y_target.  Predictions must be sliced back to the original
target count so the output shape is always (horizon, num_original_targets).

Tests verify:
 * settings.yaml declares covariates: past_future for all five models
 * stacking code patterns are present in the model source files
 * all five models removed from NO_COVARIATE_MODELS set

NOTE: Full model-instantiation tests for NBEATS, iTransformer, and
PatchTSMixer require neuralforecast/transformers + working torchvision
and only run in the Docker worker. XGBoost and Theta model-instantiation
tests are in test_xgboost_theta_covariate_stack.py.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

_TF_STUBS = ("tensorflow", "tensorboard", "tensorboard.summary")
for mod in _TF_STUBS:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import pytest
import yaml

from tempus_bench.utils.model_settings import load_capabilities_for_model, clear_model_settings_cache

MODELS_ROOT = Path(__file__).resolve().parents[2] / "tempus_bench" / "models"


# ---------------------------------------------------------------------------
# settings.yaml: covariates must be past_future
# ---------------------------------------------------------------------------
class TestSettingsYamlCovariatePastFuture:

    @pytest.mark.parametrize("model_name", [
        "xgboost", "theta", "nbeats", "itransformer", "patchtsmixer",
    ])
    def test_settings_yaml_declares_past_future(self, model_name: str):
        clear_model_settings_cache()
        cap = load_capabilities_for_model(model_name)
        assert cap.covariates == "past_future", (
            f"{model_name} settings.yaml should declare covariates: past_future, "
            f"got {cap.covariates!r}"
        )

    def test_lafn_remains_none(self):
        """LAFN is a pre-trained foundation model and cannot support covariates."""
        clear_model_settings_cache()
        cap = load_capabilities_for_model("lafn")
        assert cap.covariates == "none"


# ---------------------------------------------------------------------------
# Source-level: verify covariate stacking code patterns exist in model files
# ---------------------------------------------------------------------------
class TestCovariateStackingSourcePatterns:
    """Parse model source to verify the covariate stacking pattern is present.

    Each model must:
    1. Set self._num_original_targets in train()
    2. Concatenate x_context into y_context
    3. Slice predictions to :num_original_targets before returning
    """

    @pytest.mark.parametrize("model_dir,model_file", [
        ("nbeats", "nbeats_model.py"),
        ("itransformer", "itransformer_model.py"),
        ("patchtsmixer", "patchtsmixer_model.py"),
        ("xgboost", "xgboost_model.py"),
        ("theta", "theta_model.py"),
    ])
    def test_num_original_targets_in_init(self, model_dir: str, model_file: str):
        source = (MODELS_ROOT / model_dir / model_file).read_text()
        assert "_num_original_targets" in source, (
            f"{model_file} must define _num_original_targets"
        )

    @pytest.mark.parametrize("model_dir,model_file", [
        ("nbeats", "nbeats_model.py"),
        ("itransformer", "itransformer_model.py"),
        ("patchtsmixer", "patchtsmixer_model.py"),
        ("xgboost", "xgboost_model.py"),
        ("theta", "theta_model.py"),
    ])
    def test_concatenate_covariates_in_train(self, model_dir: str, model_file: str):
        source = (MODELS_ROOT / model_dir / model_file).read_text()
        assert "np.concatenate([y_context, x_context]" in source, (
            f"{model_file} train() must concatenate x_context into y_context"
        )

    @pytest.mark.parametrize("model_dir,model_file", [
        ("nbeats", "nbeats_model.py"),
        ("itransformer", "itransformer_model.py"),
        ("patchtsmixer", "patchtsmixer_model.py"),
        ("xgboost", "xgboost_model.py"),
        ("theta", "theta_model.py"),
    ])
    def test_slice_predictions_to_original(self, model_dir: str, model_file: str):
        source = (MODELS_ROOT / model_dir / model_file).read_text()
        assert ":num_original_targets]" in source, (
            f"{model_file} predict() must slice output to :num_original_targets"
        )


# ---------------------------------------------------------------------------
# Cross-model: none of the five appear in NO_COVARIATE_MODELS
# ---------------------------------------------------------------------------
class TestPipelineCovariateSetsNewModels:

    def test_not_in_no_covariate_set(self):
        clear_model_settings_cache()
        from tempus_bench.utils.model_settings import get_no_covariate_models
        no_cov = get_no_covariate_models()
        for name in ("xgboost", "theta", "nbeats", "itransformer", "patchtsmixer"):
            assert name not in no_cov, f"{name} should NOT be in no-covariate set"

    def test_lafn_still_in_no_covariate_set(self):
        clear_model_settings_cache()
        from tempus_bench.utils.model_settings import get_no_covariate_models
        no_cov = get_no_covariate_models()
        assert "lafn" in no_cov, "LAFN must remain in no-covariate set"
