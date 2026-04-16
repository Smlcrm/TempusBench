"""Tests for Tiny Time Mixer ↔ transformers tied-weight loading compatibility."""

from __future__ import annotations

import os
import unittest

# Prefer torch-only in ``transformers`` so a broken optional TensorFlow install does not
# block imports (see transformers ``import_utils``: USE_TORCH=1 disables TF).
os.environ.setdefault("USE_TORCH", "1")

import torch


def _pretrained_model_has_mark_tied_weights() -> bool:
    from transformers.modeling_utils import PreTrainedModel

    return hasattr(PreTrainedModel, "mark_tied_weights_as_initialized")


@unittest.skipUnless(
    _pretrained_model_has_mark_tied_weights(),
    "PreTrainedModel.mark_tied_weights_as_initialized not present in this transformers version",
)
class TestTtmTransformersTiedWeightPatch(unittest.TestCase):
    def setUp(self) -> None:
        from transformers.modeling_utils import PreTrainedModel

        self._orig_mark = PreTrainedModel.mark_tied_weights_as_initialized
        import tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model as ttm_mod

        self._ttm_mod = ttm_mod
        ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = False
        ttm_mod._patch_transformers_tiny_time_mixer_tied_weights()

    def tearDown(self) -> None:
        from transformers.modeling_utils import PreTrainedModel

        PreTrainedModel.mark_tied_weights_as_initialized = self._orig_mark
        self._ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = False

    def test_patch_skips_invalid_tied_parameter_names_without_raising(self) -> None:
        """Mimics Batch failure: ``all_tied_weights_keys`` lists a path with no ``nn.Parameter``."""

        class _LoadingInfo:
            def __init__(self) -> None:
                self.missing_keys: set[str] = set()

        class _FakeModel:
            def __init__(self) -> None:
                self.all_tied_weights_keys = {
                    "nonexistent.tied_leaf": "other",
                    "real.weight": "other",
                }
                self._real = torch.nn.Parameter(torch.tensor([2.0]))

            def get_parameter(self, name: str) -> torch.nn.Parameter:
                if name == "nonexistent.tied_leaf":
                    raise AttributeError(name)
                if name == "real.weight":
                    return self._real
                raise AttributeError(name)

            def is_remote_code(self) -> bool:
                return False

        m = _FakeModel()
        info = _LoadingInfo()
        from transformers.modeling_utils import PreTrainedModel

        PreTrainedModel.mark_tied_weights_as_initialized(m, info)
        self.assertTrue(getattr(m._real, "_is_hf_initialized", False))

    def test_patch_idempotent_second_call_is_noop_on_flag(self) -> None:
        self._ttm_mod._patch_transformers_tiny_time_mixer_tied_weights()
        # still same patched method
        from transformers.modeling_utils import PreTrainedModel

        self.assertIsNot(PreTrainedModel.mark_tied_weights_as_initialized, self._orig_mark)

    def test_remote_code_only_tied_weights_keys_dict_backfills_all_tied(self) -> None:
        """Remote-code TTM may expose only ``_tied_weights_keys`` (transformers 5.x)."""

        class _LoadingInfo:
            def __init__(self) -> None:
                self.missing_keys: set[str] = set()

        class _FakeRemoteTtm:
            def __init__(self) -> None:
                self._tied_weights_keys = {"a.weight": "b.weight"}
                self._p = torch.nn.Parameter(torch.tensor([1.0]))

            def get_parameter(self, name: str) -> torch.nn.Parameter:
                if name == "a.weight":
                    return self._p
                raise AttributeError(name)

            def is_remote_code(self) -> bool:
                return True

            def get_parameter_or_buffer(self, key: str) -> object:
                o = type("_O", (), {})()
                o._is_hf_initialized = True
                return o

        m = _FakeRemoteTtm()
        info = _LoadingInfo()
        from transformers.modeling_utils import PreTrainedModel

        PreTrainedModel.mark_tied_weights_as_initialized(m, info)
        self.assertIsInstance(m.all_tied_weights_keys, dict)
        self.assertEqual(m.all_tied_weights_keys, m._tied_weights_keys)
        self.assertTrue(getattr(m._p, "_is_hf_initialized", False))

    def test_remote_code_branch_tolerates_bad_buffer_lookup(self) -> None:
        class _LoadingInfo:
            def __init__(self) -> None:
                self.missing_keys = {"bad_missing", "ok_missing"}

        class _FakeModelRemote:
            def __init__(self) -> None:
                self.all_tied_weights_keys: dict[str, str] = {}

            def get_parameter(self, name: str) -> torch.nn.Parameter:
                raise AttributeError(name)

            def is_remote_code(self) -> bool:
                return True

            def get_parameter_or_buffer(self, key: str) -> object:
                if key == "bad_missing":
                    raise AttributeError(key)
                o = type("_O", (), {})()
                o._is_hf_initialized = False
                return o

        m = _FakeModelRemote()
        info = _LoadingInfo()
        from transformers.modeling_utils import PreTrainedModel

        PreTrainedModel.mark_tied_weights_as_initialized(m, info)
        self.assertIn("bad_missing", info.missing_keys)
        self.assertIn("ok_missing", info.missing_keys)


class TestTtmTransformersTiedWeightPatchNoMarkTiedApi(unittest.TestCase):
    """Runs on all supported ``transformers`` versions (including those without ``mark_tied_*``)."""

    def test_patch_marks_done_without_mark_tied_api_or_after_apply(self) -> None:
        import tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model as ttm_mod

        ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = False
        ttm_mod._patch_transformers_tiny_time_mixer_tied_weights()
        self.assertTrue(ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE)
        ttm_mod._patch_transformers_tiny_time_mixer_tied_weights()
        self.assertTrue(ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE)


@unittest.skipUnless(
    __import__("os").environ.get("TTM_INTEGRATION") == "1",
    "Set TTM_INTEGRATION=1 to run sktime + HF download smoke test (slow, needs network).",
)
class TestTtmSktimeIntegration(unittest.TestCase):
    def test_tiny_time_mixer_forecaster_fit_predict_smoke(self) -> None:
        import numpy as np
        import pandas as pd
        from sktime.forecasting.ttm import TinyTimeMixerForecaster

        import tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model as ttm_mod

        ttm_mod._TINY_TIME_MIXER_TRANSFORMERS_TIED_PATCH_DONE = False
        ttm_mod._patch_transformers_tiny_time_mixer_tied_weights()

        rng = np.random.default_rng(0)
        n = 256
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        y = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=[0])
        fh = list(range(1, 9))
        forecaster = TinyTimeMixerForecaster(model_path="ibm/TTM", revision="main")
        forecaster.fit(y, fh=fh)
        pred = forecaster.predict(fh=fh)
        self.assertEqual(pred.shape[0], 8)


if __name__ == "__main__":
    unittest.main()
