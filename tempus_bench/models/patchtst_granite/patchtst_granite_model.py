"""PatchTST Granite (~40M). Fine-tuned on ETTh1. Loads ibm-granite/granite-timeseries-patchtst via settings.yaml."""

from tempus_bench.models.patchtst_fm.patchtst_fm_model import PatchtstFmModel


class PatchtstGraniteModel(PatchtstFmModel):
    """PatchTST Granite variant. Inherits full logic; hf_model_name from settings."""

    def train(self, *args, **kwargs):
        """Override to pass config with n_head from num_attention_heads for PatchTSTFM compatibility."""
        from tempus_bench.models.base_model import validate_covariate_support

        from tsfm_public import PatchTSTFMConfig, PatchTSTFMForPrediction

        from transformers import AutoConfig

        x_context = kwargs.get("x_context")
        x_target = kwargs.get("x_target")
        validate_covariate_support(
            x_context, x_target,
            supports_past_only=True,
            supports_future_only=False,
            supports_both=False,
            model_name="PatchTST-FM",
        )
        hf_config = AutoConfig.from_pretrained(self.hf_model_name)
        n_head = getattr(hf_config, "num_attention_heads", 16)
        d_patch = getattr(hf_config, "patch_length", getattr(hf_config, "d_patch", 12))
        # Granite PatchTST backbone expects context_length divisible by d_patch.
        # Config has context_length=512 but 512/12 is not integer; use 504 (42*12).
        raw_ctx = getattr(hf_config, "context_length", 512)
        context_length = (raw_ctx // d_patch) * d_patch
        config = PatchTSTFMConfig(
            d_model=getattr(hf_config, "d_model", getattr(hf_config, "hidden_size", 128)),
            n_head=n_head,
            n_layer=getattr(hf_config, "n_layer", getattr(hf_config, "num_hidden_layers", 3)),
            context_length=context_length,
            prediction_length=getattr(hf_config, "prediction_length", 96),
            d_patch=d_patch,
        )
        device = getattr(self, "device", "cpu")
        self._model = PatchTSTFMForPrediction.from_pretrained(self.hf_model_name, config=config)
        self._model = self._model.to(device)
        self.is_fitted = True
        return self
