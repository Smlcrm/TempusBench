"""TTM R2.1 (~1.7M params). Loads ibm-granite/granite-timeseries-ttm-r2.1 via settings.yaml."""

from tempus_bench.models.tiny_time_mixer_r1.tiny_time_mixer_r1_model import TinyTimeMixerR1Model


class TinyTimeMixerR21Model(TinyTimeMixerR1Model):
    """Tiny Time Mixer R2.1 variant. Inherits full logic; hf_model_name from settings."""
    pass
