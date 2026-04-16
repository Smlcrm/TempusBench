"""Chronos-T5 Small (~46M params). Loads amazon/chronos-t5-small via settings.yaml."""

from tempus_bench.models.chronos_tiny.chronos_tiny_model import ChronosTinyModel


class ChronosSmallModel(ChronosTinyModel):
    """Chronos T5 small variant. Inherits full logic; hf_model_name from settings."""
    pass
