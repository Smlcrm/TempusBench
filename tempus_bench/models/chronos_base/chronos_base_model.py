"""Chronos-T5 Base (~200M params). Loads amazon/chronos-t5-base via settings.yaml."""

from tempus_bench.models.chronos_tiny.chronos_tiny_model import ChronosTinyModel


class ChronosBaseModel(ChronosTinyModel):
    """Chronos T5 base variant. Inherits full logic; hf_model_name from settings."""
    pass
