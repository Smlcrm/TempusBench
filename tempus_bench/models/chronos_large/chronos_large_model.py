"""Chronos-T5 Large (~710M params). Loads amazon/chronos-t5-large via settings.yaml."""

from tempus_bench.models.chronos_tiny.chronos_tiny_model import ChronosTinyModel


class ChronosLargeModel(ChronosTinyModel):
    """Chronos T5 large variant. Inherits full logic; hf_model_name from settings."""
    pass
