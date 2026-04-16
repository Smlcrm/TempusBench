"""Chronos-T5 Mini (~20M params). Loads amazon/chronos-t5-mini via settings.yaml."""

from tempus_bench.models.chronos_tiny.chronos_tiny_model import ChronosTinyModel


class ChronosMiniModel(ChronosTinyModel):
    """Chronos T5 mini variant. Inherits full logic; hf_model_name from settings."""
    pass
