"""Chronos-Bolt Small (~48M params). Loads amazon/chronos-bolt-small via settings.yaml."""

from tempus_bench.models.chronos_bolt_base.chronos_bolt_base_model import ChronosBoltBaseModel


class ChronosBoltSmallModel(ChronosBoltBaseModel):
    """Chronos-Bolt small variant. Inherits full logic; hf_model_name from settings."""
    pass
