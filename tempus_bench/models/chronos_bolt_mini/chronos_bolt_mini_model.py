"""Chronos-Bolt Mini (~21M params). Loads amazon/chronos-bolt-mini via settings.yaml."""

from tempus_bench.models.chronos_bolt_base.chronos_bolt_base_model import ChronosBoltBaseModel


class ChronosBoltMiniModel(ChronosBoltBaseModel):
    """Chronos-Bolt mini variant. Inherits full logic; hf_model_name from settings."""
    pass
