"""Chronos-Bolt Tiny (~9M params). Loads amazon/chronos-bolt-tiny via settings.yaml."""

from tempus_bench.models.chronos_bolt_base.chronos_bolt_base_model import ChronosBoltBaseModel


class ChronosBoltTinyModel(ChronosBoltBaseModel):
    """Chronos-Bolt tiny variant. Inherits full logic; hf_model_name from settings."""
    pass
