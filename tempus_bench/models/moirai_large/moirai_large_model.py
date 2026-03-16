"""Moirai 1.1 Large (~311M params). Loads Salesforce/moirai-1.1-R-large via size in settings."""

from tempus_bench.models.moirai_base.moirai_base_model import MoiraiBaseModel


class MoiraiLargeModel(MoiraiBaseModel):
    """Moirai large variant. Inherits full logic; size from settings.yaml."""
    pass
