"""Moirai 1.1 Small (~14M params). Loads Salesforce/moirai-1.1-R-small via size in settings."""

from tempus_bench.models.moirai_base.moirai_base_model import MoiraiBaseModel


class MoiraiSmallModel(MoiraiBaseModel):
    """Moirai small variant. Inherits full logic; size from settings.yaml."""
    pass
