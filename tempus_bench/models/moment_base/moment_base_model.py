"""MOMENT Base (~125M params). Loads AutonLab/MOMENT-1-base via settings.yaml."""

from tempus_bench.models.moment_large.moment_large_model import MomentLargeModel


class MomentBaseModel(MomentLargeModel):
    """MOMENT base variant. Inherits full logic; hf_model_name from settings."""
    pass
