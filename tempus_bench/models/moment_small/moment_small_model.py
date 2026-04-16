"""MOMENT Small (~40M params). Loads AutonLab/MOMENT-1-small via settings.yaml."""

from tempus_bench.models.moment_large.moment_large_model import MomentLargeModel


class MomentSmallModel(MomentLargeModel):
    """MOMENT small variant. Inherits full logic; hf_model_name from settings."""
    pass
