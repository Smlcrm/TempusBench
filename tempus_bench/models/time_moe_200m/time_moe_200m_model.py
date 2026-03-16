"""Time-MoE 200M params. Loads Maple728/TimeMoE-200M via settings.yaml."""

from tempus_bench.models.time_moe_50m.time_moe_50m_model import TimeMoe50mModel


class TimeMoe200mModel(TimeMoe50mModel):
    """Time-MoE 200M variant. Inherits full logic; hf_model_name from settings."""
    pass
