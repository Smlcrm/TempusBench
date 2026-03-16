"""Kairos 23M params. Loads mldi-lab/Kairos_23m via settings.yaml."""

from tempus_bench.models.kairos_50m.kairos_50m_model import Kairos50mModel


class Kairos23mModel(Kairos50mModel):
    """Kairos 23M variant. Inherits full logic; hf_model_name from settings."""
    pass
