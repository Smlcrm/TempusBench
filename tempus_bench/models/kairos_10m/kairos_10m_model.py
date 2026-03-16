"""Kairos 10M params. Loads mldi-lab/Kairos_10m via settings.yaml."""

from tempus_bench.models.kairos_50m.kairos_50m_model import Kairos50mModel


class Kairos10mModel(Kairos50mModel):
    """Kairos 10M variant. Inherits full logic; hf_model_name from settings."""
    pass
