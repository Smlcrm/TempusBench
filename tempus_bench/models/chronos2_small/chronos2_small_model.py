"""Chronos-2 Small (~40M params). Loads autogluon/chronos-2-small via settings.yaml."""

from tempus_bench.models.chronos2.chronos2_model import Chronos2Model


class Chronos2SmallModel(Chronos2Model):
    """Chronos-2 small variant. Inherits full logic; hf_model_name from settings."""
    pass
