"""Registry integrity for the data generators."""

import numpy as np
import pytest

from tempus_bench import generators


def test_metadata_has_54_generators_with_unique_ids():
    meta = generators.load_metadata()
    assert len(meta) == 54
    ids = [entry["generator_id"] for entry in meta.values()]
    assert sorted(ids) == list(range(1, 55))


@pytest.mark.parametrize("name", sorted(generators.load_metadata()))
def test_every_generator_imports_and_returns_expected_shape(name):
    meta = generators.load_metadata()[name]
    out = generators.generate(name, T=256, seed=0)
    assert isinstance(out, np.ndarray)
    assert np.all(np.isfinite(out))
    if meta["variate"] == "univariate":
        assert out.shape == (256,)
    else:
        assert out.ndim == 2 and out.shape[0] == 256


def test_same_seed_reproduces_same_series():
    a = generators.generate("random_walk", T=128, seed=7)
    b = generators.generate("random_walk", T=128, seed=7)
    c = generators.generate("random_walk", T=128, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_default_series_length_comes_from_metadata():
    out = generators.generate("linear_trend", seed=0)
    assert out.shape == (generators.load_metadata()["linear_trend"]["default_series_length"],)


def test_unknown_generator_raises_keyerror_naming_the_generator():
    with pytest.raises(KeyError, match="no_such_generator"):
        generators.generate("no_such_generator")


def test_every_entry_declares_a_primary_category_within_its_categories():
    for name, entry in generators.load_metadata().items():
        assert entry["primary_category"] == entry["categories"][0], name
