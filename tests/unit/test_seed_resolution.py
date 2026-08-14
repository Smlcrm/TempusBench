"""Per-generator seed derivation."""

import pytest

from tempus_bench import generators


def test_seed_is_deterministic():
    assert generators.resolve_seed(42, "random_walk") == generators.resolve_seed(
        42, "random_walk"
    )


def test_generators_differ_under_the_same_base_seed():
    names = sorted(generators.load_metadata())
    seeds = [generators.resolve_seed(42, name) for name in names]
    assert len(set(seeds)) == len(names)


def test_no_collisions_across_base_seeds():
    names = sorted(generators.load_metadata())
    seen = set()
    for base in range(20):
        for name in names:
            seed = generators.resolve_seed(base, name)
            assert seed not in seen, f"collision at base={base} name={name}"
            seen.add(seed)


def test_seed_is_base_times_stride_plus_generator_id():
    entry = generators.load_metadata()["random_walk"]
    expected = 3 * generators.SEED_STRIDE + entry["generator_id"]
    assert generators.resolve_seed(3, "random_walk") == expected


def test_unknown_generator_raises():
    with pytest.raises(KeyError, match="no_such_generator"):
        generators.resolve_seed(0, "no_such_generator")
