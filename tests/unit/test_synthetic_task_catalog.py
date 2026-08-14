"""The generated Tasks/Synthetic Tasks catalog stays in step with the registry.

Reads the synthetic catalog directly rather than through
``paths.load_all_task_documents``, so these tests describe the generator taskbed
alone and do not depend on the state of the application catalog.
"""

from pathlib import Path

import pytest
import yaml

from tempus_bench import generators
from tempus_bench.utils.paths import get_project_root
from tempus_bench.utils.task_yaml_loader import build_task_config_from_raw

CATALOG = Path(get_project_root()) / "Tasks" / "Synthetic Tasks"


def _raw_documents():
    for path in sorted(CATALOG.glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if doc and "task" in doc:
                    yield path, doc["task"]


def _configs():
    return [build_task_config_from_raw(raw) for _, raw in _raw_documents()]


def test_there_is_one_yaml_per_primary_category():
    primary = {
        entry["primary_category"] for entry in generators.load_metadata().values()
    }
    assert len(list(CATALOG.glob("*.yaml"))) == len(primary)


def test_every_generator_appears_exactly_once():
    names = [config.dataset_name for config in _configs()]
    assert len(names) == 54
    assert len(set(names)) == 54


def test_task_names_are_unique_so_the_catalog_loader_accepts_them():
    names = [config.task_name for config in _configs()]
    assert len(set(names)) == len(names)


def test_every_document_names_a_real_generator_and_carries_no_file():
    metadata = generators.load_metadata()
    for config in _configs():
        assert config.is_synthetic()
        assert config.dataset_name in metadata
        assert config.file_name is None


def test_each_generator_sits_under_its_primary_category():
    metadata = generators.load_metadata()
    for config in _configs():
        assert (
            config.dataset_category
            == metadata[config.dataset_name]["primary_category"]
        )


def test_declared_variables_match_the_generator_output_width():
    for config in _configs():
        series = generators.generate(config.dataset_name, T=8, seed=0)
        columns = 1 if series.ndim == 1 else series.shape[1]
        declared = len(config.target_variable_names) + len(
            config.covariate_variable_names
        )
        assert declared == columns, config.dataset_name


def test_inferred_task_mode_agrees_with_the_registry_variate():
    metadata = generators.load_metadata()
    for config in _configs():
        assert config.task_mode == metadata[config.dataset_name]["variate"]


def test_count_and_binary_targets_are_not_standardised():
    for config in _configs():
        if config.target_type != "continuous_real":
            assert config.normalization_method == "none", config.dataset_name


def test_window_exceptions_from_the_design_doc_are_applied():
    windows = {
        config.dataset_name: (config.context_window, config.forecast_horizon)
        for config in _configs()
    }
    assert windows["multi_seasonal"][0] >= 512
    assert windows["logistic_map"][1] <= 20
    assert windows["mv_leadlag"][1] == 16


def test_every_category_tag_is_recoverable_from_the_registry():
    """Tasks appear once, so per-category reporting reads tags from metadata."""
    metadata = generators.load_metadata()
    tagged_trend = {
        name for name, entry in metadata.items() if "trend" in entry["categories"]
    }
    assert len(tagged_trend) == 16
    primary_trend = {
        config.dataset_name
        for config in _configs()
        if config.dataset_category == "trend"
    }
    assert primary_trend < tagged_trend


@pytest.mark.parametrize("field", ["series_length", "target_type"])
def test_generator_only_fields_are_populated(field):
    for config in _configs():
        assert getattr(config, field) is not None, config.dataset_name
