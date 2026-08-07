"""The generated Tasks/Synthetic Tasks catalog stays in step with the registry."""

from pathlib import Path

from tempus_bench import generators
from tempus_bench.utils.paths import get_project_root
from tempus_bench.utils.task_yaml_loader import load_task_configs_from_category_yaml

CATALOG = Path(get_project_root()) / "Tasks" / "Synthetic Tasks"


def _all_configs():
    for path in sorted(CATALOG.glob("*.yaml")):
        for config in load_task_configs_from_category_yaml(path):
            yield path, config


def test_there_is_one_yaml_per_category():
    categories = {
        category
        for entry in generators.load_metadata().values()
        for category in entry["categories"]
    }
    assert len(list(CATALOG.glob("*.yaml"))) == len(categories)


def test_every_document_loads_and_names_a_real_generator():
    metadata = generators.load_metadata()
    for _, config in _all_configs():
        assert config.is_synthetic()
        assert config.dataset_name in metadata
        assert config.file_name is None


def test_every_generator_appears_in_each_of_its_categories():
    metadata = generators.load_metadata()
    seen: dict[str, set[str]] = {}
    for _, config in _all_configs():
        seen.setdefault(config.dataset_name, set()).add(config.dataset_category)
    for name, entry in metadata.items():
        assert seen[name] == set(entry["categories"]), name


def test_deduping_by_generator_recovers_exactly_54_tasks():
    names = {config.dataset_name for _, config in _all_configs()}
    assert len(names) == 54


def test_declared_variables_match_the_generator_output_width():
    for _, config in _all_configs():
        width = generators.generate(config.dataset_name, T=8, seed=0)
        columns = 1 if width.ndim == 1 else width.shape[1]
        declared = len(config.target_variable_names) + len(
            config.covariate_variable_names
        )
        assert declared == columns, config.dataset_name


def test_count_and_binary_targets_are_not_standardised():
    for _, config in _all_configs():
        if config.target_type != "continuous_real":
            assert config.normalization_method == "none", config.dataset_name


def test_window_exceptions_from_the_design_doc_are_applied():
    windows = {
        config.dataset_name: (config.context_window, config.forecast_horizon)
        for _, config in _all_configs()
    }
    assert windows["multi_seasonal"][0] >= 512
    assert windows["logistic_map"][1] <= 20
    assert windows["mv_leadlag"][1] == 16


def test_task_path_pattern_recognises_the_synthetic_catalog():
    from tempus_bench.utils.task_yaml_loader import is_synthetic_task_path

    assert is_synthetic_task_path("Synthetic Tasks")
    assert is_synthetic_task_path("Synthetic Tasks/Trend")
    assert not is_synthetic_task_path("covariate/covariate_transport_monthly")


def test_a_single_category_pattern_loads_only_that_category():
    from tempus_bench.utils.task_yaml_loader import load_synthetic_task_configs

    configs = load_synthetic_task_configs("Synthetic Tasks/Trend")
    assert configs
    assert {c.dataset_category for c in configs} == {"trend"}


def test_whole_bed_dedupes_to_one_document_per_generator():
    from tempus_bench.utils.task_yaml_loader import load_synthetic_task_configs

    configs = load_synthetic_task_configs("Synthetic Tasks")
    names = [c.dataset_name for c in configs]
    assert len(names) == len(set(names)) == 54


def test_whole_bed_keeps_each_generator_under_its_primary_category():
    from tempus_bench.utils.task_yaml_loader import load_synthetic_task_configs

    metadata = generators.load_metadata()
    for config in load_synthetic_task_configs("Synthetic Tasks"):
        assert config.dataset_category == metadata[config.dataset_name]["primary_category"]


def test_unknown_category_is_reported():
    import pytest

    from tempus_bench.utils.task_yaml_loader import load_synthetic_task_configs

    with pytest.raises(FileNotFoundError, match="Synthetic category file not found"):
        load_synthetic_task_configs("Synthetic Tasks/NoSuchCategory")
