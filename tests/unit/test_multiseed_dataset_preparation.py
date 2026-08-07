"""End-to-end check of the per-seed dataset preparation loop.

Mirrors what ``BenchmarkRunner.__enter__`` does, without spawning model
subprocesses: every configured base seed must yield its own pickled Dataset, and
the pickles must differ from each other but be reproducible.
"""

import pickle
import tempfile

import numpy as np
import pytest

from tempus_bench.pipeline.data_loader import DataLoader
from tempus_bench.utils.configs import EvaluationConfig
from tempus_bench.utils.log_manager import LogManager
from tempus_bench.utils.paths import task_dataset_filename
from tempus_bench.utils.task_yaml_loader import load_synthetic_task_configs


@pytest.fixture(autouse=True)
def _log_manager():
    LogManager.log_manager = None
    with tempfile.TemporaryDirectory() as d:
        lm = LogManager(
            logs_path=d,
            console_logging=False,
            file_logging=False,
            tensorboard_logging=False,
        )
        yield lm
        try:
            lm.close()
        except Exception:
            pass
        LogManager.log_manager = None


def _prepare(task_configs, evaluation_config, out_dir):
    """The dataset preparation loop from BenchmarkRunner.__enter__."""
    seeds = evaluation_config.seed_list()
    written = []
    for task_config in task_configs:
        base_seeds = seeds if task_config.is_synthetic() else [None]
        for base_seed in base_seeds:
            path = out_dir / task_dataset_filename(task_config.task_name, base_seed)
            loader = DataLoader(task_config, evaluation_config, base_seed=base_seed)
            with open(path, "wb") as handle:
                pickle.dump(loader.dataset, handle)
            written.append(path)
    return written


def test_each_seed_produces_its_own_dataset(tmp_path):
    configs = load_synthetic_task_configs("Synthetic Tasks/Covariate")
    evaluation = EvaluationConfig(task_path="Synthetic Tasks/Covariate", seeds=[0, 1, 2])

    written = _prepare(configs, evaluation, tmp_path)

    assert len(written) == len(configs) * 3
    assert all(path.is_file() for path in written)


def test_datasets_differ_across_seeds_but_repeat_within_a_seed(tmp_path):
    configs = [c for c in load_synthetic_task_configs("Synthetic Tasks/Covariate")][:1]
    evaluation = EvaluationConfig(task_path="Synthetic Tasks/Covariate", seeds=[0, 1])

    _prepare(configs, evaluation, tmp_path)
    first = tmp_path / task_dataset_filename(configs[0].task_name, 0)
    second = tmp_path / task_dataset_filename(configs[0].task_name, 1)

    with open(first, "rb") as handle:
        a = pickle.load(handle)
    with open(second, "rb") as handle:
        b = pickle.load(handle)

    assert not np.array_equal(np.array(a.target), np.array(b.target))
    assert a.metadata["base_seed"] == 0
    assert b.metadata["base_seed"] == 1

    # Re-preparing seed 0 reproduces it exactly.
    again = tmp_path / "again"
    again.mkdir()
    _prepare(
        configs,
        EvaluationConfig(task_path="Synthetic Tasks/Covariate", seeds=0),
        again,
    )
    with open(again / task_dataset_filename(configs[0].task_name, 0), "rb") as handle:
        repeat = pickle.load(handle)
    assert np.array_equal(np.array(a.target), np.array(repeat.target))


def test_the_whole_taskbed_prepares_under_a_single_seed(tmp_path):
    configs = load_synthetic_task_configs("Synthetic Tasks")
    evaluation = EvaluationConfig(task_path="Synthetic Tasks", seeds=0)

    written = _prepare(configs, evaluation, tmp_path)

    assert len(written) == 54
    assert len({path.name for path in written}) == 54
