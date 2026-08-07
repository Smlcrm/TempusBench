"""Pickle filenames for prepared task datasets."""

from tempus_bench.utils.paths import task_dataset_filename


def test_application_tasks_keep_the_unsuffixed_name():
    assert task_dataset_filename("Daily Corn Futures", None) == "Daily Corn Futures.pkl"


def test_generator_tasks_are_suffixed_with_the_base_seed():
    assert task_dataset_filename("Random Walk", 0) == "Random Walk__seed0.pkl"
    assert task_dataset_filename("Random Walk", 12) == "Random Walk__seed12.pkl"


def test_distinct_seeds_never_share_a_filename():
    names = {task_dataset_filename("Random Walk", seed) for seed in range(50)}
    assert len(names) == 50


def test_seeded_name_never_collides_with_the_unseeded_one():
    assert task_dataset_filename("Random Walk", 0) != task_dataset_filename(
        "Random Walk", None
    )
