"""Regenerate Tasks/Synthetic Tasks/*.yaml from tempus_bench/generators/metadata.json.

One file per category, and each generator appears exactly once, in its primary
category. The catalog loader requires task_name to be unique across all of
Tasks/, so a generator cannot be repeated under every category it is tagged
with; the full tag list stays in metadata.json for per-category reporting.

Re-run after any metadata change; never hand-edit the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tempus_bench import generators  # noqa: E402
from tempus_bench.utils.paths import get_project_root  # noqa: E402

OUT = Path(get_project_root()) / "Tasks" / "Synthetic Tasks"

CATEGORY_TITLES = {
    "stationarity": "Stationarity",
    "trend": "Trend",
    "seasonality": "Seasonality",
    "noise": "Noise",
    "memory": "Memory",
    "nonlinearity": "Nonlinearity",
    "structural_change": "Structural Change",
    "target_type": "Target Type",
    "intermittency": "Intermittency",
    "multivariate": "Multivariate",
    "covariate": "Covariate",
}


# Registry keys are snake_case; naive .title() mangles the statistical acronyms.
ACRONYMS = {
    "ar": "AR",
    "ar1": "AR(1)",
    "ar2": "AR(2)",
    "ma1": "MA(1)",
    "mv": "MV",
    "garch": "GARCH",
    "setar": "SETAR",
    "iid": "IID",
    "fgn": "fGn",
    "snr": "SNR",
    "negbin": "NegBin",
    "var": "VAR",
    "rw": "RW",
}


def _title(name: str) -> str:
    """Human-readable task name for a registry key."""
    return " ".join(ACRONYMS.get(word, word.capitalize()) for word in name.split("_"))


def _variables(name: str, entry: dict) -> tuple[list[str], list[str]]:
    """Target and covariate names, matching the generator's column order."""
    if entry["variate"] == "covariate":
        return ["y"], ["x"]
    if entry["variate"] == "multivariate":
        # The column count is a property of the generator, so ask it rather than
        # duplicating each generator's default width in the metadata.
        width = generators.generate(name, T=8, seed=0).shape[1]
        return [f"y{i + 1}" for i in range(width)], []
    return ["y"], []


def _document(name: str, entry: dict, category: str) -> dict:
    targets, covariates = _variables(name, entry)
    return {
        "task": {
            "task_name": _title(name),
            "task_description": entry["description"],
            "context_window": entry["context_window"],
            "forecast_horizon": entry["forecast_horizon"],
            "handle_missing": "interpolate",
            "normalization_method": entry["normalization_method"],
            "task_catalog": "synthetic",
            "dataset_category": category,
            "dataset_name": name,
            "target_type": entry["target_type"],
            "series_length": entry["default_series_length"],
            "target_variable_names": targets,
            "covariate_variable_names": covariates,
        }
    }


def main() -> None:
    metadata = generators.load_metadata()
    OUT.mkdir(parents=True, exist_ok=True)

    by_category: dict[str, list[dict]] = {}
    for name, entry in metadata.items():
        for category in entry["categories"]:
            if category not in CATEGORY_TITLES:
                raise SystemExit(
                    f"{name}: unknown category {category!r}; add it to CATEGORY_TITLES"
                )
        category = entry["primary_category"]
        by_category.setdefault(category, []).append(_document(name, entry, category))

    total = 0
    for category, documents in sorted(by_category.items()):
        path = OUT / f"{CATEGORY_TITLES[category]}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump_all(documents, handle, sort_keys=False, allow_unicode=True)
        total += len(documents)
        print(f"{path.name}: {len(documents)} task(s)")
    print(f"{total} documents over {len(metadata)} generators")


if __name__ == "__main__":
    main()
