"""
Build a Hugging Face Dataset Card (README.md) for TempusBench task data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from discover import DEFAULT_REPO_ID, TaskAsset, discover_task_csvs, resolve_tasks_root

ARXIV_URL = "https://arxiv.org/pdf/2604.11529"
DATASET_URL = f"https://huggingface.co/datasets/{DEFAULT_REPO_ID}"


def build_frontmatter(assets: list[TaskAsset]) -> dict:
    configs: list[dict] = []
    folder_counts: dict[str, int] = {}
    for asset in assets:
        folder_counts[asset.folder_name] = folder_counts.get(asset.folder_name, 0) + 1

    for asset in assets:
        config_name = (
            f"{asset.folder_name}_{asset.category}"
            if folder_counts[asset.folder_name] > 1
            else asset.folder_name
        )
        configs.append(
            {
                "config_name": config_name,
                "data_files": [{"split": "train", "path": asset.hf_path}],
            }
        )

    return {
        "license": "cc-by-nc-nd-4.0",
        "language": ["en"],
        "pretty_name": "TempusBench Tasks",
        "configs": configs,
    }


def build_markdown_body() -> str:
    return f"""# TempusBench — task data

Time-series task CSVs in TempusBench format (`variable_name`, `timestamps`, `values`).
Each **config** is one benchmark task folder (see YAML `configs` in the frontmatter above).

Task definitions (`task.yaml`, `metadata.json`) ship with the TempusBench package or
can be synced from this dataset under the `tasks/` prefix.

## Usage with TempusBench

```bash
pip install tempus_bench
python -m tempus_bench.run_benchmark --config tempus_bench/config/local_test.yaml
```

On first run, missing task CSVs are downloaded automatically from this dataset.

## Citation

TempusBench: [{ARXIV_URL}]({ARXIV_URL})

## License

Project default license for TempusBench: **CC-BY-NC-ND-4.0**. Individual series may
inherit upstream terms — see the TempusBench paper and source listings.

Dataset page: {DATASET_URL}
"""


def render_readme(assets: list[TaskAsset]) -> str:
    frontmatter = build_frontmatter(assets)
    yaml_str = yaml.safe_dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    return "---\n" + yaml_str + "\n---\n\n" + build_markdown_body()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, help="Path to tempus_bench/tasks")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("README.md"),
        help="Write Dataset Card here (default: ./README.md)",
    )
    args = parser.parse_args()

    root = resolve_tasks_root(args.tasks_dir)
    assets = discover_task_csvs(root)
    if not assets:
        raise SystemExit("No tasks discovered.")

    text = render_readme(assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output} with {len(assets)} config(s).")


if __name__ == "__main__":
    main()
