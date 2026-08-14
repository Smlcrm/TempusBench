"""Record of the one-time split of eval-synthetic-dataset/synthetic_generators.py
into tempus_bench/generators/.

Wrote one module per generator, a shared _common.py, and metadata.json (the
registry). Kept for auditability of that migration; it is not imported at run
time and cannot be re-run, because its input file was removed once the split was
verified to reproduce every generator's output exactly.

metadata.json is now the source of truth. To change a generator's metadata, edit
that file and re-run scripts/build_synthetic_task_yamls.py. Generator ids in it
are permanent: renumbering changes every derived seed.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "eval-synthetic-dataset" / "synthetic_generators.py"
OUT = ROOT / "tempus_bench" / "generators"

SHARED_FUNCS = {"_rng", "_t", "_ar1", "_base_signal"}
SHARED_CONSTS = {"DEFAULT_T", "PERIOD", "LONG_PERIOD", "BURN_IN"}

# Section 5 of synthetic_tasks.md: per-task window exceptions.
WINDOW_OVERRIDES = {
    "multi_seasonal": {"context_window": 512},
    "mv_leadlag": {"forecast_horizon": 16},
    "logistic_map": {"forecast_horizon": 20},
}


def _summarise(docstring: str) -> str:
    """One-line description from a generator docstring.

    Takes the first paragraph only (the rest is the DGP derivation), collapses it
    onto one line, strips RST inline literals, and drops a trailing colon left by
    a lead-in to a formula block.
    """
    block = docstring.strip().split("\n\n")[0]
    text = " ".join(line.strip() for line in block.splitlines())
    text = text.replace("``", "")
    text = re.sub(r"\s+", " ", text).strip().rstrip(":")
    if len(text) > 200:
        text = text[:200].rsplit(" ", 1)[0] + "..."
    return text


def main() -> None:
    source = SRC.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    def segment(node: ast.AST) -> str:
        """Exact source text of a top-level node."""
        return "".join(lines[node.lineno - 1 : node.end_lineno])

    funcs: dict[str, ast.FunctionDef] = {}
    consts: dict[str, ast.Assign] = {}
    tasks_node: ast.Assign | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "TASKS":
                tasks_node = node
            else:
                consts[name] = node

    if tasks_node is None:
        raise SystemExit("TASKS registry not found in the source file")

    # Each value is a dict(fn=<name>, variate=..., ...) call, so it is not a
    # literal and ast.literal_eval cannot read it. Walk the keywords instead:
    # `fn` is a bare Name referring to the generator function, the rest are
    # literals.
    registry: dict[str, dict] = {}
    for key_node, value_node in zip(tasks_node.value.keys, tasks_node.value.values):
        name = ast.literal_eval(key_node)
        spec: dict = {}
        for keyword in value_node.keywords:
            if keyword.arg == "fn":
                spec["function"] = keyword.value.id
            else:
                spec[keyword.arg] = ast.literal_eval(keyword.value)
        registry[name] = spec

    OUT.mkdir(parents=True, exist_ok=True)

    # --- _common.py ---------------------------------------------------------
    common = [
        '"""Shared constants and helpers for the data generators.\n\n',
        "Extracted verbatim from the original synthetic_generators.py so the\n",
        "per-generator modules stay byte-identical in behaviour.\n",
        '"""\n\n',
        "from __future__ import annotations\n\nimport numpy as np\n\n",
    ]
    for name in sorted(SHARED_CONSTS):
        common.append(segment(consts[name]))
    for name in sorted(SHARED_FUNCS):
        common.append("\n\n" + segment(funcs[name]).rstrip() + "\n")
    (OUT / "_common.py").write_text("".join(common), encoding="utf-8")

    # --- one module per generator -------------------------------------------
    metadata: dict[str, dict] = {}
    for gen_id, (name, spec) in enumerate(registry.items(), start=1):
        fn_name = spec["function"]
        body = segment(funcs[fn_name]).rstrip() + "\n"
        used = sorted(
            symbol
            for symbol in SHARED_FUNCS | SHARED_CONSTS
            if re.search(rf"\b{symbol}\b", body)
        )
        header = [
            f'"""Generator: {name}.\n\n',
            "See eval-synthetic-dataset/synthetic_tasks.md for the design rationale\n",
            "and the full data-generating process.\n",
            '"""\n\n',
            "from __future__ import annotations\n\nimport numpy as np\n",
        ]
        if used:
            header.append(f"\nfrom ._common import {', '.join(used)}\n")
        (OUT / f"{name}.py").write_text("".join(header) + "\n\n" + body, encoding="utf-8")

        docstring = ast.get_docstring(funcs[fn_name]) or ""
        summary = _summarise(docstring)
        entry = {
            "generator_id": gen_id,
            "function": fn_name,
            "primary_category": spec["categories"][0],
            "categories": spec["categories"],
            "variate": spec["variate"],
            "target_type": spec["target_type"],
            "description": summary,
            "default_series_length": 2048,
            "context_window": 512,
            "forecast_horizon": 64,
            # Normalising a count, binary or strictly positive target would
            # destroy the property the task exists to test.
            "normalization_method": (
                "standard" if spec["target_type"] == "continuous_real" else "none"
            ),
        }
        entry.update(WINDOW_OVERRIDES.get(name, {}))
        metadata[name] = entry

    (OUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(metadata)} generator modules + metadata.json to {OUT}")


if __name__ == "__main__":
    main()
