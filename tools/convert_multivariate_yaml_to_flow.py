import os
import sys
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq, CommentedMap


def set_flow_on_sequences(node: Any) -> None:
    """Recursively set flow style on all sequences while keeping mappings block style.

    This preserves values and keys; only list formatting changes.
    """
    if isinstance(node, CommentedSeq):
        node.fa.set_flow_style()
        for item in list(node):
            set_flow_on_sequences(item)
    elif isinstance(node, CommentedMap):
        for _, value in list(node.items()):
            set_flow_on_sequences(value)
    elif isinstance(node, dict):
        for _, value in list(node.items()):
            set_flow_on_sequences(value)
    elif isinstance(node, list):
        # Fallback in case loader returns plain list
        for item in list(node):
            set_flow_on_sequences(item)


def convert_file(path: str) -> None:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=2, offset=0)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    set_flow_on_sequences(data)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: convert_multivariate_yaml_to_flow.py <configs_dir>")
        return 2
    target_dir = argv[1]
    changed = 0
    for root, _dirs, files in os.walk(target_dir):
        for fn in files:
            if not fn.endswith(".yaml"):
                continue
            convert_file(os.path.join(root, fn))
            changed += 1
    print(f"Converted {changed} YAML file(s) to flow style lists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


