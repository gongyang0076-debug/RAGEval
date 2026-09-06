"""Import modules without running CLI entrypoints or requesting models."""

import importlib
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    modules = ["app"]
    for folder in ("config", "src"):
        for path in sorted((root / folder).rglob("*.py")):
            relative = path.relative_to(root).with_suffix("")
            parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
            modules.append(".".join(parts))
    for module in modules:
        importlib.import_module(module)
    print(f"IMPORT CHECK: PASS ({len(modules)} modules)")


if __name__ == "__main__":
    main()
