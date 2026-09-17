# 文件作用：尝试导入配置、业务模块和页面入口。
# 为什么有它：在本地和 CI 提前发现缺依赖、导入错误或模块入口问题。
"""Import modules without running CLI entrypoints or requesting models."""

import importlib
from pathlib import Path
import sys


# 做什么：逐个导入 app、config 和 src 下的 Python 模块。
# 为什么需要：在 CI 和本地尽早发现依赖缺失或导入错误。
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
