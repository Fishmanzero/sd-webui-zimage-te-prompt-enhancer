# -*- coding: utf-8 -*-
"""纯 stdlib 自检运行器。不导入 torch，不依赖 WebUI。

用法：
    python tests/run_selftest.py
    <venv>/python.exe tests/run_selftest.py

任一用例失败则退出码为 1。
"""

import importlib
import os
import sys

_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_HERE)

for path in (_ROOT, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

_MODULES = ("test_presets", "test_textproc", "test_cache", "test_config")

_results = []


def check(name, condition, detail=""):
    ok = bool(condition)
    _results.append((ok, name, detail))
    suffix = "" if ok else f"   <-- {detail}"
    print(("  PASS  " if ok else "  FAIL  ") + name + suffix)
    return ok


def main():
    for module_name in _MODULES:
        print(module_name)
        module = importlib.import_module(module_name)
        module.run(check)

    failed = [item for item in _results if not item[0]]
    total = len(_results)
    print()
    if failed:
        print(f"{len(failed)} / {total} 项失败")
        for _, name, detail in failed:
            print(f"  - {name}  {detail}")
        return 1

    print(f"全部通过（{total} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
