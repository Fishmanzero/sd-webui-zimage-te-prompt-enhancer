# -*- coding: utf-8 -*-
"""lib_ztpe —— 纯 Python 层。

刻意保持空的 `__init__`：不在这里 import 任何子模块。这样
`tests/run_selftest.py` 能在没有 torch、没有 WebUI 的系统 Python 下
直接 `import lib_ztpe.textproc` 而不会触发任何重依赖。

需要 torch 的部分在 `lib_ztpe.rt` 子包里，由 scripts/ 惰性导入。
"""
