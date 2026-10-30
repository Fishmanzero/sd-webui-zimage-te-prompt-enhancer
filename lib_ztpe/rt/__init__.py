# -*- coding: utf-8 -*-
"""lib_ztpe.rt —— 需要 torch 与 Forge backend 的运行时层。

同样保持空 `__init__`。子模块只能由 scripts/ 在函数内部导入，
避免 torch / backend 导入异常导致整个脚本被 Forge 静默丢弃。
"""
