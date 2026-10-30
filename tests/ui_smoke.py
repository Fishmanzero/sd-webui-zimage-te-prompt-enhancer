# -*- coding: utf-8 -*-
"""无头 UI 构建自检。

用桩替身顶掉 modules.scripts / modules.shared，在真实 gradio 里把面板构建一遍，
用来抓 Gradio 参数误用（组件关键字、choices 顺序、事件绑定等）。

这个脚本需要 gradio，所以不进 run_selftest.py；用 venv 的 python 单独跑：

    <venv>/python.exe tests/ui_smoke.py
"""

import importlib
import os
import sys
import types

_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")

for path in (_ROOT, _SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------- 桩
def _install_stubs():
    modules = types.ModuleType("modules")
    scripts_mod = types.ModuleType("modules.scripts")
    shared_mod = types.ModuleType("modules.shared")

    class Script:
        section = None
        create_group = True
        sorting_priority = 0

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = object()

    class OptionInfo:
        def __init__(self, default, label="", component=None, component_args=None,
                     onchange=None, *args, **kwargs):
            self.default = default
            self.label = label
            self.component = component
            self.component_args = component_args or {}

    shared_mod.OptionInfo = OptionInfo
    shared_mod.options_templates = {}

    def options_section(section, options):
        return dict(options)

    shared_mod.options_section = options_section

    class _State:
        interrupted = False
        skipped = False
        textinfo = ""

    class _Opts:
        pass

    shared_mod.opts = _Opts()
    shared_mod.state = _State()

    modules.scripts = scripts_mod
    modules.shared = shared_mod
    sys.modules["modules"] = modules
    sys.modules["modules.scripts"] = scripts_mod
    sys.modules["modules.shared"] = shared_mod


def main():
    _install_stubs()

    import gradio as gr

    module = importlib.import_module("zimage_te_pe")
    from lib_ztpe import config, presets

    print(f"  扩展模块导入 OK，版本 {module.VERSION}")

    script = module.ZImageTEPromptEnhancer()
    assert script.title() == module.EXT_NAME
    assert script.show(False) is not None

    built = {}
    for is_img2img in (False, True):
        with gr.Blocks():
            controls = script.ui(is_img2img)
        built[is_img2img] = len(controls)
        assert len(controls) == len(config.FIELDS), (
            f"ui() 返回 {len(controls)} 个组件，config.FIELDS 有 {len(config.FIELDS)} 项"
        )

    print(f"  txt2img / img2img 面板构建 OK，各 {built[False]} 个控件")

    # 注册进去的设置项应覆盖 config.FIELDS 与几个仅 opts 的键
    templates = sys.modules["modules.shared"].options_templates
    registered = set(templates)
    missing = {f"ztpe_{name}" for name in config.FIELDS} - registered
    assert not missing, f"缺少设置项：{sorted(missing)}"
    for extra in ("ztpe_time_budget", "ztpe_force_safe", "ztpe_allow_gguf_dequant",
                  "ztpe_allow_base_model", "ztpe_allow_cpu"):
        assert extra in registered, f"缺少设置项：{extra}"

    # 镜像 modules/ui_settings.py:39-46 的类型分派。默认值是 float 又没指定
    # component 时，create_setting_component 会直接抛 "bad options item type"，
    # 整个设置页构建失败、WebUI 起不来 —— 这个坑必须在测试里挡住。
    for key, info in templates.items():
        if info.component is not None:
            continue
        kind = type(info.default)
        assert kind in (str, int, bool), (
            f"{key} 的默认值是 {kind.__name__} 且没有指定 component；"
            "create_setting_component 只接受 str/int/bool"
        )
    print(f"  设置项注册与类型 OK，共 {len(registered)} 项")

    # 从空参数构造的 cfg 应落在预期默认值上
    cfg = config.from_args([], sys.modules["modules.shared"].opts)
    assert cfg.max_new == 160 and cfg.temperature == 0.7 and cfg.top_p == 0.8
    assert cfg.preset in presets.PRESETS
    print("  from_args 默认值 OK")

    # 预置模板应能在两个语种下取到
    for lang in ("auto", "en", "zh"):
        text = module._template_for(presets.DEFAULT_PRESET, lang)
        assert "{prompt}" in text, lang
    print("  模板预置 OK")

    print("UI 自检全部通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
