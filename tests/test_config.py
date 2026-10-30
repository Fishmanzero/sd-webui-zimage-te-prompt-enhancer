# -*- coding: utf-8 -*-
"""参数强制转换自检。重点是旧版 int(seed) 在空值上崩溃。"""

from lib_ztpe import config, presets


class _Opts:
    """模拟 shared.opts：只提供部分 key，其余走 getattr 默认值。"""

    ztpe_max_new = 128
    ztpe_temperature = 0.55
    ztpe_time_budget = 30.0
    ztpe_force_safe = True


def run(check):
    check("to_int(None) 回落", config.to_int(None, -1) == -1)
    check("to_int('') 回落", config.to_int("", -1) == -1)
    check("to_int('abc') 回落", config.to_int("abc", -1) == -1)
    check("to_int('12.7') 取整", config.to_int("12.7", -1) == 12)
    check("to_int(True) 为 1", config.to_int(True, -1) == 1)

    check("to_float(None) 回落", config.to_float(None, 0.5) == 0.5)
    check("to_float('x') 回落", config.to_float("x", 0.5) == 0.5)

    check("to_bool('true')", config.to_bool("true", False) is True)
    check("to_bool('off')", config.to_bool("off", True) is False)
    check("to_bool(None) 回落", config.to_bool(None, True) is True)

    cfg = config.from_args([], opts=None)
    check("空参数时用 dataclass 默认值", cfg.max_new == 160 and cfg.temperature == 0.7 and cfg.top_p == 0.8)
    check("默认语言 auto", cfg.language == "auto")
    check("默认布局 official", cfg.chat_layout == "official")
    check("默认 hires follow-main", cfg.hires_mode == "follow-main")

    cfg2 = config.from_args([], opts=_Opts())
    check("opts 补齐 max_new", cfg2.max_new == 128)
    check("opts 补齐 temperature", abs(cfg2.temperature - 0.55) < 1e-9)
    check("opts 补齐 time_budget", abs(cfg2.time_budget - 30.0) < 1e-9)
    check("opts 补齐 force_safe", cfg2.force_safe is True)

    args = [True, "light", "自定义模板", "en", "user", 0, -1.0, 3.0, 0.5, "", "per-image", "leave", False, False, False, "fp32"]
    cfg3 = config.from_args(args)
    check("UI 参数逐项落到字段", cfg3.enabled is True and cfg3.preset == "light" and cfg3.template == "自定义模板")
    check("max_new 下限夹紧", cfg3.max_new == 32)
    check("temperature 下限夹紧", cfg3.temperature == 0.0)
    check("top_p 上限夹紧", cfg3.top_p == 1.0)
    check("rep_penalty 下限夹紧", cfg3.rep_penalty == 1.0)
    check("空 seed 回落 -1", cfg3.seed == -1, "旧版 int('') 崩溃回归")
    check("非法语言回落 auto", config.from_args([False, "official", "", "klingon"]).language == "auto")
    check("非法布局回落 official", config.from_args([False, "official", "", "auto", "nope"]).chat_layout == "official")
    check("非法批量模式回落 single",
          config.from_args([False, "official", "", "auto", "official", 160, 0.7, 0.8, 1.05, -1, "nope"]).batch_mode == "single")

    check("FIELDS 与 GenCfg 字段一致",
          set(config.FIELDS) <= set(config.field_names()),
          f"{set(config.FIELDS) - set(config.field_names())}")

    check("template_text 优先用户模板",
          cfg3.template_text(_FakePresets(), "en") == "自定义模板")
    check("template_text 回落预设",
          config.GenCfg().template_text(_FakePresets(), "en") == "PRESET:official:en")

    # 回归：面板会把预设内容预填进模板框，若不识别"这就是内置文本"，
    # auto 语言下会永远用预填的那份（通常中文），英文输入先出中文再重试。
    prefilled = config.GenCfg(preset="official", template=presets.OFFICIAL_ZH)
    check("预填的内置模板按语种重新解析",
          prefilled.template_text(presets, "en") == presets.OFFICIAL_EN,
          "auto/en 下仍用中文模板")
    check("预填的英文模板也按语种解析",
          config.GenCfg(preset="official", template=presets.OFFICIAL_EN)
          .template_text(presets, "zh") == presets.OFFICIAL_ZH)

    edited = config.GenCfg(preset="official", template="我自己的模板 {prompt}")
    check("真正编辑过的模板优先",
          edited.template_text(presets, "en") == "我自己的模板 {prompt}")


class _FakePresets:
    @staticmethod
    def for_language(key, lang):
        return f"PRESET:{key}:{lang}"

    @staticmethod
    def is_builtin(text):
        return False
