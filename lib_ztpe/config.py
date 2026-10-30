# -*- coding: utf-8 -*-
"""运行参数：把 gradio 组件值强制转换成可信的 GenCfg。

纯 Python，不依赖 gradio / torch。gradio 的 Number/Slider 可能给出 None、
空字符串、浮点甚至非法文本，这里统一兜底，避免旧版 `int(seed)` 直接抛
TypeError 导致整个 process() 被跳过。
"""

from dataclasses import dataclass, fields

#: 与 scripts/zimage_te_pe.py 中 ui() 的返回值顺序必须逐项一致。
FIELDS = (
    "enabled",
    "preset",
    "template",
    "language",
    "chat_layout",
    "max_new",
    "temperature",
    "top_p",
    "rep_penalty",
    "seed",
    "batch_mode",
    "hires_mode",
    "use_cache",
    "strip_meta",
    "write_original",
    "compute_dtype",
)


@dataclass
class GenCfg:
    enabled: bool = False
    preset: str = "official"
    template: str = ""
    language: str = "auto"
    chat_layout: str = "official"
    max_new: int = 160
    temperature: float = 0.7
    top_p: float = 0.8
    rep_penalty: float = 1.05
    seed: int = -1
    batch_mode: str = "single"
    hires_mode: str = "follow-main"
    use_cache: bool = True
    strip_meta: bool = True
    write_original: bool = True
    compute_dtype: str = "auto"

    #: 以下不来自 UI，只来自 shared.opts。
    time_budget: float = 45.0
    force_safe: bool = False
    allow_gguf_dequant: bool = False
    allow_base_model: bool = False
    allow_cpu: bool = False

    def template_text(self, presets, lang):
        """优先用用户编辑过的模板，否则按预设 + 语种取。

        面板会预填某个预设的内容；若把它无条件当成覆盖，auto 语言下就永远用
        预填的那份（通常是中文），英文输入会先出中文、判语种不符、再重试一次。
        所以只有"不等于任何内置文本"时才视为用户覆盖。
        """
        override = (self.template or "").strip()
        if override and not presets.is_builtin(override):
            return override
        return presets.for_language(self.preset, lang)


def to_int(value, default):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def to_str(value, default):
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _clamp(value, low, high):
    return max(low, min(high, value))


def _opt(opts, name, default):
    """shared.opts 里读一个值；读不到就用默认值。

    已确认 Options.__getattr__ 在 data 未命中时会回落到 OptionInfo.default，
    所以导入期注册的项可以直接读；这里仍然兜一层，兼容更老的 Forge。
    """
    if opts is None:
        return default
    return getattr(opts, name, default)


def from_args(args, opts=None):
    """把 UI 返回的参数列表与 shared.opts 合成一个 GenCfg。"""
    cfg = GenCfg()
    args = tuple(args or ())

    for index, name in enumerate(FIELDS):
        value = args[index] if index < len(args) else _opt(opts, f"ztpe_{name}", getattr(cfg, name))
        default = getattr(cfg, name)
        field_type = type(default)
        if field_type is bool:
            setattr(cfg, name, to_bool(value, default))
        elif field_type is int:
            setattr(cfg, name, to_int(value, default))
        elif field_type is float:
            setattr(cfg, name, to_float(value, default))
        else:
            setattr(cfg, name, to_str(value, default))

    cfg.time_budget = _clamp(to_float(_opt(opts, "ztpe_time_budget", 45.0), 45.0), 5.0, 600.0)
    cfg.force_safe = to_bool(_opt(opts, "ztpe_force_safe", False), False)
    cfg.allow_gguf_dequant = to_bool(_opt(opts, "ztpe_allow_gguf_dequant", False), False)
    cfg.allow_base_model = to_bool(_opt(opts, "ztpe_allow_base_model", False), False)
    cfg.allow_cpu = to_bool(_opt(opts, "ztpe_allow_cpu", False), False)

    cfg.max_new = _clamp(cfg.max_new, 32, 512)
    cfg.temperature = _clamp(cfg.temperature, 0.0, 1.5)
    cfg.top_p = _clamp(cfg.top_p, 0.1, 1.0)
    cfg.rep_penalty = _clamp(cfg.rep_penalty, 1.0, 1.4)

    if cfg.language not in ("auto", "en", "zh"):
        cfg.language = "auto"
    if cfg.chat_layout not in ("official", "user", "parity"):
        cfg.chat_layout = "official"
    if cfg.batch_mode not in ("single", "per-image"):
        cfg.batch_mode = "single"
    if cfg.hires_mode not in ("follow-main", "rewrite-separately", "leave"):
        cfg.hires_mode = "follow-main"
    if cfg.compute_dtype not in ("auto", "fp16", "bf16", "fp32"):
        cfg.compute_dtype = "auto"

    return cfg


def field_names():
    return tuple(f.name for f in fields(GenCfg))
