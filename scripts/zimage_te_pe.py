# -*- coding: utf-8 -*-
"""Z-Image TE Prompt Enhancer for SD WebUI Forge Neo.

在生成图片之前，复用 Forge 已经加载在显存里的 Z-Image 文本编码器（Qwen3-4B）
改写提示词，不额外加载任何 LLM。

面板位置：txt2img / img2img 页面下方独立分区「Z-Image TE Prompt Enhancer」。

模块级只导入纯 Python 层；torch 与 backend 一律在函数内惰性导入，
避免任何导入异常导致整个脚本被 Forge 静默丢弃。
"""

import functools
import os
import sys
import time
import traceback

import gradio as gr

_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

from lib_ztpe import cache as cache_mod      # noqa: E402
from lib_ztpe import config, diag, presets, textproc   # noqa: E402

from modules import scripts, shared          # noqa: E402

EXT_NAME = "Z-Image TE Prompt Enhancer"
VERSION = "2.0.0"

print(f"[ZTPE] loaded v{VERSION} ({os.path.basename(__file__)})")

_CACHE = cache_mod.LRUCache(maxsize=64)


def _rt():
    """惰性导入 torch 层。"""
    from lib_ztpe.rt import decode, locate, memory, probe
    return locate, memory, probe, decode


def _gpu_lock():
    """复用 Forge 自己的队列锁。

    预览 / 诊断 / KV 校验都会调 `load_model_gpu`，如果在出图途中执行，可能把
    正在采样的扩散模型挤出显存。Forge 的生成按钮就是在 `queue_lock` 里跑的，
    这里跟随同一把锁；拿不到就退化成空上下文。
    """
    try:
        from modules import call_queue
        return call_queue.queue_lock
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _with_gpu_lock(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _gpu_lock():
            return fn(*args, **kwargs)
    return wrapper


def _prepare(sd, cfg):
    """解析链路 → 把 TE 调到计算设备 → 刷新 device/dtype。

    第三步不能省：Forge 会在两次生成之间把文本编码器卸到 CPU（CPU 副本是
    fp32），`resolve()` 读到的 `device`/`storage_dtype` 是"已卸下"的陈旧值，
    不刷新就会误报"文本编码器在 CPU 上"，把自己禁用掉。
    """
    locate, memory, _probe, _decode = _rt()
    chain = locate.resolve(sd, cfg.compute_dtype)
    memory.ensure_resident(sd)
    chain.sync()
    return chain


# ---------------------------------------------------------------- 设置项

def _register_settings():
    try:
        shared.options_templates.update(shared.options_section(
            ("zimage_te_pe", "Z-Image TE Prompt Enhancer"),
            {
                "ztpe_enabled": shared.OptionInfo(False, "默认启用改写"),
                "ztpe_preset": shared.OptionInfo(presets.DEFAULT_PRESET, "改写模板", gr.Dropdown,
                                                 lambda: {"choices": presets.names()}),
                "ztpe_template": shared.OptionInfo("", "模板覆盖（留空则用预设）"),
                "ztpe_language": shared.OptionInfo("auto", "输出语言", gr.Radio,
                                                   {"choices": ["auto", "en", "zh"]}),
                "ztpe_chat_layout": shared.OptionInfo("official", "Chat 布局", gr.Radio,
                                                      {"choices": ["official", "user", "parity"]}),
                "ztpe_max_new": shared.OptionInfo(160, "最大生成 token", gr.Slider,
                                                  {"minimum": 32, "maximum": 512, "step": 8}),
                "ztpe_temperature": shared.OptionInfo(0.7, "temperature（官方 0.7）", gr.Slider,
                                                      {"minimum": 0.0, "maximum": 1.5, "step": 0.05}),
                "ztpe_top_p": shared.OptionInfo(0.8, "top_p（官方 0.8）", gr.Slider,
                                                {"minimum": 0.1, "maximum": 1.0, "step": 0.05}),
                "ztpe_rep_penalty": shared.OptionInfo(1.05, "重复惩罚（1.0 关闭）", gr.Slider,
                                                      {"minimum": 1.0, "maximum": 1.4, "step": 0.01}),
                "ztpe_seed": shared.OptionInfo(-1, "改写种子（-1 跟随图片种子）"),
                "ztpe_batch_mode": shared.OptionInfo("single", "批量行为", gr.Radio,
                                                     {"choices": ["single", "per-image"]}),
                "ztpe_hires_mode": shared.OptionInfo("follow-main", "hires.fix 提示词", gr.Radio,
                                                     {"choices": ["follow-main", "rewrite-separately", "leave"]}),
                "ztpe_use_cache": shared.OptionInfo(True, "缓存相同输入的改写结果"),
                "ztpe_strip_meta": shared.OptionInfo(True, "剔除 8K/杰作类 meta 标签"),
                "ztpe_write_original": shared.OptionInfo(True, "把原始提示词写入 infotext"),
                "ztpe_compute_dtype": shared.OptionInfo("auto", "计算精度", gr.Radio,
                                                        {"choices": ["auto", "fp16", "bf16", "fp32"]}),
                # 注意：create_setting_component 只认 str/int/bool 的默认值，
                # float 默认值必须显式给 component，否则整个设置页构建会抛
                # "bad options item type" 并让 WebUI 起不来。
                "ztpe_time_budget": shared.OptionInfo(45.0, "单条改写时间预算（秒）", gr.Slider,
                                                      {"minimum": 5, "maximum": 300, "step": 5}),
                "ztpe_force_safe": shared.OptionInfo(False, "强制安全模式（不用 KV cache）"),
                "ztpe_allow_gguf_dequant": shared.OptionInfo(False, "允许 GGUF 一次性反量化（+约 0.8 GB）"),
                "ztpe_allow_base_model": shared.OptionInfo(False, "疑似 base 模型时不再告警"),
                "ztpe_allow_cpu": shared.OptionInfo(False, "允许 CPU 解码（约 1 token/s）"),
            },
        ))
    except Exception:
        print("[ZTPE] 设置项注册失败（不影响使用）")
        traceback.print_exc()


_register_settings()


# ---------------------------------------------------------------- 小工具

def _opt(name, default):
    return getattr(shared.opts, name, default)


def _template_for(preset_key, language):
    lang = "zh" if language in (None, "", "auto") else language
    return presets.for_language(preset_key, lang)


def _snapshot(p):
    return {
        "prompt": getattr(p, "prompt", None),
        "all_prompts": getattr(p, "all_prompts", None),
        "main_prompt": getattr(p, "main_prompt", None),
        "hr_prompt": getattr(p, "hr_prompt", None),
        "all_hr_prompts": getattr(p, "all_hr_prompts", None),
    }


def _restore(p, snapshot):
    for name, value in snapshot.items():
        try:
            setattr(p, name, value)
        except Exception:
            pass


def _param(p, key, value):
    try:
        p.extra_generation_params[key] = value
    except Exception:
        pass


def _chain_id(sd):
    """缓存身份。刻意不含 patcher 的 id：LoRA 每轮都会 clone 出新 patcher，
    带进去会让结果缓存永远不命中。"""
    info = getattr(sd, "sd_checkpoint_info", None)
    title = getattr(info, "title", "") or getattr(info, "name", "") or ""
    short_hash = getattr(sd, "sd_model_hash", "") or ""
    return (id(sd), title, short_hash)


def _seed_for(cfg, seeds, index):
    if cfg.seed >= 0:
        return cfg.seed + index
    if seeds:
        return int(seeds[min(index, len(seeds) - 1)])
    return 1234


def _report(chain, caps, residency):
    from lib_ztpe.rt import memory as mem

    free = mem.free_bytes(chain.device)
    checks = [
        ("ok" if caps.kv else ("warn" if caps.ok else "fail"), "KV cache",
         "可用（增量解码）" if caps.kv else "不可用"),
        ("ok" if caps.chat_template else "warn", "chat template",
         "存在" if caps.chat_template else "缺失，改用内置 ChatML"),
    ]

    rows = [
        ("engine", chain.describe()),
        ("device", f"{chain.device}   storage {str(chain.storage_dtype).replace('torch.', '')}"
                   f"   compute {str(chain.compute_dtype).replace('torch.', '')}"),
        ("projection", caps.projection + (f" ({caps.quant})" if caps.quant else "")),
        ("stop ids", ", ".join(str(x) for x in sorted(caps.stop_ids))),
        ("tokenizer", f"{type(chain.tokenizer).__name__}"
                      f"   enable_thinking: {'yes' if caps.think_kwarg else 'no'}"),
        ("residency", f"就位 {residency:.2f}s   free {diag.humanize_bytes(free)}"),
    ]

    notes = list(caps.notes)
    if str(chain.storage_dtype) == "torch.float32":
        notes.append(
            "TE 以 fp32 存放（4B 约 16 GB）。Forge 默认应为 fp16 —— 若启动日志里的 "
            "「Qwen3 Data Type」显示 float16，说明是加载后被上游转成 fp32 的，"
            "加 --fp16-text-enc 也无效；只有显示 float32 时才需要检查命令行参数。"
        )

    if caps.mode == "disabled":
        return diag.render({
            "verdict": "DISABLED",
            "headline": caps.reasons[0] if caps.reasons else "未知原因",
            "rows": rows,
            "checks": checks,
            "notes": notes,
            "fix": caps.reasons[1:] or ["确认当前 checkpoint 是 Z-Image 且 qwen_3_4b 已加载。"],
        })

    verdict = "FAST PATH (KV cache)" if caps.kv else "SAFE MODE (no KV cache) — 约慢 20 倍"
    if not caps.chat_template:
        verdict += " / LOW QUALITY (no chat template)"

    return diag.render({
        "verdict": verdict,
        "rows": rows,
        "checks": checks,
        "notes": notes,
        # 安全模式下 caps.ok 仍为 True，降级原因只记在 reasons 里；
        # 不传出去的话用户就看不到为什么没用上 KV。
        "fix": caps.reasons or None,
    })


# ---------------------------------------------------------------- Script

class ZImageTEPromptEnhancer(scripts.Script):
    section = "zimage_te_pe"
    create_group = True
    sorting_priority = 100

    def __init__(self):
        self._last_rewritten = None
        self._last_chain_id = None

    def title(self):
        return EXT_NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ------------------------------------------------------------ UI
    def ui(self, is_img2img):
        tab = "img2img" if is_img2img else "txt2img"

        with gr.Accordion(EXT_NAME, open=False, elem_id=f"ztpe_{tab}_accordion"):
            gr.Markdown(
                "复用当前已加载的 **Qwen3-4B 文本编码器**改写提示词，不额外加载 LLM。"
                "首次使用请先点「运行诊断」。"
            )

            with gr.Row():
                enabled = gr.Checkbox(label="本次启用", value=_opt("ztpe_enabled", False))
                status = gr.Markdown(value="", elem_id=f"ztpe_{tab}_status")

            with gr.Row():
                preset = gr.Dropdown(
                    label="改写模板",
                    choices=[(presets.label(key), key) for key in presets.names()],
                    value=_opt("ztpe_preset", presets.DEFAULT_PRESET),
                )
                language = gr.Radio(
                    label="输出语言",
                    choices=[("auto（跟随输入）", "auto"), ("英文", "en"), ("中文", "zh")],
                    value=_opt("ztpe_language", "auto"),
                )

            template_box = gr.Textbox(
                label="模板内容（{prompt} 为占位符）",
                value=_opt("ztpe_template", "") or _template_for(_opt("ztpe_preset", presets.DEFAULT_PRESET),
                                                                 _opt("ztpe_language", "auto")),
                lines=10,
                max_lines=28,
            )

            with gr.Row():
                reset_btn = gr.Button("重置为预设", size="sm")
                save_btn = gr.Button("保存为默认", size="sm")

            with gr.Accordion("高级", open=False):
                with gr.Row():
                    max_new = gr.Slider(label="最大生成 token", minimum=32, maximum=512, step=8,
                                        value=_opt("ztpe_max_new", 160))
                    temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.5, step=0.05,
                                            value=_opt("ztpe_temperature", 0.7))
                    top_p = gr.Slider(label="top_p", minimum=0.1, maximum=1.0, step=0.05,
                                      value=_opt("ztpe_top_p", 0.8))
                    rep_penalty = gr.Slider(label="重复惩罚", minimum=1.0, maximum=1.4, step=0.01,
                                            value=_opt("ztpe_rep_penalty", 1.05))

                with gr.Row():
                    seed = gr.Number(label="改写种子（-1 跟随图片种子）", value=_opt("ztpe_seed", -1), precision=0)
                    batch_mode = gr.Radio(label="批量行为",
                                          choices=[("整批共用一次", "single"), ("每张单独改写", "per-image")],
                                          value=_opt("ztpe_batch_mode", "single"))
                    hires_mode = gr.Radio(label="hires.fix 提示词",
                                          choices=[("跟随主提示词", "follow-main"),
                                                   ("单独改写", "rewrite-separately"),
                                                   ("保持原样", "leave")],
                                          value=_opt("ztpe_hires_mode", "follow-main"))

                with gr.Row():
                    chat_layout = gr.Radio(label="Chat 布局",
                                           choices=[("官方（模板进 system）", "official"),
                                                    ("模板进 user", "user"),
                                                    ("复刻编码路径", "parity")],
                                           value=_opt("ztpe_chat_layout", "official"))
                    compute_dtype = gr.Radio(label="计算精度",
                                             choices=["auto", "fp16", "bf16", "fp32"],
                                             value=_opt("ztpe_compute_dtype", "auto"))

                with gr.Row():
                    use_cache = gr.Checkbox(label="缓存相同输入", value=_opt("ztpe_use_cache", True))
                    strip_meta = gr.Checkbox(label="剔除 meta 标签", value=_opt("ztpe_strip_meta", True))
                    write_info = gr.Checkbox(label="原文写入 infotext", value=_opt("ztpe_write_original", True))

            with gr.Accordion("预览与诊断", open=False):
                preview_in = gr.Textbox(label="测试用提示词", lines=2,
                                        placeholder="仅用于试跑改写效果，不影响实际生成")
                with gr.Row():
                    load_btn = gr.Button("载入当前提示词", size="sm")
                    preview_btn = gr.Button("预览改写", variant="primary", size="sm")
                    diag_btn = gr.Button("运行诊断", size="sm")
                    kv_btn = gr.Button("KV 一致性校验", size="sm")
                    probe_btn = gr.Button("指令跟随探测", size="sm")

                preview_out = gr.Textbox(label="输出", lines=6, interactive=False, show_copy_button=True)
                diag_out = gr.Textbox(label="诊断报告", lines=12, interactive=False, show_copy_button=True)

            controls = [
                enabled, preset, template_box, language, chat_layout,
                max_new, temperature, top_p, rep_penalty, seed,
                batch_mode, hires_mode, use_cache, strip_meta, write_info, compute_dtype,
            ]

            # ------------------------------------------------ 事件
            def _on_preset(name, lang):
                # 显式换预设 / 点重置：直接覆盖，这是用户的明确意图。
                return gr.update(value=_template_for(name, lang))

            def _on_language(name, lang, current):
                # 切语言时只在模板框仍是内置文本时才覆盖，否则会吃掉用户的手改。
                if current and not presets.is_builtin(current):
                    return gr.update()
                return gr.update(value=_template_for(name, lang))

            preset.change(_on_preset, inputs=[preset, language], outputs=[template_box],
                          show_progress=False)
            language.change(_on_language, inputs=[preset, language, template_box],
                            outputs=[template_box], show_progress=False)

            reset_btn.click(_on_preset, inputs=[preset, language], outputs=[template_box],
                            show_progress=False)

            def _save_default(text):
                try:
                    shared.opts.ztpe_template = text or ""
                    # 只赋值不会落盘 —— 必须显式 save，否则重启就丢。
                    shared.opts.save(shared.config_filename)
                    _CACHE.clear()
                    return "已保存为默认模板（已写入配置文件）。"
                except Exception as exc:
                    return f"保存失败：{exc}"

            save_btn.click(_save_default, inputs=[template_box], outputs=[status],
                           show_progress=False)

            load_btn.click(
                fn=lambda current: current,
                inputs=[preview_in],
                outputs=[preview_in],
                js=f"(x) => [ztpe_get_prompt('{tab}')]",
                show_progress=False,
            )

            @_with_gpu_lock
            def _preview(prompt_text, *args):
                text = (prompt_text or "").strip()
                if not text:
                    return "测试用提示词为空。先点「载入当前提示词」或手动输入。"
                args = list(args)
                if args:
                    args[0] = True
                cfg = config.from_args(args, shared.opts)
                try:
                    _, _, probe, decode = _rt()
                    chain = _prepare(shared.sd_model, cfg)
                    caps = probe.probe(chain, cfg, fresh=True)
                    if not caps.ok:
                        return f"[不可用] {caps.reasons[0] if caps.reasons else '未知原因'}"
                    lang = textproc.resolve_lang(cfg.language, text)
                    template = cfg.template_text(presets, lang)
                    result = decode.generate(chain, caps, cfg, text, template,
                                             seed=_seed_for(cfg, [], 0))
                    if not result.ok:
                        detail = (result.meta or {}).get("traceback", "")
                        suffix = f"\n\n{detail}" if detail else ""
                        return f"{text}\n\n[提示] {result.error}{suffix}"
                    meta = result.meta
                    status = (f"[{meta.get('mode')}] {meta.get('tokens')} tokens / "
                              f"{meta.get('seconds', 0):.1f}s "
                              f"({meta.get('speed', 0):.1f} tok/s)  "
                              f"lang={meta.get('lang')}  finish={meta.get('finish')}")
                    if meta.get("truncated"):
                        status += "  注意：结果被截断（打满 max_new 或超时），可提高「最大生成 token」"
                    return f"{result.text}\n\n{status}"
                except Exception:
                    return "预览失败：\n" + traceback.format_exc()

            preview_btn.click(_preview, inputs=[preview_in] + controls, outputs=[preview_out])

            @_with_gpu_lock
            def _diagnose(*args):
                cfg = config.from_args(args, shared.opts)
                try:
                    locate, memory, probe, _decode = _rt()
                except Exception:
                    return "无法导入运行时模块：\n" + traceback.format_exc()

                try:
                    chain = locate.resolve(shared.sd_model, cfg.compute_dtype)
                except locate.ResolveError as exc:
                    return diag.disabled(exc.message,
                                         fix=["加载 Z-Image / Z-Image-Turbo 后再试。"])
                except Exception:
                    return "解析链路失败：\n" + traceback.format_exc()

                started = time.time()
                try:
                    memory.ensure_resident(shared.sd_model)
                except Exception as exc:
                    return f"文本编码器无法就位：{exc}"
                residency = time.time() - started

                # 搬到计算设备之后必须重读：此前拿到的是"已卸到 CPU"的陈旧值。
                chain.sync()

                try:
                    caps = probe.probe(chain, cfg, fresh=True)
                except Exception:
                    return "能力探测失败：\n" + traceback.format_exc()

                return _report(chain, caps, residency)

            diag_btn.click(_diagnose, inputs=controls, outputs=[diag_out])

            @_with_gpu_lock
            def _consistency(*args):
                cfg = config.from_args(args, shared.opts)
                try:
                    _, _, probe, decode = _rt()
                    chain = _prepare(shared.sd_model, cfg)
                    caps = probe.probe(chain, cfg, fresh=True)
                    if not caps.kv:
                        return "KV 不可用，无法做一致性校验。先看「运行诊断」。"
                    started = time.time()
                    delta_max, delta_mean, length = decode.consistency_check(chain)
                    elapsed = time.time() - started
                    verdict = "一致" if delta_max < 5e-3 else "偏差偏大"
                    return (f"{verdict}\n"
                            f"  探针长度   {length} tokens\n"
                            f"  max|Δlogit|  {delta_max:.3e}\n"
                            f"  mean|Δlogit| {delta_mean:.3e}\n"
                            f"  耗时       {elapsed:.2f}s\n"
                            f"  阈值       5e-3")
                except Exception:
                    return "一致性校验失败：\n" + traceback.format_exc()

            kv_btn.click(_consistency, inputs=controls, outputs=[diag_out])

            @_with_gpu_lock
            def _instruct(*args):
                cfg = config.from_args(args, shared.opts)
                try:
                    _, _, probe, decode = _rt()
                    chain = _prepare(shared.sd_model, cfg)
                    caps = probe.probe(chain, cfg, fresh=True)
                    if not caps.ok:
                        return f"[不可用] {caps.reasons[0] if caps.reasons else '未知原因'}"
                    text = decode.greedy_probe(
                        chain, caps, "ZEBRA",
                        "Repeat the following word exactly, output only the word: {prompt}",
                        max_new=12,
                    )
                    ok = "ZEBRA" in (text or "").upper()
                    head = "指令跟随正常" if ok else "指令跟随可疑（可能是 base 模型）"
                    return f"{head}\n  输出: {text!r}"
                except Exception:
                    return "指令探测失败：\n" + traceback.format_exc()

            probe_btn.click(_instruct, inputs=controls, outputs=[diag_out])

        return controls

    # ------------------------------------------------------------ 生成钩子
    def process(self, p, *args):
        cfg = config.from_args(args, shared.opts)
        if not cfg.enabled:
            self._last_rewritten = None
            return

        # 先清空，避免上一轮的改写结果残留到 before_process_batch 的守卫里，
        # 造成"被其它脚本覆盖"的假告警。
        self._last_rewritten = None
        snapshot = _snapshot(p)
        try:
            self._run(p, cfg)
        except Exception as exc:
            _restore(p, snapshot)
            self._last_rewritten = None
            if type(exc).__name__ == "ResolveError":
                # 不是 Z-Image 属于正常跳过，不该每次出图都刷一整段 traceback。
                print(f"[ZTPE] 已跳过：{exc}")
                return
            _param(p, "ZTPE error", f"{type(exc).__name__}: {exc}")
            print(f"[ZTPE] 改写已跳过：{type(exc).__name__}: {exc}")
            traceback.print_exc()

    def _run(self, p, cfg):
        _, _, probe, _ = _rt()

        chain = _prepare(shared.sd_model, cfg)

        # 换模型时让能力探测缓存失效（chain_id 含 checkpoint 标题/哈希与 patcher）。
        chain_id = _chain_id(shared.sd_model)
        if chain_id != self._last_chain_id:
            probe.clear_cache()
            self._last_chain_id = chain_id

        caps = probe.probe(chain, cfg)
        if not caps.ok:
            print(f"[ZTPE] 已跳过：{caps.reasons[0] if caps.reasons else '未知原因'}")
            return

        if not caps.chat_template and not cfg.allow_base_model:
            print("[ZTPE] 文本编码器没有 chat template，可能是 base 模型；"
                  "改写质量可能明显下降。可在设置里关闭该告警。")

        originals = list(getattr(p, "all_prompts", None) or [])
        if not originals:
            return

        seeds = list(getattr(p, "all_seeds", None) or [])
        errors = []

        if cfg.batch_mode == "per-image":
            rewritten = []
            for index, original in enumerate(originals):
                if getattr(shared.state, "interrupted", False):
                    rewritten.extend(originals[index:])
                    break
                text, error = self._one(chain, caps, cfg, original,
                                        _seed_for(cfg, seeds, index), chain_id)
                rewritten.append(text)
                if error:
                    errors.append(error)
        else:
            # 整批共用：按"去重后的提示词"各改写一次，再映射回原顺序。
            # 不能只改写 originals[0] —— 那会把 prompt matrix / 通配符产生的
            # 其它提示词整条丢掉（README 又建议把本插件排在 dynamic-prompts 之后，
            # 两者叠加等于默认配置下静默吞掉所有变化）。
            first_index = {}
            for index, original in enumerate(originals):
                first_index.setdefault(original, index)

            mapping = {}
            for original, index in first_index.items():
                if getattr(shared.state, "interrupted", False):
                    break
                text, error = self._one(chain, caps, cfg, original,
                                        _seed_for(cfg, seeds, index), chain_id)
                mapping[original] = text
                if error:
                    errors.append(error)

            rewritten = [mapping.get(original, original) for original in originals]

        if len(rewritten) != len(originals):
            raise RuntimeError("改写结果数量与提示词数量不一致")

        p.all_prompts = rewritten
        p.prompt = rewritten[0]
        p.main_prompt = rewritten[0]

        if getattr(p, "enable_hr", False):
            self._sync_hires(p, chain, caps, cfg, originals, rewritten, seeds, chain_id)

        _param(p, "ZTPE", (
            f"Qwen3-4B TE v{VERSION}; preset={cfg.preset}; T={cfg.temperature:g}; "
            f"top_p={cfg.top_p:g}; max={cfg.max_new}; lang={cfg.language}; mode={caps.mode}"
        ))
        if cfg.write_original:
            _param(p, "ZTPE original", originals[0].replace("\n", " "))

        if errors:
            message = "; ".join(dict.fromkeys(errors))
            _param(p, "ZTPE warning", message)
            print(f"[ZTPE] {message}")

        self._last_rewritten = list(rewritten)

    def _one(self, chain, caps, cfg, prompt, seed, chain_id):
        """返回 (文本, 错误)。失败时文本为原文。"""
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt, ""

        lang = textproc.resolve_lang(cfg.language, prompt)
        template = cfg.template_text(presets, lang)

        key = None
        if cfg.use_cache:
            key = cache_mod.make_key(
                chain_id, prompt, template, lang, seed,
                cfg.max_new, round(cfg.temperature, 3), round(cfg.top_p, 3),
                round(cfg.rep_penalty, 3), cfg.compute_dtype,
                # chat_layout 与 strip_meta 都会改变最终文本，必须进 key，
                # 否则切换布局做 A/B 时会拿到上一次的缓存。
                cfg.chat_layout, bool(cfg.strip_meta),
            )
            hit = _CACHE.get(key)
            if not cache_mod.is_miss(hit):
                return hit, ""

        _, _, _, decode = _rt()
        result = decode.generate(chain, caps, cfg, prompt, template, seed=seed)

        if result.ok and (result.meta or {}).get("truncated"):
            print(f"[ZTPE] 改写被截断（finish={(result.meta or {}).get('finish')}），"
                  f"结果可能不完整；可提高「最大生成 token」或时间预算。")

        if not result.ok:
            return prompt, result.error

        if cfg.use_cache and key is not None:
            _CACHE.put(key, result.text)

        return result.text, ""

    def _sync_hires(self, p, chain, caps, cfg, originals, rewritten, seeds, chain_id):
        if cfg.hires_mode == "leave":
            return

        current = getattr(p, "all_hr_prompts", None)
        if not isinstance(current, list) or len(current) != len(originals):
            return

        if cfg.hires_mode == "follow-main":
            p.all_hr_prompts = [
                rewritten[i] if current[i] == originals[i] else current[i]
                for i in range(len(originals))
            ]
        else:
            updated = []
            for index, one in enumerate(current):
                text, _error = self._one(chain, caps, cfg, one,
                                         _seed_for(cfg, seeds, index) + 7919, chain_id)
                updated.append(text)
            p.all_hr_prompts = updated

        hr_prompt = getattr(p, "hr_prompt", None)
        if isinstance(hr_prompt, list):
            p.hr_prompt = list(p.all_hr_prompts)
        elif isinstance(hr_prompt, str) and hr_prompt:
            p.hr_prompt = p.all_hr_prompts[0]

    def before_process_batch(self, p, *args, **kwargs):
        """一致性守卫：改写结果若被后续脚本覆写，让现象可见而不是静默失效。"""
        expected = self._last_rewritten
        if not expected:
            return

        current = list(getattr(p, "prompts", None) or [])
        if not current:
            return

        batch = int(kwargs.get("batch_number", 0) or 0)
        size = len(current)
        window = expected[batch * size:(batch + 1) * size]

        if window and current != window:
            print("[ZTPE] 改写结果被其它脚本覆盖。"
                  "请到 Settings → Scripts 把本扩展排到 dynamic-prompts / wildcards 之后。")
            self._last_rewritten = None
