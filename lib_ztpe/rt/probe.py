# -*- coding: utf-8 -*-
"""能力探测。

判定顺序刻意如此：结构 → 形状 → 导入面 → 词表投影 → tokenizer → stop ids
→ KV 功能探测。KV 探测最贵（约 1 秒、约 3 MB），放在最后，前面任何一步失败
都不必再花这个时间。
"""

from dataclasses import dataclass, field


@dataclass
class Caps:
    ok: bool = False
    kv: bool = False
    mode: str = "disabled"          # kv / safe / disabled
    projection: str = "unknown"
    quant: str = ""
    chat_template: bool = False
    think_kwarg: bool = False
    stop_ids: frozenset = frozenset()
    reasons: list = field(default_factory=list)
    notes: list = field(default_factory=list)


_LAYER_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "input_layernorm",
    "post_attention_layernorm",
)


def _dig(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _structural(chain):
    """返回问题列表；空列表表示结构可复用。"""
    problems = []
    cfg = chain.cfg

    if getattr(chain.lm, "norm", None) is None:
        problems.append("lm.norm 缺失（无法做最终归一化）")

    weight = getattr(getattr(chain.lm, "embed_tokens", None), "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        problems.append("embed_tokens.weight 形状异常")

    expected = int(getattr(cfg, "num_hidden_layers", 0))
    if expected and chain.n_layers != expected:
        problems.append(f"层数 {chain.n_layers} != 配置的 {expected}")

    if chain.n_layers == 0:
        problems.append("lm.layers 为空")
        return problems

    layer0 = chain.layers[0]
    for path in _LAYER_PATHS:
        if _dig(layer0, path) is None:
            problems.append(f"layer0.{path} 缺失")

    n_heads = int(getattr(cfg, "num_attention_heads", 0))
    n_kv = int(getattr(cfg, "num_key_value_heads", 0))
    head_dim = int(getattr(cfg, "head_dim", 0))

    if n_kv and n_heads and n_heads % n_kv != 0:
        problems.append(f"num_attention_heads {n_heads} 不能被 num_key_value_heads {n_kv} 整除")

    inner = getattr(_dig(layer0, "self_attn"), "inner_size", None)
    if inner is not None and head_dim and int(inner) != head_dim * n_heads:
        problems.append(f"self_attn.inner_size {inner} != head_dim*heads {head_dim * n_heads}")

    if weight is not None and getattr(weight, "ndim", 0) == 2:
        vocab = int(getattr(cfg, "vocab_size", 0))
        hidden = int(getattr(cfg, "hidden_size", 0))
        if vocab and hidden and tuple(weight.shape) != (vocab, hidden):
            problems.append(f"embed 形状 {tuple(weight.shape)} != ({vocab}, {hidden})")

    return problems


def _probe_think_kwarg(tokenizer):
    if not getattr(tokenizer, "chat_template", None):
        return False
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return True
    except TypeError:
        return False
    except Exception:
        return False


def _stop_ids(tokenizer):
    ids = set()

    for token, fallback in (("<|im_end|>", 151645), ("<|endoftext|>", 151643)):
        found = None
        try:
            value = tokenizer.convert_tokens_to_ids(token)
            if isinstance(value, int) and value >= 0:
                found = value
        except Exception:
            pass
        ids.add(found if found is not None else fallback)

    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0:
        ids.add(eos)
    elif isinstance(eos, (list, tuple)):
        ids.update(int(item) for item in eos if isinstance(item, int) and item >= 0)

    return frozenset(ids)


#: 能力探测要跑一次 4 token 的 prefill 才能确认 KV 可用，没必要每次出图都重跑。
#: 键里不含 projector（它每轮重建），但含 clip 与模型 id；换模型时由调用方 clear。
_CACHE = {}


def clear_cache():
    _CACHE.clear()


def probe(chain, cfg, fresh=False):
    """带缓存的能力探测。诊断面板请传 fresh=True 以拿到当下真实状态。"""
    key = (
        id(chain.sd),
        bool(cfg.force_safe),
        cfg.compute_dtype,
        bool(cfg.allow_gguf_dequant),
        bool(cfg.allow_cpu),
    )
    if not fresh:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

    caps = _probe(chain, cfg)
    _CACHE.clear()
    _CACHE[key] = caps
    return caps


def _probe(chain, cfg):
    from . import decode, memory

    caps = Caps()

    # ---------------------------------------------------------- 1. 结构
    problems = _structural(chain)
    caps.reasons.extend(problems)

    # ---------------------------------------------------------- 2. 导入面
    try:
        import backend.nn.llm.llama as llama_mod
        llama_mod.precompute_freqs_cis
        llama_mod.attention_function
    except Exception as exc:
        problems.append(f"backend.nn.llm.llama 导入失败：{exc}")
        caps.reasons.append(problems[-1])

    # ---------------------------------------------------------- 3. 词表投影
    try:
        caps.projection = chain.projector.refresh(cfg.allow_gguf_dequant)
        if caps.projection == "gguf":
            caps.quant = "gguf"
        elif caps.projection.startswith("tied/"):
            caps.quant = "fp8"
    except Exception as exc:
        caps.projection = "unsupported"
        caps.reasons.append(f"词表投影不可用：{exc}")

    # ---------------------------------------------------------- 4. tokenizer
    caps.chat_template = bool(getattr(chain.tokenizer, "chat_template", None))
    caps.think_kwarg = _probe_think_kwarg(chain.tokenizer)
    caps.stop_ids = _stop_ids(chain.tokenizer)

    if not caps.chat_template:
        if cfg.allow_base_model:
            caps.notes.append(
                "tokenizer 没有 chat_template，按设置忽略该告警；仍使用内置 ChatML 框法。"
            )
        else:
            caps.notes.append(
                "tokenizer 没有 chat_template，改用内置 ChatML 框法；"
                "该文本编码器可能是 base 变体，改写质量可能明显下降。"
            )

    # ---------------------------------------------------------- 5. 设备
    cpu_only = memory.is_cpu(chain.device)
    if cpu_only:
        if cfg.allow_cpu:
            caps.notes.append("文本编码器在 CPU 上，解码约 1 token/s，已把上限压到 32。")
        else:
            caps.reasons.append(
                "文本编码器在 CPU 上（约 1 token/s）。如确认要用，请在设置里打开「允许 CPU 解码」。"
            )

    # ---------------------------------------------------------- 6. KV 功能探测
    blocking = bool(problems) or caps.projection == "unsupported" or (cpu_only and not cfg.allow_cpu)

    if cfg.force_safe:
        caps.kv = False
        caps.notes.append("已按设置强制走安全路径（不使用 KV cache）。")
    elif blocking:
        caps.kv = False
    else:
        ok, detail = decode.kv_selftest(chain)
        caps.kv = ok
        if ok:
            caps.notes.append(f"KV 探测通过：{detail}")
        else:
            caps.reasons.append(f"KV 不可用，已降级安全模式：{detail}")

    caps.ok = not blocking
    caps.mode = "disabled" if blocking else ("kv" if caps.kv else "safe")

    if caps.mode == "safe":
        caps.notes.append("安全模式下每个 token 都要重跑全量前向，速度约为快路径的 1/20。")

    return caps
