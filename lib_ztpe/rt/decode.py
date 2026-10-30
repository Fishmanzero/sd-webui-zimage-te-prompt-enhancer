# -*- coding: utf-8 -*-
"""生成后端：KV-cache 增量解码（快路径）+ 全量前向（安全路径）。

为什么必须自己写层循环
----------------------
`Llama2_.forward` 在 backend/nn/llm/llama.py:518 把 `past_key_value` 硬编码成
None，并且只返回 `(x, intermediate)` 二元组 —— 它的 `**kwargs` 会把
`past_key_values=` 悄悄吞掉。所以复用它的 forward 就拿不到增量解码。

但底层是完整支持的：`Attention.forward`（:279-324）接受 `(k, v, write_index)`
三元组，容量足够时**原地写入**并把 k/v 切片成有效长度送 attention，
`TransformerBlock.forward`（:351-377）原样透传。于是这里绕开 `Llama2_.forward`，
自己跑 36 层循环。

三个不能省的细节
----------------
1. `lm.norm` 只能做一次。`Llama2_.forward` 内部已经做过，绕开它之后由我们负责。
   旧版在 forward 之外又 norm 一次，logits 直接被毁。
2. `freqs_cis` 必须用**绝对**位置。`Llama2_.forward` 里 `arange(0, seq_len)`
   只对全量前向成立，增量解码时必须传当前真实位置。
3. prefill 必须传 causal mask，decode 必须不传（单 query，所有 key 都有效）。

为什么用 torch.no_grad 而不是 torch.inference_mode
---------------------------------------------------
`inference_mode` 造出来的张量带 "inference" 标记，**只能在 inference_mode 内部
做就地操作**；一旦流出到外面，任何 in-place 都会抛
`RuntimeError: Inference tensors do not track version counter.`

这里的前向要和 Forge 自己的显存调度交错（`load_model_gpu` 会搬权重、cast 权重，
这些调用发生在 Forge 的 `@torch.inference_mode()` 里），KV 缓存也是我们预分配后
就地写入的。两边的 inference 边界一旦错位就会踩上面那个错。改用 `no_grad`
不会生成 inference 张量，同样不建计算图，代价只是少了一点版本计数器的优化 ——
对一次 4B 前向来说可以忽略。
"""

import functools
import threading
import time
import traceback
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .. import presets, textproc
from . import locate, memory

#: 串行化所有前向。预览按钮与正在进行的生成可能分属两个线程，
#: 同时在一份权重上跑前向既会互相踩显存，也会让 Forge 的模型调度状态错乱。
_LOCK = threading.RLock()


def _locked(fn):
    """把一次完整的前向序列串行化。RLock 允许嵌套。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _LOCK:
            return fn(*args, **kwargs)
    return wrapper


class CacheOverflow(RuntimeError):
    pass


# ---------------------------------------------------------------- 状态

def _state():
    try:
        from modules import shared
        return shared.state
    except Exception:
        return None


def interrupted():
    state = _state()
    return bool(getattr(state, "interrupted", False)) if state is not None else False


def set_textinfo(message):
    state = _state()
    if state is not None:
        try:
            state.textinfo = message
        except Exception:
            pass


# ---------------------------------------------------------------- 基础件

def align(value, step=64):
    value = int(value)
    return ((value + step - 1) // step) * step


def _llama():
    import backend.nn.llm.llama as module
    return module


def causal_mask(size, dtype, device):
    """逐字照抄 backend/nn/llm/llama.py:489，保证与编码路径数值一致。"""
    return torch.empty(size, size, dtype=dtype, device=device).fill_(
        torch.finfo(dtype).min / 4
    ).triu_(1)


def _freqs(chain, llama_mod, position_ids):
    cfg = chain.cfg
    return llama_mod.precompute_freqs_cis(
        int(cfg.head_dim),
        position_ids,
        cfg.rope_theta,
        rope_scale=getattr(cfg, "rope_scale", None),
        rope_dims=getattr(cfg, "rope_dims", None),
        device=chain.device,
        interleaved_mrope=getattr(cfg, "interleaved_mrope", False),
    )


@dataclass
class KVCache:
    kv: list
    cap: int
    last_hidden: object = None

    @property
    def pos(self):
        return int(self.kv[0][2])


@dataclass
class GenResult:
    text: str
    error: str = ""
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.error


# ---------------------------------------------------------------- 词表投影

class Projector:
    """tied embedding 投影：hidden -> vocab logits。

    Qwen3_4B 没有 lm_head（llama.py:463-464 受 config.lm_head 保护，而
    Qwen3_4BConfig 没这个字段），只能拿 embed_tokens.weight 当输出矩阵。

    注意：`get_weight_and_bias` 在有 weight_function（TE LoRA）时会 clone 权重
    （operations.py:92）。每次生成只 refresh 一次，绝不逐 token 调用，
    否则每步克隆 780 MB。
    """

    def __init__(self, lm):
        self.lm = lm
        self.weight = None
        self.plain = False
        self.gguf = False
        self.quantized = False
        self.cast = None
        self.kind = "unknown"
        self.allow_dequant = False
        self.normal_weight = None

    def refresh(self, allow_gguf_dequant=False):
        # 允许反量化一旦放开就保持放开：probe 放行后，kv_selftest / greedy_probe
        # 会再调一次 refresh，不能用默认参数把刚放行的权重重新拦下来。
        allow = bool(allow_gguf_dequant) or bool(self.allow_dequant)
        self.allow_dequant = allow

        weight = None
        try:
            from backend.operations import get_weight_and_bias
            weight, _ = get_weight_and_bias(self.lm.embed_tokens)
        except Exception:
            weight = getattr(self.lm.embed_tokens, "weight", None)

        if weight is None:
            raise RuntimeError("embed_tokens 没有 weight，无法做词表投影。")

        previous = self.weight
        self.weight = weight
        self.gguf = getattr(weight, "gguf_cls", None) is not None
        self.plain = isinstance(weight, torch.Tensor) and not self.gguf
        self.cast = None
        # 权重没换就保留上次克隆出来的普通副本，避免每轮重新拷 780 MB。
        if previous is not weight:
            self.normal_weight = None

        if self.gguf:
            self.kind = "gguf"
        elif self.plain and weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.kind = "tied"
        elif self.plain:
            self.kind = f"tied/{str(weight.dtype).replace('torch.', '')}"
        else:
            self.quantized = True
            self.kind = "quantized"

        if (self.gguf or self.quantized) and not allow:
            raise RuntimeError(
                "文本编码器使用 GGUF/量化权重，tied embedding 无法直接做词表投影。"
            )
        return self.kind

    def _materialize(self):
        weight = self.weight
        if self.gguf:
            try:
                from backend.operations import dequantize_tensor
                return dequantize_tensor(weight).to(torch.float16)
            except Exception as exc:
                raise RuntimeError(f"GGUF 反量化失败：{exc}") from exc
        if isinstance(weight, torch.Tensor):
            return weight.to(torch.float16)
        for name in ("dequantize", "to_dense", "dequantize_tensor"):
            fn = getattr(weight, name, None)
            if callable(fn):
                try:
                    return fn().to(torch.float16)
                except Exception:
                    continue
        raise RuntimeError(f"不支持的权重类型：{type(weight).__name__}")

    def _clean_copies(self, hidden, weight):
        """克隆出普通张量。只在真的踩到 inference 错误时调用一次。

        两个必须记住的点：

        1. **`Parameter.is_inference()` 不可靠。** Forge 用 `set_data` 换过 TE 的
           权重，impl 上的 inference 标记和 `is_inference()` 的返回值会对不上：
           实测它返回 False，而 `weight.t()`（创建视图会 bump 版本计数器）
           照样抛 "Inference tensors do not track version counter."。
           所以这里**不做判断**，直接克隆。
        2. 克隆必须在 `inference_mode` **之外**执行，否则克隆出来的仍是
           inference tensor —— 这是上一版兜底无效的原因。
        """
        print("[ZTPE] 词表投影遇到 inference tensor，克隆权重后重试（仅一次）")
        with torch.inference_mode(False):
            if self.normal_weight is None:
                self.normal_weight = weight.clone()
            hidden = hidden.clone()
        return hidden, self.normal_weight

    def __call__(self, hidden):
        weight = self.weight
        if not (self.plain and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            if self.cast is None:
                self.cast = self._materialize()
            weight = self.cast

        # 已经降级过一次就直接用普通副本，不再重复付克隆开销。
        if self.normal_weight is not None:
            weight = self.normal_weight

        try:
            return self._project(hidden, weight)
        except RuntimeError as exc:
            if "inference" not in str(exc).lower():
                raise
            hidden, weight = self._clean_copies(hidden, weight)
            return self._project(hidden, weight)

    @staticmethod
    def _project(hidden, weight):
        # 必须压成一维：hidden 是 (1,1,H)，直接返回会得到 (1,1,V)，而采样侧
        # 全部按词表向量处理（cut[0]、scatter(0, ...)、multinomial）。
        return F.linear(hidden.to(weight.dtype), weight).float().reshape(-1)


# ---------------------------------------------------------------- 前向

@torch.no_grad()
def prefill(chain, ids_tensor, cap):
    """一次性吃下整段前缀，建立 KV 缓存。"""
    llama_mod = _llama()
    cfg = chain.cfg
    device, dtype = chain.device, chain.compute_dtype
    n_kv = int(cfg.num_key_value_heads)
    head_dim = int(cfg.head_dim)
    length = int(ids_tensor.shape[1])

    if length > cap:
        raise CacheOverflow(f"前缀长度 {length} 超过缓存容量 {cap}")

    kv = [
        (
            torch.zeros(1, n_kv, cap, head_dim, dtype=dtype, device=device),
            torch.zeros(1, n_kv, cap, head_dim, dtype=dtype, device=device),
            0,
        )
        for _ in range(chain.n_layers)
    ]

    x = chain.lm.embed_tokens(ids_tensor.to(device)).to(dtype)
    freqs = _freqs(chain, llama_mod, torch.arange(length, device=device).unsqueeze(0))
    mask = causal_mask(length, x.dtype, device)
    attention = llama_mod.attention_function

    for index, layer in enumerate(chain.layers):
        x, kv[index] = layer(
            x=x,
            attention_mask=mask,
            freqs_cis=freqs,
            optimized_attention=attention,
            past_key_value=kv[index],
        )

    hidden = chain.lm.norm(x[:, -1:, :])   # 只做一次
    return KVCache(kv=kv, cap=cap, last_hidden=hidden)


@torch.no_grad()
def kv_step(chain, cache, token_id):
    """喂入单个 token，返回该位置的 hidden（未投影）。"""
    llama_mod = _llama()
    cfg = chain.cfg
    device, dtype = chain.device, chain.compute_dtype
    position = cache.pos

    if position + 1 > cache.cap:
        raise CacheOverflow(f"KV 缓存溢出：需要 {position + 1}，容量 {cache.cap}")

    ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    x = chain.lm.embed_tokens(ids).to(dtype)
    freqs = _freqs(chain, llama_mod, torch.tensor([[position]], device=device))
    attention = llama_mod.attention_function

    for index, layer in enumerate(chain.layers):
        # 单 query，所有缓存位置都有效，不需要任何 mask。
        x, cache.kv[index] = layer(
            x=x,
            attention_mask=None,
            freqs_cis=freqs,
            optimized_attention=attention,
            past_key_value=cache.kv[index],
        )

    hidden = chain.lm.norm(x)              # 只做一次
    cache.last_hidden = hidden
    return hidden


@torch.no_grad()
def safe_forward(chain, ids):
    """无 KV 的全量前向。同时是 KV 路径的对照 oracle。"""
    llama_mod = _llama()
    cfg = chain.cfg
    device, dtype = chain.device, chain.compute_dtype
    length = len(ids)

    ids_tensor = torch.tensor([list(ids)], dtype=torch.long, device=device)
    x = chain.lm.embed_tokens(ids_tensor).to(dtype)
    freqs = _freqs(chain, llama_mod, torch.arange(length, device=device).unsqueeze(0))
    mask = causal_mask(length, x.dtype, device)
    attention = llama_mod.attention_function

    for layer in chain.layers:
        x, _ = layer(
            x=x,
            attention_mask=mask,
            freqs_cis=freqs,
            optimized_attention=attention,
            past_key_value=None,
        )

    return chain.lm.norm(x[:, -1:, :])     # 只做一次


def check_alignment(cache):
    """所有层的写入位置必须一致。错位会表现为"通顺但跑题"，最难排查。"""
    positions = {int(item[2]) for item in cache.kv}
    if len(positions) != 1:
        raise RuntimeError(f"KV 各层写入位置不一致：{sorted(positions)}")
    return positions.pop()


# ---------------------------------------------------------------- 采样

def sample(logits, cfg, recent, generator):
    # 兜底压平：无论上游给的是 (V,) 还是 (1,1,V)，这里都按一维词表向量处理。
    # 不压平会导致 cut[0]=False 把整行清空、scatter/索引越界（CUDA 上还会触发
    # device-side assert 污染上下文）。
    logits = logits.reshape(-1).float().clone()

    if cfg.rep_penalty > 1.0 and recent:
        window = recent[-64:]
        indices = torch.tensor(sorted(set(window)), dtype=torch.long, device=logits.device)
        values = logits[indices]
        logits[indices] = torch.where(
            values > 0, values / cfg.rep_penalty, values * cfg.rep_penalty
        )

    if cfg.temperature <= 1e-4:
        return int(logits.argmax().item())

    probs = torch.softmax(logits / cfg.temperature, dim=-1)

    if 0.0 < cfg.top_p < 1.0:
        sorted_probs, sorted_index = torch.sort(probs, descending=True)
        cut = torch.cumsum(sorted_probs, dim=-1) > cfg.top_p
        cut[0] = False
        sorted_probs = sorted_probs.masked_fill(cut, 0.0)
        total = sorted_probs.sum()
        if total <= 0:
            return int(sorted_index[0].item())
        probs = torch.zeros_like(probs).scatter(0, sorted_index, sorted_probs / total)

    # 采样统一在 CPU 上做：Generator 与张量同设备即可，151936 个 float 的拷贝
    # 约 0.6 MB，开销可忽略，换来跨 GPU / torch 版本的种子可复现。
    pick = torch.multinomial(probs.cpu(), 1, generator=generator)
    return int(pick.item())


# ---------------------------------------------------------------- chat 组装

def build_chat(chain, caps, system, user, layout):
    if layout == "parity":
        template = getattr(chain.engine, "llama_template", None) or \
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        try:
            return template.format(user)
        except Exception:
            return template.replace("{}", user)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user or " "})

    if caps.chat_template:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if caps.think_kwarg:
            kwargs["enable_thinking"] = False
        try:
            return chain.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            pass

    parts = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
    parts.append(f"<|im_start|>user\n{user or ' '}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


# ---------------------------------------------------------------- 生成

def generate(chain, caps, cfg, prompt, template, seed=1234, progress=None):
    """对单条 prompt 执行改写。失败时 `GenResult.text` 回退为原文。"""
    lang = textproc.resolve_lang(cfg.language, prompt)

    with _LOCK:
        result = _generate_once(chain, caps, cfg, prompt, template, lang, seed, progress, False)
        if result.reason == "wrong_language":
            # 语种不符只重试一次，显式追加语言指令；二次仍不符则交付原文。
            retry = _generate_once(chain, caps, cfg, prompt, template, lang, seed, progress, True)
            if retry.ok:
                return retry
        return result


def _generate_once(chain, caps, cfg, prompt, template, lang, seed, progress, strict):
    """先按 no_grad 跑；若踩到 inference tensor，再重试一次。

    注意：实测 inference_mode **并不能**解决所有情况 —— `F.linear` 在
    inference_mode 内部同样会抛 "Inference tensors do not track version counter."。
    真正的修复在 `Projector.__call__` 里（clone 剥掉标记）。这里保留一次重试
    只是兜底，覆盖 Projector 之外可能出现的同类错误。
    """
    try:
        return _decode_once(chain, caps, cfg, prompt, template, lang, seed, progress, strict)
    except RuntimeError as exc:
        if "inference tensor" not in str(exc).lower():
            raise
        print("[ZTPE] 检测到 inference tensor，改用 inference_mode 重试一次")
        with torch.inference_mode():
            return _decode_once(chain, caps, cfg, prompt, template, lang, seed, progress, strict)


@torch.no_grad()
def _decode_once(chain, caps, cfg, prompt, template, lang, seed, progress, strict):
    protected, table = textproc.protect_tags(prompt)

    body = template
    if strict and lang in presets.LANG_DIRECTIVE:
        body = template + presets.LANG_DIRECTIVE[lang]

    system, user = textproc.build_messages(body, protected, cfg.chat_layout)
    chat = build_chat(chain, caps, system, user, cfg.chat_layout)

    try:
        ids = locate.tokenize(chain, chat)
    except Exception as exc:
        return GenResult(prompt, f"分词失败：{exc}", "tokenize_failed")

    if not ids:
        return GenResult(prompt, "分词结果为空", "empty_ids")

    max_context = int(getattr(chain.cfg, "max_position_embeddings", 40960))
    if len(ids) + 8 >= max_context:
        return GenResult(prompt, f"输入过长（{len(ids)} token）", "input_too_long")

    use_kv = bool(caps.kv)
    max_new = int(cfg.max_new)
    if memory.is_cpu(chain.device):
        # CPU 解码约 1 token/s，默认 160 token 会挂十几分钟。probe 里的提示
        # 必须名副其实 —— 这里真的把上限压下来。
        max_new = min(max_new, 32)
    capacity = align(len(ids) + max_new + 8, 64)

    needed = memory.cache_bytes(
        chain.n_layers, chain.cfg.num_key_value_heads, chain.cfg.head_dim,
        capacity, chain.compute_dtype,
    )
    if use_kv and not memory.fits(chain.device, needed):
        free = memory.free_bytes(chain.device)
        per_token = needed / max(capacity, 1)
        allowed = int((free - 0.6 * 1024 ** 3) / per_token) - len(ids) - 8 if per_token > 0 else 0
        if allowed < 32:
            return GenResult(prompt, "显存不足，已跳过改写", "oom_precheck")
        max_new = min(max_new, allowed)
        capacity = align(len(ids) + max_new + 8, 64)

    try:
        chain.projector.refresh(cfg.allow_gguf_dequant)
    except Exception as exc:
        return GenResult(prompt, str(exc), "projection_unsupported")

    generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    started = time.time()
    produced = []
    mode = "kv" if use_kv else "safe"
    # 记录结束原因。截断（打满 max_new 或超时间预算）的结果不应被当成完整改写，
    # 至少要让调用方知道。
    finish = "max_new"

    try:
        if use_kv:
            cache = prefill(chain, torch.tensor([ids], device=chain.device), capacity)
            logits = chain.projector(cache.last_hidden)
        else:
            cache = None
            logits = chain.projector(safe_forward(chain, ids))

        for step in range(max_new):
            token = sample(logits, cfg, produced, generator)
            if token in caps.stop_ids:
                finish = "stop"
                break

            produced.append(token)

            if step % 4 == 0:
                if interrupted():
                    finish = "interrupted"
                    break
                set_textinfo(f"Z-Image TE: {step}/{max_new} tokens ({mode})")
                if progress is not None:
                    try:
                        progress(step, max_new)
                    except Exception:
                        pass

            if time.time() - started > cfg.time_budget:
                finish = "time_budget"
                break

            if use_kv:
                if step % 16 == 0:
                    check_alignment(cache)
                logits = chain.projector(kv_step(chain, cache, token))
            else:
                logits = chain.projector(safe_forward(chain, ids + produced))

        cache = None

    except CacheOverflow as exc:
        return GenResult(prompt, str(exc), "cache_overflow")
    except torch.cuda.OutOfMemoryError:
        # 必须写在 except Exception 之前 —— 旧版就是被内层包装成 RuntimeError 后
        # 这个分支永远进不来。
        memory.soft_release()
        return GenResult(prompt, "改写过程中显存不足。请降低最大生成 token。", "oom")
    except Exception as exc:
        detail = traceback.format_exc()
        # inference tensor 相关的错误交给 _generate_once 用 inference_mode 重试，
        # 这里不能吞掉，否则外层永远看不到。
        if "inference tensor" in str(exc).lower():
            raise
        print(f"[ZTPE] 生成失败：\n{detail}")
        return GenResult(prompt, f"{type(exc).__name__}: {exc}", "exception",
                         {"traceback": detail})

    if not produced:
        return GenResult(prompt, "模型没有产生任何输出 token", "empty_output")

    elapsed = time.time() - started
    raw = locate.detokenize(chain, produced)
    text = textproc.clean(raw)
    text, missing = textproc.restore_tags(text, table)

    if cfg.strip_meta:
        text = textproc.strip_meta_tags(text)

    if not textproc.tags_intact(text, table):
        return GenResult(prompt, "标签还原失败，已回退原文", "tags_lost")

    reason = textproc.check_output(protected, text, lang)
    if reason:
        return GenResult(
            prompt,
            f"改写结果不可用（{textproc.REASON_TEXT.get(reason, reason)}），已回退原文",
            reason,
        )

    return GenResult(
        text,
        "",
        "",
        {
            "tokens": len(produced),
            "seconds": elapsed,
            "mode": mode,
            "lang": lang,
            "prompt_tokens": len(ids),
            "speed": (len(produced) / elapsed) if elapsed > 0 else 0.0,
            "finish": finish,
            "truncated": finish in ("max_new", "time_budget"),
        },
    )


# ---------------------------------------------------------------- 自检

@torch.no_grad()
@_locked
def greedy_probe(chain, caps, prompt, template="Repeat exactly: {prompt}", max_new=12):
    """纯贪心、不做任何清洗与退化检测的探针，用于判断 TE 是否具备指令跟随能力。"""
    protected, _table = textproc.protect_tags(prompt)
    system, user = textproc.build_messages(template, protected, "official")
    chat = build_chat(chain, caps, system, user, "official")

    ids = locate.tokenize(chain, chat)
    if not ids:
        return ""

    chain.projector.refresh()

    if caps.kv:
        capacity = align(len(ids) + max_new + 8, 64)
        cache = prefill(chain, torch.tensor([ids], device=chain.device), capacity)
        logits = chain.projector(cache.last_hidden)
    else:
        cache = None
        logits = chain.projector(safe_forward(chain, ids))

    out = []
    for _ in range(int(max_new)):
        token = int(logits.argmax().item())
        if token in caps.stop_ids:
            break
        out.append(token)
        if caps.kv:
            logits = chain.projector(kv_step(chain, cache, token))
        else:
            logits = chain.projector(safe_forward(chain, ids + out))

    return locate.detokenize(chain, out)


@torch.no_grad()
@_locked
def kv_selftest(chain, length=4, capacity=16):
    """用 4 个 dummy token 验证 KV 机制真的通。约 1 秒、约 3 MB。"""
    try:
        chain.projector.refresh()
    except Exception as exc:
        return False, f"投影不可用：{exc}"

    device = chain.device
    ids = torch.ones((1, length), dtype=torch.long, device=device)

    try:
        cache = prefill(chain, ids, capacity)
    except Exception as exc:
        return False, f"prefill 失败：{type(exc).__name__}: {exc}"

    if cache.pos != length:
        return False, f"prefill 写入位置 {cache.pos}，期望 {length}"

    expected = (1, int(chain.cfg.num_key_value_heads), capacity, int(chain.cfg.head_dim))
    shape = tuple(cache.kv[0][0].shape)
    if shape != expected:
        return False, f"KV 形状 {shape}，期望 {expected}"

    try:
        kv_step(chain, cache, 1)
    except Exception as exc:
        return False, f"decode 失败：{type(exc).__name__}: {exc}"

    if cache.pos != length + 1:
        return False, f"decode 后位置 {cache.pos}，期望 {length + 1}"

    return True, f"prefill+decode 正常（cap={capacity}, 位置={cache.pos}）"


@torch.no_grad()
@_locked
def consistency_check(chain, text=None):
    """对比"增量解码"与"全量前向"的 logits，返回 (max_delta, mean_delta, 探针长度)。

    必须真的走一次 kv_step —— 只比 prefill 和全量前向的话，两边都是整段计算，
    增量解码最容易错的地方（绝对位置 RoPE、decode 不传 mask）根本没被检验。
    """
    probe = text or "A quiet room with a wooden table and soft window light."
    ids = locate.tokenize(chain, probe)
    if len(ids) < 2:
        ids = (ids or [1]) * 2

    chain.projector.refresh()
    capacity = align(len(ids) + 8, 64)

    # 前缀走 prefill，最后一个 token 走 kv_step —— 与真实生成路径一致。
    prefix, last = ids[:-1], ids[-1]
    cache = prefill(chain, torch.tensor([prefix], device=chain.device), capacity)
    logits_kv = chain.projector(kv_step(chain, cache, last))

    logits_ref = chain.projector(safe_forward(chain, ids))

    delta = (logits_kv.float() - logits_ref.float()).abs()
    return float(delta.max().item()), float(delta.mean().item()), len(ids)
