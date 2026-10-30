# -*- coding: utf-8 -*-
"""把当前模型解析成一条确定的、类型化的链路。

不再像旧版那样用 hasattr 白名单到处试探 —— 那条路径既慢又会误命中。
Z-Image 的链路在 Forge 里是固定的：

    shared.sd_model (ZImage)
      └─ .text_processing_engine_qwen          Qwen34BEngine
           └─ .text_encoder                    SDClipModel
                └─ .transformer                Qwen3_4B
                     └─ .model                 Llama2_   ← 真正要跑前向的东西

`Chain` 刻意不缓存：`p.sd_model.forge_objects` 每轮都会被重新赋值
（modules/processing.py），LoRA 也会重打 patcher，缓存必然变脏。
"""

from dataclasses import dataclass, field

import torch


class ResolveError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Chain:
    sd: object
    engine: object
    clip: object
    sdclip: object
    wrapper: object
    lm: object
    tokenizer: object
    layers: object
    cfg: object
    device: torch.device
    storage_dtype: torch.dtype
    compute_dtype: torch.dtype
    requested_dtype: str = "auto"
    projector: object = field(default=None)

    def sync(self):
        """重新读取权重当前的设备与精度。

        Forge 会在两次生成之间把文本编码器卸到 CPU（CPU 副本是 fp32），
        下次用到时再搬回 GPU。若在 `load_model_gpu` 之前就读 device/dtype，
        拿到的会是"已卸下"的陈旧值 —— 诊断会误报"文本编码器在 CPU 上"。
        """
        weight = getattr(getattr(self.lm, "embed_tokens", None), "weight", None)
        if weight is None:
            return self
        self.device = getattr(weight, "device", self.device)
        self.storage_dtype = getattr(weight, "dtype", self.storage_dtype)
        self.compute_dtype = _compute_dtype(weight, self.requested_dtype)
        return self

    @property
    def n_layers(self):
        try:
            return len(self.layers)
        except Exception:
            return int(getattr(self.cfg, "num_hidden_layers", 0))

    def describe(self):
        names = [
            type(self.sd).__name__,
            type(self.engine).__name__,
            type(self.sdclip).__name__,
            type(self.wrapper).__name__,
            type(self.lm).__name__,
        ]
        return " -> ".join(names) + f" ({self.n_layers} layers)"


def _resolve_tokenizer(sd, engine):
    """按可靠性排序取 tokenizer，优先裸 HF tokenizer。"""
    candidates = []

    clip = getattr(getattr(sd, "forge_objects", None), "clip", None)
    holder = getattr(clip, "tokenizer", None)
    if holder is not None:
        for name in ("qwen3", "qwen3_4b", "qwen_3_4b", "qwen"):
            candidates.append(getattr(holder, name, None))

    wrapper = getattr(engine, "tokenizer", None)
    candidates.append(getattr(wrapper, "tokenizer", None))
    candidates.append(wrapper)

    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "encode") and hasattr(candidate, "decode"):
            return candidate

    raise ResolveError(
        "no_tokenizer",
        "找不到 Qwen3 的 tokenizer。请确认 qwen_3_4b 的 tokenizer 已随 checkpoint 加载。",
    )


def _compute_dtype(weight, requested="auto"):
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if requested in mapping:
        return mapping[requested]

    dtype = getattr(weight, "dtype", None)
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    # fp8 / gguf / 量化布局：激活值统一用 fp16，权重由 Projector 单独处理。
    return torch.float16


def resolve(sd_model, compute_dtype="auto"):
    if sd_model is None:
        raise ResolveError("no_model", "尚未加载任何模型。")

    if type(sd_model).__name__ != "ZImage":
        raise ResolveError(
            "not_zimage",
            f"当前模型不是 Z-Image（{type(sd_model).__name__}）。本扩展只在 Z-Image / Z-Image-Turbo 下工作。",
        )

    engine = getattr(sd_model, "text_processing_engine_qwen", None)
    if engine is None:
        raise ResolveError(
            "no_engine",
            "Z-Image 引擎缺少 text_processing_engine_qwen —— Forge 版本与本扩展不匹配。",
        )

    clip = getattr(getattr(sd_model, "forge_objects", None), "clip", None)
    sdclip = getattr(engine, "text_encoder", None)
    if sdclip is None:
        raise ResolveError("no_sdclip", "text_processing_engine_qwen 上没有 text_encoder。")

    wrapper = getattr(sdclip, "transformer", None)
    if wrapper is None:
        raise ResolveError("no_transformer", "SDClipModel 上没有 transformer。")

    # 正常是 Qwen3_4B.model -> Llama2_；某些构建可能直接挂 Llama2_。
    lm = getattr(wrapper, "model", wrapper)

    layers = getattr(lm, "layers", None)
    cfg = getattr(lm, "config", None)
    embed = getattr(lm, "embed_tokens", None)
    weight = getattr(embed, "weight", None)

    if layers is None or cfg is None or weight is None:
        raise ResolveError(
            "lm_type",
            f"文本编码器结构为 {type(lm).__name__}，不是可复用的 Llama2_ —— 无法驱动其前向。",
        )

    tokenizer = _resolve_tokenizer(sd_model, engine)

    device = getattr(weight, "device", torch.device("cpu"))
    storage_dtype = getattr(weight, "dtype", torch.float16)

    chain = Chain(
        sd=sd_model,
        engine=engine,
        clip=clip,
        sdclip=sdclip,
        wrapper=wrapper,
        lm=lm,
        tokenizer=tokenizer,
        layers=layers,
        cfg=cfg,
        device=device,
        storage_dtype=storage_dtype,
        compute_dtype=_compute_dtype(weight, compute_dtype),
        requested_dtype=compute_dtype,
    )

    from .decode import Projector
    chain.projector = Projector(lm)
    return chain


def tokenize(chain, text):
    """返回一维 token id 列表。"""
    tokenizer = chain.tokenizer

    if callable(tokenizer):
        try:
            out = tokenizer(text, add_special_tokens=False)
            ids = out["input_ids"] if isinstance(out, dict) else out
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            if ids and isinstance(ids[0], int):
                return list(ids)
        except Exception:
            pass

    out = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(out, "ids"):
        out = out.ids
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def detokenize(chain, ids):
    try:
        return chain.tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        return chain.tokenizer.decode(ids)
