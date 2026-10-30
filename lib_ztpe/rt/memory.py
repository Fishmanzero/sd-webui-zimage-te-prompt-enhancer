# -*- coding: utf-8 -*-
"""显存调度。全部走 Forge 自己的 memory_management，不自作主张。

两条硬规则（旧版都踩了）：

1. 传给 `load_model_gpu` 的必须是 **ModelPatcher**，不是 `CLIP` 包装对象。
   `CLIP` 没有 `model_patches_models()`，会静默失败。
2. 生成结束后 **不要** 调 `torch.cuda.empty_cache()`。它会把 Forge 的缓存块
   清掉，导致下一轮重新从 CPU 搬模型，每次出图都多几秒。
"""

import torch


class TextEncoderMemoryError(RuntimeError):
    pass


def _mm():
    from backend import memory_management
    return memory_management


def clip_patcher(sd):
    """取文本编码器的 ModelPatcher；每次都重新取，不缓存。"""
    objects = getattr(sd, "forge_objects", None)
    clip = getattr(objects, "clip", None)
    patcher = getattr(clip, "patcher", None)
    if patcher is None:
        raise TextEncoderMemoryError("找不到文本编码器的 ModelPatcher（forge_objects.clip.patcher）。")
    return patcher


def ensure_resident(sd):
    """把文本编码器调到计算设备。已常驻时是 no-op。

    与 backend/diffusion_engine/zimage.py 的 get_learned_conditioning 一致。
    """
    patcher = clip_patcher(sd)
    mm = _mm()
    try:
        mm.load_model_gpu(patcher)
    except RuntimeError as exc:
        # Forge 自己的 get_learned_conditioning 是 @torch.inference_mode() 的，
        # 权重可能因此被 materialize 成 inference tensor。这类张量只允许在
        # inference_mode 内做就地操作，在外面搬动会抛
        # "Inference tensors do not track version counter."。
        # 重试一次，这次按 Forge 的调用约定进入 inference_mode。
        if "inference tensor" not in str(exc).lower():
            raise
        with torch.inference_mode():
            mm.load_model_gpu(patcher)
    return patcher


def free_bytes(device):
    try:
        return float(_mm().get_free_memory(device))
    except Exception:
        return 0.0


def soft_release():
    """OOM 之后的温和回收。绝不用 torch.cuda.empty_cache()。"""
    try:
        _mm().soft_empty_cache()
    except Exception:
        pass


def cache_bytes(n_layers, n_kv_heads, head_dim, capacity, dtype):
    itemsize = torch.tensor([], dtype=dtype).element_size() if dtype is not None else 2
    return 2 * int(n_layers) * int(n_kv_heads) * int(head_dim) * int(capacity) * itemsize


def fits(device, needed, reserve=0.6 * 1024 ** 3):
    free = free_bytes(device)
    if free <= 0:
        return True  # 读不到就不拦，交给真实分配去报错
    return free > needed + reserve


def is_cpu(device):
    try:
        return torch.device(device).type == "cpu"
    except Exception:
        return True
