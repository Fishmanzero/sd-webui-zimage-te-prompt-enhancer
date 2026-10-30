# -*- coding: utf-8 -*-
"""KV 路径数值等价自检（不需要任何 checkpoint）。

用 Forge 自己的 `Llama2_` 造一个 2 层、64 隐藏维的极小模型，然后对比三条路径
最后一个位置的 logits：

    A. `Llama2_.forward`   —— Forge 原生全量前向（基准）
    B. `decode.safe_forward` —— 我们的全量前向（无 KV）
    C. `decode.prefill` + `kv_step` —— 我们的增量解码

A≈B 说明"绕开 forward 自己跑层循环 + 只 norm 一次"是对的；
B≈C 说明 KV 缓存、绝对位置 RoPE、mask 处理都是对的。

需要 torch 与 Forge backend，所以不进 run_selftest.py：

    <venv>/python.exe tests/torch_selftest.py

backend 导入失败时本脚本会打印 SKIP 并退出 0（例如缺少 gguf 依赖的环境）。
"""

import os
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.realpath(__file__))
_ROOT = os.path.dirname(_HERE)
_FORGE = os.path.dirname(os.path.dirname(_ROOT))

for path in (_ROOT, _FORGE):
    if path not in sys.path:
        sys.path.insert(0, path)


def _stub_missing_gguf():
    """部分环境没有装 gguf 包，但 backend 的导入链会碰到它。

    这里只补上导入期需要的壳子；本测试完全不涉及 GGUF 权重。
    """
    try:
        import gguf  # noqa: F401
        return
    except Exception:
        pass

    class _Any:
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return 0

        def __call__(self, *args, **kwargs):
            return self

    def _module(name):
        module = types.ModuleType(name)
        module.__path__ = []

        def _getattr(item):
            if item.startswith("__"):
                raise AttributeError(item)
            return _Any()

        module.__getattr__ = _getattr
        sys.modules[name] = module

    for name in ("gguf", "gguf.constants", "gguf.gguf_reader", "gguf.vocab", "gguf.quants"):
        _module(name)

    loader = types.ModuleType("backend.loader_gguf")
    loader.dequantize = lambda weight, **kwargs: weight
    loader.get_orig_shape = lambda weight: tuple(weight.shape)
    sys.modules["backend.loader_gguf"] = loader


@dataclass
class TinyConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    transformer_type: str = "llama"
    head_dim: int = 16
    rms_norm_add: bool = False
    mlp_activation: str = "silu"
    qkv_bias: bool = False
    rope_dims: object = None
    q_norm: str = "gemma3"
    k_norm: str = "gemma3"
    rope_scale: object = None
    final_norm: bool = True


def _max_diff(a, b):
    """NaN 必须表现为失败，不能悄悄被 max() 吃掉。"""
    import torch
    delta = (a.float() - b.float()).abs()
    if not torch.isfinite(delta).all():
        return float("inf")
    return float(delta.max().item())


def _init_weights(model):
    """Forge 的 ops 不会给合成模型做初始化，权重可能是未初始化的垃圾值。"""
    import torch
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                torch.nn.init.normal_(param, std=0.02)
            else:
                torch.nn.init.ones_(param)


def main():
    _stub_missing_gguf()

    try:
        import torch
        import backend.nn.llm.llama as llama_mod
        from backend import operations
    except Exception as exc:
        print(f"SKIP: 无法导入 Forge backend（{type(exc).__name__}: {exc}）")
        return 0

    from lib_ztpe.rt import decode
    from lib_ztpe.config import GenCfg

    torch.manual_seed(0)
    config = TinyConfig()

    # Forge 的 nn.RMSNorm 是打过补丁的（接受 add= 参数），补丁只在
    # using_forge_operations 上下文里生效，所以建模必须包在里面。
    with operations.using_forge_operations(
        device=torch.device("cpu"), dtype=torch.float32, manual_cast_enabled=False
    ):
        lm = llama_mod.Llama2_(config).eval()
    _init_weights(lm)

    projector = decode.Projector(lm)
    projector.refresh()

    device = torch.device("cpu")

    class _FakeTokenizer:
        """只为让 greedy_probe 跑通，字符级映射，不追求可读。"""

        chat_template = None

        def encode(self, text, add_special_tokens=False):
            return [(ord(char) % 120) + 1 for char in text][:48]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr((int(i) - 1) % 128) for i in ids)

    chain = SimpleNamespace(
        lm=lm,
        layers=lm.layers,
        cfg=config,
        device=device,
        compute_dtype=torch.float32,
        projector=projector,
        n_layers=len(lm.layers),
        storage_dtype=torch.float32,
        tokenizer=_FakeTokenizer(),
        engine=None,
    )

    prefix = [1, 5, 9, 12, 7, 3, 21]
    failures = []

    def check(name, condition, detail=""):
        print(("  PASS  " if condition else "  FAIL  ") + name + ("" if condition else f"   <-- {detail}"))
        if not condition:
            failures.append(name)

    with torch.inference_mode():
        # ---------------- A: Forge 原生 forward
        reference_hidden = lm(torch.tensor([prefix]))[0][:, -1:, :]
        reference = projector(reference_hidden)

        # ---------------- B: 我们的全量前向
        safe_logits = projector(decode.safe_forward(chain, prefix))
        delta_ab = _max_diff(reference, safe_logits)
        check("safe_forward 与 Llama2_.forward 一致（max|Δ|<1e-4）", delta_ab < 1e-4, f"max|Δ|={delta_ab:.3e}")

        # ---------------- C: prefill
        cache = decode.prefill(chain, torch.tensor([prefix]), cap=64)
        kv_logits = projector(cache.last_hidden)
        delta_ac = _max_diff(reference, kv_logits)
        check("prefill 与 Llama2_.forward 一致（max|Δ|<1e-4）", delta_ac < 1e-4, f"max|Δ|={delta_ac:.3e}")
        check("prefill 写入位置正确", cache.pos == len(prefix), f"pos={cache.pos}")

        # ---------------- D: 逐步增量解码，逐步与全量前向对齐
        produced = []
        worst = 0.0
        for _ in range(6):
            token = int(kv_logits.argmax().item())
            produced.append(token)

            kv_logits = projector(decode.kv_step(chain, cache, token))
            full_logits = projector(decode.safe_forward(chain, prefix + produced))
            worst = max(worst, _max_diff(kv_logits, full_logits))

        check("6 步增量解码逐步对齐全量前向（max|Δ|<1e-4）", worst < 1e-4, f"max|Δ|={worst:.3e}")
        check("增量解码后位置正确", cache.pos == len(prefix) + len(produced),
              f"pos={cache.pos} 期望 {len(prefix) + len(produced)}")
        check("各层写入位置一致", decode.check_alignment(cache) == cache.pos)

        # ---------------- E: 缓存容量不足要报错，而不是静默重分配
        try:
            tight = decode.prefill(chain, torch.tensor([prefix]), cap=len(prefix))
            for _ in range(3):
                token = int(kv_logits.argmax().item())
                kv_logits = projector(decode.kv_step(chain, tight, token))
            check("容量耗尽抛 CacheOverflow", False, "没有抛异常")
        except decode.CacheOverflow:
            check("容量耗尽抛 CacheOverflow", True)

        # ---------------- F: 贪心探针可跑通
        caps = SimpleNamespace(kv=True, stop_ids=frozenset(),
                               chat_template=False, think_kwarg=False)
        text = decode.greedy_probe(
            chain, caps, "ZEBRA",
            template="Repeat exactly: {prompt}",
            max_new=4,
        )
        check("greedy_probe 返回字符串", isinstance(text, str), type(text).__name__)

    # ---------------- G: 不能产生 inference tensor
    # 必须在 inference_mode 之外检查，否则测到的是调用方的环境而不是我们的装饰器。
    # inference tensor 只能在 inference_mode 内部做就地操作，一旦流出到 Forge 的
    # 显存调度里就会抛 "Inference tensors do not track version counter."。
    # 这条断言锁住"解码路径必须用 no_grad 而非 inference_mode"。
    probe_cache = decode.prefill(chain, torch.tensor([prefix]), cap=32)
    check("prefill 的 hidden 不是 inference tensor",
          not probe_cache.last_hidden.is_inference())
    check("KV 缓冲不是 inference tensor",
          not probe_cache.kv[0][0].is_inference() and not probe_cache.kv[0][1].is_inference())
    hidden_after = decode.kv_step(chain, probe_cache, 5)
    check("kv_step 的 hidden 不是 inference tensor",
          not hidden_after.is_inference())
    del probe_cache

    # ---------------- H: Chain.sync() 必须能纠正陈旧的 device/dtype
    # Forge 会在两次生成之间把 TE 卸到 CPU（CPU 副本是 fp32），此时 resolve()
    # 读到的是陈旧值。sync() 必须在搬到计算设备后被调用并刷新它们。
    from lib_ztpe.rt import locate as locate_mod
    chain_obj = locate_mod.Chain(
        sd=None, engine=None, clip=None, sdclip=None, wrapper=lm, lm=lm,
        tokenizer=chain.tokenizer, layers=lm.layers, cfg=config,
        device=torch.device("meta"), storage_dtype=torch.float16,
        compute_dtype=torch.float16,
    )
    check("sync 前 device 是陈旧值", chain_obj.device.type == "meta")
    chain_obj.sync()
    check("sync 修正 device", chain_obj.device == lm.embed_tokens.weight.device)
    check("sync 修正 storage_dtype",
          chain_obj.storage_dtype == lm.embed_tokens.weight.dtype,
          f"{chain_obj.storage_dtype} vs {lm.embed_tokens.weight.dtype}")

    # ---------------- I: inference 错误必须触发 inference_mode 重试
    # _decode_once 内部会把异常吞成 GenResult，所以它必须对 inference 相关错误
    # 重新抛出，否则外层永远等不到重试机会。
    calls = {"n": 0}
    real_decode_once = decode._decode_once

    def _fake_decode_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Inference tensors do not track version counter.")
        return decode.GenResult("重试成功")

    decode._decode_once = _fake_decode_once
    try:
        result = decode._generate_once(None, None, None, "", "", "en", 1, None, False)
        check("inference 错误触发重试并成功", result.ok and result.text == "重试成功",
              f"ok={result.ok} calls={calls['n']}")
        check("重试只发生一次", calls["n"] == 2, f"calls={calls['n']}")
    finally:
        decode._decode_once = real_decode_once

    # 普通异常不应被重试
    calls["n"] = 0

    def _bad_decode_once(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("something else broke")

    decode._decode_once = _bad_decode_once
    try:
        try:
            decode._generate_once(None, None, None, "", "", "en", 1, None, False)
            check("非 inference 错误直接抛出", False, "没有抛异常")
        except RuntimeError:
            check("非 inference 错误直接抛出", calls["n"] == 1, f"calls={calls['n']}")
    finally:
        decode._decode_once = real_decode_once

    # ---------------- J: 投影必须能绕开 inference 错误
    # 真机堆栈显示失败点是 `weight.t()`，而 `weight.is_inference()` 返回 False ——
    # Forge 用 set_data 换过 TE 权重，impl 上的标记与 is_inference() 不一致。
    # 所以兜底不能依赖 is_inference()，必须无条件克隆出普通副本。
    state = {"n": 0}
    real_linear = decode.F.linear

    def _flaky_linear(x, w, *args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("Inference tensors do not track version counter.")
        return real_linear(x, w, *args, **kwargs)

    decode.F.linear = _flaky_linear
    try:
        projector.normal_weight = None
        out = projector(torch.randn(1, 1, config.hidden_size))
        check("投影遇 inference 错误自动降级", tuple(out.shape) == (config.vocab_size,),
              str(tuple(out.shape)))
        check("降级后缓存了普通权重副本", projector.normal_weight is not None)

        before = state["n"]
        projector(torch.randn(1, 1, config.hidden_size))
        check("降级后不再重复克隆/重试", state["n"] - before == 1, f"calls={state['n'] - before}")
    finally:
        decode.F.linear = real_linear
        projector.normal_weight = None

    # ---------------- K: 采样必须接受三维 logits
    # 投影输出是 (1,1,V)；sample 若按一维处理，cut[0]=False 会清空整行，
    # scatter(0, ...) 和 logits[indices] 直接越界（CUDA 上还会 device-side assert）。
    sample_cfg = GenCfg(temperature=0.7, top_p=0.9, rep_penalty=1.05)
    logits3 = torch.randn(1, 1, config.vocab_size)
    try:
        token = decode.sample(logits3, sample_cfg, [], torch.Generator().manual_seed(0))
        check("sample 接受三维 logits", 0 <= token < config.vocab_size, str(token))
    except Exception as exc:
        check("sample 接受三维 logits", False, f"{type(exc).__name__}: {exc}")

    try:
        token = decode.sample(logits3, sample_cfg, [1, 2, 3], torch.Generator().manual_seed(0))
        check("sample 带重复惩罚也接受三维 logits", 0 <= token < config.vocab_size, str(token))
    except Exception as exc:
        check("sample 带重复惩罚也接受三维 logits", False, f"{type(exc).__name__}: {exc}")

    check("投影输出是一维词表向量",
          tuple(projector(torch.randn(1, 1, config.hidden_size)).shape) == (config.vocab_size,))

    # ---------------- L: 一致性校验必须真的走 kv_step
    try:
        delta_max, delta_mean, probe_len = decode.consistency_check(chain)
        check("consistency_check 走 kv_step 且数值一致", delta_max < 1e-4,
              f"max|Δ|={delta_max:.3e} len={probe_len}")
    except Exception as exc:
        check("consistency_check 走 kv_step 且数值一致", False, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} 项失败：{failures}")
        return 1
    print("KV 数值等价自检全部通过")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
