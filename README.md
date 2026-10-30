# Z-Image TE Prompt Enhancer

在生成图片之前，**复用 Forge 已经加载在显存里的 Z-Image 文本编码器（Qwen3-4B）**
改写提示词。不加载第二个 LLM，不走云端 API，不新增任何 Python 依赖。

用 **KV-cache 增量解码**：整段前缀只前向一次，之后逐 token 复用缓存。
这是本扩展与同类实现最本质的差别 —— 同样 160 个 token，全量重算要几十秒，
增量解码只要几秒。

---

## 安装

```
sd-webui-forge-neo/
└── extensions/
    └── zimage-te-prompt-enhancer/
        ├── scripts/zimage_te_pe.py     ← 必须在 scripts/ 下
        ├── lib_ztpe/
        ├── javascript/
        └── metadata.ini
```

**完全重启 WebUI**（不是 Reload UI）。启动日志里应出现：

```
[ZTPE] loaded v2.0.0 (zimage_te_pe.py)
```

看不到这行说明脚本没被加载：检查是否多套了一层目录、控制台是否有红色 traceback、
启动参数是否带了 `--disable-extra-extensions`。

---

## 面板在哪里

不是独立标签页。它注册了自己的分区，出现在 txt2img / img2img 页面下方
「Z-Image TE Prompt Enhancer」处，默认折叠。

---

## 快速开始

1. 加载 Z-Image / Z-Image-Turbo，选好 `qwen_3_4b` 文本编码器与 VAE。
2. 展开面板 → 「预览与诊断」→ 点 **运行诊断**。
   期望看到 `VERDICT: FAST PATH (KV cache)`，并且 KV cache 一栏是 `[OK]`。
3. 点 **KV 一致性校验**，期望 `max|Δlogit| < 5e-3`。
4. 点 **载入当前提示词** → **预览改写**，确认效果与耗时。
5. 勾选 **本次启用**，正常 Generate。

---

## 参数

### 面板

| 参数 | 说明 |
|---|---|
| 本次启用 | 只影响这一次运行；默认值在设置页 |
| 改写模板 | 预设 + 可编辑文本框；`{prompt}` 为占位符 |
| 输出语言 | `auto` 跟随输入语种；`en` / `zh` 强制 |
| 最大生成 token | 默认 160，**主要耗时开关**。打满会被截断，预览与日志会明确标注 |
| temperature / top_p | 默认 **0.7 / 0.8**，与官方 Space 一致 |
| 重复惩罚 | 1.0 关闭，默认 1.05 |
| 改写种子 | -1 跟随本次图片种子（`p.all_seeds[i]`），保证同种子可复现 |
| 批量行为 | 整批共用一次（按**去重后**的提示词各改写一次，省时）／每张单独改写 |
| hires.fix 提示词 | 跟随主提示词／单独改写／保持原样 |
| Chat 布局 | 官方（模板进 system）／模板进 user／复刻编码路径 |

### 设置页

`Settings → Z-Image TE Prompt Enhancer` 里的值会持久化，面板默认值从这里读。
另有几个只在设置页出现的开关：时间预算、强制安全模式、允许 GGUF 反量化、
允许 CPU 解码。

---

## 关于「官方 PE」预设

`official` 预设是 Hugging Face Space `Tongyi-MAI/Z-Image-Turbo` 的 `pe.py`
里 `prompt_template` 的**逐字原文**（中文），没有改写、没有精简。
`tests/test_presets.py` 里用 sha256 锁住了它，改动会立刻被测试发现。

英文版 `official (translated)` 是本项目对该原文的翻译，标签里明确标注，
不冒充官方。

顺带说明：官方 Space 的改写走的是云端 `qwen3-max-preview`，而且它 UI 里的
增强开关目前是注释掉的、`generate(enhance=False)` 默认关闭。所以本地 4B 的
结果请按「能用」来预期，不要按「等效官方」来预期。

---

## 语言策略

- `auto`：输入里汉字占比 > 15% 判为中文，否则英文。
- `official` 预设自带中英两份，切换语言时直接换模板。
- 其余预设保持原文，在末尾追加语言指令。
- 输出语种不符时会**重试一次**（追加更强的语言指令），二次仍不符则回退原文 ——
  宁可不出改写，也不交付语种错的改写。

---

## 性能与显存

文本编码器本来就为编码常驻显存，所以本扩展的**增量**开销很小：

- KV 缓存：约 100 MB（36 层 × 8 KV 头 × 128 维 × 704 容量 × fp16）
- fp8 编码器：一次性 cast 约 780 MB 瞬时

解码速度由显存带宽决定，4B fp16 大致：

| 显卡档位 | 速度 | 160 token 耗时 |
|---|---|---|
| 3060 级 | 20–40 tok/s | 4–8 s |
| 4090 / 5090 级 | 80–120 tok/s | 1.5–2 s |

若诊断显示 `SAFE MODE`，说明 KV 探测失败，速度约为快路径的 1/20，
此时建议把最大生成 token 降到 96 以下。

---

## 排错

诊断面板会给出六类明确结论，每类都附一行 `fix:`：

| 结论 | 含义 | 处理 |
|---|---|---|
| `DISABLED / 不是 Z-Image` | 当前模型不是 Z-Image | 换 checkpoint；本扩展不做跨模型适配 |
| `DISABLED / 缺少 text_processing_engine_qwen` | Forge 版本不匹配 | 升级/回退 Forge |
| `DISABLED / 词表投影不可用` | GGUF 或 QuantizedTensor 权重 | 换 fp16/bf16 的 `qwen_3_4b`，或在设置里允许一次性反量化 |
| `SAFE MODE` | KV 探测失败 | 可用但慢；降低 token 上限 |
| `... / LOW QUALITY (no chat template)` | tokenizer 没有 chat template，疑似 base 变体 | 换 instruct 版 TE，或在设置里关闭告警 |
| `改写结果被其它脚本覆盖` | 执行顺序被别的扩展抢了 | 见下一节 |

### 和其它改写脚本一起用

`sd-dynamic-prompts`、wildcards 这类扩展也用 `process()` 钩子，而 `process()`
的执行顺序由 `Settings → Scripts` 里的排序决定（**不是**由 `sorting_priority`
决定，那个只影响 UI 面板顺序）。如果它们排在本扩展之后，会用自己的展开结果
覆盖掉改写结果。

本扩展在 `before_process_batch` 里做了一次一致性守卫：一旦发现结果被覆盖，
会在控制台打印 `[ZTPE] 改写结果被其它脚本覆盖`，不会静默失效。
解决办法是把本扩展拖到它们**之后**。

---

## 自检

三层，按需要的深度挑：

```bash
# 1. 纯文本逻辑，不需要 torch / WebUI
python tests/run_selftest.py

# 2. 无头 UI 构建，需要 gradio
<venv>/python.exe tests/ui_smoke.py

# 3. KV 路径数值等价，需要 torch + Forge backend（不需要 checkpoint）
<venv>/python.exe tests/torch_selftest.py
```

第 3 项会用 Forge 自己的 `Llama2_` 造一个 2 层小模型，对比三条路径：
`Llama2_.forward`（基准）、本扩展的全量前向、本扩展的增量解码。
三者最后一个位置的 logits 必须一致到 1e-4。这是在没有 Z-Image checkpoint
的情况下能拿到的最强保证。

---

## 已知限制

1. **质量上限**：官方用 `qwen3-max-preview`，本地 4B 明显更弱。
2. **深度耦合内部结构**：KV 路径依赖 `Llama2_` / `Attention` / `TransformerBlock`
   的字段与签名。Forge 升级后靠功能探测自动降级，不会崩，但可能变慢。
3. **量化编码器**：fp8 可用；GGUF / QuantizedTensor 默认不可用。
4. **base 变体**：指令跟随弱，改写质量差，面板会告警。
5. **跨 GPU 复现**：采样用 CPU Generator，同种子可复现；但 logits 仍有
   ~1e-3 级差异。
6. **不改负向提示词**（与官方 PE 行为一致）。

失败时统一行为：打印一行原因 → 回退原文 → 正常出图，绝不中断生成。

---

## 许可

AGPL-3.0-or-later，与 Forge 一致。官方 PE 模板语义源自
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo)（Apache-2.0）。
