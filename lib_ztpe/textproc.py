# -*- coding: utf-8 -*-
"""提示词处理：模板渲染、标签保护、输出清洗、退化检测、语种判定。

纯 Python，不依赖 torch / modules.*，可脱离 WebUI 单测。
"""

import json
import re

# ---------------------------------------------------------------- 语种

_CJK = re.compile(r"[\u4e00-\u9fff]")


def cjk_ratio(text):
    """汉字占非空白字符的比例。"""
    if not text:
        return 0.0
    total = sum(1 for c in text if not c.isspace())
    if total == 0:
        return 0.0
    return len(_CJK.findall(text)) / total


def resolve_lang(policy, prompt):
    """auto 时按输入语种决定输出语种。"""
    if policy in ("zh", "en"):
        return policy
    return "zh" if cjk_ratio(prompt) > 0.15 else "en"


# ---------------------------------------------------------------- Chat 构造

def build_messages(template, prompt, layout="official"):
    """返回 (system, user)。

    官方 Space 的做法是：模板（含 {prompt} 插值）作为 system 消息，
    user 轮为单个空格。`layout` 可切到其它布局用于排查。

    - official：system 承载模板，user 留空
    - user：无 system 消息，模板整体进 user 轮
    - parity：与 parity 相同，但调用方应改用编码路径的 llama_template 框法
    """
    rendered = template.replace("{prompt}", prompt) if "{prompt}" in template else \
        f"{template.rstrip()}\n\n用户输入 prompt: {prompt}"

    if layout == "user" or layout == "parity":
        return None, rendered

    if "{prompt}" in template:
        return rendered, " "
    return template, prompt


# ---------------------------------------------------------------- 标签保护

#: A1111 / Forge 的 extra-network 标签与提示词编辑语法。这些不是画面内容，
#: 交给 LLM 会被改写甚至删除，必须在改写前替换成惰性哨兵、改写后原样还原。
PROTECT_RE = re.compile(
    r"<(?:lora|lyco|locon|loha|dora|hypernet|embedding|ti|network|prompt|controlnet)"
    r"[:\-][^<>\n]{1,512}>"
    r"|\[[^\[\]\n]{1,200}:[^\[\]\n]{1,200}(?::[^\[\]\n]{1,60})?\]",
    re.IGNORECASE,
)

#: 哨兵正文刻意用非数字前缀。旧写法是纯编号 ⟦0⟧，一旦模型丢了方括号只剩 "0"，
#: 宽松匹配会把提示词里任何一处 "0"（例如 <lora:cat:0.8> 的权重、或 "0.8 秒"）
#: 当成哨兵就地替换成整个标签。带上 ZTPE 前缀后这种误命中基本不可能。
_SENTINEL = "\u27e6ZTPE{}\u27e7"


def protect_tags(prompt):
    """把标签换成哨兵，返回 (新文本, {哨兵: 原文})。"""
    table = {}

    def _repl(match):
        key = _SENTINEL.format(len(table))
        table[key] = match.group(0)
        return key

    return PROTECT_RE.sub(_repl, prompt), table


def restore_tags(text, table):
    """三趟还原，返回 (新文本, 缺失列表)。

    1. 哨兵完整存在 → 直接替换；
    2. 模型保留了编号但丢了方括号 → 宽松匹配后就地替换；
    3. 都没找到 → 追加到末尾。A1111 的 extra networks 与位置无关
       （modules/extra_networks.py 按标签抽取），追加语义等价、无损。
    """
    missing = []
    for sentinel, original in table.items():
        if sentinel in text:
            text = text.replace(sentinel, original)
            continue

        # 数字边界不能少：哨兵正文是 ZTPE1、ZTPE10…，纯前缀匹配会让
        # ZTPE1 命中 ZTPE10（标签数 ≥ 11 时必现）。
        loose = re.search(r"(?<!\d)" + re.escape(sentinel[1:-1]) + r"(?!\d)",
                          text, re.IGNORECASE)
        if loose is not None:
            text = text[:loose.start()] + original + text[loose.end():]
            continue

        missing.append(original)

    if missing:
        text = text.rstrip(" ,，") + " " + " ".join(missing)

    return text, missing


def tags_intact(text, table):
    """还原后置条件：每个原始标签恰好出现一次。"""
    return all(text.count(original) == 1 for original in table.values())


# ---------------------------------------------------------------- 清洗

_THINK_CLOSED = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_IM = re.compile(r"<\|im_(?:start|end)\|>|<\|endoftext\|>")

#: 只删「整行就是角色名」的情况。旧版用 \n*(assistant|user|system)\n* 无锚定，
#: 会把 "solar system"、"operating system" 里的普通单词一起吃掉。
_ROLE_LINE = re.compile(r"(?im)^[ \t]*(?:assistant|user|system)[ \t]*:?[ \t]*$")

_FENCE_HEAD = re.compile(r"^\s*```[a-zA-Z0-9_-]*[ \t]*\r?\n?")
_FENCE_TAIL = re.compile(r"\r?\n?[ \t]*```\s*$")

_JSON_REVISION = re.compile(r'"revised_prompt"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)

_PREFIXES = (
    "改写后的prompt：", "改写后的prompt:", "改写后的提示词：", "改写后的提示词:",
    "最终prompt：", "最终prompt:", "最终描述：", "最终描述:",
    "修改后的prompt：", "修改后的prompt:",
    "输出：", "输出:",
    "final prompt:", "revised prompt:", "rewritten prompt:", "enhanced prompt:",
    "output:", "prompt:",
)

_QUOTE_PAIRS = (
    ('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u300c", "\u300d"),
    ("\u300e", "\u300f"), ("`", "`"),
)


def _maybe_json(text):
    """官方 app.py 会找 ```json 里的 revised_prompt；这里做同样的兜底。"""
    if "revised_prompt" not in text:
        return text
    match = _JSON_REVISION.search(text)
    if match is None:
        return text
    try:
        return json.loads('"' + match.group(1) + '"')
    except Exception:
        return match.group(1)


def clean(text):
    """剥离 think 块、特殊 token、角色标记、代码围栏、前导标签、包裹引号。"""
    if not text:
        return ""

    text = _THINK_CLOSED.sub("", text)
    text = _THINK_OPEN.sub("", text)
    text = _IM.sub("", text)
    text = _maybe_json(text)
    text = _ROLE_LINE.sub("", text)
    text = text.strip()

    if text.startswith("```"):
        text = _FENCE_HEAD.sub("", text)
        text = _FENCE_TAIL.sub("", text)
        text = text.strip()

    lowered = text.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    if len(text) > 2:
        for left, right in _QUOTE_PAIRS:
            if text[0] == left and text[-1] == right:
                # 只有这对引号确实是唯一的一对时才剥。官方模板要求画面里的文字
                # 用英文双引号包裹，结果里本来就可能有 "HELLO" 这种内层引号。
                if left == right:
                    strip = text.count(left) == 2
                else:
                    strip = text.count(left) == 1 and text.count(right) == 1
                if strip:
                    text = text[1:-1].strip()
                break

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- meta 标签

META_TAGS = frozenset({
    "8k", "4k", "16k", "masterpiece", "best quality", "high quality",
    "ultra detailed", "ultra-detailed", "highly detailed", "trending on artstation",
    "award winning", "award-winning",
    "杰作", "最佳画质", "高质量", "超精细", "超高清", "大师作品",
})


def strip_meta_tags(text):
    """按逗号切分并剔除纯 meta 片段，保留原有分隔符（全角/半角不互转）。"""
    if not text:
        return ""

    pieces = re.split(r"([,\uff0c])", text)
    kept = []
    for i in range(0, len(pieces), 2):
        segment = pieces[i]
        separator = pieces[i + 1] if i + 1 < len(pieces) else ""
        if segment.strip().lower().strip(".。 ") in META_TAGS:
            continue
        kept.append((segment, separator))

    if not kept:
        return ""

    out = []
    for index, (segment, separator) in enumerate(kept):
        out.append(segment)
        if index < len(kept) - 1:
            # 沿用该片段原本的分隔符，全角逗号不会被改写成半角。
            out.append(separator or ",")

    return "".join(out).strip()


# ---------------------------------------------------------------- 退化检测

_META_LEAK = re.compile(
    # 英文短语必须加词边界：不加的话 "here is" 会命中 "T(here is)"，
    # "There is a red sign…" 这种极常见的英文改写会被整条丢弃。
    r"(?i)(\bas an ai\b|\bi cannot\b|\bi'm sorry\b|\bi am sorry\b|\bhere is\b|\bhere's the\b)"
    r"|用户输入|以下是|抱歉|作为一个ai",
)


def check_output(original, result, lang):
    """返回 None 表示可用，否则返回原因代码。"""
    if not result:
        return "empty"

    # 阈值取 8 而非 16：极短输入（如"一只猫"）的合理改写也可能只有十来个字。
    if len(result) < max(8, int(0.25 * len(original))):
        return "too_short"

    if not re.search(r"[\u4e00-\u9fffA-Za-z]", result):
        return "no_letters"

    if re.search(r"(.{2,24}?)\1{4,}", result):
        return "repetition"

    if _META_LEAK.search(result):
        return "meta_leak"

    if "\u27e6" in result or "\u27e7" in result:
        return "sentinel_leak"

    if lang == "zh" and cjk_ratio(result) < 0.08:
        return "wrong_language"

    if lang == "en" and cjk_ratio(result) > 0.35:
        return "wrong_language"

    return None


REASON_TEXT = {
    "empty": "模型没有输出任何内容",
    "too_short": "改写结果过短",
    "no_letters": "改写结果不含任何文字",
    "repetition": "改写结果出现复读",
    "meta_leak": "改写结果泄漏了元话语",
    "sentinel_leak": "改写结果残留了内部哨兵",
    "wrong_language": "改写结果语种不符",
}
