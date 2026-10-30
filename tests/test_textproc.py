# -*- coding: utf-8 -*-
"""文本处理自检。重点是旧版踩过的坑：吃掉 "system" 一词。"""

from lib_ztpe import textproc as tp


def run(check):
    # ---------------------------------------------------------- 清洗
    check(
        "回归：句中 system 不被删除",
        tp.clean("a photo of the solar system in space") == "a photo of the solar system in space",
        repr(tp.clean("a photo of the solar system in space")),
    )
    check(
        "回归：operating system 不被删除",
        tp.clean("an operating system ui mockup") == "an operating system ui mockup",
    )
    check(
        "整行角色标记被删",
        tp.clean("一只猫\nassistant\n在窗台") == "一只猫 在窗台",
    )
    check(
        "think 块被删",
        tp.clean("<think>hmm</think>改写后的prompt：一只猫") == "一只猫",
    )
    check("未闭合 think 整段丢弃", tp.clean("<think>没写完") == "")
    check("im 特殊 token 被删", tp.clean("一只猫<|im_end|>") == "一只猫")
    check("代码围栏被剥", tp.clean("```\n一只猫\n```") == "一只猫")
    check("带语言标记的围栏被剥", tp.clean("```text\n一只猫\n```") == "一只猫")
    check("包裹引号被剥", tp.clean('"一只猫"') == "一只猫")
    check("中文引号被剥", tp.clean("\u201c一只猫\u201d") == "一只猫")
    check(
        "带内层引号时保留外层引号",
        tp.clean('"A" and "B"') == '"A" and "B"',
        repr(tp.clean('"A" and "B"')),
    )
    check(
        "画面文字引号不被破坏",
        tp.clean('"HELLO" 字样的招牌，白色背景') == '"HELLO" 字样的招牌，白色背景',
    )
    check("前导标签被剥", tp.clean("最终prompt：一只猫") == "一只猫")
    check("英文前导标签被剥", tp.clean("Final prompt: a cat") == "a cat")
    check("空白被压缩", tp.clean("一只猫\n\n   在   窗台") == "一只猫 在 窗台")
    check("空输入返回空", tp.clean("") == "" and tp.clean(None) == "")
    check(
        "JSON 包裹的 revised_prompt 被取出",
        tp.clean('```json\n{"revised_prompt": "a cat on a sill"}\n```') == "a cat on a sill",
    )

    # ---------------------------------------------------------- 标签保护
    original = "一只猫 <lora:cat:0.8> 在窗台 [from:to:0.5]"
    protected, table = tp.protect_tags(original)
    check("标签被替换成哨兵", "<lora:cat:0.8>" not in protected and len(table) == 2)
    restored, missing = tp.restore_tags(protected, table)
    check("哨兵完整时原样还原", restored == original and not missing)

    restored2, missing2 = tp.restore_tags("一只猫 在窗台", table)
    check("哨兵全丢失时追加到末尾", restored2.endswith("<lora:cat:0.8> [from:to:0.5]"))
    check("追加后标签仍完整", tp.tags_intact(restored2, table))

    loose_text = "一只猫 " + list(table.keys())[0][1:-1] + " 在窗台"
    restored3, missing3 = tp.restore_tags(loose_text, table)
    check("括号丢失但编号残留时就地还原", "<lora:cat:0.8>" in restored3)
    check("宽松还原不会重复追加该标签", tp.tags_intact(restored3, table))

    # 回归：哨兵正文若为纯数字，模型丢掉方括号后，正文里任何一处 "0"
    # （例如 LoRA 权重 0.8）都会被当成哨兵，把整个标签插进数字中间。
    table_num = tp.protect_tags("一只猫 <lora:cat:0.8> 在窗台")[1]
    restored_num, _ = tp.restore_tags("一只猫 0.8 在窗台", table_num)
    check("数字不会被误判为哨兵", restored_num == "一只猫 0.8 在窗台 <lora:cat:0.8>", restored_num)

    # 回归：哨兵正文是 ZTPE0、ZTPE1…ZTPE10，纯前缀匹配会让 ZTPE1 命中 ZTPE10。
    # 标签数 ≥ 11 时必现，所以这里造 11 个标签并把方括号全部去掉。
    many = " ".join(f"<lora:t{i}:0.5>" for i in range(11))
    protected_many, table_many = tp.protect_tags(many)
    loose_many = protected_many.replace("\u27e6", "").replace("\u27e7", "")
    restored_many, _ = tp.restore_tags(loose_many, table_many)
    check("11 个哨兵的前缀不互相干扰", tp.tags_intact(restored_many, table_many),
          restored_many[:80])

    no_tags, empty_table = tp.protect_tags("一只猫在窗台")
    check("无标签时表为空", not empty_table and no_tags == "一只猫在窗台")

    # ---------------------------------------------------------- meta 标签
    check(
        "meta 标签剔除且保留全角逗号",
        tp.strip_meta_tags("一只猫，8K，杰作，窗台") == "一只猫，窗台",
        repr(tp.strip_meta_tags("一只猫，8K，杰作，窗台")),
    )
    check("meta 标签剔除（半角）", tp.strip_meta_tags("a cat, 8k, masterpiece, a sill") == "a cat, a sill")
    check("全是 meta 时返回空", tp.strip_meta_tags("8K, masterpiece") == "")
    check("无 meta 时原样返回", tp.strip_meta_tags("一只猫，窗台") == "一只猫，窗台")

    # ---------------------------------------------------------- 语种
    check("cjk_ratio 全中文", tp.cjk_ratio("一只猫") == 1.0)
    check("cjk_ratio 全英文", tp.cjk_ratio("a cat") == 0.0)
    check("cjk_ratio 混合", abs(tp.cjk_ratio("中文abc") - 0.4) < 1e-9)
    check("cjk_ratio 空串", tp.cjk_ratio("") == 0.0)
    check("auto 中文输入判 zh", tp.resolve_lang("auto", "一只猫在窗台") == "zh")
    check("auto 英文输入判 en", tp.resolve_lang("auto", "a cat on a sill") == "en")
    check("显式 en 覆盖 auto", tp.resolve_lang("en", "一只猫") == "en")
    check("显式 zh 覆盖 auto", tp.resolve_lang("zh", "a cat") == "zh")

    # ---------------------------------------------------------- 退化检测
    check("空输出判退化", tp.check_output("一只猫", "", "zh") == "empty")
    check("过短判退化", tp.check_output("一只猫在窗台上晒太阳", "猫", "zh") == "too_short")
    check("无字母判退化", tp.check_output("一只猫", "1234 5678 90!!", "zh") == "no_letters")
    check("复读判退化", tp.check_output("猫", "一只猫一只猫一只猫一只猫一只猫", "zh") == "repetition")
    check("元话语泄漏判退化",
          tp.check_output("一只猫", "抱歉，我无法完成这个请求，请提供更多信息谢谢", "zh") == "meta_leak")
    check("哨兵泄漏判退化",
          tp.check_output("一只猫", "一只猫 \u27e60\u27e7 在窗台", "zh") == "sentinel_leak")
    check("语种不符判退化",
          tp.check_output("一只猫在窗台上晒太阳", "a cat sitting on a wooden windowsill", "zh") == "wrong_language")
    check("正常改写通过",
          tp.check_output("一只猫", "一只橘色短毛猫蹲坐在木质窗台上，侧光勾勒毛发边缘", "zh") is None)
    check("英文正常改写通过",
          tp.check_output("a cat", "a ginger shorthair cat sitting on a wooden windowsill, rim light", "en") is None)
    # 回归：_META_LEAK 里的 "here is" 没有词边界时会命中 "T(here is)"，
    # "There is …" 这种极常见的英文改写会被整条丢弃。
    check("There is 不误判为元话语",
          tp.check_output("a cat", "There is a red sign above the doorway", "en") is None)
    check("there's the 不误判",
          tp.check_output("a cat", "there's the old bridge in the distance", "en") is None)
    check("Where is 不误判",
          tp.check_output("a cat", "Where is the light coming from, soft shadows", "en") is None)
    check("真正的元话语仍被拦截",
          tp.check_output("a cat", "Here is the rewritten prompt you asked for", "en") == "meta_leak")

    # ---------------------------------------------------------- chat 构造
    system, user = tp.build_messages("指令 {prompt}", "一只猫", "official")
    check("official 布局：模板进 system", system == "指令 一只猫" and user == " ")

    system2, user2 = tp.build_messages("指令", "一只猫", "official")
    check("official 布局：无占位符时 user 承载提示词", system2 == "指令" and user2 == "一只猫")

    system3, user3 = tp.build_messages("指令 {prompt}", "一只猫", "user")
    check("user 布局：无 system", system3 is None and user3 == "指令 一只猫")
