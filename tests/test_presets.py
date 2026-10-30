# -*- coding: utf-8 -*-
"""预设模板自检。

`OFFICIAL_ZH_SHA256` 是官方原文的快照。它一旦变化就说明有人改动了"官方"文本 ——
这正是需要人工确认的事，所以用哈希锁住，而不是靠肉眼比对。
"""

import hashlib

from lib_ztpe import presets

#: 取自 HF Space Tongyi-MAI/Z-Image-Turbo 的 pe.py prompt_template（去首尾换行）。
OFFICIAL_ZH_SHA256 = "2e82e37d19a33699e9c1b87ec324fb69eb203355700601afe1b0bbc30864eb22"


def run(check):
    check(
        "官方中文原文与快照一致",
        hashlib.sha256(presets.OFFICIAL_ZH.encode("utf-8")).hexdigest() == OFFICIAL_ZH_SHA256,
        "有人改动了 OFFICIAL_ZH，请确认是否为有意修改后更新快照",
    )

    check("官方原文含人格化开头", presets.OFFICIAL_ZH.startswith("你是一位被关在逻辑牢笼里的幻视艺术家"))
    check("官方原文以占位符结尾", presets.OFFICIAL_ZH.rstrip().endswith("用户输入 prompt: {prompt}"))
    check("官方英译存在且不同", presets.OFFICIAL_EN and presets.OFFICIAL_EN != presets.OFFICIAL_ZH)

    check("每个预设都含 {prompt}", all("{prompt}" in text for text in presets.PRESETS.values()))

    hashes = {key: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for key, text in presets.PRESETS.items()}
    check("预设哈希两两不同", len(set(hashes.values())) == len(hashes))
    check("预设哈希前 16 位也两两不同",
          len({value[:16] for value in hashes.values()}) == len(hashes),
          "旧版 template[:64] 的碰撞回归")

    check("render 替换占位符", presets.render(presets.LIGHT, "一只猫").endswith("一只猫"))
    check("render 无占位符时追加", "一只猫" in presets.render("无占位模板", "一只猫"))
    check("render 不误伤用户花括号",
          presets.render(presets.LIGHT, "a {cat} on {mat}").count("{cat}") == 1)

    check("official 切 en 用英译", presets.for_language("official", "en") == presets.OFFICIAL_EN)
    check("official 切 zh 用原文", presets.for_language("official", "zh") == presets.OFFICIAL_ZH)
    check("非官方预设追加英文指令",
          presets.for_language("light", "en").endswith(presets.LANG_DIRECTIVE["en"]))
    check("未知预设回落官方", presets.for_language("nope", "zh") == presets.OFFICIAL_ZH)

    check("choices 覆盖全部预设", len(presets.choices()) == len(presets.PRESETS))
    check("默认预设在注册表内", presets.DEFAULT_PRESET in presets.PRESETS)
