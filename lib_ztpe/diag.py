# -*- coding: utf-8 -*-
"""诊断报告的纯文本渲染。

只吃一个 dict，不碰 torch，因此可以在没有 WebUI 的环境里单测排版。
运行期探测（locate / probe / decode）负责填 dict，这里只负责排版。
"""

_MARK = {
    "ok": "[OK]  ",
    "warn": "[WARN]",
    "fail": "[FAIL]",
    "skip": "[SKIP]",
}


def render(report):
    lines = [f"VERDICT: {report.get('verdict', 'UNKNOWN')}"]

    headline = report.get("headline")
    if headline:
        lines.append(f"  {headline}")

    for key, value in report.get("rows", []):
        lines.append(f"  {key:<11} {value}")

    for status, name, detail in report.get("checks", []):
        lines.append(f"  {_MARK.get(status, '[????]')} {name:<18} {detail}")

    for note in report.get("notes", []):
        lines.append(f"  {note}")

    fixes = report.get("fix") or []
    if fixes:
        lines.append("")
        for item in fixes:
            lines.append(f"  fix: {item}")

    return "\n".join(lines)


def disabled(reason, fix=None):
    return render({
        "verdict": "DISABLED",
        "headline": f"reason: {reason}",
        "fix": fix or [],
    })


def humanize_bytes(value):
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
