"""人格采集器 · 与 AstrBot 运行时无关的纯函数。

把这些函数单独拆出来，好处是：
- 单元测试不需要 AstrBot 环境（iSH / CI 上都能跑）；
- 逻辑出错时能立刻定位，不用去翻插件框架代码。
"""

from __future__ import annotations

from typing import Any

# 指令名
CMD = "人格采集"
# 指令别名（含带斜杠的常见写法）
CMD_ALIASES = {
    "/人格采集",
    "人设采集",
    "/人设采集",
    "persona_harvest",
    "/persona_harvest",
}


def split_qq_list(raw: Any) -> list[str]:
    """把配置里的 QQ 列表解析成去重后的字符串列表。

    支持 list，也支持用逗号 / 顿号 / 分号 / 空格分隔的字符串。
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.replace("，", ",").replace("；", ",").replace(";", ",")
        raw = raw.replace(" ", ",").replace("\t", ",")
        raw = [part for part in raw.split(",") if part.strip()]
    if not isinstance(raw, (list, tuple, set)):
        return []
    result: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def strip_command_prefix(text: str) -> str:
    """去掉指令前缀，返回后面的参数部分。"""
    out = (text or "").strip()
    # 长的前缀优先，避免「人格采集」先匹配掉「/人格采集」的一部分
    for prefix in sorted({CMD, *CMD_ALIASES}, key=len, reverse=True):
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
            break
    return out.lstrip("/").strip()


def extract_text(raw: Any) -> str:
    """从 OneBot 消息段数组里提取纯文本。"""
    if isinstance(raw, str):
        return raw.strip()
    if not isinstance(raw, list):
        return ""
    parts: list[str] = []
    for seg in raw:
        if not isinstance(seg, dict):
            continue
        if seg.get("type") == "text":
            data = seg.get("data") or {}
            text = data.get("text") if isinstance(data, dict) else None
            if text:
                parts.append(str(text))
    return "".join(parts).strip()


def strip_wrapper(text: str) -> str:
    """剥掉模型可能可能加的 markdown 代码块围栏和前言。"""
    out = (text or "").strip()
    if out.startswith("```"):
        lines = out.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        out = "\n".join(lines).strip()
    for prefix in (
        "以下是人设提示词：",
        "以下是人格设定提示词：",
        "人格设定提示词：",
        "```text",
    ):
        if out.startswith(prefix):
            out = out[len(prefix) :].strip()
    return out


def make_summary(system_prompt: str) -> str:
    """从人格正文里取一行做摘要。"""
    for line in (system_prompt or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        if stripped:
            return stripped[:80]
    return (system_prompt or "")[:80]
