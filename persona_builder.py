"""人格采集器 · 语料 → 人格设定提示词。

这个模块负责两件事：

1. :func:`build_corpus`  —— 把原始聊天记录整理成适合喂给大模型的语料文本。
   大段无关闲聊会被裁掉，重复的口头禅会被统计出来，但**不删原文**，
   让模型自己判断哪些是特征。

2. :func:`build_persona_prompt` —— 构造让大模型生成人格设定的提示词。
   输出要求是一段可以直接当 system prompt 用的第三人称人格描述。

之所以把「统计」和「生成」分开：统计是纯 Python 逻辑，可以离线单测；
生成只依赖大模型，出错时容易定位。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

try:  # 作为插件包加载
    from .storage import Message
except ImportError:  # 独立运行单元测试时
    from storage import Message

# 一轮人格生成最多喂给模型多少条消息。
MAX_CORPUS_MESSAGES = 800
# 语料里保留的最大字符数（粗略按中文 1 字 1 token 估，留出 prompt 余量）。
MAX_CORPUS_CHARS = 60_000
# 判定「这条消息对人格没贡献」的长度阈值。
MIN_USEFUL_CHARS = 2


@dataclass
class CorpusStats:
    """语料统计，用于喂给模型做参考，也用于生成报告。"""

    total_messages: int
    """参与统计的消息总数。"""
    used_messages: int
    """实际进入语料的条数。"""
    total_chars: int
    sessions: int
    first_at: float
    last_at: float
    avg_len: float
    top_phrases: list[tuple[str, int]]
    """高频短语 / 口头禅（按出现次数降序）。"""
    sample_dialogs: list[tuple[str, str]]
    """少量代表性对话（发送者 → 文本），给模型做语感参考。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "used_messages": self.used_messages,
            "total_chars": self.total_chars,
            "sessions": self.sessions,
            "first_at": self.first_at,
            "last_at": self.last_at,
            "avg_len": self.avg_len,
            "top_phrases": [{"text": t, "count": c} for t, c in self.top_phrases],
            "sample_dialogs": [
                {"speaker": s, "text": t} for s, t in self.sample_dialogs
            ],
        }


# 常见的中文口头禅 / 语气词，用于统计说话风格。
_PHRASE_PATTERNS = [
    "哈哈",
    "哈哈哈",
    "hh",
    "233",
    "草",
    "笑死",
    "绝了",
    "离谱",
    "真的假的",
    "我靠",
    "我天",
    "救命",
    "qwq",
    "awa",
    "www",
    "呜",
    "额",
    "嗯",
    "好的",
    "行吧",
    "可以的",
    "没问题",
    "收到",
    "懂了",
    "然后",
    "就是",
    "其实",
    "感觉",
    "觉得",
    "好像",
    "可能",
    "应该",
    "对了",
    "顺便",
    "话说",
    "反正",
    "总之",
    "老子",
    "我超",
    "服了",
    "6",
    "666",
    "好家伙",
    "绷不住了",
    "泪目",
    "破防",
    "爱了",
    "牛",
    "厉害",
    "漂亮",
    "感谢",
    "谢谢",
    "晚安",
    "早安",
    "在吗",
    "咋",
    "咋样",
    "咋办",
    "咋整",
    "整",
    "得",
    "的",
    "了",
    "啊",
    "呀",
    "吧",
    "呢",
    "嘛",
    "哦",
    "噢",
    "唔",
]

# 判定「消息里主要是链接 / 表情 / 无意义内容」的正则。
_NOISE_PATTERNS = [
    re.compile(r"^\s*https?://\S+\s*$"),
    re.compile(r"^\s*\[.{0,10}(图片|表情|动画表情|语音|文件|视频).{0,10}\]\s*$"),
    re.compile(r"^[\s\W_]+$"),
]

# 归一化：把长串数字 / 字母折叠，避免「2024年」和「2025年」被当成两个特征。
_NORMALIZE_PATTERNS = [
    (re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"), "<日期>"),
    (re.compile(r"\d{1,2}:\d{2}(:\d{2})?"), "<时间>"),
    (re.compile(r"\d+"), "<数字>"),
    (re.compile(r"https?://\S+"), "<链接>"),
    (re.compile(r"@[\w一-鿿]+"), "@某人"),
    (re.compile(r"\s+"), " "),
]


def _is_noise(text: str) -> bool:
    """判断消息是否几乎不含有效信息。"""
    stripped = text.strip()
    if len(stripped) < MIN_USEFUL_CHARS:
        return True
    return any(p.match(stripped) for p in _NOISE_PATTERNS)


def _normalize(text: str) -> str:
    """把文本归一化，用于口头禅统计。"""
    out = text
    for pattern, repl in _NORMALIZE_PATTERNS:
        out = pattern.sub(repl, out)
    return out.strip()


def _extract_phrases(messages: list[Message]) -> list[tuple[str, int]]:
    """统计高频短语 / 语气词。"""
    counter: Counter[str] = Counter()
    for msg in messages:
        text = msg.text
        for phrase in _PHRASE_PATTERNS:
            count = text.count(phrase)
            if count:
                counter[phrase] += count
    # 过滤掉只出现一次且本身很短的，避免噪声
    items = [
        (phrase, count)
        for phrase, count in counter.items()
        if count >= 2 or len(phrase) >= 3
    ]
    items.sort(key=lambda item: (-item[1], -len(item[0])))
    return items[:30]


def _pick_samples(messages: list[Message], limit: int = 30) -> list[tuple[str, str]]:
    """挑一些有代表性的消息做样例。

    挑法：优先挑长度中位数附近的、有实质内容的句子，
    这样模型看到的是「典型发言」而不是一连串「嗯」「好」。
    """
    candidates = [m for m in messages if not _is_noise(m.text)]
    if not candidates:
        return []
    lengths = sorted(len(m.text) for m in candidates)
    median = lengths[len(lengths) // 2]

    candidates.sort(key=lambda m: abs(len(m.text) - median))
    chosen: list[tuple[str, str]] = []
    seen: set[str] = set()
    for msg in candidates:
        key = _normalize(msg.text)
        if not key or key in seen:
            continue
        seen.add(key)
        chosen.append((msg.sender_name or msg.sender_id, msg.text.strip()))
        if len(chosen) >= limit:
            break
    return chosen


def build_corpus(
    messages: list[Message],
    max_messages: int = MAX_CORPUS_MESSAGES,
    max_chars: int = MAX_CORPUS_CHARS,
) -> tuple[str, CorpusStats]:
    """把消息列表整理成语料文本 + 统计信息。

    Args:
        messages: 按时间正序的消息。
        max_messages: 最多使用多少条。
        max_chars: 语料最大字符数。

    Returns:
        ``(corpus_text, stats)``。
    """
    total = len(messages)
    if total == 0:
        empty = CorpusStats(0, 0, 0, 0, 0.0, 0.0, 0.0, [], [])
        return "", empty

    # 丢掉噪声，保留其余
    useful = [m for m in messages if not _is_noise(m.text)]
    # 太长的话优先保留最近的部分（最新消息更能反映当前说话方式），
    # 但前面也要留一些，所以从两头取。
    if len(useful) > max_messages:
        head = useful[: max_messages // 3]
        tail = useful[-(max_messages - len(head)) :]
        useful = head + tail

    lines: list[str] = []
    used_chars = 0
    used = 0
    for msg in useful:
        name = (msg.sender_name or msg.sender_id or "?").replace("\n", " ")
        text = msg.text.strip().replace("\n", " ")
        if not text:
            continue
        line = f"{name}：{text}"
        if used_chars + len(line) > max_chars:
            break
        lines.append(line)
        used_chars += len(line)
        used += 1

    corpus = "\n".join(lines)
    sessions = len({m.session_id for m in messages})
    first_at = min(m.created_at for m in messages)
    last_at = max(m.created_at for m in messages)
    avg_len = (sum(len(m.text) for m in messages) / total) if total else 0.0

    stats = CorpusStats(
        total_messages=total,
        used_messages=used,
        total_chars=sum(len(m.text) for m in messages),
        sessions=sessions,
        first_at=first_at,
        last_at=last_at,
        avg_len=round(avg_len, 1),
        top_phrases=_extract_phrases(messages),
        sample_dialogs=_pick_samples(messages),
    )
    return corpus, stats


def build_persona_prompt(
    corpus: str,
    stats: CorpusStats,
    target_qq: str,
    nickname: str = "",
) -> str:
    """构造「让大模型写人格设定」的提示词。

    Args:
        corpus: 语料正文。
        stats: 语料统计。
        target_qq: 被采集者 QQ 号。
        nickname: 被采集者的昵称（可空）。

    Returns:
        完整的提示词。
    """
    name = nickname or f"QQ {target_qq}"
    phrase_lines = "\n".join(
        f"  - 「{text}」出现 {count} 次" for text, count in stats.top_phrases
    ) or "  - （无明显口头禅）"
    sample_lines = "\n".join(
        f"  - {speaker}：{text}" for speaker, text in stats.sample_dialogs
    ) or "  - （无）"

    return f"""下面是一段真实聊天记录语料，全部来自同一个人（昵称：{name}，QQ 号：{target_qq}）。

## 语料统计
- 消息条数：{stats.total_messages} 条（本次分析使用 {stats.used_messages} 条）
- 覆盖会话数：{stats.sessions} 个（群聊 + 私聊）
- 消息平均长度：{stats.avg_len} 字
- 出现最多的语气词 / 口头禅：
{phrase_lines}

## 典型发言样例
{sample_lines}

## 完整语料
{corpus}

---

请基于以上语料，为这个人写一份**人格设定提示词（system prompt）**，要求：

1. 用**第三人称**描述这个人（写「他/她」或直接用名字，不要写「你」），因为这份提示词会用于扮演该人物。
2. 覆盖以下维度（有语料依据的才写，没依据的不要编）：
   - 身份与背景（自述过的工作、身份、经历）
   - 性格特征（脾气、说话风格、内在 / 外在）
   - 说话方式（语气、口头禅、常用句式、emoji 习惯、标点习惯、是否爱用网络用语）
   - 兴趣爱好与关注点
   - 价值观 / 立场 / 忌讳
   - 情绪模式与反应习惯
3. 尽量**引用语料里的原话**作为例证（用引号标出），让设定有据可依。
4. 语气上要写成**可直接使用的指令式设定**，像给 AI 的行为准则，而不是人物小传散文。
5. 如果某个维度语料里没有体现，就不要写那个维度。
6. 输出**纯提示词正文**，不要写「以下是人格设定」这类开场白，不要用 Markdown 代码块包裹，不要输出解释性后记。
7. 长度控制在 600–1200 字。

直接开始输出人格设定提示词。"""
