"""人格采集器 · 单元测试（不依赖 AstrBot 运行时）。

运行：
    python3 -m pytest tests/ -v
或：
    python3 tests/test_persona_harvester.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persona_builder import (  # noqa: E402
    build_corpus,
    build_persona_prompt,
)
from storage import Message, Persona, Storage  # noqa: E402
from text_utils import (  # noqa: E402
    extract_text,
    make_summary,
    split_qq_list,
    strip_command_prefix,
    strip_wrapper,
)


def make_msg(
    idx: int,
    text: str,
    sender: str = "10001",
    name: str = "小明",
    group: str = "20002",
    ts: float = 1_700_000_000.0,
) -> Message:
    return Message(
        id=idx,
        sender_id=sender,
        sender_name=name,
        group_id=group,
        session_id=group,
        is_private=False,
        message_id=f"m{idx}",
        text=text,
        created_at=ts + idx,
    )


# ----------------------------------------------------------------------
# 存储层
# ----------------------------------------------------------------------
def test_storage_add_and_dedupe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp))
        assert asyncio.run(storage.add_message(make_msg(1, "你好啊朋友"))) is True
        # 相同 message_id 不重复入库
        assert asyncio.run(storage.add_message(make_msg(1, "你好啊朋友"))) is False
        assert asyncio.run(storage.count_messages()) == 1
        storage.close()


def test_storage_rejects_empty_and_truncates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp))
        assert asyncio.run(storage.add_message(make_msg(1, "   "))) is False
        long_text = "字" * 5000
        assert asyncio.run(storage.add_message(make_msg(2, long_text))) is True
        msgs = asyncio.run(storage.get_all_messages())
        assert len(msgs[0].text) <= 4001
        storage.close()


def test_storage_filter_and_stats() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp))
        asyncio.run(storage.add_message(make_msg(1, "第一条", sender="111")))
        asyncio.run(storage.add_message(make_msg(2, "第二条", sender="222")))
        asyncio.run(storage.add_message(make_msg(3, "第三条", sender="111")))

        assert asyncio.run(storage.count_messages()) == 3
        assert asyncio.run(storage.count_messages("111")) == 2

        msgs = asyncio.run(storage.get_messages(sender_id="111", order="old"))
        assert [m.text for m in msgs] == ["第一条", "第三条"]

        stats = asyncio.run(storage.get_stats("111"))
        assert stats["count"] == 2
        assert stats["sessions"] == 1
        # 「第一条」3 字 + 「第三条」3 字
        assert stats["chars"] == 6

        assert asyncio.run(storage.delete_messages(sender_id="111")) == 2
        assert asyncio.run(storage.count_messages()) == 1
        storage.close()


def test_storage_persists_across_reopen() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp))
        asyncio.run(storage.add_message(make_msg(1, "持久化测试")))
        storage.close()

        storage2 = Storage(Path(tmp))
        assert asyncio.run(storage2.count_messages()) == 1
        storage2.close()


def test_persona_versions_are_incremented() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = Storage(Path(tmp))
        for i in range(3):
            asyncio.run(
                storage.save_persona(
                    Persona(
                        id=0,
                        persona_id="harvest_10001",
                        target_qq="10001",
                        system_prompt=f"第 {i} 版",
                        summary="摘要",
                        stats={"total_messages": i},
                        created_at=0.0,
                    ),
                ),
            )
        personas = asyncio.run(storage.get_personas("10001"))
        assert len(personas) == 3
        # 最新的排在最前
        assert personas[0].version == 3
        assert personas[0].system_prompt == "第 2 版"
        latest = asyncio.run(storage.get_latest_persona("10001"))
        assert latest is not None
        assert latest.version == 3
        storage.close()


# ----------------------------------------------------------------------
# 语料构建
# ----------------------------------------------------------------------
def test_build_corpus_filters_noise() -> None:
    msgs = [
        make_msg(1, "这是一句有内容的话"),
        make_msg(2, "嗯"),
        make_msg(3, "https://example.com/very/long/link"),
        make_msg(4, "[图片]"),
        make_msg(5, "哈哈哈笑死我了"),
    ]
    corpus, stats = build_corpus(msgs)
    assert "这是一句有内容的话" in corpus
    assert "哈哈哈笑死我了" in corpus
    # 噪声不应进入语料
    assert "https://example.com" not in corpus
    assert "[图片]" not in corpus
    # 统计用的是全部消息
    assert stats.total_messages == 5
    # 「嗯」只有 1 个字、纯链接、纯表情标记都属于噪声，实际入语料的是 2 条
    assert stats.used_messages == 2


def test_build_corpus_keeps_both_ends_when_truncating() -> None:
    msgs = [make_msg(i, f"消息{i}内容") for i in range(100)]
    corpus, stats = build_corpus(msgs, max_messages=30, max_chars=100000)
    assert stats.used_messages == 30
    # 头部和尾部都要保留
    assert "消息0内容" in corpus
    assert "消息99内容" in corpus
    # 中间被丢掉了
    assert "消息50内容" not in corpus


def test_build_corpus_respects_char_limit() -> None:
    msgs = [make_msg(i, "字" * 50) for i in range(100)]
    corpus, stats = build_corpus(msgs, max_messages=1000, max_chars=500)
    assert len(corpus) <= 501
    assert stats.used_messages < 100


def test_build_corpus_empty_input() -> None:
    corpus, stats = build_corpus([])
    assert corpus == ""
    assert stats.total_messages == 0


def test_phrase_extraction_counts_repeats() -> None:
    msgs = [make_msg(i, "哈哈哈这个真的绝了") for i in range(5)]
    _, stats = build_corpus(msgs)
    phrases = dict(stats.top_phrases)
    assert phrases.get("哈哈哈") == 5
    assert phrases.get("绝了") == 5


def test_sample_dialogs_dedupe() -> None:
    msgs = [make_msg(i, "完全一样的一句话") for i in range(10)]
    _, stats = build_corpus(msgs)
    assert len(stats.sample_dialogs) == 1


# ----------------------------------------------------------------------
# 提示词
# ----------------------------------------------------------------------
def test_persona_prompt_contains_required_sections() -> None:
    msgs = [make_msg(i, f"第{i}句话，语气词哈哈哈") for i in range(20)]
    corpus, stats = build_corpus(msgs)
    prompt = build_persona_prompt(corpus, stats, "10001", "小明")

    assert "10001" in prompt
    assert "小明" in prompt
    assert "第三人称" in prompt
    assert "语料统计" in prompt
    assert "完整语料" in prompt
    # 语料正文要真的塞进去
    assert "第0句话" in prompt


def test_persona_prompt_without_nickname() -> None:
    msgs = [make_msg(1, "测试内容")]
    corpus, stats = build_corpus(msgs)
    prompt = build_persona_prompt(corpus, stats, "99999", "")
    assert "QQ 99999" in prompt


# ----------------------------------------------------------------------
# main.py 里的纯函数
# ----------------------------------------------------------------------
def test_extract_text_from_segments() -> None:
    assert extract_text("纯字符串") == "纯字符串"
    assert extract_text([{"type": "text", "data": {"text": "你好"}}]) == "你好"
    assert (
        extract_text(
            [
                {"type": "text", "data": {"text": "你好"}},
                {"type": "image", "data": {"file": "x.jpg"}},
                {"type": "text", "data": {"text": "世界"}},
            ],
        )
        == "你好世界"
    )
    assert extract_text(None) == ""
    assert extract_text([]) == ""


def test_strip_wrapper_removes_fences_and_preamble() -> None:
    assert strip_wrapper("```text\n人格正文\n```") == "人格正文"
    assert strip_wrapper("```\n人格正文\n```") == "人格正文"
    assert strip_wrapper("以下是人格设定提示词：\n人格正文") == "人格正文"
    assert strip_wrapper("人格正文") == "人格正文"


def test_make_summary_picks_first_line() -> None:
    assert make_summary("# 标题\n第二行") == "标题"
    assert make_summary("\n\n第一行\n第二行") == "第一行"
    assert make_summary("") == ""


def teststrip_command_prefix() -> None:
    assert strip_command_prefix("/人格采集 列表") == "列表"
    assert strip_command_prefix("人格采集 生成 123") == "生成 123"
    assert strip_command_prefix("persona_harvest 列表") == "列表"
    assert strip_command_prefix("/人格采集") == ""


def test_split_qq_list_handles_strings_and_lists() -> None:
    assert split_qq_list(["1", "2", "1"]) == ["1", "2"]
    assert split_qq_list("1,2，3") == ["1", "2", "3"]
    assert split_qq_list(None) == []
    assert split_qq_list(123) == []


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
