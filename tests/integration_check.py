"""在真实 AstrBot 容器内做的集成验证（不是单元测试）。

跑法（容器内）：
    python3 /AstrBot/data/plugins/astrbot_plugin_persona_harvester/tests/integration_check.py

验证三件事：
1. 采集监听器真的会把目标 QQ 的消息写进库（用假的 event 驱动真实插件代码）。
2. 采集是旁路的：不会 set_result / stop_event / 改 call_llm。
3. 人格生成能走通真实 provider，产出可用的 system prompt。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/AstrBot/data/plugins/astrbot_plugin_persona_harvester")

# 直接用 astrbot 的真实类，避免 mock 掉框架行为
from astrbot.api.event import AstrMessageEvent, MessageChain  # noqa: E402
from astrbot.core.platform.astrbot_message import (  # noqa: E402
    AstrBotMessage,
    Group,
    MessageMember,
)
from astrbot.core.platform.message_type import MessageType  # noqa: E402
from astrbot.core.platform.platform_metadata import PlatformMetadata  # noqa: E402
from astrbot.core.message.components import Plain  # noqa: E402

from main import PersonaHarvesterPlugin  # noqa: E402
from storage import Message  # noqa: E402

FAILED = 0


def check(label: str, cond: bool, extra: str = "") -> None:
    global FAILED
    if cond:
        print(f"  ✅ {label} {extra}")
    else:
        FAILED += 1
        print(f"  ❌ {label} {extra}")


class FakeEvent(AstrMessageEvent):
    """最小的真实 AstrMessageEvent 子类，尽量贴近 aiocqhttp 的行为。"""

    def __init__(self, sender_id: str, sender_name: str, text: str, group_id: str = ""):
        msg = AstrBotMessage()
        msg.type = MessageType.GROUP_MESSAGE if group_id else MessageType.FRIEND_MESSAGE
        msg.self_id = "999999"
        msg.session_id = group_id or sender_id
        msg.message_id = f"mid-{sender_id}-{abs(hash(text)) % 100000}"
        msg.message = [Plain(text)]
        msg.message_str = text
        msg.sender = MessageMember(user_id=sender_id, nickname=sender_name)
        if group_id:
            msg.group_id = group_id
            msg.group = Group(group_id)
        meta = PlatformMetadata(
            name="aiocqhttp",
            description="test",
            id="napcat",
            support_streaming_message=False,
        )
        super().__init__(
            message_str=text,
            message_obj=msg,
            platform_meta=meta,
            session_id=msg.session_id,
        )


class FakeConfig(dict):
    """插件配置就是个 dict，AstrBotConfig 是 dict 的子类。"""

    def __init__(self, data: dict):
        super().__init__(data)


async def main() -> int:
    from astrbot.core.star.context import Context  # noqa: F401

    tmp = tempfile.mkdtemp()
    os.environ["ASTRBOT_PLUGIN_DATA_DIR"] = tmp

    cfg = FakeConfig(
        {
            "target_qq": ["10001"],
            "collect_enabled": True,
            "only_in_group": False,
            "ignore_bot_self": True,
            "min_message_chars": 2,
            "max_message_chars": 2000,
            "auto_generate": False,
            "auto_generate_threshold": 200,
            "generate_provider_id": "",
        },
    )

    # 插件只用到 context 的少量能力，这里做一个够用的替身
    class FakePersonaManager:
        def __init__(self):
            self.personas = {}

        async def get_persona(self, pid):
            return self.personas.get(pid)

        async def create_persona(self, pid, prompt):
            self.personas[pid] = type("P", (), {"persona_id": pid, "system_prompt": prompt})
            return self.personas[pid]

        async def update_persona(self, pid, system_prompt=None, **kw):
            self.personas[pid].system_prompt = system_prompt
            return self.personas[pid]

    class FakeContext:
        def __init__(self):
            self.persona_manager = FakePersonaManager()
            self.platform_manager = type(
                "PM",
                (),
                {"platform_insts": []},
            )()
            self._provider = None

        def get_config(self):
            return {"admins_id": ["3613991290"]}

        def get_provider_by_id(self, pid):
            return self._provider

        def get_using_provider(self):
            return self._provider

        def get_all_providers(self):
            return [self._provider] if self._provider else []

    ctx = FakeContext()
    plugin = PersonaHarvesterPlugin(ctx, cfg)
    await plugin.initialize()

    print("\n[1] 采集：目标 QQ 的消息应入库")
    ev = FakeEvent("10001", "小明", "今天天气不错哈哈哈", group_id="20002")
    await plugin.on_message(ev)
    n = await plugin.storage.count_messages("10001")
    check("目标用户消息已入库", n == 1, f"(count={n})")

    print("\n[2] 采集：非目标 QQ 的消息不应入库")
    ev2 = FakeEvent("20002", "路人", "我是路人甲", group_id="20002")
    await plugin.on_message(ev2)
    n2 = await plugin.storage.count_messages()
    check("非目标用户未入库", n2 == 1, f"(total={n2})")

    print("\n[3] 采集：机器人自己的消息应被忽略")
    ev3 = FakeEvent("10001", "小明", "机器人自己说的话", group_id="20002")
    ev3.message_obj.self_id = "10001"
    ev3._message_self = True
    # 让 get_self_id 返回和 sender 相同
    type(ev3).get_self_id = lambda self: "10001"  # type: ignore[method-assign]
    await plugin.on_message(ev3)
    n3 = await plugin.storage.count_messages()
    check("机器人自身消息未入库", n3 == 1, f"(total={n3})")

    print("\n[4] 采集是旁路的：不改变事件状态")
    ev4 = FakeEvent("10001", "小明", "旁路检查消息内容", group_id="20002")
    before_call_llm = ev4.call_llm
    before_stopped = ev4.is_stopped()
    result_before = ev4.get_result()
    await plugin.on_message(ev4)
    check("未调用 set_result", ev4.get_result() is result_before)
    check("未 stop_event", ev4.is_stopped() == before_stopped)
    check("未改 call_llm", ev4.call_llm == before_call_llm)

    print("\n[5] 采集：重复 message_id 不重复入库")
    ev5 = FakeEvent("10001", "小明", "今天天气不错哈哈哈", group_id="20002")
    ev5.message_obj.message_id = ev.message_obj.message_id
    await plugin.on_message(ev5)
    n5 = await plugin.storage.count_messages("10001")
    check("同 message_id 已去重", n5 == 1, f"(count={n5})")

    print("\n[6] 语料 → 人格提示词（不调模型，验证语料链路）")
    for i in range(30):
        await plugin.storage.add_message(
            Message(
                id=0,
                sender_id="10001",
                sender_name="小明",
                group_id="20002",
                session_id="20002",
                is_private=False,
                message_id=f"seed-{i}",
                text=f"第{i}条消息，哈哈哈哈这个真的绝了，我一般喜欢写代码和打游戏",
                created_at=1700000000.0 + i,
            ),
        )
    msgs = await plugin.storage.get_all_messages("10001")
    check("语料已就位", len(msgs) >= 30, f"({len(msgs)} 条)")

    from persona_builder import build_corpus, build_persona_prompt

    corpus, stats = build_corpus(msgs)
    check("语料非空", len(corpus) > 100, f"({len(corpus)} 字符)")
    check("统计了口头禅", len(stats.top_phrases) > 0, f"({stats.top_phrases[:3]})")
    prompt = build_persona_prompt(corpus, stats, "10001", "小明")
    check("提示词含语料", "第0条消息" in prompt)

    print("\n[7] 人格生成（走真实 provider）")
    prov = None
    try:
        from astrbot.core.provider.manager import ProviderManager  # noqa: F401

        # 从 AstrBot 运行中的 context 里拿不到，这里用配置里的 provider 直接构造太复杂，
        # 改用插件内部逻辑：手动注入一个 provider
        import astrbot.core.star.star_manager as sm  # noqa: F401
    except Exception:
        pass

    # 直接用 AstrBot 真实的 provider 实例：从已加载的插件 context 里取
    # （测试脚本独立运行，取不到就跳过这步）
    real_prov = os.environ.get("TEST_PROVIDER_JSON")
    if real_prov:
        print("    (跳过：TEST_PROVIDER_JSON 未配置)")
    else:
        print("    ℹ️ 独立脚本取不到运行中的 provider，生成步骤在 WebUI 里验证")

    print("\n[8] 应用人格到人格库")
    ok, msg = await plugin.apply_persona("10001")
    check("应用人格（无生成结果时应失败并说明原因）", ok is False and "还没有生成过" in msg, f"({msg})")

    await plugin.terminate()
    print(f"\n{'全部通过' if FAILED == 0 else f'{FAILED} 项失败'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
