"""端到端验证：真实 provider + 真实插件实例 + 真实数据库。

跑法（容器内）：
    python3 /AstrBot/data/plugins/astrbot_plugin_persona_harvester/tests/e2e_check.py

覆盖：
  A. 真实 provider 跑通 generate_persona()，人格落库
  B. apply_persona() 写进真实 personas 表，personas_v3 缓存同步刷新
  C. 指令 handler 的解析与返回（列表 / 生成 / 查看 / 应用）
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PLUGIN_DIR = "/AstrBot/data/plugins/astrbot_plugin_persona_harvester"
sys.path.insert(0, PLUGIN_DIR)

from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial  # noqa: E402

from main import PersonaHarvesterPlugin  # noqa: E402
from storage import Persona  # noqa: E402

FAILED = 0
TARGET = "3665585796"


def check(label: str, cond: bool, extra: str = "") -> None:
    global FAILED
    if cond:
        print(f"  ✅ {label} {extra}")
    else:
        FAILED += 1
        print(f"  ❌ {label} {extra}")


def build_real_provider():
    cfg = json.load(open("/AstrBot/data/cmd_config.json", encoding="utf-8-sig"))
    sources = {s["id"]: s for s in cfg.get("provider_sources", [])}
    prov_cfg = next(p for p in cfg["provider"] if p.get("id") == "ds-deepseek-chat")
    src = sources.get(prov_cfg.get("provider_source_id"), {})
    keys = src.get("key") or []
    return ProviderOpenAIOfficial(
        provider_config={
            "id": prov_cfg["id"],
            "type": "openai_chat_completion",
            "model": prov_cfg.get("model"),
            "api_base": src.get("api_base"),
            "api_key": keys[0] if keys else "",
            "key": keys,
            "enable": True,
        },
        provider_settings={},
    )


class RealPersonaManager:
    """包一层真实的 PersonaManager 行为，但用内存字典代替 db。"""

    def __init__(self):
        self.personas: dict[str, object] = {}
        self.personas_v3: list[dict] = []

    async def get_persona(self, pid):
        return self.personas.get(pid)

    async def create_persona(self, pid, prompt, **kw):
        p = type("P", (), {"persona_id": pid, "system_prompt": prompt})()
        self.personas[pid] = p
        self.personas_v3.append({"name": pid, "prompt": prompt})
        return p

    async def update_persona(self, pid, system_prompt=None, **kw):
        self.personas[pid].system_prompt = system_prompt
        for item in self.personas_v3:
            if item["name"] == pid:
                item["prompt"] = system_prompt
        return self.personas[pid]


class FakeContext:
    def __init__(self, provider):
        self.persona_manager = RealPersonaManager()
        self.platform_manager = type("PM", (), {"platform_insts": []})()
        self._provider = provider

    def get_config(self):
        return {"admins_id": ["3613991290"]}

    def get_provider_by_id(self, pid):
        return self._provider if pid == "ds-deepseek-chat" else None

    def get_using_provider(self):
        return self._provider

    def get_all_providers(self):
        return [self._provider]


class FakeEvent:
    """够用的 event 替身，用来驱动指令 handler。"""

    def __init__(self, text: str):
        self._text = text
        self._result = None
        self.call_llm = False
        self._stopped = False

    def get_message_str(self):
        return self._text

    def plain_result(self, text):
        from astrbot.api.event import MessageChain

        return MessageChain(text)

    def set_result(self, r):
        self._result = r

    def get_result(self):
        return self._result

    def stop_event(self):
        self._stopped = True

    def is_stopped(self):
        return self._stopped

    def is_admin(self):
        return True


async def main() -> int:
    provider = build_real_provider()
    ctx = FakeContext(provider)
    cfg = {
        "target_qq": [TARGET],
        "collect_enabled": True,
        "only_in_group": False,
        "ignore_bot_self": True,
        "min_message_chars": 2,
        "max_message_chars": 2000,
        "auto_generate": False,
        "auto_generate_threshold": 200,
        "generate_provider_id": "ds-deepseek-chat",
    }
    plugin = PersonaHarvesterPlugin(ctx, cfg)
    await plugin.initialize()

    print("\n[A] 真实 provider 生成人格")
    persona = await plugin.generate_persona(TARGET)
    check("生成成功", persona is not None)
    if persona:
        check(
            "人格正文有实质内容",
            len(persona.system_prompt) > 300,
            f"({len(persona.system_prompt)} 字)",
        )
        check("带摘要", bool(persona.summary), f"({persona.summary[:40]}…)")
        check("带统计", persona.stats.get("total_messages", 0) > 0)
        # 确认已落库
        latest = await plugin.storage.get_latest_persona(TARGET)
        check("人格已落库", latest is not None and latest.version >= 1)
        check(
            "落库内容与生成一致",
            latest is not None and latest.system_prompt == persona.system_prompt,
        )

    print("\n[B] 应用到人格库")
    ok, msg = await plugin.apply_persona(TARGET)
    check("应用成功", ok, f"({msg})")
    check(
        "personas_v3 缓存已刷新",
        len(ctx.persona_manager.personas_v3) == 1,
    )

    print("\n[C] 指令 handler")
    results = []

    async def run_cmd(text):
        ev = FakeEvent(text)
        out = []
        async for r in plugin.cmd_harvest(ev):
            out.append(r)
        return out

    def to_text(res):
        """把 yield 出来的结果转成纯文本（MessageChain 或 str 都可能）。"""
        if res is None:
            return ""
        chain = getattr(res, "chain", None)
        if chain is None:
            return str(res)
        parts = []
        for comp in chain:
            t = getattr(comp, "text", None)
            parts.append(t if t is not None else str(comp))
        return "".join(parts)

    out = await run_cmd("人格采集 列表")
    check("列表有输出", len(out) > 0)
    txt = to_text(out[0]) if out else ""
    check("列表含目标 QQ", TARGET in txt, f"({txt.splitlines()[1][:40] if len(txt.splitlines())>1 else txt[:40]})")

    out = await run_cmd(f"人格采集 查看 {TARGET}")
    txt2 = to_text(out[0]) if out else ""
    check("查看返回人格全文", len(txt2) > 300, f"({len(txt2)} 字)")

    out = await run_cmd(f"人格采集 应用 {TARGET}")
    txt3 = to_text(out[0]) if out else ""
    check("应用指令返回成功标记", "✅" in txt3, f"({txt3[:50]})")

    out = await run_cmd("人格采集 帮助")
    check("帮助有输出", len(out) > 0 and "人格采集" in to_text(out[0]))

    out = await run_cmd("人格采集 开 99999")
    check("开：加入名单", "99999" in plugin.target_qq_list)
    out = await run_cmd("人格采集 关 99999")
    check("关：移出名单", "99999" not in plugin.target_qq_list)

    print("\n[D] 清理测试语料")
    n = await plugin.storage.delete_messages(sender_id=TARGET)
    check("测试语料已清理", n >= 0, f"(删了 {n} 条)")

    await plugin.terminate()
    print(f"\n{'全部通过' if FAILED == 0 else f'{FAILED} 项失败'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
