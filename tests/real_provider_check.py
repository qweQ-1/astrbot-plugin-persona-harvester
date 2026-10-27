"""用**真实 provider** 验证人格生成链路（容器内跑）。

独立脚本拿不到 AstrBot 运行中的 Context，所以这里自己按 cmd_config.json
构造一个真实的 openai_chat_completion provider，然后直接调插件的
generate_persona()，验证：
  语料 -> build_corpus -> build_persona_prompt -> provider.text_chat -> 存版本
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PLUGIN_DIR = "/AstrBot/data/plugins/astrbot_plugin_persona_harvester"
sys.path.insert(0, PLUGIN_DIR)

from astrbot.core.provider.sources.openai_source import (  # noqa: E402
    ProviderOpenAIOfficial,
)

from persona_builder import build_corpus, build_persona_prompt  # noqa: E402
from storage import Storage  # noqa: E402

TARGET = "3665585796"


async def main() -> int:
    cfg = json.load(open("/AstrBot/data/cmd_config.json", encoding="utf-8-sig"))
    sources = {s["id"]: s for s in cfg.get("provider_sources", [])}
    prov_cfg = None
    for p in cfg.get("provider", []):
        if p.get("id") == "ds-deepseek-chat":
            prov_cfg = p
            break
    if prov_cfg is None:
        print("❌ 找不到 ds-deepseek-chat provider 配置")
        return 1
    src = sources.get(prov_cfg.get("provider_source_id"), {})
    api_base = src.get("api_base") or prov_cfg.get("api_base")
    keys = src.get("key") or prov_cfg.get("key") or []
    if isinstance(keys, str):
        keys = [keys]
    api_key = keys[0] if keys else (src.get("api_key") or prov_cfg.get("api_key"))
    print(f"provider: {prov_cfg['id']} model={prov_cfg.get('model')}")
    print(f"api_base: {api_base}  key_present={bool(api_key)}")

    prov = ProviderOpenAIOfficial(
        provider_config={
            "id": prov_cfg["id"],
            "type": "openai_chat_completion",
            "model": prov_cfg.get("model"),
            "api_base": api_base,
            "api_key": api_key,
            "key": keys,
            "enable": True,
        },
        provider_settings={},
    )

    storage = Storage(Path("/AstrBot/data/plugin_data/astrbot_plugin_persona_harvester"))
    messages = await storage.get_all_messages(TARGET)
    print(f"语料条数: {len(messages)}")
    if not messages:
        print("❌ 没有语料")
        return 1

    corpus, stats = build_corpus(messages)
    print(f"语料字符: {len(corpus)}  统计: {stats.total_messages} 条 / {stats.sessions} 会话")
    print(f"口头禅 top5: {stats.top_phrases[:5]}")

    prompt = build_persona_prompt(corpus, stats, TARGET, "虎支")
    print(f"提示词长度: {len(prompt)}")

    print("\n调用 provider 生成中…")
    resp = await prov.text_chat(prompt=prompt)
    text = (getattr(resp, "completion_text", "") or "").strip()
    print(f"返回长度: {len(text)}")
    print("\n" + "=" * 60)
    print(text[:3000])
    print("=" * 60)

    from text_utils import strip_wrapper, make_summary

    final = strip_wrapper(text)
    print(f"\n剥壳后长度: {len(final)}")
    print(f"摘要: {make_summary(final)}")
    if len(final) < 200:
        print("❌ 生成内容过短")
        return 1
    print("✅ 人格生成链路正常")
    storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
