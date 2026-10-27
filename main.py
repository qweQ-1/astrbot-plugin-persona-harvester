"""AstrBot 人格采集器 · 按 QQ 号锁定采集聊天记录并整理成人格设定提示词。

工作方式
--------
1. **持续采集**：插件监听所有群聊 / 私聊消息，只要发送者的 QQ 号在锁定名单里，
   就把这条消息存进本地 SQLite。采集是**旁路**的——不回复、不打断、不改写结果，
   群里的其它插件和机器人默认回复链路都照常工作。
2. **回溯补采**：用 OneBot 的历史消息接口，把指定 QQ 号在某个群 / 私聊里
   之前说过的话一并拉回来，这样人格样本才够厚。
3. **整理成人格**：把采集到的语料交给大模型，生成一份第三人称的人格设定提示词，
   存成可追溯的版本；也可以一键写进 AstrBot 的人格库直接生效。

指令（仅管理员）
----------------
- ``/人格采集 开 <QQ号>``            把 QQ 号加入运行时锁定名单
- ``/人格采集 关 <QQ号>``            移出运行时锁定名单
- ``/人格采集 列表``                  查看锁定名单与采集统计
- ``/人格采集 补采 <QQ号> <群号> [条数]``   回溯拉取历史消息
- ``/人格采集 生成 <QQ号>``           用采集到的语料生成人格设定
- ``/人格采集 查看 <QQ号>``           查看已生成的人格设定全文
- ``/人格采集 应用 <QQ号>``           把最新人格写入 AstrBot 人格库
- ``/人格采集 清空 <QQ号>``           清掉该 QQ 号的采集语料

数据
----
语料与人格都存在 ``data/plugin_data/astrbot_plugin_persona_harvester/`` 下，
插件重装不丢数据。

隐私提醒
--------
插件会把**他人**的聊天记录落盘。默认只在本机存储、不外发；
只有执行「生成」时才会把语料片段发给当前配置的 LLM provider。
请确保使用者已获得相关用户的同意。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star, StarTools

try:  # 作为插件包加载
    from .persona_builder import build_corpus, build_persona_prompt
    from .storage import Message, Persona, Storage
    from .text_utils import (
        CMD,
        CMD_ALIASES,
        extract_text,
        make_summary,
        split_qq_list,
        strip_command_prefix,
        strip_wrapper,
    )
except ImportError:  # 独立运行单元测试时
    from persona_builder import build_corpus, build_persona_prompt
    from storage import Message, Persona, Storage
    from text_utils import (
        CMD,
        CMD_ALIASES,
        extract_text,
        make_summary,
        split_qq_list,
        strip_command_prefix,
        strip_wrapper,
    )

PLUGIN_NAME = "astrbot_plugin_persona_harvester"
PLUGIN_VERSION = "v1.0.0"

# 采集监听器的优先级：尽量早，早于其它插件的回复链路，
# 这样即使别的插件提前 stop_event，语料也不会漏采。
COLLECT_PRIORITY = 900_000

# 生成人格的超时（秒）
GENERATE_TIMEOUT = 300
# 补采历史时单次请求的条数上限
BACKFILL_MAX = 500


def _plugin_data_dir() -> Path:
    """返回插件数据目录：AstrBot 约定为 ``data/plugin_data/<插件名>/``。"""
    candidates: list[Path] = []
    env_dir = os.environ.get("ASTRBOT_PLUGIN_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME)
    candidates.append(Path("/AstrBot/data/plugin_data") / PLUGIN_NAME)
    for path in candidates:
        if path.parent.exists():
            path.mkdir(parents=True, exist_ok=True)
            return path
    fallback = candidates[-1]
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class PersonaHarvesterPlugin(Star):
    """按 QQ 号锁定采集聊天记录，整理成人格设定提示词。"""

    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.storage = Storage(_plugin_data_dir())
        # 运行时动态维护的锁定名单（配置里的 + 指令加的）
        self._extra_targets: list[str] = []
        # 只保护「正在生成中」这个集合的锁，不在 LLM 调用期间持锁
        self._state_lock = asyncio.Lock()
        self._generating: set[str] = set()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        self._extra_targets = []
        logger.info(
            f"[{PLUGIN_NAME}] 初始化完成，锁定 QQ: {self.target_qq_list or '（空）'}",
        )

    async def terminate(self) -> None:
        self.storage.close()
        logger.info(f"[{PLUGIN_NAME}] 已卸载")

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        try:
            value = self.config.get(key, default)
        except Exception:  # noqa: BLE001 - 配置异常不应让插件崩溃
            return default
        return default if value is None else value

    @property
    def target_qq_list(self) -> list[str]:
        """当前锁定的 QQ 号列表（配置 + 指令动态添加）。"""
        result = split_qq_list(self._cfg("target_qq", []))
        for item in self._extra_targets:
            if item not in result:
                result.append(item)
        return result

    @property
    def collect_enabled(self) -> bool:
        return bool(self._cfg("collect_enabled", True))

    @property
    def only_in_group(self) -> bool:
        """只采集群聊消息（关闭则群聊 + 私聊都采集）。"""
        return bool(self._cfg("only_in_group", False))

    @property
    def ignore_bot_self(self) -> bool:
        """忽略机器人自己发出的消息。"""
        return bool(self._cfg("ignore_bot_self", True))

    @property
    def max_message_chars(self) -> int:
        """单条消息最多保留多少字。"""
        try:
            value = int(self._cfg("max_message_chars", 2000))
        except (TypeError, ValueError):
            value = 2000
        return max(50, min(4000, value))

    @property
    def min_message_chars(self) -> int:
        """短于这个字数的消息直接丢弃（「嗯」「好」这类没有信息量）。"""
        try:
            value = int(self._cfg("min_message_chars", 2))
        except (TypeError, ValueError):
            value = 2
        return max(0, min(200, value))

    @property
    def auto_generate(self) -> bool:
        """每采集够一批消息就自动生成新的人格版本。"""
        return bool(self._cfg("auto_generate", False))

    @property
    def auto_generate_threshold(self) -> int:
        """自动生成的触发条数阈值。"""
        try:
            value = int(self._cfg("auto_generate_threshold", 200))
        except (TypeError, ValueError):
            value = 200
        return max(20, min(100000, value))

    @property
    def generate_provider_id(self) -> str:
        """指定用于生成人格的 provider id；留空则用会话当前 provider。"""
        return str(self._cfg("generate_provider_id", "") or "").strip()

    def is_target(self, sender_id: str) -> bool:
        """判断发送者是否在锁定名单里。"""
        if not sender_id:
            return False
        return str(sender_id) in self.target_qq_list

    # ------------------------------------------------------------------
    # 采集
    # ------------------------------------------------------------------
    @filter.event_message_type(
        filter.EventMessageType.ALL,
        priority=COLLECT_PRIORITY,
        alias="persona_harvester_collect",
    )
    async def on_message(self, event: AstrMessageEvent) -> None:
        """旁路监听所有消息，采集被锁定用户的发言。

        这里刻意 **不** 调用 ``event.set_result`` / ``event.stop_event``，
        也不 ``should_call_llm(False)``：

        - 不设结果 → 其它插件和默认回复链路照常工作；
        - 不 stop → 不会误伤同一条消息的其它订阅者；
        - 不改 call_llm → 群里 @ 机器人时仍然正常回复。

        采集只写本地库，不产生任何外部副作用。
        """
        if not self.collect_enabled:
            return
        if self.ignore_bot_self and str(event.get_self_id()) == str(
            event.get_sender_id(),
        ):
            return

        sender_id = str(event.get_sender_id() or "")
        if not self.is_target(sender_id):
            return

        is_private = bool(event.is_private_chat())
        if self.only_in_group and is_private:
            return

        text = (event.get_message_str() or "").strip()
        if not text:
            return
        if len(text) < self.min_message_chars:
            return
        if len(text) > self.max_message_chars:
            text = text[: self.max_message_chars]

        group_id = str(event.get_group_id() or "")
        session_id = group_id if group_id else sender_id
        message_id = ""
        try:
            message_id = str(getattr(event.message_obj, "message_id", "") or "")
        except Exception:  # noqa: BLE001 - 拿不到 message_id 不影响采集
            message_id = ""

        record = Message(
            id=0,
            sender_id=sender_id,
            sender_name=str(event.get_sender_name() or sender_id),
            group_id=group_id,
            session_id=session_id,
            is_private=is_private,
            message_id=message_id,
            text=text,
            created_at=time.time(),
        )

        written = await self.storage.add_message(record)
        if not written:
            return

        if self.auto_generate:
            await self._maybe_auto_generate(sender_id)

    async def _maybe_auto_generate(self, sender_id: str) -> None:
        """采集条数达到阈值时自动生成一版人格。"""
        threshold = self.auto_generate_threshold
        total = await self.storage.count_messages(sender_id)
        if total < threshold or total % threshold != 0:
            return
        async with self._state_lock:
            if sender_id in self._generating:
                return
            self._generating.add(sender_id)
        try:
            await self.generate_persona(sender_id, notify=True)
        except Exception as e:  # noqa: BLE001 - 自动生成失败不应影响采集
            logger.error(f"[{PLUGIN_NAME}] 自动生成人格失败: {e}")
        finally:
            async with self._state_lock:
                self._generating.discard(sender_id)

    # ------------------------------------------------------------------
    # 回溯补采（OneBot 历史消息）
    # ------------------------------------------------------------------
    def _get_cq_client(self) -> Any | None:
        """拿到 OneBot 客户端，用于调用 ``get_group_msg_history`` 等动作。"""
        try:
            for platform in self.context.platform_manager.platform_insts:
                if platform.meta().name != "aiocqhttp":
                    continue
                client = platform.get_client()
                if client is not None:
                    return client
        except Exception as e:  # noqa: BLE001 - 平台实例还没起来时忽略
            logger.debug(f"[{PLUGIN_NAME}] 获取 OneBot 客户端失败: {e}")
        return None

    async def backfill(
        self,
        target_qq: str,
        group_id: str = "",
        limit: int = 100,
    ) -> dict[str, Any]:
        """回溯拉取历史消息并入库。

        Args:
            target_qq: 目标 QQ 号。
            group_id: 群号；留空表示私聊历史。
            limit: 最多拉取多少条。

        Returns:
            ``{"ok": bool, "added": int, "scanned": int, "error": str}``
        """
        client = self._get_cq_client()
        if client is None:
            return {
                "ok": False,
                "added": 0,
                "scanned": 0,
                "error": "未找到 OneBot(aiocqhttp) 客户端，无法调用历史消息接口",
            }

        limit = max(1, min(BACKFILL_MAX, int(limit)))
        try:
            if group_id:
                raw = await client.call_action(
                    "get_group_msg_history",
                    group_id=int(group_id),
                    message_seq=0,
                    count=limit,
                    reverseOrder=False,
                )
            else:
                raw = await client.call_action(
                    "get_friend_msg_history",
                    user_id=int(target_qq),
                    count=limit,
                    reverseOrder=False,
                )
        except Exception as e:  # noqa: BLE001 - 接口失败要如实回报
            return {
                "ok": False,
                "added": 0,
                "scanned": 0,
                "error": f"调用历史消息接口失败: {e}",
            }

        messages = raw if isinstance(raw, list) else []
        added = 0
        scanned = 0
        for item in messages:
            if not isinstance(item, dict):
                continue
            scanned += 1
            sender = str((item.get("sender") or {}).get("user_id", "") or "")
            if sender != str(target_qq):
                continue
            text = extract_text(item.get("message"))
            if not text or len(text) < self.min_message_chars:
                continue
            if len(text) > self.max_message_chars:
                text = text[: self.max_message_chars]
            sender_info = item.get("sender") or {}
            gid = str(item.get("group_id", "") or group_id)
            ok = await self.storage.add_message(
                Message(
                    id=0,
                    sender_id=sender,
                    sender_name=str(
                        sender_info.get("card")
                        or sender_info.get("nickname")
                        or sender,
                    ),
                    group_id=gid,
                    session_id=gid or sender,
                    is_private=not bool(gid),
                    message_id=str(item.get("message_id", "") or ""),
                    text=text,
                    created_at=float(item.get("time", 0) or time.time()),
                ),
            )
            if ok:
                added += 1

        return {"ok": True, "added": added, "scanned": scanned, "error": ""}

    # ------------------------------------------------------------------
    # 人格生成
    # ------------------------------------------------------------------
    async def generate_persona(
        self,
        target_qq: str,
        notify: bool = False,
    ) -> Persona | None:
        """用采集到的语料生成一份人格设定提示词。

        Args:
            target_qq: 目标 QQ 号。
            notify: 生成后是否主动通知管理员。

        Returns:
            生成的人格；失败返回 None。
        """
        target_qq = str(target_qq).strip()
        messages = await self.storage.get_all_messages(target_qq)
        if not messages:
            logger.info(f"[{PLUGIN_NAME}] {target_qq} 没有任何语料，无法生成")
            return None

        corpus, stats = build_corpus(messages)
        if not corpus.strip():
            logger.info(f"[{PLUGIN_NAME}] {target_qq} 语料清洗后为空，无法生成")
            return None

        nickname = ""
        for msg in messages:
            if msg.sender_name and msg.sender_name != target_qq:
                nickname = msg.sender_name
                break

        prompt = build_persona_prompt(corpus, stats, target_qq, nickname)

        provider = self._resolve_provider()
        if provider is None:
            logger.warning(f"[{PLUGIN_NAME}] 没有可用的 chat provider，无法生成人格")
            return None

        try:
            response = await asyncio.wait_for(
                provider.text_chat(prompt=prompt),
                timeout=GENERATE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(f"[{PLUGIN_NAME}] 生成人格超时（{GENERATE_TIMEOUT}s）")
            return None
        except Exception as e:  # noqa: BLE001 - provider 异常要如实回报
            logger.error(f"[{PLUGIN_NAME}] 生成人格失败: {e}")
            return None

        text = (getattr(response, "completion_text", "") or "").strip()
        if not text:
            logger.warning(f"[{PLUGIN_NAME}] provider 返回了空内容")
            return None

        # 模型偶尔会裹上 markdown 代码块或加一句前言，这里剥掉。
        system_prompt = strip_wrapper(text)
        summary = make_summary(system_prompt)

        persona = Persona(
            id=0,
            persona_id=f"harvest_{target_qq}",
            target_qq=target_qq,
            system_prompt=system_prompt,
            summary=summary,
            stats=stats.to_dict(),
            created_at=time.time(),
        )
        await self.storage.save_persona(persona)
        logger.info(
            f"[{PLUGIN_NAME}] 已为 {target_qq} 生成人格 v{persona.version}"
            f"（{stats.total_messages} 条语料）",
        )

        if notify:
            await self._notify(target_qq, persona)
        return persona

    def _resolve_provider(self) -> Any | None:
        """挑一个可用的 chat provider。"""
        provider_id = self.generate_provider_id
        if provider_id:
            try:
                prov = self.context.get_provider_by_id(provider_id)
            except Exception:  # noqa: BLE001 - provider 不存在时回退
                prov = None
            if prov is not None:
                return prov
            logger.warning(
                f"[{PLUGIN_NAME}] 配置的 provider {provider_id} 不存在，回退到当前 provider",
            )

        try:
            prov = self.context.get_using_provider()
        except Exception as e:  # noqa: BLE001 - 类型不对时回退
            logger.warning(f"[{PLUGIN_NAME}] 获取当前 provider 失败: {e}")
            prov = None
        if prov is not None:
            return prov

        for candidate in self.context.get_all_providers():
            try:
                meta = candidate.meta()
            except Exception:  # noqa: BLE001 - 跳过读不到 meta 的 provider
                continue
            if getattr(meta, "provider_type", None) is not None:
                return candidate
        return None

    async def _notify(self, target_qq: str, persona: Persona) -> None:
        """把生成结果主动发给管理员所在的私聊会话。"""
        try:
            admins = self.context.get_config().get("admins_id", [])
            if not admins:
                return
            platform_id = self._aiocqhttp_platform_id() or "napcat"
            umo = f"{platform_id}:FriendMessage:{admins[0]}"
            await StarTools.send_message(
                umo,
                MessageChain(
                    f"[{PLUGIN_NAME}] 已为 QQ {target_qq} 生成人格 v{persona.version}\n"
                    f"{persona.summary}",
                ),
            )
        except Exception as e:  # noqa: BLE001 - 通知失败不影响生成结果
            logger.debug(f"[{PLUGIN_NAME}] 通知生成结果失败: {e}")

    def _aiocqhttp_platform_id(self) -> str:
        """拿到 aiocqhttp 适配器的 id（用于拼 unified_msg_origin）。"""
        try:
            for platform in self.context.platform_manager.platform_insts:
                if platform.meta().name == "aiocqhttp":
                    return str(platform.meta().id)
        except Exception:  # noqa: BLE001 - 拿不到就用默认值
            pass
        return ""

    async def apply_persona(self, target_qq: str) -> tuple[bool, str]:
        """把最新人格写入 AstrBot 人格库。

        用 ``context.persona_manager`` 而不是直接写 db，这样内存里的
        ``personas_v3`` 缓存会同步刷新，人格立刻生效，不用重启 AstrBot。
        """
        persona = await self.storage.get_latest_persona(target_qq)
        if persona is None:
            return False, f"QQ {target_qq} 还没有生成过人格"

        persona_id = persona.persona_id
        try:
            mgr = getattr(self.context, "persona_manager", None)
            if mgr is None:
                return False, "当前 AstrBot 版本没有提供人格管理器"
            try:
                existing = await mgr.get_persona(persona_id)
            except ValueError:
                existing = None
            if existing is not None:
                await mgr.update_persona(
                    persona_id,
                    system_prompt=persona.system_prompt,
                )
                return True, f"已更新人格 {persona_id}（v{persona.version}）"
            await mgr.create_persona(persona_id, persona.system_prompt)
            return True, f"已创建人格 {persona_id}（v{persona.version}）"
        except ValueError as e:
            # ID 冲突等
            return False, str(e)
        except Exception as e:  # noqa: BLE001 - 数据库异常要如实回报
            logger.error(f"[{PLUGIN_NAME}] 写入人格库失败: {e}")
            return False, f"写入人格库失败: {e}"

    # ------------------------------------------------------------------
    # 指令
    # ------------------------------------------------------------------
    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command(CMD, alias=CMD_ALIASES)
    async def cmd_harvest(self, event: AstrMessageEvent):
        """人格采集指令：开 / 关 / 列表 / 补采 / 生成 / 查看 / 应用 / 清空。"""
        args = strip_command_prefix(event.get_message_str() or "")
        if not args:
            yield event.plain_result(self._help_text())
            return

        parts = args.split()
        action = parts[0]
        rest = parts[1:]

        if action in ("帮助", "help", "?", "菜单"):
            yield event.plain_result(self._help_text())
            return

        if action in ("开", "添加", "add", "on"):
            if not rest:
                yield event.plain_result("用法：/人格采集 开 <QQ号>")
                return
            added: list[str] = []
            for qq in rest:
                if qq not in self._extra_targets:
                    self._extra_targets.append(qq)
                    added.append(qq)
            if not added:
                yield event.plain_result(f"这些 QQ 已经在锁定名单里了：{', '.join(rest)}")
                return
            yield event.plain_result(
                f"已加入锁定名单：{', '.join(added)}\n"
                f"当前名单：{', '.join(self.target_qq_list)}\n"
                "提示：名单只在插件运行期间有效，持久化请写进插件配置的 target_qq。",
            )
            return

        if action in ("关", "移除", "remove", "off"):
            if not rest:
                yield event.plain_result("用法：/人格采集 关 <QQ号>")
                return
            removed: list[str] = []
            for qq in rest:
                if qq in self._extra_targets:
                    self._extra_targets.remove(qq)
                    removed.append(qq)
            if not removed:
                yield event.plain_result(
                    f"这些 QQ 不在运行时锁定名单里：{', '.join(rest)}\n"
                    "若要永久移除，请改插件配置里的 target_qq。",
                )
                return
            yield event.plain_result(f"已移出锁定名单：{', '.join(removed)}")
            return

        if action in ("列表", "list", "状态", "status"):
            targets = self.target_qq_list
            if not targets:
                yield event.plain_result(
                    "当前没有锁定任何 QQ 号。\n"
                    "用 /人格采集 开 <QQ号> 添加，或在插件配置里填写 target_qq。",
                )
                return
            lines = ["锁定名单与采集统计："]
            for qq in targets:
                stats = await self.storage.get_stats(qq)
                persona = await self.storage.get_latest_persona(qq)
                lines.append(
                    f"  • QQ {qq}：{stats['count']} 条 / {stats['sessions']} 个会话"
                    f" / {stats['chars']} 字"
                    + (
                        f" / 已有人格 v{persona.version}"
                        if persona
                        else " / 尚未生成人格"
                    ),
                )
            total = await self.storage.count_messages()
            lines.append(f"库内总计：{total} 条")
            yield event.plain_result("\n".join(lines))
            return

        if action in ("补采", "backfill", "拉取"):
            if len(rest) < 2:
                yield event.plain_result("用法：/人格采集 补采 <QQ号> <群号> [条数]")
                return
            qq = rest[0]
            group = rest[1]
            limit = int(rest[2]) if len(rest) > 2 and rest[2].isdigit() else 100
            result = await self.backfill(qq, group, limit)
            if not result["ok"]:
                yield event.plain_result(f"补采失败：{result['error']}")
                return
            yield event.plain_result(
                f"补采完成：扫描 {result['scanned']} 条，新增入库 {result['added']} 条。",
            )
            return

        if action in ("生成", "generate", "gen"):
            if not rest:
                yield event.plain_result("用法：/人格采集 生成 <QQ号>")
                return
            qq = rest[0]
            async with self._state_lock:
                if qq in self._generating:
                    busy = True
                else:
                    busy = False
                    self._generating.add(qq)
            if busy:
                yield event.plain_result(f"QQ {qq} 的人格正在生成中，请稍候。")
                return
            try:
                yield event.plain_result(f"正在根据 QQ {qq} 的语料生成人格设定…")
                persona = await self.generate_persona(qq)
                if persona is None:
                    yield event.plain_result(
                        f"生成失败：QQ {qq} 可能还没有足够语料，或 provider 调用出错。"
                        "详见 AstrBot 日志。",
                    )
                    return
                yield event.plain_result(
                    f"✅ 已生成 {persona.persona_id} v{persona.version}\n"
                    f"{persona.summary}\n\n"
                    f"共分析 {persona.stats.get('total_messages', 0)} 条消息。\n"
                    "用 /人格采集 查看 查看全文，/人格采集 应用 写入人格库。",
                )
            finally:
                async with self._state_lock:
                    self._generating.discard(qq)
            return

        if action in ("查看", "show", "view"):
            if not rest:
                yield event.plain_result("用法：/人格采集 查看 <QQ号>")
                return
            persona = await self.storage.get_latest_persona(rest[0])
            if persona is None:
                yield event.plain_result(f"QQ {rest[0]} 还没有生成过人格设定。")
                return
            yield event.plain_result(
                f"{persona.persona_id} v{persona.version}\n"
                f"（{persona.stats.get('total_messages', 0)} 条语料）\n\n"
                f"{persona.system_prompt}",
            )
            return

        if action in ("应用", "apply"):
            if not rest:
                yield event.plain_result("用法：/人格采集 应用 <QQ号>")
                return
            ok, message = await self.apply_persona(rest[0])
            yield event.plain_result(("✅ " if ok else "❌ ") + message)
            return

        if action in ("清空", "clear", "删除"):
            if not rest:
                yield event.plain_result("用法：/人格采集 清空 <QQ号>")
                return
            count = await self.storage.delete_messages(sender_id=rest[0])
            yield event.plain_result(f"已清空 QQ {rest[0]} 的 {count} 条采集语料。")
            return

        yield event.plain_result(self._help_text())

    def _help_text(self) -> str:
        return f"""【{PLUGIN_NAME}】人格采集器

用法：
/{CMD} 开 <QQ号>            把 QQ 号加入锁定名单
/{CMD} 关 <QQ号>            移出锁定名单
/{CMD} 列表                  查看锁定名单与采集统计
/{CMD} 补采 <QQ号> <群号> [条数]   回溯拉取历史消息
/{CMD} 生成 <QQ号>          用采集语料生成人格设定
/{CMD} 查看 <QQ号>          查看已生成的人格设定全文
/{CMD} 应用 <QQ号>          把最新人格写入 AstrBot 人格库
/{CMD} 清空 <QQ号>          清空该 QQ 号的采集语料

采集是旁路的：不会打断群聊、不会触发回复。"""
