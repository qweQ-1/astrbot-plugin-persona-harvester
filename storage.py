"""人格采集器 · 数据模型与存储层。

设计要点
--------
- 所有数据存放在 AstrBot 的 ``data/plugin_data/<plugin_name>/`` 目录下，插件卸载重装不丢数据。
- 使用单个 SQLite 文件做索引，正文直接落在同一张表里（消息量级不大，够用）。
- 采用 ``asyncio.Lock`` 串行化所有写操作，避免多事件并发写库导致锁冲突。
- 采集库与「人格成品」分离：
  - ``messages`` 表存原始语料（按 QQ 号锁定采集来的聊天记录）
  - ``personas`` 表存整理出来的人格设定提示词（可多版本，保留历史）
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 单条消息最大长度，防止把整张合并转发 / 超长文本塞进库里。
MAX_MESSAGE_CHARS = 4000
# 采集库总量上限（条）。超过后不再写入新消息，防止磁盘被撑爆。
MAX_TOTAL_MESSAGES = 200_000
# 人格版本保留数量上限。
MAX_PERSONA_VERSIONS = 20


def _now() -> float:
    return time.time()


@dataclass
class Message:
    """一条被采集到的聊天记录。"""

    id: int
    sender_id: str
    sender_name: str
    group_id: str
    """群号；私聊为空串。"""
    session_id: str
    """会话 id：群聊为 group_id，私聊为 sender_id。"""
    is_private: bool
    message_id: str
    text: str
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "group_id": self.group_id,
            "session_id": self.session_id,
            "is_private": self.is_private,
            "message_id": self.message_id,
            "text": self.text,
            "created_at": self.created_at,
        }


@dataclass
class Persona:
    """一份整理好的人格设定提示词。"""

    id: int
    persona_id: str
    """逻辑标识，形如 ``harvest_3665585796``。"""
    target_qq: str
    """这份人格对应的被采集者 QQ 号。"""
    system_prompt: str
    """人格设定提示词正文。"""
    summary: str
    """一句话摘要。"""
    stats: dict[str, Any] = field(default_factory=dict)
    """生成时使用的统计信息。"""
    created_at: float = 0.0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "persona_id": self.persona_id,
            "target_qq": self.target_qq,
            "system_prompt": self.system_prompt,
            "summary": self.summary,
            "stats": self.stats,
            "created_at": self.created_at,
            "version": self.version,
        }


class Storage:
    """SQLite 存储层。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "harvester.db"
        self._lock = asyncio.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # 初始化 / 关闭
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL DEFAULT '',
                    group_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    is_private INTEGER NOT NULL DEFAULT 0,
                    message_id TEXT NOT NULL DEFAULT '',
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """,
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_sender
                    ON messages(sender_id, created_at)
                """,
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at)
                """,
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    persona_id TEXT NOT NULL,
                    target_qq TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    stats TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1
                )
                """,
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_personas_qq
                    ON personas(target_qq, created_at)
                """,
            )

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - 关闭失败不应让插件卸载流程报错
            pass

    # ------------------------------------------------------------------
    # 消息采集
    # ------------------------------------------------------------------
    async def add_message(self, message: Message) -> bool:
        """写入一条消息。已存在相同 message_id 时跳过。

        Returns:
            是否真的写入了新记录。
        """
        text = (message.text or "").strip()
        if not text:
            return False
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "…"

        async with self._lock:
            try:
                if message.message_id:
                    row = self._conn.execute(
                        "SELECT id FROM messages WHERE message_id = ? LIMIT 1",
                        (message.message_id,),
                    ).fetchone()
                    if row:
                        return False

                total = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM messages",
                ).fetchone()["c"]
                if total >= MAX_TOTAL_MESSAGES:
                    return False

                self._conn.execute(
                    """
                    INSERT INTO messages (
                        sender_id, sender_name, group_id, session_id,
                        is_private, message_id, text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.sender_id,
                        message.sender_name,
                        message.group_id,
                        message.session_id,
                        1 if message.is_private else 0,
                        message.message_id,
                        text,
                        message.created_at or _now(),
                    ),
                )
                self._conn.commit()
            except sqlite3.Error:
                return False
        return True

    async def count_messages(self, sender_id: str | None = None) -> int:
        async with self._lock:
            if sender_id:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM messages WHERE sender_id = ?",
                    (sender_id,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) AS c FROM messages",
                ).fetchone()
        return int(row["c"])

    async def get_messages(
        self,
        sender_id: str | None = None,
        limit: int = 500,
        offset: int = 0,
        order: str = "new",
        keyword: str = "",
        session_id: str = "",
    ) -> list[Message]:
        """按条件取消息列表。"""
        where: list[str] = []
        params: list[Any] = []
        if sender_id:
            where.append("sender_id = ?")
            params.append(sender_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if keyword:
            where.append("text LIKE ?")
            params.append(f"%{keyword}%")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        direction = "ASC" if order == "old" else "DESC"
        limit = max(1, min(5000, int(limit)))
        offset = max(0, int(offset))
        async with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM messages {clause}
                ORDER BY created_at {direction}, id {direction}
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    async def get_all_messages(
        self,
        sender_id: str | None = None,
        limit: int = 20000,
    ) -> list[Message]:
        """取全部消息（按时间正序），用于人格生成。"""
        where = ""
        params: list[Any] = []
        if sender_id:
            where = "WHERE sender_id = ?"
            params.append(sender_id)
        limit = max(1, min(200000, int(limit)))
        async with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM messages {where}
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [self._row_to_message(r) for r in rows]

    async def get_stats(self, sender_id: str | None = None) -> dict[str, Any]:
        """统计信息：条数、会话数、时间跨度、字数。"""
        where = ""
        params: list[Any] = []
        if sender_id:
            where = "WHERE sender_id = ?"
            params.append(sender_id)
        async with self._lock:
            row = self._conn.execute(
                f"""
                SELECT COUNT(*) AS c,
                       COUNT(DISTINCT session_id) AS s,
                       COALESCE(SUM(LENGTH(text)), 0) AS chars,
                       MIN(created_at) AS t0,
                       MAX(created_at) AS t1
                FROM messages {where}
                """,
                params,
            ).fetchone()
        return {
            "count": int(row["c"]),
            "sessions": int(row["s"]),
            "chars": int(row["chars"]),
            "first_at": float(row["t0"] or 0.0),
            "last_at": float(row["t1"] or 0.0),
        }

    async def get_sessions(self, sender_id: str | None = None) -> list[dict[str, Any]]:
        """会话列表（用于展示采集自哪些群 / 私聊）。"""
        where = ""
        params: list[Any] = []
        if sender_id:
            where = "WHERE sender_id = ?"
            params.append(sender_id)
        async with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT session_id, group_id, is_private,
                       COUNT(*) AS c,
                       MIN(created_at) AS t0,
                       MAX(created_at) AS t1
                FROM messages {where}
                GROUP BY session_id
                ORDER BY c DESC
                """,
                params,
            ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "group_id": r["group_id"],
                "is_private": bool(r["is_private"]),
                "count": int(r["c"]),
                "first_at": float(r["t0"] or 0.0),
                "last_at": float(r["t1"] or 0.0),
            }
            for r in rows
        ]

    async def delete_messages(
        self,
        sender_id: str | None = None,
        session_id: str = "",
    ) -> int:
        """删除消息。返回删除条数。"""
        where: list[str] = []
        params: list[Any] = []
        if sender_id:
            where.append("sender_id = ?")
            params.append(sender_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        async with self._lock:
            cur = self._conn.execute(f"DELETE FROM messages {clause}", params)
            self._conn.commit()
            return cur.rowcount or 0

    # ------------------------------------------------------------------
    # 人格成品
    # ------------------------------------------------------------------
    async def save_persona(self, persona: Persona) -> Persona:
        """保存一个人格版本。同一 persona_id 保留多个历史版本。"""
        async with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM personas WHERE persona_id = ?",
                (persona.persona_id,),
            ).fetchone()
            version = int(row["v"]) + 1
            cur = self._conn.execute(
                """
                INSERT INTO personas (
                    persona_id, target_qq, system_prompt, summary, stats, created_at, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persona.persona_id,
                    persona.target_qq,
                    persona.system_prompt,
                    persona.summary,
                    json.dumps(persona.stats, ensure_ascii=False),
                    persona.created_at or _now(),
                    version,
                ),
            )
            self._conn.commit()
            persona.id = int(cur.lastrowid or 0)
            persona.version = version
        return persona

    async def get_personas(self, target_qq: str = "") -> list[Persona]:
        where = ""
        params: list[Any] = []
        if target_qq:
            where = "WHERE target_qq = ?"
            params.append(target_qq)
        async with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM personas {where}
                ORDER BY created_at DESC, id DESC
                LIMIT 200
                """,
                params,
            ).fetchall()
        return [self._row_to_persona(r) for r in rows]

    async def get_latest_persona(self, target_qq: str) -> Persona | None:
        rows = await self.get_personas(target_qq)
        return rows[0] if rows else None

    async def get_persona_by_id(self, persona_id: str) -> Persona | None:
        async with self._lock:
            row = self._conn.execute(
                "SELECT * FROM personas WHERE persona_id = ? ORDER BY version DESC LIMIT 1",
                (persona_id,),
            ).fetchone()
        return self._row_to_persona(row) if row else None

    async def delete_personas(self, target_qq: str = "") -> int:
        where = ""
        params: list[Any] = []
        if target_qq:
            where = "WHERE target_qq = ?"
            params.append(target_qq)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        async with self._lock:
            cur = self._conn.execute(f"DELETE FROM personas {clause}", params)
            self._conn.commit()
            return cur.rowcount or 0

    # ------------------------------------------------------------------
    # 行转换
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> Message:
        return Message(
            id=int(row["id"]),
            sender_id=row["sender_id"],
            sender_name=row["sender_name"] or "",
            group_id=row["group_id"] or "",
            session_id=row["session_id"] or "",
            is_private=bool(row["is_private"]),
            message_id=row["message_id"] or "",
            text=row["text"] or "",
            created_at=float(row["created_at"] or 0.0),
        )

    @staticmethod
    def _row_to_persona(row: sqlite3.Row) -> Persona:
        try:
            stats = json.loads(row["stats"] or "{}")
        except (TypeError, ValueError):
            stats = {}
        return Persona(
            id=int(row["id"]),
            persona_id=row["persona_id"],
            target_qq=row["target_qq"] or "",
            system_prompt=row["system_prompt"] or "",
            summary=row["summary"] or "",
            stats=stats if isinstance(stats, dict) else {},
            created_at=float(row["created_at"] or 0.0),
            version=int(row["version"] or 1),
        )
