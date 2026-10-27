# 人格采集器 · AstrBot 插件

按 **QQ 号锁定**采集指定用户的聊天记录，持续收集，并整理成一份**第三人称人格设定提示词**，
可一键写进 AstrBot 的人格库直接生效。

> ⚠️ **隐私提醒**
> 本插件会把**他人**的聊天记录落盘保存。默认只在本机存储、不外发；
> 只有执行「生成」时才会把语料片段发给当前配置的 LLM provider。
> 请确保使用者已获得相关用户的同意。

## 特性

| 能力 | 说明 |
| --- | --- |
| **按 QQ 号锁定** | 只有名单里的 QQ 号发出的消息才会入库 |
| **持续采集** | 插件常驻，每条新消息实时入库，自动去重（按 `message_id`） |
| **旁路运行** | 不回复、不打断、不 `stop_event`、不改 `call_llm`——群里的其它插件和机器人默认回复链路完全不受影响 |
| **回溯补采** | 通过 OneBot `get_group_msg_history` / `get_friend_msg_history` 把历史消息也拉回来 |
| **语料清洗** | 自动剔除「嗯」「[图片]」纯链接等无信息量内容；统计口头禅、平均句长、典型发言样例 |
| **人格生成** | 交给大模型产出 600–1200 字的第三人称人格设定，带语料原话例证 |
| **版本化** | 每次生成都存一个新版本，可回看历史 |
| **一键生效** | 「应用」把最新人格写进 AstrBot 人格库，`personas_v3` 缓存同步刷新，**不用重启** |

## 安装

在 AstrBot 的「插件管理」里用仓库地址安装：

```
https://github.com/qweQ-1/astrbot-plugin-persona-harvester
```

或手动把本目录放到 `data/plugins/astrbot_plugin_persona_harvester/` 后重载插件。

## 配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `target_qq` | `[]` | 需要采集的 QQ 号列表。留空则不采集任何人 |
| `collect_enabled` | `true` | 总开关 |
| `only_in_group` | `false` | 开启后只采群聊，不采私聊 |
| `ignore_bot_self` | `true` | 忽略机器人自己发的消息 |
| `min_message_chars` | `2` | 短于这个字数的消息直接丢弃 |
| `max_message_chars` | `2000` | 单条消息最多保留多少字 |
| `auto_generate` | `false` | 每采集够一批就自动生成人格（会消耗 token） |
| `auto_generate_threshold` | `200` | 自动生成的触发条数阈值 |
| `generate_provider_id` | `""` | 指定生成用的 provider；留空则用会话当前模型 |

> `target_qq` 也可以用指令临时添加（仅运行时有效，重启后失效）。
> 要持久化就写进配置里。

## 指令

全部指令**仅管理员**可用。

| 指令 | 说明 |
| --- | --- |
| `/人格采集 开 <QQ号>` | 把 QQ 号加入运行时锁定名单 |
| `/人格采集 关 <QQ号>` | 移出运行时锁定名单 |
| `/人格采集 列表` | 查看锁定名单与采集统计 |
| `/人格采集 补采 <QQ号> <群号> [条数]` | 回溯拉取历史消息（条数上限 500） |
| `/人格采集 生成 <QQ号>` | 用采集语料生成人格设定 |
| `/人格采集 查看 <QQ号>` | 查看已生成的人格设定全文 |
| `/人格采集 应用 <QQ号>` | 把最新人格写入 AstrBot 人格库 |
| `/人格采集 清空 <QQ号>` | 清空该 QQ 号的采集语料 |

指令别名：`人设采集` / `persona_harvest`。

> AstrBot 的指令需要带唤醒前缀（当前实例是 `/可可莉`）或 @ 机器人，例如：
> `/可可莉 人格采集 列表`

## 工作流程

```
群里有人发言
   └─► 采集监听器（priority 900000，旁路）
        └─► 发送者 QQ 在名单里？
             ├─ 否 → 直接返回，什么都不做
             └─ 是 → 清洗 + 去重 → 写 SQLite
                        └─► （可选）达到阈值 → 自动调模型生成新人格版本

/人格采集 生成 <QQ号>
   └─► 读全部语料 → build_corpus() 清洗统计 → build_persona_prompt()
        └─► 调 provider.text_chat() → 剥围栏 → 存成新版本
             └─► /人格采集 应用 <QQ号> → 写入 personas → personas_v3 刷新 → 立即生效
```

## 数据存储

```
data/plugin_data/astrbot_plugin_persona_harvester/
└── harvester.db          # SQLite：messages 表（语料） + personas 表（人格版本）
```

插件卸载重装**不会**丢数据。要彻底清除，用 `/人格采集 清空 <QQ号>` 或直接删库文件。

## 开发

```bash
# 跑单元测试（不需要 AstrBot 环境）
python3 tests/test_persona_harvester.py
```

测试覆盖：存储去重 / 截断 / 持久化 / 统计、语料清洗与截断策略、口头禅统计、
提示词结构、OneBot 消息段取文本、模型输出剥壳、指令前缀解析、QQ 列表解析。

## 文件结构

```
astrbot_plugin_persona_harvester/
├── main.py             # 插件入口：采集监听、指令、人格生成
├── storage.py          # SQLite 存储层
├── persona_builder.py  # 语料清洗 + 提示词构造
├── text_utils.py       # 与 AstrBot 无关的纯函数
├── metadata.yaml       # 插件元数据
├── _conf_schema.json   # 配置面板 schema
└── tests/
```

## 兼容性

- AstrBot `>=4.27.2`
- 平台：`aiocqhttp`（OneBot v11 / NapCat）
- 「回溯补采」依赖 OneBot 的 `get_group_msg_history` / `get_friend_msg_history`，
  其它 OneBot 实现可能不支持，此时会明确报错而不是静默失败。

## License

MIT
