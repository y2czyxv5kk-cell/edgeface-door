"""工具注册表：大模型可调用的工具定义与实际执行。

LLM 根据用户自然语言决定调用哪个工具，本模块只负责执行并把结果
以文本形式返回。工具是显式白名单，LLM 无法调用未注册的能力。
"""
import json
import os
from dataclasses import dataclass
from typing import Dict

from edgeface.door import FakeDoor
from edgeface.log import RecognitionLog

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class ToolSpec:
    """一个工具的完整定义。"""
    name: str
    description: str
    parameters: dict      # JSON Schema（OpenAI function 格式）
    handler: object       # 执行函数 func(**args) -> str


class ToolRegistry:
    """工具集合：生成 ollama/OpenAI 格式工具定义 + 按名字分发执行。

    工具：
      open_door    开门（触发继电器/FakeDoor 脉冲）
      recent_logs  查询最近识别记录
      list_users   列出注册用户
    """

    def __init__(self, door=None, log: RecognitionLog = None,
                 id_map: Dict[int, str] = None):
        """Args:
            door: DoorController/FakeDoor；None 用 FakeDoor
            log: 识别日志；None 用默认路径
            id_map: 注册用户映射；None 从特征库读取
        """
        self.door = door or FakeDoor()
        self.log = log or RecognitionLog()
        self.id_map = id_map if id_map is not None else self._load_id_map()

        self._tools = {
            "open_door": ToolSpec(
                name="open_door",
                description=(
                    "打开门禁（触发继电器开门脉冲，门保持打开约 1 秒）。"
                    "只有当用户明确要求开门时才调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "开门原因（可选，用于日志）",
                        },
                    },
                    "required": [],
                },
                handler=self._open_door,
            ),
            "recent_logs": ToolSpec(
                name="recent_logs",
                description=(
                    "查询门禁系统的识别记录。当用户问"
                    "「谁来过」「今天有谁」「识别记录」等问题时调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "description": "查询最近几条记录（默认 5，最大 50）",
                        },
                    },
                    "required": [],
                },
                handler=self._recent_logs,
            ),
            "list_users": ToolSpec(
                name="list_users",
                description=(
                    "查询门禁系统中已注册的人员名单。当用户问"
                    "「谁能开门」「注册了哪些人」「有哪些人」等问题时调用。"
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "按姓名过滤（可选，不填则列出全部）",
                        },
                    },
                    "required": [],
                },
                handler=self._list_users,
            ),
        }

    def specs(self) -> list:
        """生成 chat(tools=...) 需要的工具定义列表。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, args: dict) -> str:
        """执行工具调用，返回结果文本（会作为 tool 消息喂回 LLM）。

        未知工具/参数错误/执行异常都返回错误文本而非抛异常，
        让 LLM 能向用户解释。
        """
        spec = self._tools.get(name)
        if spec is None:
            return f"错误：工具 {name} 不存在（可用：{list(self._tools.keys())}）"
        try:
            args = args or {}
            return str(spec.handler(**args))
        except TypeError as e:
            return f"错误：工具 {name} 的参数不正确（{e}）"
        except Exception as e:
            return f"错误：工具 {name} 执行失败（{e}）"

    # ---- 工具实现 ----
    def _open_door(self, reason: str = "") -> str:
        """执行开门脉冲。FakeDoor 打印日志；板端 DoorController 触发 GPIO。"""
        self.door.trigger()
        return ("开门成功（继电器脉冲已触发）"
                + (f"，原因：{reason}" if reason else ""))

    def _recent_logs(self, n: int = 5) -> str:
        """返回最近 n 条识别记录。"""
        n = max(1, min(int(n), 50))  # 参数范围 1~50
        records = self.log.recent(n)
        if not records:
            return "暂无识别记录。"
        import datetime
        lines = []
        for r in records:
            t = datetime.datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
            lines.append(f"[{t}] {r['name']}（相似度 {r['sim']:.2f}）")
        return "\n".join(lines)

    def _list_users(self, keyword: str = "") -> str:
        """列出注册用户，可按姓名关键词过滤。"""
        names = sorted(self.id_map.values())
        if keyword:
            names = [n for n in names if keyword in n]
        if not names:
            return "没有符合条件的注册用户。" if keyword else "特征库为空，还没有注册用户。"
        return f"已注册 {len(names)} 人：" + "、".join(names)

    @staticmethod
    def _load_id_map() -> Dict[int, str]:
        """从 assets/face_db/id_map.json 读取注册用户映射。"""
        path = os.path.join(_PROJECT_ROOT, "assets", "face_db", "id_map.json")
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
