"""对话循环：用户输入 → LLM 决定调工具 → 执行 → 生成回复。

一轮对话流程（流式）：
  ① system + 历史 + 用户消息流式发给 LLM（附工具定义），
     文本逐块透传给 UI（首个 token 即上屏，无需等整段生成）
  ② LLM 返回 tool_calls → ToolRegistry 逐个执行
  ③ 工具结果直接作为回复（跳过再生成确认语的那轮推理，
     约省一半时间；工具结果本身已是通顺中文）

后端容错：按 [ollama → llama.cpp] 顺序尝试，失败自动切换。
llama.cpp 后端懒启动：ollama 健康时完全不碰 llama-server，
避免白白等待模型加载（旧版无条件双拉起，且从不释放）。

小模型有时把工具调用写成文本而非结构化输出，由
_extract_tool_calls 兜底解析（仅限已注册工具名）；再不行按
关键词规则路由（_rule_intent），保证门禁操作类意图可执行。

安全护栏：疑问句（谁/哪些/名单/吗/为什么…）一律不执行开门，
物理动作不允许被询问式语句触发。事件日志写 assets/logs/agent.log。
"""
import json
import logging
import os
import re

from .llm import OllamaClient, ollama_healthy, ToolCall, ChatReply
from .llama_backend import LlamaCppClient
from .tools import ToolRegistry

_PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def _make_logger():
    """agent 事件日志（assets/logs/agent.log）：诊断路由/工具执行用。"""
    logger = logging.getLogger("edgebrain.agent")
    if logger.handlers:
        return logger
    path = os.path.join(_PROJECT_ROOT, "assets", "logs", "agent.log")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError:
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                           "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_log = _make_logger()

SYSTEM_PROMPT = (
    "你是 EdgeFace 智能门禁的语音助手，只能通过调用工具操作门禁系统。\n"
    "用户意图与工具的对应关系：\n"
    "- 用户要求开门（如「开门」「帮我开一下门」）→ 调用 open_door\n"
    "- 用户问识别记录（如「谁来过」「今天有谁」）→ 调用 recent_logs\n"
    "- 用户问注册名单（如「哪些人能开门」「注册了谁」）→ 调用 list_users\n"
    "与门禁无关的闲聊直接简短回答，不要调用工具。\n"
    "回答用中文，一句话即可。"
)

# 规则兜底：LLM 未调用工具时按关键词直接路由（顺序敏感，
# 查询类在前，避免"哪些人能开门"被"开门"关键词误路由）
_RULE_INTENTS = [
    (("谁来过", "今天有谁", "识别记录", "来过"), "recent_logs"),
    (("名单", "谁能", "哪些人", "注册了"), "list_users"),
    (("开门", "打开门", "开一下门", "解锁"), "open_door"),
]

# 安全护栏：文本呈询问形式时禁止触发开门（物理动作，疑问句不执行）
_OPEN_QUESTION_MARKERS = ("谁", "哪些", "哪几个", "名单", "记录",
                          "吗", "呢", "为什么", "怎么", "如何", "是不是")

# 疑问句被拦下开门时的统一答复
_OPEN_REFUSAL = ("抱歉，没有执行开门：查询或疑问请换一种问法；"
                 "开门请给出明确指令（如「开门」）。")

# 疑似"工具调用写成文本"的起始特征：出现 {（JSON 形式，流式分块
# 可能只剩单字符）或 [工具名]( 函数形式即开始缓冲（不显示原始
# 调用串，避免 UI 闪现 {"name": ...}）
_TOOL_TEXT_RE = re.compile(r'\{|\[(\w+)\]\(')


class EdgeAgent:
    """端侧门禁助手：LLM + 工具调用。"""

    def __init__(self, model: str = "qwen2.5:1.5b",
                 registry: ToolRegistry = None,
                 system_prompt: str = SYSTEM_PROMPT,
                 backend: str = "auto"):
        """Args:
            model: ollama 模型名（backend="llama" 时忽略）
            registry: 工具注册表；None 自动创建（FakeDoor + 默认日志）
            system_prompt: 系统提示词
            backend: "auto"=ollama 优先失败回落 llama.cpp；
                     "ollama" / "llama" 强制指定
        """
        self.registry = registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.model = model

        # 后端懒初始化：__init__ 不做任何网络/进程操作，
        # 首次对话时按序尝试并缓存（llama.cpp 只在 ollama 不可用时拉起）
        self._backend = backend
        self._ollama = None      # OllamaClient / None（未尝试或已放弃）
        self._ollama_dead = False
        self._llama = None       # LlamaCppClient（懒 spawn，成功后缓存）
        self._last_error = None

    # ---- 主入口 ----
    def chat_stream(self, user_text: str, history: list = None):
        """流式处理用户的一句话，逐块产出事件。

        事件协议（生成器 yield）：
          ("text", s)        可见文本块（LLM 逐 token 输出）
          ("replace", s)     工具已执行：用 s 替换之前已显示的文本
        最终显示内容 = 最后一个 replace 的文本，或全部 text 拼接。

        Args:
            user_text: 用户输入
            history: 之前对话历史（OpenAI 格式，不含 system）；None 新会话
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        if _log is not None:
            _log.info("用户: %s (历史 %d 条)", user_text, len(history or []))

        texts = []      # 已显示的文本
        pending = []    # 疑似工具调用文本，暂缓显示（流结束再定去留）
        tool_calls = []
        for event, value in self._stream_with_fallback(messages):
            if event == "text":
                if pending:
                    pending.append(value)
                    continue
                m = _TOOL_TEXT_RE.search(value)
                if m:
                    if m.start() > 0:
                        yield ("text", value[:m.start()])
                    pending.append(value[m.start():])
                else:
                    texts.append(value)
                    yield (event, value)
            elif event == "tool_calls":
                tool_calls.extend(value)

        streamed_all = "".join(texts) + "".join(pending)
        if _log is not None:
            _log.info("模型输出: %s", streamed_all[:200] or "（空）")
            if tool_calls:
                _log.info("模型 tool_calls: %s",
                          [(tc.name, tc.arguments) for tc in tool_calls])

        if not tool_calls:
            # 小模型有时把工具调用写成文本，正则兜底（仅已注册工具名）
            tool_calls = self._extract_tool_calls(
                ChatReply(content=streamed_all))

        # 安全护栏：疑问句不执行开门（物理动作）。
        # 拦掉 open_door 后若还有别的调用则照常执行；没有则落入
        # 下面的规则兜底（如"有哪些人能开门"→名单），规则分支
        # 同样受护栏保护，不会重新触发开门。
        if self._is_open_question(user_text) and \
                any(tc.name == "open_door" for tc in tool_calls):
            tool_calls = [tc for tc in tool_calls if tc.name != "open_door"]
            if _log is not None:
                _log.info("护栏：疑问句拦截 open_door")

        if not tool_calls:
            # 关键词规则兜底：LLM 没调工具时保证门禁命令可执行
            rule = self._rule_intent(user_text)
            if rule:
                if rule[0] == "open_door" and self._is_open_question(user_text):
                    if _log is not None:
                        _log.info("护栏：疑问句拦截规则路由 open_door")
                    yield ("replace", _OPEN_REFUSAL)
                    return
                if _log is not None:
                    _log.info("规则兜底: %s", rule[0])
                yield ("replace", self.registry.dispatch(rule[0], rule[1]))
                return
            # 无工具、无规则：缓冲文本不是调用串，是正常回答，放行显示
            if pending:
                yield ("text", "".join(pending))
            return  # 流式文本即最终答复

        # 执行工具，结果直接作为回复（跳过确认语推理轮）
        if _log is not None:
            _log.info("执行工具: %s", [(tc.name, tc.arguments)
                                       for tc in tool_calls])
        results = [self.registry.dispatch(tc.name, tc.arguments)
                   for tc in tool_calls]
        out = "\n".join(r for r in results if r) or "（工具执行完成）"
        yield ("replace", out)

    def chat(self, user_text: str, history: list = None) -> str:
        """非流式兼容接口：收集 chat_stream 的全部事件返回最终文本。"""
        try:
            parts = []
            for event, value in self.chat_stream(user_text, history):
                if event == "replace":
                    parts = [value]   # 工具结果替换已流出的文本
                else:
                    parts.append(value)
            return "".join(parts) or "（空回复，请重试）"
        except Exception as e:
            return f"⚠️ 出错了：{e}"

    # ---- 后端管理 ----
    def _stream_with_fallback(self, messages):
        """按 [ollama → llama.cpp] 顺序取流。

        某后端在**首个事件之前**失败才切换下一后端；已流出部分
        文本后失败则直接抛异常（半截输出无法干净重试）。
        """
        # ① ollama（auto 模式下探测一次，失败后本次进程内不再尝试）
        if self._backend in ("auto", "ollama") and not self._ollama_dead:
            if self._ollama is None:
                if self._backend == "ollama" or ollama_healthy():
                    self._ollama = OllamaClient(model=self.model)
                else:
                    self._ollama_dead = True
            if self._ollama is not None:
                try:
                    if _log is not None:
                        _log.info("backend=ollama")
                    gen = self._ollama.chat_stream(
                        messages, tools=self.registry.specs())
                    first = next(gen)   # 首个 chunk 前失败 → 换后端
                    yield first
                    yield from gen
                    return
                except Exception as e:
                    if _log is not None:
                        _log.info("ollama 失败: %s", e)
                    self._last_error = e
                    self._ollama_dead = True

        # ② llama.cpp（首次使用时拉起子进程，之后复用）
        if self._backend in ("auto", "llama"):
            if self._llama is None:
                try:
                    self._llama = LlamaCppClient()
                except Exception as e:
                    self._last_error = e
            if self._llama is not None:
                if _log is not None:
                    _log.info("backend=llama.cpp")
                gen = self._llama.chat_stream(
                    messages, tools=self.registry.specs())
                yield from gen   # 已无后备，失败直接抛出
                return
        raise RuntimeError(f"本地大模型不可用：{self._last_error}")

    def _extract_tool_calls(self, reply) -> list:
        """从回复中提取工具调用。

        优先用结构化 tool_calls；小模型有时把调用写成文本，用正则兜底：
          - {"name": "xxx", "arguments": {...}}  JSON 形式
          - [xxx]()                               函数形式
        文本解析出的调用只认已注册的工具名（白名单）。
        """
        if reply.tool_calls:
            return reply.tool_calls

        content = reply.content or ""
        known = set(self.registry._tools.keys())
        calls = []

        # JSON 形式：{"name": "list_users", "arguments": {...}}
        for m in re.finditer(
                r'\{"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{[^{}]*\})\s*\}',
                content):
            name, raw_args = m.group(1), m.group(2)
            if name not in known:
                continue
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(name=name, arguments=args))

        # 函数形式：[list_users]()
        for m in re.finditer(r'\[(\w+)\]\(', content):
            name = m.group(1)
            if name in known:
                calls.append(ToolCall(name=name, arguments={}))

        return calls

    @staticmethod
    def _rule_intent(text: str):
        """按关键词匹配意图，返回 (工具名, 参数)；无匹配返回 None。"""
        for words, tool in _RULE_INTENTS:
            if any(w in text for w in words):
                return (tool, {})
        return None

    @staticmethod
    def _is_open_question(text: str) -> bool:
        """文本是否呈询问形式（疑问词/语气词），用于禁止疑问句开门。"""
        return any(m in text for m in _OPEN_QUESTION_MARKERS)

    def close(self) -> None:
        """释放后端资源（llama.cpp 子进程等）。"""
        for backend in (self._ollama, self._llama):
            if backend is None:
                continue
            try:
                backend.close()
            except Exception:
                pass
