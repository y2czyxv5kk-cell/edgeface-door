"""llama.cpp 后端：llama-server 二进制 + OpenAI 兼容 API。

llama-server.exe 是独立进程：加载 GGUF 模型 → 在本地端口提供
/v1/chat/completions 接口（原生支持工具调用）。本类负责：
拉起子进程、等待模型加载完成、走 OpenAI 协议对话、释放时关闭进程。

GGUF 模型放在 assets/models/llm/（如 qwen2.5-0.5b-instruct-q4_k_m.gguf）。
"""
import os
import subprocess
import sys
import time

import httpx

from .llm import ChatReply, ToolCall

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# llama.cpp 的 exe 依赖同目录的 CPU DLL，须整目录部署
_DEFAULT_EXE = os.path.join(_PROJECT_ROOT, "assets", "models", "llm",
                            "llama", "llama-server.exe")
_LLM_DIR = os.path.join(_PROJECT_ROOT, "assets", "models", "llm")
_GGUF_BIG = os.path.join(_LLM_DIR, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
_GGUF_SMALL = os.path.join(_LLM_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")


def _pick_default_gguf() -> str:
    """默认模型：1.5B 优先（工具调用更稳），不存在则用 0.5B。"""
    for p in (_GGUF_BIG, _GGUF_SMALL):
        if os.path.isfile(p):
            return p
    return _GGUF_BIG  # 都不存在时交给构造函数报错


class LlamaCppClient:
    """llama-server 子进程管理 + OpenAI 兼容对话客户端。

    接口与 OllamaClient 一致（chat 返回 ChatReply）。
    """

    def __init__(self, model_path: str = None, server_exe: str = None,
                 port: int = 8181, n_ctx: int = 2048, n_threads: int = None,
                 load_timeout: float = 180.0):
        """Args:
            model_path: GGUF 模型路径
            server_exe: llama-server.exe 路径
            port: 本地服务端口（8181，避开 ollama 的 11434）
            n_ctx: 上下文窗口长度
            n_threads: 推理线程数；None 自动
            load_timeout: 等待模型加载完成的超时（秒）
        """
        self.model_path = model_path or _pick_default_gguf()
        self.server_exe = server_exe or _DEFAULT_EXE
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"

        if not os.path.isfile(self.server_exe):
            raise FileNotFoundError(
                f"找不到 llama-server.exe：{self.server_exe}"
            )
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"找不到 GGUF 模型：{self.model_path}"
            )

        # 拉起 server；若加载失败（常见于内存不足）且用的是大模型，
        # 自动回退到 0.5B 重试
        self._proc = self._spawn_server(n_ctx, n_threads)
        try:
            self._wait_ready(load_timeout)
        except RuntimeError:
            if self.model_path == _GGUF_BIG and os.path.isfile(_GGUF_SMALL):
                print(f"[warn] 1.5B 模型加载失败（内存不足？），回退到 0.5B")
                self.model_path = _GGUF_SMALL
                self._proc = self._spawn_server(n_ctx, n_threads)
                self._wait_ready(load_timeout)
            else:
                raise
        self._client = httpx.Client(base_url=self.base_url, timeout=120.0)

    def _spawn_server(self, n_ctx: int, n_threads: int) -> subprocess.Popen:
        """拉起 llama-server 子进程。Windows 下隐藏控制台窗口。"""
        cmd = [self.server_exe, "-m", self.model_path,
               "--port", str(self.port),
               "--host", "127.0.0.1",
               "-c", str(n_ctx)]
        if n_threads:
            cmd += ["-t", str(n_threads)]
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, **kwargs)

    def _wait_ready(self, timeout: float) -> None:
        """轮询 /health 直到模型加载完成（加载期间该接口返回 503）。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server 启动后立即退出")
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError(f"llama-server 在 {timeout}s 内未就绪")

    def chat(self, messages: list, tools: list = None) -> ChatReply:
        """发一轮对话（OpenAI 兼容协议，原生支持工具调用）。"""
        payload = {
            "model": "local-model",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.2,   # 工具调用场景用低温度更稳定
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        r = self._client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]

        tool_calls = []
        for tc in (msg.get("tool_calls") or []):
            args = tc["function"].get("arguments") or "{}"
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(
                name=tc["function"]["name"], arguments=args,
                id=tc.get("id", "")))
        return ChatReply(content=(msg.get("content") or "").strip(),
                         tool_calls=tool_calls)

    def chat_stream(self, messages: list, tools: list = None):
        """流式对话（SSE）：逐 chunk yield ("text", 文本)；结束时若模型
        请求了工具，再 yield ("tool_calls", [ToolCall, ...])。

        llama-server 流式协议：`data: {...}` 每行一个 chunk，
        delta.content 为增量文本，工具调用出现在结尾 chunk 的
        delta.tool_calls 或 message.tool_calls（版本差异，均兼容）。
        """
        payload = {
            "model": "local-model",
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.2,   # 工具调用场景用低温度更稳定
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        import json as _json

        def _parse_args(raw):
            """arguments 可能是 JSON 字符串（流式分片）或已解析的 dict。"""
            if not isinstance(raw, str):
                return raw or {}
            try:
                return _json.loads(raw)
            except _json.JSONDecodeError:
                return {}

        complete = []   # 完整 message.tool_calls（部分版本收尾 chunk 里给全量）
        fragments = {}  # delta.tool_calls 分片，按 index 拼 arguments
        with self._client.stream("POST", "/v1/chat/completions",
                                 json=payload) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = _json.loads(data)
                except _json.JSONDecodeError:
                    continue
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    yield ("text", content)
                for tc in (delta.get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    slot = fragments.setdefault(tc.get("index", 0), {
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "arguments": "",
                    })
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    slot["arguments"] += fn.get("arguments") or ""
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                for tc in ((choice.get("message") or {}).get("tool_calls") or []):
                    fn = tc.get("function") or {}
                    complete.append(ToolCall(
                        name=fn.get("name", ""),
                        arguments=_parse_args(fn.get("arguments")),
                        id=tc.get("id", "")))

        if not complete:
            complete = [ToolCall(name=s["name"],
                                 arguments=_parse_args(s["arguments"]),
                                 id=s["id"])
                        for s in fragments.values()]
        if complete:
            yield ("tool_calls", complete)

    def close(self) -> None:
        """关闭 server 子进程。"""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
