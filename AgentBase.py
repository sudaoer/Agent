from pathlib import Path
from collections import deque
from contextlib import contextmanager
from openai import OpenAI
import httpx
from typing import Any, Callable, Optional, Iterable
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from .ToolBase import ToolBase
from ..ConfigMgr.configMgr import get_max_concurrent_generations, normalize_base_url


def get_local_config(config_name: str) -> dict[str, str]:
    default_config = Path(__file__).resolve().parent / "secret.json"
    if not default_config.exists():
        raise FileNotFoundError(f"Local config file not found: {default_config}")
    with open(default_config, "r") as f:
        config_data = json.load(f)
    if config_name not in config_data:
        raise ValueError(f"Config '{config_name}' not found in local config file")
    return config_data[config_name]


class AgentBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_system_prompt(self, system_prompt: str) -> None:
        pass

    @abstractmethod
    def chat(self, messages: Any) -> str:
        pass


class AgentBase_OpenAIBackend(AgentBase):
    _logger = logging.getLogger(__name__)
    _endpoint_limiters: dict[str, "_EndpointConcurrencyLimiter"] = {}
    _endpoint_limiters_lock = threading.Lock()

    class _EndpointConcurrencyLimiter:
        def __init__(self, max_concurrent: int):
            self.max_concurrent = max_concurrent
            self._active_count = 0
            self._next_ticket = 0
            self._waiting_tickets: deque[int] = deque()
            self._condition = threading.Condition()

        @contextmanager
        def acquire(self):
            ticket = self._enqueue_ticket()
            try:
                self._wait_for_turn(ticket)
                yield
            finally:
                self._release_slot()

        def _enqueue_ticket(self) -> int:
            with self._condition:
                ticket = self._next_ticket
                self._next_ticket += 1
                self._waiting_tickets.append(ticket)
                return ticket

        def _wait_for_turn(self, ticket: int) -> None:
            with self._condition:
                while True:
                    is_front = len(self._waiting_tickets) > 0 and self._waiting_tickets[0] == ticket
                    has_capacity = self._active_count < self.max_concurrent
                    if is_front and has_capacity:
                        self._waiting_tickets.popleft()
                        self._active_count += 1
                        return
                    self._condition.wait()

        def _release_slot(self) -> None:
            with self._condition:
                self._active_count -= 1
                self._condition.notify_all()

    def __init__(
        self,
        name: str,
        base_url: str,
        model_name: str,
        api_key: Optional[
            str
        ] = None,  # lm-studio或其他本地部署的模型可能不需要api_key，但是仍然需要传参
        proxy: Optional[str] = "http://127.0.0.1:7890",
    ):

        self.name = name
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = "test_api_key" if api_key is None else api_key
        self.normalized_base_url = normalize_base_url(self.base_url)
        self.max_concurrent_generations = get_max_concurrent_generations(
            self.base_url,
            self.model_name,
        )

        parsed_base_url = urlparse(self.base_url)
        if parsed_base_url.hostname in {"127.0.0.1", "localhost"}:
            proxy = None

        self.httpx_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.openai_client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, http_client=self.httpx_client
        )

        self.system_prompts: list[str] = []

        # 检查模型是否可用
        model_list = self.openai_client.models.list()
        if self.model_name not in [model.id for model in model_list.data]:
            raise ValueError(
                f"Model '{self.model_name}' is not available in the model list: {[model.id for model in model_list.data]}"
            )

        self.tool_list: list[ToolBase] = []  # 存储注册的工具
        self.tool_list_jsonready_cache: list[dict[str, Any]] = (
            []
        )  # 存储工具列表的JSON-ready版本，每个元素是一个字典，包含工具名称和描述，用于传给模型
        self.token_usage: dict[str, int] = {}
        self.history: list[dict[str, Any]] = []
        self.event_callback: Optional[Callable[[str, dict[str, Any]], None]] = None

    @classmethod
    def _get_endpoint_limiter(
        cls,
        normalized_base_url: str,
        max_concurrent_generations: int | None,
    ) -> "_EndpointConcurrencyLimiter | None":
        if max_concurrent_generations is None:
            return None
        with cls._endpoint_limiters_lock:
            limiter = cls._endpoint_limiters.get(normalized_base_url)
            if limiter is None:
                limiter = cls._EndpointConcurrencyLimiter(max_concurrent_generations)
                cls._endpoint_limiters[normalized_base_url] = limiter
                return limiter
            limiter.max_concurrent = max_concurrent_generations
            return limiter

    @contextmanager
    def _acquire_generation_slot(self):
        limiter = self._get_endpoint_limiter(
            self.normalized_base_url,
            self.max_concurrent_generations,
        )
        if limiter is None:
            yield
            return
        self._logger.debug(
            "Waiting for generation slot on %s (limit=%s)",
            self.normalized_base_url,
            self.max_concurrent_generations,
        )
        with limiter.acquire():
            yield

    # 获取当前Agent自创建以来的token消耗量
    def get_token_consumption(self) -> dict[str, int]:
        return self.token_usage

    # 设置系统提示词
    def add_system_prompt(self, system_prompt: str) -> None:
        self.system_prompts.append(system_prompt)

    def set_event_callback(
        self, callback: Optional[Callable[[str, dict[str, Any]], None]]
    ) -> None:
        self.event_callback = callback

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    def write_history_log(self, log_path: Optional[str]) -> None:
        if log_path is not None:

            # 递归遍历history中的每个str，如果这个str可以被解读为一个json对象，就将这个str替换成这个json对象
            def try_parse_json(obj: Any) -> Any:
                if isinstance(obj, str):
                    try:
                        return json.loads(obj)
                    except Exception:
                        return obj
                elif isinstance(obj, list):
                    return [try_parse_json(item) for item in obj]  # type: ignore
                elif isinstance(obj, dict):
                    return {key: try_parse_json(value) for key, value in obj.items()}  # type: ignore
                else:
                    return obj

            with open(log_path, "w") as f:
                json.dump(try_parse_json(self.history), f, indent=4, ensure_ascii=False)

    # 进行一次对话，输入为用户消息，输出为模型回复，并且处理工具调用并更新对话历史
    def _normalize_user_message(
        self, messages: str | list[dict[str, Any]]
    ) -> dict[str, Any]:
        if isinstance(messages, str):
            return {"role": "user", "content": messages, "_message_source": "user"}
        else:
            normalized_content: list[dict[str, Any]] = []
            for item in messages:

                normalized_item = dict(item)
                item_type = normalized_item.get("type")
                if item_type not in {"text", "image_url"}:
                    raise ValueError(f"Unsupported user content type: {item_type}")
                if item_type == "image_url":
                    image_url = normalized_item.get("image_url")
                    if not isinstance(image_url, dict):
                        raise ValueError("image_url content must contain an object.")
                    url = image_url.get("url") # type: ignore
                    if not isinstance(url, str) or not url.startswith("data:image/"):
                        raise ValueError(
                            "Only base64 image data URLs are supported for image input."
                        )
                normalized_content.append(normalized_item)
            return {
                "role": "user",
                "content": normalized_content,
                "_message_source": "user",
            }

        raise TypeError("messages must be a string or a multimodal content list.")

    def _strip_internal_fields(self, obj: Any) -> Any:
        if isinstance(obj, list):
            return [self._strip_internal_fields(item) for item in obj]  # type: ignore 评价为掩耳盗铃
        if isinstance(obj, dict):
            return {
                key: self._strip_internal_fields(value)
                for key, value in obj.items()  # type: ignore
                if not key.startswith("_")  # type: ignore
            }  # type: ignore
        return obj

    def chat(self, messages: Any, log_path: Optional[str] = None) -> str:
        # 如果没有历史记录，则使用系统提示词作为对话的开头
        if len(self.history) == 0:
            self.history = [
                {"role": "system", "content": "\n".join(self.system_prompts)}
            ]
        user_message = self._normalize_user_message(messages)
        self.history.append(user_message)
        self.emit_event("user_message", {"content": user_message["content"]})

        response = self._post_chatHistory(self.history, log_path=log_path)

        while (
            response.choices[0].message.tool_calls is not None
            and len(response.choices[0].message.tool_calls) > 0
        ):
            self.history.append(response.choices[0].message.model_dump())
            self.emit_event(
                "assistant_tool_calls",
                {
                    "tool_calls": [
                        tool_call.model_dump()
                        for tool_call in response.choices[0].message.tool_calls
                    ]
                },
            )
            for tool_call in response.choices[0].message.tool_calls:
                self.history = self._handle_tool_call(
                    tool_call.model_dump(), self.history
                )
            response = self._post_chatHistory(self.history, log_path=log_path)

        assistant_reply = response.choices[0].message.content
        assert assistant_reply is not None, "模型回复的内容为None"
        self.history.append(response.choices[0].message.model_dump())  # type: ignore
        self.emit_event("assistant_message", {"content": assistant_reply})
        self.write_history_log(log_path)
        return assistant_reply

    # 获取当前Agent的对话历史，返回一个列表，每个元素是一个字典
    # 分别表示消息的角色（如"user"或"assistant"）和消息内容
    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    from openai.types.chat.chat_completion import ChatCompletion

    def _post_chatHistory(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        log_path: Optional[str] = None,
    ) -> ChatCompletion:

        self.write_history_log(log_path)
        api_history = self._strip_internal_fields(history)
        while True:
            with self._acquire_generation_slot():
                time_start = time.perf_counter()
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=api_history,  # type: ignore
                    tools=self.tool_list_jsonready_cache,  # type: ignore
                    extra_body=extra_body,
                )
                time_end = time.perf_counter()
            this_usage = self._handle_usage(response.usage)
            # 计算token生成速度
            completion_tokens = this_usage.get("completion_tokens", 0)
            time_cost = time_end - time_start
            tokens_per_second = completion_tokens / time_cost if time_cost > 0 else 0
            self._logger.info(
                f"{tokens_per_second:.2f} tokens/s, use time {time_cost:.2f}s"
            )
            self.emit_event(
                "model_response",
                {
                    "usage": this_usage,
                    "time_cost": time_cost,
                    "tokens_per_second": tokens_per_second,
                },
            )
            # 检测toolcall是否合法
            all_tool_calls_valid = True
            for tool_call in response.choices[0].message.tool_calls or []:
                tool_arg = tool_call.function.arguments  # type: ignore
                try:
                    tool_arg = json.loads(tool_arg)  # type: ignore
                    if not isinstance(tool_arg, dict):  # type: ignore
                        raise ValueError("工具参数应该是一个JSON对象格式的字符串")
                except Exception as e:
                    all_tool_calls_valid = False
                    self._logger.error(
                        f"Invalid tool call arguments: {tool_arg}, error: {e}\nRetrying..."
                    )
                    break
            if all_tool_calls_valid:
                break

        return response

    from openai.types.completion_usage import CompletionUsage

    def _handle_usage(self, usage: Optional[CompletionUsage]) -> dict[str, Any]:
        if usage is None:
            return {}
        usage_dict = usage.model_dump()
        for k, v in usage_dict.items():
            if isinstance(v, int):
                self.token_usage[k] = self.token_usage.get(k, 0) + v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():  # type: ignore
                    if isinstance(sub_v, int):
                        self.token_usage[sub_k] = self.token_usage.get(sub_k, 0) + sub_v  # type: ignore
        return usage_dict

    def register_tool(self, tool: ToolBase) -> None:
        self.tool_list.append(tool)
        self.tool_list_jsonready_cache.append(tool.to_dict())
        extra_prompt = tool.extra_system_prompt()
        if extra_prompt:
            self.add_system_prompt(extra_prompt)

    def register_tool_list(self, tools: Iterable[ToolBase]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def _handle_tool_call(
        self, tool_call: dict[str, Any], ctx: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tool_name = tool_call["function"]["name"]
        tool_args: dict[str, Any] | str
        try:
            tool_args = ToolBase.parse_toolcall_arguments(tool_call)
        except Exception:
            tool_args = tool_call["function"]["arguments"]
        self.emit_event(
            "tool_call_start",
            {
                "tool_name": tool_name,
                "arguments": tool_args,
                "tool_call_id": tool_call["id"],
            },
        )
        for tool in self.tool_list:
            if tool.toolName == tool_name:
                new_ctx = tool.execute(tool_call, ctx)
                tool_result = ""
                for message in reversed(new_ctx):
                    if (
                        isinstance(message, dict) # type: ignore
                        and message.get("role") == "tool"
                        and message.get("tool_call_id") == tool_call["id"]
                    ):
                        tool_result = str(message.get("content", ""))
                        break
                self.emit_event(
                    "tool_call_end",
                    {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "tool_call_id": tool_call["id"],
                        "result": tool_result,
                    },
                )
                return new_ctx
        error_msg = f"Tool '{tool_name}' not found in registered tools"
        self._logger.error(error_msg)
        self.emit_event(
            "tool_call_end",
            {
                "tool_name": tool_name,
                "arguments": tool_args,
                "tool_call_id": tool_call["id"],
                "result": error_msg,
            },
        )
        ctx.append(
            {
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call["id"],
            }
        )
        return ctx
