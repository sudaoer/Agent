from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from openai import OpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage
import httpx
from typing import Any, Optional, Iterable, cast
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from urllib.parse import urlparse
from .ToolBase import ToolBase
from ..ConfigMgr.configMgr import (
    get_max_concurrent_generations,
    get_proxy_port,
    is_self_hosted_endpoint,
    normalize_base_url,
)


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

    def chat_stream(
        self,
        messages: Any,
        *,
        log_path: str | None = None,
        extra_body: dict[str, Any] | None = None,
        stream_options: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support chat_stream()."
        )


class Agent_OpenAIChat_API_Backend(AgentBase):
    _logger = logging.getLogger(__name__)
    _endpoint_limiters: dict[str, "_EndpointConcurrencyLimiter"] = {}
    _endpoint_limiters_lock = threading.Lock()
    _ALIYUN_EXPLICIT_CACHE_ROLES = {"system", "user", "assistant", "tool"}

    class _EndpointConcurrencyLimiter:
        _tls = threading.local()
        _DYNAMIC_INCREASE_THRESHOLD = 20

        def __init__(self, max_concurrent: int, *, dynamic: bool = False):
            self.max_concurrent = max_concurrent
            self._dynamic = dynamic
            self._active_count = 0
            self._next_ticket = 0
            self._waiting_tickets: deque[int] = deque()
            self._blocked_until = 0.0
            self._consecutive_rate_limits = 0
            self._successes_at_capacity = 0
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
                    now = time.monotonic()
                    is_front = (
                        len(self._waiting_tickets) > 0
                        and self._waiting_tickets[0] == ticket
                    )
                    has_capacity = self._active_count < self.max_concurrent
                    wait_seconds = max(0.0, self._blocked_until - now)
                    if is_front and has_capacity and wait_seconds <= 0:
                        self._waiting_tickets.popleft()
                        self._active_count += 1
                        type(self)._tls._was_at_capacity = (
                            self._active_count >= self.max_concurrent
                        )
                        return
                    self._condition.wait(
                        timeout=wait_seconds if wait_seconds > 0 else None
                    )

        def _release_slot(self) -> None:
            with self._condition:
                self._active_count -= 1
                self._condition.notify_all()

        def record_rate_limit(
            self,
            retry_after_seconds: float | None,
        ) -> float:
            with self._condition:
                self._consecutive_rate_limits += 1
                self._successes_at_capacity = 0
                exponential_backoff = min(
                    0.5 * (2 ** (self._consecutive_rate_limits - 1)),
                    8.0,
                )
                wait_seconds = max(
                    retry_after_seconds or 0.0,
                    exponential_backoff,
                )
                self._blocked_until = max(
                    self._blocked_until,
                    time.monotonic() + wait_seconds,
                )
                if self._dynamic and self.max_concurrent > 1:
                    old = self.max_concurrent
                    self.max_concurrent = max(1, self.max_concurrent * 9 // 10)
                    if self.max_concurrent != old:
                        _logger = logging.getLogger(__name__)
                        _logger.warning(
                            "Dynamic concurrency: reduced max_concurrent %d -> %d",
                            old,
                            self.max_concurrent,
                        )
                self._condition.notify_all()
                return wait_seconds

        def record_success(self) -> None:
            with self._condition:
                self._consecutive_rate_limits = 0
                if self._dynamic:
                    was_at_capacity = getattr(
                        type(self)._tls, "_was_at_capacity", False
                    )
                    type(self)._tls._was_at_capacity = False
                    if was_at_capacity:
                        self._successes_at_capacity += 1
                        if (
                            self._successes_at_capacity
                            >= self._DYNAMIC_INCREASE_THRESHOLD
                        ):
                            self._successes_at_capacity = 0
                            old = self.max_concurrent
                            self.max_concurrent += 1
                            _logger = logging.getLogger(__name__)
                            _logger.warning(
                                "Dynamic concurrency: increased max_concurrent %d -> %d",
                                old,
                                self.max_concurrent,
                            )
                self._condition.notify_all()

    def __init__(
        self,
        name: str,
        base_url: str,
        model_name: str,
        api_key: Optional[
            str
        ] = None,  # lm-studio或其他本地部署的模型可能不需要api_key，但是仍然需要传参
        proxy: Optional[str] = None,
    ):

        self.name = name
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = "test_api_key" if api_key is None else api_key
        self.normalized_base_url = normalize_base_url(self.base_url)
        self._is_self_hosted = is_self_hosted_endpoint(self.base_url)
        self.max_concurrent_generations = get_max_concurrent_generations(
            self.base_url,
            self.model_name,
        )

        parsed_base_url = urlparse(self.base_url)
        if parsed_base_url.hostname in {"127.0.0.1", "localhost"}:
            proxy = None
        elif proxy is None:
            port = get_proxy_port()
            if port is not None:
                proxy = f"http://127.0.0.1:{port}"

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

    @classmethod
    def _get_endpoint_limiter(
        cls,
        normalized_base_url: str,
        max_concurrent_generations: int | None,
        dynamic: bool,
    ) -> "_EndpointConcurrencyLimiter | None":
        if max_concurrent_generations is None:
            return None
        with cls._endpoint_limiters_lock:
            limiter = cls._endpoint_limiters.get(normalized_base_url)
            if limiter is None:
                limiter = cls._EndpointConcurrencyLimiter(
                    max_concurrent_generations,
                    dynamic=dynamic,
                )
                cls._endpoint_limiters[normalized_base_url] = limiter
                return limiter
            limiter.max_concurrent = max(
                limiter.max_concurrent, max_concurrent_generations
            )
            return limiter

    @contextmanager
    def _acquire_generation_slot(self):
        limiter = self._get_endpoint_limiter(
            self.normalized_base_url,
            self.max_concurrent_generations,
            dynamic=not self._is_self_hosted,
        )
        if limiter is None:
            yield
            return
        self._logger.debug(
            "Waiting for generation slot on %s (limit=%s, dynamic=%s)",
            self.normalized_base_url,
            limiter.max_concurrent,
            limiter._dynamic,  # pyright: ignore[reportPrivateUsage]
        )
        with limiter.acquire():
            yield

    def _get_active_endpoint_limiter(self) -> "_EndpointConcurrencyLimiter | None":
        return self._get_endpoint_limiter(
            self.normalized_base_url,
            self.max_concurrent_generations,
            dynamic=not self._is_self_hosted,
        )

    @staticmethod
    def _extract_retry_after_seconds(response: httpx.Response | None) -> float | None:
        if response is None:
            return None
        retry_after = response.headers.get("retry-after")
        if retry_after is None:
            return None
        try:
            parsed = float(retry_after.strip())
        except ValueError:
            return None
        return max(0.0, parsed)

    # 获取当前Agent自创建以来的token消耗量
    def get_token_consumption(self) -> dict[str, int]:
        return self.token_usage

    # 设置系统提示词
    def add_system_prompt(self, system_prompt: str) -> None:
        self.system_prompts.append(system_prompt)

    def _is_aliyun_compatible_host(self) -> bool:
        hostname = urlparse(self.base_url).hostname
        return isinstance(hostname, str) and "aliyun" in hostname.lower()

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

    def _normalize_user_message(
        self, messages: str | list[dict[str, Any]]
    ) -> dict[str, Any]:
        if isinstance(messages, str):
            return {"role": "user", "content": messages}
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
                    url = image_url.get("url")  # type: ignore
                    if not isinstance(url, str) or not url.startswith("data:image/"):
                        raise ValueError(
                            "Only base64 image data URLs are supported for image input."
                        )
                normalized_content.append(normalized_item)
            return {
                "role": "user",
                "content": normalized_content,
            }

    def _build_aliyun_explicit_cache_messages(
        self, history: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        request_messages = deepcopy(history)
        system_message_index: int | None = None

        for index, message in enumerate(request_messages):
            if message.get("role") != "system":
                continue
            if self._mark_aliyun_explicit_cache_message(message):
                system_message_index = index
                break

        for index in range(len(request_messages) - 1, -1, -1):
            if index == system_message_index:
                continue
            message = request_messages[index]
            if message.get("role") not in self._ALIYUN_EXPLICIT_CACHE_ROLES:
                continue
            self._mark_aliyun_explicit_cache_message(message)
            return request_messages

        return request_messages

    def _mark_aliyun_explicit_cache_message(self, message: dict[str, Any]) -> bool:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            return True

        if isinstance(content, list) and len(cast(list[Any], content)) > 0:
            if not all(isinstance(item, dict) for item in content):  # type: ignore
                self._logger.debug(
                    "Skipping Aliyun explicit cache marker because the message content list is not dict-only."
                )
                return False
            content_blocks = cast(list[dict[str, Any]], content)
            content_blocks[-1]["cache_control"] = {"type": "ephemeral"}
            return True

        self._logger.debug(
            "Skipping Aliyun explicit cache marker because %s message content is not safely rewritable.",
            message.get("role"),
        )
        return False

    def _prepare_chat_request(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return history, dict(extra_body) if extra_body is not None else {}

    def _build_chat_completion_kwargs(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        *,
        stream: bool = False,
        stream_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        request_history, request_extra_body = self._prepare_chat_request(
            history, extra_body
        )
        request_messages = request_history
        if self._is_aliyun_compatible_host():
            request_messages = self._build_aliyun_explicit_cache_messages(
                request_history
            )

        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": request_messages,
            "extra_body": request_extra_body or None,
        }
        if self.tool_list_jsonready_cache:
            request_kwargs["tools"] = self.tool_list_jsonready_cache
        if stream:
            resolved_stream_options = {"include_usage": True}
            if stream_options is not None:
                resolved_stream_options.update(stream_options)
            request_kwargs["stream"] = True
            request_kwargs["stream_options"] = resolved_stream_options
        return request_kwargs

    def _emit_model_response_metrics(
        self,
        usage_dict: dict[str, Any],
        *,
        time_start: float,
        time_end: float,
    ) -> None:
        completion_tokens = usage_dict.get("completion_tokens", 0)
        time_cost = time_end - time_start
        tokens_per_second = completion_tokens / time_cost if time_cost > 0 else 0
        self._logger.debug(
            f"{tokens_per_second:.2f} tokens/s, use time {time_cost:.2f}s"
        )

    @staticmethod
    def _tool_calls_are_valid(response: ChatCompletion) -> bool:
        for tool_call in response.choices[0].message.tool_calls or []:
            tool_arg = tool_call.function.arguments  # type: ignore[assignment]
            try:
                parsed_tool_arg = json.loads(tool_arg)  # type: ignore[arg-type]
                if not isinstance(parsed_tool_arg, dict):
                    raise ValueError("工具参数应该是一个JSON对象格式的字符串")
            except Exception:
                return False
        return True

    @staticmethod
    def _extract_reasoning_content(delta: Any) -> str:
        model_extra = getattr(delta, "model_extra", None)
        if isinstance(model_extra, dict):
            model_extra_dict = cast(dict[str, Any], model_extra)
            value = model_extra_dict.get("reasoning_content")
            if isinstance(value, str):
                return value
        value = getattr(delta, "reasoning_content", None)
        if isinstance(value, str):
            return value
        return ""

    def _ensure_history_initialized(self) -> None:
        if len(self.history) == 0:
            self.history = [
                {"role": "system", "content": "\n".join(self.system_prompts)}
            ]

    # 进行一次对话，输入为用户消息，输出为模型回复，并且处理工具调用并更新对话历史
    def chat(self, messages: Any, log_path: Optional[str] = None) -> str:
        self._ensure_history_initialized()
        user_message = self._normalize_user_message(messages)
        self.history.append(user_message)

        response = self._post_chatHistory(self.history, log_path=log_path)

        while (
            response.choices[0].message.tool_calls is not None
            and len(response.choices[0].message.tool_calls) > 0
        ):
            self.history.append(response.choices[0].message.model_dump())

            for tool_call in response.choices[0].message.tool_calls:
                self.history = self._handle_tool_call(
                    tool_call.model_dump(), self.history
                )
            response = self._post_chatHistory(self.history, log_path=log_path)

        assistant_reply = response.choices[0].message.content
        assert assistant_reply is not None, "模型回复的内容为None"
        self.history.append(response.choices[0].message.model_dump())  # type: ignore
        self.write_history_log(log_path)
        return assistant_reply

    def chat_stream(
        self,
        messages: Any,
        *,
        log_path: Optional[str] = None,
        extra_body: Optional[dict[str, Any]] = None,
        stream_options: Optional[dict[str, Any]] = None,
    ) -> str:
        self._ensure_history_initialized()
        user_message = self._normalize_user_message(messages)
        self.history.append(user_message)

        assistant_reply, reasoning_content = self._stream_chatHistory(
            self.history,
            extra_body=extra_body,
            log_path=log_path,
            stream_options=stream_options,
        )
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_reply,
        }
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content
        self.history.append(assistant_message)
        self.write_history_log(log_path)
        return assistant_reply

    # 获取当前Agent的对话历史，返回一个列表，每个元素是一个字典
    # 分别表示消息的角色（如"user"或"assistant"）和消息内容
    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    def _post_chatHistory(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        log_path: Optional[str] = None,
    ) -> ChatCompletion:

        self.write_history_log(log_path)
        request_kwargs = self._build_chat_completion_kwargs(history, extra_body)
        rate_limit_attempts = 0
        while True:
            try:
                with self._acquire_generation_slot():
                    time_start = time.perf_counter()
                    response = cast(
                        ChatCompletion,
                        self.openai_client.chat.completions.create(**request_kwargs),
                    )
                    time_end = time.perf_counter()
            except RateLimitError as exc:
                rate_limit_attempts += 1
                limiter = self._get_active_endpoint_limiter()
                retry_after_seconds = self._extract_retry_after_seconds(exc.response)
                wait_seconds = max(
                    retry_after_seconds or 0.0,
                    min(0.5 * (2 ** (rate_limit_attempts - 1)), 8.0),
                )
                if limiter is not None:
                    wait_seconds = limiter.record_rate_limit(retry_after_seconds)
                self._logger.warning(
                    "Rate limited on %s, retrying in %.2fs (attempt=%s)",
                    self.normalized_base_url,
                    wait_seconds,
                    rate_limit_attempts,
                )
                time.sleep(wait_seconds)
                continue
            limiter = self._get_active_endpoint_limiter()
            if limiter is not None:
                limiter.record_success()
            rate_limit_attempts = 0
            this_usage = self._handle_usage(response.usage)
            self._emit_model_response_metrics(
                this_usage,
                time_start=time_start,
                time_end=time_end,
            )
            if self._tool_calls_are_valid(response):
                break
            tool_calls = response.choices[0].message.tool_calls or []
            invalid_arg = "<missing>"
            if len(tool_calls) > 0:
                tool_call_payload = tool_calls[0].model_dump()
                raw_function_payload = tool_call_payload.get("function")
                if isinstance(raw_function_payload, dict):
                    function_payload = cast(dict[str, Any], raw_function_payload)
                    arguments: Any = function_payload.get("arguments")
                    if isinstance(arguments, str):
                        invalid_arg = arguments
                    elif arguments is not None:
                        invalid_arg = str(arguments)
            self._logger.error(
                "Invalid tool call arguments: %s\nRetrying...", invalid_arg
            )

        return response

    def _stream_chatHistory(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        log_path: Optional[str] = None,
        stream_options: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        self.write_history_log(log_path)
        request_kwargs = self._build_chat_completion_kwargs(
            history,
            extra_body,
            stream=True,
            stream_options=stream_options,
        )
        rate_limit_attempts = 0
        while True:
            try:
                with self._acquire_generation_slot():
                    time_start = time.perf_counter()
                    stream_response = cast(
                        Iterable[ChatCompletionChunk],
                        self.openai_client.chat.completions.create(**request_kwargs),
                    )
                    assistant_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    usage_dict: dict[str, Any] = {}
                    try:
                        for chunk in stream_response:
                            if chunk.usage is not None:
                                usage_dict = self._handle_usage(chunk.usage)
                            for choice in chunk.choices:
                                delta = choice.delta
                                reasoning_delta = self._extract_reasoning_content(delta)
                                if reasoning_delta:
                                    reasoning_parts.append(reasoning_delta)
                                if (
                                    delta.tool_calls is not None
                                    or delta.function_call is not None
                                    or choice.finish_reason == "tool_calls"
                                ):
                                    raise RuntimeError(
                                        "chat_stream() v1 does not support streamed tool calls."
                                    )
                                content_delta = delta.content
                                if content_delta:
                                    assistant_parts.append(content_delta)
                    finally:
                        close_stream = getattr(stream_response, "close", None)
                        if callable(close_stream):
                            close_stream()
                    time_end = time.perf_counter()
            except RateLimitError as exc:
                rate_limit_attempts += 1
                limiter = self._get_active_endpoint_limiter()
                retry_after_seconds = self._extract_retry_after_seconds(exc.response)
                wait_seconds = max(
                    retry_after_seconds or 0.0,
                    min(0.5 * (2 ** (rate_limit_attempts - 1)), 8.0),
                )
                if limiter is not None:
                    wait_seconds = limiter.record_rate_limit(retry_after_seconds)
                self._logger.warning(
                    "Rate limited on %s, retrying in %.2fs (attempt=%s)",
                    self.normalized_base_url,
                    wait_seconds,
                    rate_limit_attempts,
                )
                time.sleep(wait_seconds)
                continue
            limiter = self._get_active_endpoint_limiter()
            if limiter is not None:
                limiter.record_success()
            rate_limit_attempts = 0
            self._emit_model_response_metrics(
                usage_dict,
                time_start=time_start,
                time_end=time_end,
            )
            return "".join(assistant_parts), "".join(reasoning_parts)

    @staticmethod
    def _accumulate_usage_counts(
        target: dict[str, int], usage_dict: dict[str, Any]
    ) -> None:
        for key, value in usage_dict.items():
            if isinstance(value, int):
                target[key] = target.get(key, 0) + value
                continue
            if not isinstance(value, dict):
                continue
            value = cast(dict[str, Any], value)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, int):
                    target[sub_key] = target.get(sub_key, 0) + sub_value

    def _handle_usage(self, usage: Optional[CompletionUsage]) -> dict[str, Any]:
        if usage is None:
            return {}
        usage_dict = usage.model_dump()
        self._accumulate_usage_counts(self.token_usage, usage_dict)
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
        for tool in self.tool_list:
            if tool.toolName == tool_name:
                new_ctx = tool.execute(tool_call, ctx)
                for message in reversed(new_ctx):
                    if (
                        isinstance(message, dict)  # type: ignore
                        and message.get("role") == "tool"
                        and message.get("tool_call_id") == tool_call["id"]
                    ):
                        break
                return new_ctx
        error_msg = f"Tool '{tool_name}' not found in registered tools"
        self._logger.error(error_msg)
        ctx.append(
            {
                "role": "tool",
                "content": error_msg,
                "tool_call_id": tool_call["id"],
            }
        )
        return ctx
