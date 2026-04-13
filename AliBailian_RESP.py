import json
import logging
import warnings
from typing import Any, Callable, Iterable, Optional, cast

from openai import OpenAI

from .AgentBase import AgentBase, get_local_config
from .ToolBase import ToolBase


class AliBailian_RESP_Agent(AgentBase):
    _logger = logging.getLogger(__name__)

    def __init__(self):
        cfg = get_local_config("AliBailian-qwen3.6-plus")
        self.model_name = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self._logger.setLevel(logging.DEBUG)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.system_prompts: list[str] = []
        self.history: list[dict[str, Any]] = []
        self.tool_list: list[ToolBase] = []
        self.tool_list_jsonready_cache: list[dict[str, Any]] = []
        self.token_usage: dict[str, int] = {}
        self.event_callback: Optional[Callable[[str, dict[str, Any]], None]] = None
        self.round_logs: list[dict[str, Any]] = []

    def add_system_prompt(self, system_prompt: str) -> None:
        if system_prompt.strip() != "":
            self.system_prompts.append(system_prompt)

    def set_event_callback(
        self, callback: Optional[Callable[[str, dict[str, Any]], None]]
    ) -> None:
        self.event_callback = callback

    def emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event_type, payload)

    def get_token_consumption(self) -> dict[str, int]:
        return self.token_usage

    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    def write_history_log(self, log_path: Optional[str]) -> None:
        if log_path is None:
            return

        def try_parse_json(obj: Any) -> Any:
            if isinstance(obj, str):
                try:
                    return json.loads(obj)
                except Exception:
                    return obj
            if isinstance(obj, list):
                return [try_parse_json(item) for item in obj]
            if isinstance(obj, dict):
                return {key: try_parse_json(value) for key, value in obj.items()}
            return obj

        with open(log_path, "w") as f:
            json.dump(
                {
                    "rounds": try_parse_json(self.round_logs),
                    "history": try_parse_json(self.history),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    def _append_round_log(
        self,
        request: dict[str, Any],
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        request_for_log = dict(request)
        request_for_log.pop("tools", None)
        request_for_log.pop("tool_choice", None)
        round_log: dict[str, Any] = {
            "round_index": len(self.round_logs) + 1,
            "request": request_for_log,
        }
        if response is not None:
            response_for_log = dict(response)
            response_for_log.pop("tools", None)
            response_for_log.pop("tool_choice", None)
            round_log["response"] = response_for_log
        if error is not None:
            round_log["error"] = error
        self.round_logs.append(round_log)

    def register_tool(self, tool: ToolBase) -> None:
        self.tool_list.append(tool)
        self.tool_list_jsonready_cache.append(self._convert_tool_schema(tool.to_dict()))
        extra_prompt = tool.extra_system_prompt()
        if extra_prompt:
            self.add_system_prompt(extra_prompt)

    def register_tool_list(self, tools: Iterable[ToolBase]) -> None:
        for tool in tools:
            self.register_tool(tool)

    def _dump_response(self, response: Any) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            response = response.model_dump()  # type: ignore

        if not isinstance(response, dict):
            raise ValueError("Invalid response format: response is not a dictionary.")
        return cast(dict[str, Any], response)

    def _convert_tool_schema(self, tool_schema: dict[str, Any]) -> dict[str, Any]:
        if tool_schema.get("type") != "function":
            raise ValueError(f"Unsupported tool type for RESP: {tool_schema}")
        function_obj = tool_schema.get("function")
        if not isinstance(function_obj, dict):
            raise ValueError(f"Invalid function tool schema: {tool_schema}")
        function_schema = cast(dict[str, Any], function_obj)
        return {
            "type": "function",
            "name": function_schema["name"],
            "description": function_schema.get("description", ""),
            "parameters": function_schema.get(
                "parameters",
                {"type": "object", "properties": {}},
            ),
        }

    def _normalize_user_message(
        self, messages: str | list[dict[str, Any]]
    ) -> dict[str, Any]:
        if isinstance(messages, str):
            user_message = messages.strip()
            if user_message == "":
                raise ValueError("Input message should not be empty.")
            return {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            }

        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Input message list should not be empty.")

        normalized_content: list[dict[str, Any]] = []
        for item in messages:
            if not isinstance(item, dict):
                raise ValueError("Each message content item must be a dictionary.")
            normalized_item = dict(item)
            item_type = normalized_item.get("type")

            if item_type == "text":
                text = normalized_item.get("text")
                if not isinstance(text, str) or text.strip() == "":
                    raise ValueError("Text content must contain a non-empty string.")
                normalized_content.append({"type": "input_text", "text": text})
                continue

            if item_type == "input_text":
                text = normalized_item.get("text")
                if not isinstance(text, str) or text.strip() == "":
                    raise ValueError(
                        "input_text content must contain a non-empty string."
                    )
                normalized_content.append({"type": "input_text", "text": text})
                continue

            if item_type == "image_url":
                image_url = normalized_item.get("image_url")
                if isinstance(image_url, dict):
                    image_url = image_url.get("url")
                if not isinstance(image_url, str) or image_url.strip() == "":
                    raise ValueError("image_url content must contain a non-empty URL.")
                normalized_content.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": "auto",
                    }
                )
                continue

            if item_type == "input_image":
                image_url = normalized_item.get("image_url")
                if not isinstance(image_url, str) or image_url.strip() == "":
                    raise ValueError(
                        "input_image content must contain a non-empty URL."
                    )
                detail = normalized_item.get("detail", "auto")
                normalized_content.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detail,
                    }
                )
                continue

            raise ValueError(f"Unsupported user content type: {item_type}")

        return {"role": "user", "content": normalized_content}

    def _build_history_item(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("role") != "user":
            return message

        content = message["content"]
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
            and content[0].get("type") == "input_text"
        ):
            text = content[0].get("text")
            if isinstance(text, str):
                return {"role": "user", "content": text}
        return {"role": "user", "content": content}

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip() != "":
            return output_text.strip()

        output_raw = response.get("output")
        if not isinstance(output_raw, list):
            return ""
        output = cast(list[Any], output_raw)

        text_parts: list[str] = []
        for output_item_raw in output:
            if not isinstance(output_item_raw, dict):
                continue
            output_item = cast(dict[str, Any], output_item_raw)
            if output_item.get("type") != "message":
                continue
            content_raw = output_item.get("content")
            if not isinstance(content_raw, list):
                continue
            content = cast(list[Any], content_raw)
            for content_item_raw in content:
                if not isinstance(content_item_raw, dict):
                    continue
                content_item = cast(dict[str, Any], content_item_raw)
                text = content_item.get("text")
                if isinstance(text, str) and text.strip() != "":
                    text_parts.append(text.strip())

        return "\n".join(text_parts).strip()

    def _extract_function_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        output_raw = response.get("output")
        if not isinstance(output_raw, list):
            return []
        function_calls: list[dict[str, Any]] = []
        for output_item_raw in output_raw:
            if not isinstance(output_item_raw, dict):
                continue
            output_item = cast(dict[str, Any], output_item_raw)
            if output_item.get("type") == "function_call":
                function_calls.append(output_item)
        return function_calls

    def _append_response_output_to_history(self, response: dict[str, Any]) -> None:
        response_id = response.get("id")
        output_raw = response.get("output")
        if not isinstance(output_raw, list):
            return
        for output_item_raw in output_raw:
            if not isinstance(output_item_raw, dict):
                continue
            output_item = cast(dict[str, Any], output_item_raw)
            item_copy = dict(output_item)
            if response_id is not None and "response_id" not in item_copy:
                item_copy["response_id"] = response_id
            self.history.append(item_copy)

    def _stringify_tool_result(self, result: Any) -> str:
        if isinstance(result, str):
            return result

        if isinstance(result, list):
            text_parts: list[str] = []
            image_omitted = False
            content_like = True
            for item in result:
                if not isinstance(item, dict):
                    content_like = False
                    break
                item_type = item.get("type")
                if item_type in {"text", "input_text"}:
                    text = item.get("text")
                    if isinstance(text, str) and text.strip() != "":
                        text_parts.append(text.strip())
                    continue
                if item_type in {"image_url", "input_image", "file", "input_file"}:
                    image_omitted = True
                    continue
                content_like = False
                break

            if content_like:
                if text_parts:
                    if image_omitted:
                        text_parts.append(
                            "[tool image output omitted for RESP compatibility]"
                        )
                    return "\n".join(text_parts)
                if image_omitted:
                    return (
                        "Tool returned image content only; image omitted because RESP "
                        "function_call_output must be string-only."
                    )

        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    def _handle_usage(self, usage: Any) -> dict[str, Any]:
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif isinstance(usage, dict):
            usage_dict = usage
        else:
            return {}

        if not isinstance(usage_dict, dict):
            return {}

        for k, v in usage_dict.items():
            if isinstance(v, int):
                self.token_usage[k] = self.token_usage.get(k, 0) + v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, int):
                        self.token_usage[sub_k] = self.token_usage.get(sub_k, 0) + sub_v
        return cast(dict[str, Any], usage_dict)

    def _build_tool_call_info(self, function_call: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": function_call["call_id"],
            "type": "function",
            "function": {
                "name": function_call["name"],
                "arguments": function_call["arguments"],
            },
        }

    def _handle_tool_call(
        self, function_call: dict[str, Any]
    ) -> tuple[dict[str, Any], str]:
        tool_call_info = self._build_tool_call_info(function_call)
        tool_name = tool_call_info["function"]["name"]
        tool_args: dict[str, Any] | str
        try:
            tool_args = ToolBase.parse_toolcall_arguments(tool_call_info)
        except Exception:
            tool_args = tool_call_info["function"]["arguments"]

        self.emit_event(
            "tool_call_start",
            {
                "tool_name": tool_name,
                "arguments": tool_args,
                "tool_call_id": tool_call_info["id"],
            },
        )

        tool_result_text = ""
        for tool in self.tool_list:
            if tool.toolName != tool_name:
                continue
            temp_ctx: list[dict[str, Any]] = []
            new_ctx = tool.execute(tool_call_info, temp_ctx)
            tool_payload = ""
            for message in reversed(new_ctx):
                if (
                    isinstance(message, dict)
                    and message.get("role") == "tool"
                    and message.get("tool_call_id") == tool_call_info["id"]
                ):
                    tool_payload = self._stringify_tool_result(
                        message.get("content", "")
                    )
                    break
            tool_result_text = tool_payload
            self.emit_event(
                "tool_call_end",
                {
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "tool_call_id": tool_call_info["id"],
                    "result": tool_result_text,
                },
            )
            return (
                {
                    "type": "function_call_output",
                    "call_id": function_call["call_id"],
                    "output": tool_result_text,
                },
                tool_result_text,
            )

        error_msg = f"Tool '{tool_name}' not found in registered tools"
        self._logger.error(error_msg)
        self.emit_event(
            "tool_call_end",
            {
                "tool_name": tool_name,
                "arguments": tool_args,
                "tool_call_id": tool_call_info["id"],
                "result": error_msg,
            },
        )
        return (
            {
                "type": "function_call_output",
                "call_id": function_call["call_id"],
                "output": error_msg,
            },
            error_msg,
        )

    def _build_request_kwargs(
        self, model_input: list[dict[str, Any]]
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": model_input,
        }
        if self.system_prompts:
            request_kwargs["instructions"] = "\n".join(self.system_prompts)
        if self.tool_list_jsonready_cache:
            request_kwargs["tools"] = self.tool_list_jsonready_cache
        return request_kwargs

    def chat(self, messages: Any, log_path: Optional[str] = None) -> str:
        user_message = self._normalize_user_message(messages)
        self.history.append(user_message)
        self.emit_event("user_message", {"content": user_message["content"]})
        self.write_history_log(log_path)

        while True:
            model_input = [self._build_history_item(message) for message in self.history]
            request_kwargs = self._build_request_kwargs(model_input)
            try:
                response = self.client.responses.create(  # type: ignore[arg-type]
                    **request_kwargs
                )
                response_dict = self._dump_response(response)
                if response_dict.get("error") is not None:
                    raise RuntimeError(
                        json.dumps(response_dict["error"], ensure_ascii=False)
                    )
            except Exception as exc:
                self._append_round_log(request=request_kwargs, error=str(exc))
                self.write_history_log(log_path)
                raise

            self._append_round_log(request=request_kwargs, response=response_dict)

            self._append_response_output_to_history(response_dict)
            usage_dict = self._handle_usage(response_dict.get("usage"))
            self.emit_event("model_response", {"usage": usage_dict})
            self.write_history_log(log_path)

            function_calls = self._extract_function_calls(response_dict)
            if not function_calls:
                assistant_reply = self._extract_response_text(response_dict)
                self.emit_event("assistant_message", {"content": assistant_reply})
                self.write_history_log(log_path)
                return assistant_reply

            self.emit_event("assistant_tool_calls", {"tool_calls": function_calls})
            for function_call in function_calls:
                tool_output, _tool_result_text = self._handle_tool_call(function_call)
                self.history.append(tool_output)
            self.write_history_log(log_path)
