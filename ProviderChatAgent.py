# pyright: strict
from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Optional, cast

from .CoTChatAgent import CoTChatAgent


class OpenAICompatibleChatAgent(CoTChatAgent):
    """Default OpenAI-compatible chat agent."""


class DeepSeekChatAgent(CoTChatAgent):
    def _dump_assistant_message_for_history(self, message: Any) -> dict[str, Any]:
        payload = super()._dump_assistant_message_for_history(message)
        reasoning_content = self._extract_reasoning_content(message)
        if reasoning_content:
            payload["reasoning_content"] = reasoning_content
        return payload


class AliyunChatAgent(CoTChatAgent):
    _EXPLICIT_CACHE_ROLES: ClassVar[set[str]] = {
        "system",
        "user",
        "assistant",
        "tool",
    }

    def _prepare_chat_request(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        request_history, request_extra_body = super()._prepare_chat_request(
            history,
            extra_body,
        )
        return (
            self._build_explicit_cache_messages(request_history),
            request_extra_body,
        )

    def _build_explicit_cache_messages(
        self,
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        request_messages = deepcopy(history)
        system_message_index: int | None = None

        for index, message in enumerate(request_messages):
            if message.get("role") != "system":
                continue
            if self._mark_explicit_cache_message(message):
                system_message_index = index
                break

        for index in range(len(request_messages) - 1, -1, -1):
            if index == system_message_index:
                continue
            message = request_messages[index]
            if message.get("role") not in self._EXPLICIT_CACHE_ROLES:
                continue
            self._mark_explicit_cache_message(message)
            return request_messages

        return request_messages

    def _mark_explicit_cache_message(self, message: dict[str, Any]) -> bool:
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

        if isinstance(content, list):
            content_items = cast(list[Any], content)
            if len(content_items) == 0:
                self._logger.debug(
                    "Skipping Aliyun explicit cache marker because %s message content is not safely rewritable.",
                    message.get("role"),
                )
                return False
            if not all(isinstance(item, dict) for item in content_items):
                self._logger.debug(
                    "Skipping Aliyun explicit cache marker because the message content list is not dict-only."
                )
                return False
            content_blocks = cast(list[dict[str, Any]], content_items)
            content_blocks[-1]["cache_control"] = {"type": "ephemeral"}
            return True

        self._logger.debug(
            "Skipping Aliyun explicit cache marker because %s message content is not safely rewritable.",
            message.get("role"),
        )
        return False
