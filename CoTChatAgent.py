# 对于有思维链且可以在思维链中调用工具的模型，在tool call上需要特殊处理
from typing import Any, Optional

from .AgentBase import Agent_OpenAIChat_API_Backend


class CoTChatAgent(Agent_OpenAIChat_API_Backend):
    def __init__(
        self,
        name: str,
        base_url: str,
        model_name: str,
        api_key: str = "test_api_key",
    ):
        super().__init__(name, base_url, model_name, api_key)
        self.history: list[dict[str, Any]] = []

    def _prepare_chat_request(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prepared_extra_body = dict(extra_body) if extra_body is not None else {}
        prepared_extra_body["thinking"] = {"type": "enabled"}
        return history, prepared_extra_body
