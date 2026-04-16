# 对于有思维链且可以在思维链中调用工具的模型，在tool call上需要特殊处理
from copy import deepcopy
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
        prepared_extra_body["enable_thinking"] = True
        history_tmp = deepcopy(history)
        start_del = False
        # 删除最后一条真实用户消息之前所有的reasoning_content
        for i in range(len(history) - 1, -1, -1):
            if history_tmp[i]["role"] == "user":
                start_del = True
            if start_del and "reasoning_content" in history_tmp[i]:
                del history_tmp[i]["reasoning_content"]
        return history_tmp, prepared_extra_body
