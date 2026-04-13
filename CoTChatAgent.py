# 对于有思维链且可以在思维链中调用工具的模型，在tool call上需要特殊处理
from typing import Any, Optional

from openai.types.chat.chat_completion import ChatCompletion
from .AgentBase import AgentBase_OpenAIBackend


class CoTChatAgent(AgentBase_OpenAIBackend):
    def __init__(
        self,
        name: str,
        base_url: str,
        model_name: str,
        api_key: str = "test_api_key",
    ):
        super().__init__(name, base_url, model_name, api_key)
        self.history: list[dict[str, Any]] = []

    def _post_chatHistory(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        log_path: Optional[str] = None,
    ) -> ChatCompletion:
        extra_body = extra_body or {}
        extra_body["enable_thinking"] = True
        history_tmp = history
        start_del = False
        # 删除最后一条真实用户消息之前所有的reasoning_content
        for i in range(len(history) - 1, -1, -1):
            if (
                history[i]["role"] == "user"
            ):
                start_del = True
            if start_del and "reasoning_content" in history[i]:
                del history_tmp[i]["reasoning_content"]
        return super()._post_chatHistory(history_tmp, extra_body, log_path=log_path)
