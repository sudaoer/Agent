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
        # 删除最后一条user消息之前所有的reasoning_content
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                start_del = True
            if start_del and "reasoning_content" in history[i]:
                del history_tmp[i]["reasoning_content"]
        return super()._post_chatHistory(history_tmp, extra_body, log_path=log_path)


def get_time() -> str:
    import datetime

    return f"现在是北京时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


import os

if __name__ == "__main__":
    END_POINT = "http://127.0.0.1:1234/v1"
    MODEL = "qwen/qwen3.5-9b"
    agent = CoTChatAgent(
        name="TestAgent",
        base_url=END_POINT,
        model_name=MODEL,
        api_key=os.environ.get("DS_KEY", "test"),
    )

    system_prompt = "你是我的人工智能助手，协助我解答问题和完成任务。"
    agent.add_system_prompt(system_prompt)

    agent.register_tool(
        tool_name="get_current_time",
        tool_function=get_time,
        tool_description="获取当前时间，无需参数",
    )

    response = agent.chat("现在美国几点了")
    print("模型回复：", response)

    response = agent.chat("现在英国又是几点")
    print("模型回复：", response)

    print("对话历史：", agent.get_history())
    print(f"Token消耗： {agent.get_token_consumption()}")
