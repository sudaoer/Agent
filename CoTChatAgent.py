# 对于有思维链且可以在思维链中调用工具的模型，在tool call上需要特殊处理
from typing import Any
from AgentBase import AgentBase_OpenAIBackend
import json


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

    def chat(self, messages: str) -> str:
        # 如果没有历史记录，则使用系统提示词作为对话的开头
        if len(self.history) == 0:
            self.history = [{"role": "system", "content": self.system_prompt}]
        self.history.append({"role": "user", "content": messages})

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=self.history,  # type: ignore
            tools=self.tool_list_jsonready_cache,  # type: ignore
        )
        super()._handle_usage(response.usage)

        while response.choices[0].message.tool_calls is not None:
            self.history.append(
                {
                    "role": "assistant",
                    "content": response.choices[0].message.content,
                    "reasoning_content": response.choices[0].message.reasoning_content,  # type: ignore
                    "tool_calls": response.choices[0].message.tool_calls,
                }
            )
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name  # type: ignore
                tool_args = tool_call.function.arguments  # type: ignore
                assert (
                    type(tool_name) == str  # type: ignore
                ), f"工具名称应该是字符串，但得到的类型是{type(tool_name)}"  # type: ignore
                assert (
                    type(tool_args) == str  # type: ignore
                ), f"工具参数应该是字符串格式的JSON，但得到的类型是{type(tool_args)}"  # type: ignore
                try:
                    tool_result = super()._handle_tool_call(
                        tool_name, json.loads(tool_args)
                    )
                except Exception as e:
                    tool_result = f"发生错误：{str(e)}"

                self.history.append(
                    {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id,
                    }
                )
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.history,  # type: ignore
                tools=self.tool_list_jsonready_cache,  # type: ignore
            )
            super()._handle_usage(response.usage)

        assistant_reply = response.choices[0].message.content
        assert assistant_reply is not None, "模型回复的内容为None"
        self.history.append({"role": "assistant", "content": assistant_reply, "reasoning_content": response.choices[0].message.reasoning_content})  # type: ignore
        return assistant_reply

    def get_history(self) -> list[dict[str, Any]]:
        return self.history


def get_time() -> str:
    import datetime

    return f"现在是北京时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


import os

if __name__ == "__main__":
    END_POINT = "https://api.deepseek.com"
    MODEL = "deepseek-reasoner"
    agent = CoTChatAgent(
        name="TestAgent",
        base_url=END_POINT,
        model_name=MODEL,
        api_key=os.environ.get("DS_KEY", "test"),
    )

    system_prompt = "你是我的人工智能助手，协助我解答问题和完成任务。"
    agent.set_system_prompt(system_prompt)

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
