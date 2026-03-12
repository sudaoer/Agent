# 不带思维链的agent，因此不需要额外维护思维链的工具调用

from AgentBase import AgentBase_OpenAIBackend
import json


class NormalChatAgent(AgentBase_OpenAIBackend):
    def __init__(
        self, name: str, base_url: str, model_name: str, api_key: str = "test_api_key"
    ):
        super().__init__(name, base_url, model_name, api_key)
        self.history = []
        self.token_consumption = {}

    def chat(self, messages: str) -> str:
        # 如果没有历史记录，则使用系统提示词作为对话的开头
        if not self.history:
            self.history = [{"role": "system", "content": self.system_prompt}]
        self.history.append({"role": "user", "content": messages})

        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=self.history,
            tools=self.tool_list_jsonready_cache,
        )

        # 处理可能的工具调用
        if response.choices[0].message.tool_calls:
            for tool_call in response.choices[0].message.tool_calls:
                tool_name = tool_call.function.name
                tool_params = tool_call.function.arguments

                print(f"正在调用工具：{tool_name}，参数：{tool_params}")

                # 找到对应的工具函数并调用
                tool_result = super()._handle_tool_call(
                    tool_name, json.loads(tool_params)
                )
                # 将模型回复加入到历史记录中
                self.history.append(
                    response.choices[0].message,
                )

                # 将工具调用结果作为助手消息添加到历史记录中
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"工具调用结果：{tool_result}",
                    }
                )
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                tools=self.tool_list_jsonready_cache,
            )
        assistant_reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    def get_history(self) -> list[dict[str, str]]:
        return self.history

def get_time() -> str:
    import datetime

    return f"当前时间是：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


if __name__ == "__main__":
    END_POINT = "https://api.deepseek.com"
    MODEL = "deepseek-chat"
    agent = NormalChatAgent(
        name="TestAgent",
        base_url=END_POINT,
        model_name=MODEL,
        api_key="",
    )

    system_prompt = "你是我的人工智能助手，协助我解答问题和完成任务。"
    agent.set_system_prompt(system_prompt)

    agent.register_tool(
        tool_name="get_current_time",
        tool_function=get_time,
        tool_description="获取当前时间，无需参数",
    )

    user_message = "现在几点了？"
    response = agent.chat(user_message)
    print("模型回复：", response)

    print("对话历史：", agent.get_history())
