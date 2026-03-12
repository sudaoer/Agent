from openai import OpenAI
import httpx
from typing import Callable, Any, Optional
import inspect
from typing import get_type_hints
import json


def parse_callable_to_openai_params(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        param_info: dict[str, str] = {
            "type": (
                hints.get(name, None).__name__  # type: ignore
                if hints.get(name, None)
                else "string"  # pyright: ignore[reportOptionalMemberAccess]
            ),
            "description": f"Parameter '{name}' of type {hints.get(name, None).__name__ if hints.get(name, None) else 'string'}",  # pyright: ignore[reportOptionalMemberAccess]
        }

        #将param_info中的类型名转换成json schema支持的类型名
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        if param_info["type"] in type_mapping:
            param_info["type"] = type_mapping[param_info["type"]]

        properties[name] = param_info

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class AgentBase_OpenAIBackend:
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

        self.httpx_client = httpx.Client(proxy=proxy) if proxy else httpx.Client()
        self.openai_client = OpenAI(
            base_url=self.base_url, api_key=self.api_key, http_client=self.httpx_client
        )

        # 检查模型是否可用
        model_list = self.openai_client.models.list()
        if self.model_name not in [model.id for model in model_list.data]:
            raise ValueError(
                f"Model '{self.model_name}' is not available in the model list: {[model.id for model in model_list.data]}"
            )

        self.tool_list: list[dict[str, Any]] = (
            []
        )  # 存储注册的工具信息，每个元素是一个字典，包含工具名称、描述和函数
        self.tool_list_jsonready_cache: list[dict[str, Any]] = (
            []
        )  # 存储工具列表的JSON-ready版本，每个元素是一个字典，包含工具名称和描述，用于传给模型
        self.token_usage: dict[str, int] = {}
        self.history: list[dict[str, Any]] = []

    # 获取当前Agent自创建以来的token消耗量
    def get_token_consumption(self) -> dict[str, int]:
        return self.token_usage

    # 设置系统提示词
    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    # 进行一次对话，输入为用户消息，输出为模型回复，并且处理工具调用并更新对话历史
    def chat(self, messages: str) -> str:
        # 如果没有历史记录，则使用系统提示词作为对话的开头
        if len(self.history) == 0:
            self.history = [{"role": "system", "content": self.system_prompt}]
        self.history.append({"role": "user", "content": messages})

        response = self._post_chatHistory(self.history)

        while response.choices[0].message.tool_calls is not None and len(response.choices[0].message.tool_calls) > 0:
            self.history.append(
                response.choices[0].message.model_dump()  # type: ignore
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
                    tool_result = self._handle_tool_call(
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
            response = self._post_chatHistory(self.history)

        assistant_reply = response.choices[0].message.content
        assert assistant_reply is not None, "模型回复的内容为None"
        self.history.append(response.choices[0].message.model_dump())  # type: ignore
        return assistant_reply

    # 获取当前Agent的对话历史，返回一个列表，每个元素是一个字典，包含"role"和"content"两个键（可能还有reasoning_content）
    # 分别表示消息的角色（如"user"或"assistant"）和消息内容
    def get_history(self) -> list[dict[str, str]]:
        return self.history

    # 注册一个工具，工具由工具名称和一个函数组成，函数接受一个字典参数（包含工具调用的必要信息），返回一个字符串结果
    def register_tool(
        self,
        tool_name: str,
        tool_description: str,
        tool_function: Callable[..., str],
    ) -> None:
        self.tool_list.append(
            {
                "name": tool_name,
                "description": tool_description,
                "function": tool_function,
            }
        )

        self.tool_list_jsonready_cache.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": parse_callable_to_openai_params(tool_function),
                },
            }
        )

    from openai.types.chat.chat_completion import ChatCompletion

    def _post_chatHistory(self, history: list[dict[str, Any]]) -> ChatCompletion:
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=history,  # type: ignore
            tools=self.tool_list_jsonready_cache,  # type: ignore
        )
        self._handle_usage(response.usage)
        return response

    from openai.types.completion_usage import CompletionUsage

    def _handle_usage(self, usage: Optional[CompletionUsage]) -> None:
        if usage is None:
            return
        usage_dict = usage.model_dump()
        for k, v in usage_dict.items():
            if isinstance(v, int):
                self.token_usage[k] = self.token_usage.get(k, 0) + v

    def _handle_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        for tool in self.tool_list:
            if tool["name"] == tool_name:
                return tool["function"](**tool_args)
        raise ValueError(f"Tool '{tool_name}' not found in registered tools")
