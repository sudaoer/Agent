from pathlib import Path
from openai import OpenAI
import httpx
from typing import Any, Optional
import json
import time
import logging
from abc import ABC, abstractmethod
from .ToolBase import ToolBase


def get_local_config(config_name: str) -> dict[str, str]:
    default_config = Path(__file__).resolve().parent / "secret.json"
    if not default_config.exists():
        raise FileNotFoundError(f"Local config file not found: {default_config}")
    with open(default_config, "r") as f:
        config_data = json.load(f)
    if config_name not in config_data:
        raise ValueError(f"Config '{config_name}' not found in local config file")
    return config_data[config_name]


class AgentBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set_system_prompt(self, system_prompt: str) -> None:
        pass

    @abstractmethod
    def chat(self, messages: str) -> str:
        pass


class AgentBase_OpenAIBackend(AgentBase):
    _logger = logging.getLogger(__name__)

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

        self.tool_list: list[ToolBase] = []  # 存储注册的工具
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

    # 进行一次对话，输入为用户消息，输出为模型回复，并且处理工具调用并更新对话历史
    def chat(self, messages: str, log_path: Optional[str] = None) -> str:
        # 如果没有历史记录，则使用系统提示词作为对话的开头
        if len(self.history) == 0:
            self.history = [{"role": "system", "content": self.system_prompt}]
        self.history.append({"role": "user", "content": messages})

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

    # 获取当前Agent的对话历史，返回一个列表，每个元素是一个字典
    # 分别表示消息的角色（如"user"或"assistant"）和消息内容
    def get_history(self) -> list[dict[str, Any]]:
        return self.history

    from openai.types.chat.chat_completion import ChatCompletion

    def _post_chatHistory(
        self,
        history: list[dict[str, Any]],
        extra_body: Optional[dict[str, Any]] = None,
        log_path: Optional[str] = None,
    ) -> ChatCompletion:

        self.write_history_log(log_path)
        while True:
            time_start = time.perf_counter()
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=history,  # type: ignore
                tools=self.tool_list_jsonready_cache,  # type: ignore
                extra_body=extra_body,
            )
            time_end = time.perf_counter()
            this_usage = self._handle_usage(response.usage)
            # 计算token生成速度
            completion_tokens = this_usage.get("completion_tokens", 0)
            time_cost = time_end - time_start
            tokens_per_second = completion_tokens / time_cost if time_cost > 0 else 0
            self._logger.info(
                f"{tokens_per_second:.2f} tokens/s, use time {time_cost:.2f}s"
            )
            # 检测toolcall是否合法
            all_tool_calls_valid = True
            for tool_call in response.choices[0].message.tool_calls or []:
                tool_arg = tool_call.function.arguments  # type: ignore
                try:
                    tool_arg = json.loads(tool_arg)  # type: ignore
                    if not isinstance(tool_arg, dict):  # type: ignore
                        raise ValueError("工具参数应该是一个JSON对象格式的字符串")
                except Exception as e:
                    all_tool_calls_valid = False
                    self._logger.error(
                        f"Invalid tool call arguments: {tool_arg}, error: {e}\nRetrying..."
                    )
                    break
            if all_tool_calls_valid:
                break

        return response

    from openai.types.completion_usage import CompletionUsage

    def _handle_usage(self, usage: Optional[CompletionUsage]) -> dict[str, Any]:
        if usage is None:
            return {}
        usage_dict = usage.model_dump()
        for k, v in usage_dict.items():
            if isinstance(v, int):
                self.token_usage[k] = self.token_usage.get(k, 0) + v
            elif isinstance(v, dict):
                for sub_k, sub_v in v.items():  # type: ignore
                    if isinstance(sub_v, int):
                        self.token_usage[sub_k] = self.token_usage.get(sub_k, 0) + sub_v  # type: ignore
        return usage_dict

    def register_tool(self, tool: ToolBase) -> None:
        self.tool_list.append(tool)
        self.tool_list_jsonready_cache.append(tool.to_dict())

    def _handle_tool_call(
        self, tool_call: dict[str, Any], ctx: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        tool_name = tool_call["function"]["name"]
        for tool in self.tool_list:
            if tool.toolName == tool_name:
                return tool.execute(tool_call, ctx)
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
