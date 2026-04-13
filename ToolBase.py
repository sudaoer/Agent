from abc import ABC, abstractmethod
from typing import Any, Optional
import json


# 参考opencode的工具定义，定义一个抽象的工具类，所有的工具都应该继承这个类，并实现execute方法
class ToolBase(ABC):
    toolName: str
    toolDescription: str
    paramSchema: dict[str, tuple[type, str]]  # 参数名-参数类型和描述的字典

    # 定义一个抽象方法，所有的工具都应该实现这个方法，接受一个参数字典和一个上下文列表，返回新的上下文
    @abstractmethod
    def execute(
        self, tool_call_info: dict[str, Any], ctx: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        pass

    @staticmethod
    def pyType_to_jsonType(py_type: type) -> str:
        if py_type == str:
            return "string"
        elif py_type == int:
            return "integer"
        elif py_type == float:
            return "number"
        elif py_type == bool:
            return "boolean"
        else:
            raise ValueError(f"Unsupported type: {py_type}")

    @staticmethod
    def parse_toolcall_arguments(tool_call_info: dict[str, Any]) -> dict[str, Any]:
        arguments = tool_call_info["function"]["arguments"]
        try:
            arguments = json.loads(arguments)  # type: ignore
            if not isinstance(arguments, dict):  # type: ignore
                raise ValueError("工具参数应该是一个JSON对象格式的字符串")
            return arguments  # type: ignore
        except Exception as e:
            raise ValueError(f"Invalid tool call arguments: {arguments}, error: {e}")

    def to_dict(self) -> dict[str, Any]:
        properties = {}
        for param_name, (param_type, param_description) in self.paramSchema.items():
            properties[param_name] = {
                "type": self.pyType_to_jsonType(param_type),
                "description": param_description,
            }

        return {
            "type": "function",
            "function": {
                "name": self.toolName,
                "description": self.toolDescription,
                "parameters": {"type": "object", "properties": properties},
            },
        }

    # 额外的系统提示词
    def extra_system_prompt(self) -> Optional[str]:
        return None
