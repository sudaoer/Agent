# 对于不需要对上下文进行特殊操作的工具，可以直接继承这个类


from typing import Any
from abc import abstractmethod
from .ToolBase import ToolBase


class ToolNormal(ToolBase):
    def execute(
        self, tool_call_info: dict[str, Any], ctx: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:

        # 直接将工具调用信息添加到上下文中，工具的输出由模型根据工具调用信息生成
        ctx.append(
            self.build_tool_message(
                tool_call_info["id"],
                self.run(self.parse_toolcall_arguments(tool_call_info)),
            )
        )
        return ctx

    # 这个方法由具体的工具实现，接受工具调用信息，返回工具的输出字符串
    @abstractmethod
    def run(self, tool_call_arg: dict[str, Any]) -> str:
        pass
