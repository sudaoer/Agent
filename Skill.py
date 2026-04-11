from typing import Any, Optional
from . import ToolNormal


# 本项目的skill只向agent提供经验性的知识，不提供额外的工具
# 所有skill必须在注册前加载好内容，并在extra_system_prompt中提供给agent有哪些skill和skill的简略介绍
class Skill(ToolNormal):
    toolName = "skill"
    toolDescription = """
Give you some experience or knowledge that may be helpful for you to complete the task. 
This is not a real tool, but just a way to provide some information to you. 
You can use this information to help you decide what tools to use and how to use them.
""".strip()
    paramSchema = {
        "name": (str, "the name of the skill"),
    }

    def __init__(self, skill_folder_path: str):
        self.skill_content = ""  # todo

    def extra_system_prompt(self) -> Optional[str]:
        return ""  # todo

    def run(self, tool_call_arg: dict[str, Any]) -> str:
        return ""  # todo
