from typing import Any

from .CoTChatAgent import CoTChatAgent


class VisionChatAgent(CoTChatAgent):
    def chat(self, messages: Any, log_path: str | None = None) -> str:
        return super().chat(messages, log_path=log_path)
