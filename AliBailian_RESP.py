# 使用阿里百炼平台的resp api，以使用百炼平台额外提供的服务，如网页搜索等
import json
import warnings

from openai import OpenAI
from typing import Any, cast
import logging
from .AgentBase import AgentBase, get_local_config


# 专用于实现网页搜索功能，虽然比较简单，但可以利用百炼平台的搜索工具，避免自己去调用搜索引擎api的麻烦，同时成本也比较低廉（4元/1000次）
class AliBailian_RESP_Agent(AgentBase):
    _logger = logging.getLogger(__name__)

    def __init__(self):

        cfg = get_local_config("AliBailian-qwen3.6-plus")
        self.model_name = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self._logger.setLevel(logging.DEBUG)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.system_prompts: list[str] = []
        self.history: list[dict[str, str]] = []

    def add_system_prompt(self, system_prompt: str) -> None:
        if system_prompt.strip() != "":
            self.system_prompts.append(system_prompt)

    def _dump_response(self, response: Any) -> dict[str, Any]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            response = response.model_dump()  # type: ignore

        if not isinstance(response, dict):
            raise ValueError("Invalid response format: response is not a dictionary.")
        return cast(dict[str, Any], response)

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip() != "":
            return output_text.strip()

        output_raw = response.get("output")
        if not isinstance(output_raw, list):
            raise ValueError(
                "Invalid response format: 'output' field is missing or empty."
            )
        output = cast(list[Any], output_raw)
        if len(output) == 0:
            raise ValueError(
                "Invalid response format: 'output' field is missing or empty."
            )

        text_parts: list[str] = []
        for output_item_raw in output:
            if not isinstance(output_item_raw, dict):
                continue
            output_item = cast(dict[str, Any], output_item_raw)
            content_raw = output_item.get("content")
            if not isinstance(content_raw, list):
                continue
            content = cast(list[Any], content_raw)
            for content_item_raw in content:
                if not isinstance(content_item_raw, dict):
                    continue
                content_item = cast(dict[str, Any], content_item_raw)
                text = content_item.get("text")
                if isinstance(text, str) and text.strip() != "":
                    text_parts.append(text.strip())

        if len(text_parts) == 0:
            raise ValueError("Invalid response format: no text content found in output.")

        return "\n".join(text_parts)

    def chat(self, messages: str) -> str:
        user_message = messages.strip()
        if user_message == "":
            raise ValueError("Input message should not be empty.")

        model_input: list[dict[str, str]] = []
        if len(self.system_prompts) > 0:
            model_input.append(
                {"role": "system", "content": "\n".join(self.system_prompts)}
            )
        model_input.extend(self.history)
        model_input.append({"role": "user", "content": user_message})

        response = self.client.responses.create(
            model=self.model_name,
            input=model_input,  # type: ignore
        )
        response = self._dump_response(response)

        self._logger.debug(f"{type(response)}")
        if "error" in response and response["error"] is not None:
            return json.dumps(response["error"], ensure_ascii=False)

        assistant_reply = self._extract_response_text(response)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply

    # 使用qwen自己提供的搜索功能，这个比较便宜，4元/1000次
    def search(self, query: str) -> Any:
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": "You are an AI assistant designed to help users perform web searches. Please treat the user's input as the search query, use appropriate keywords to conduct the search, and provide a summary of the search results, removing any irrelevant formatting information. Read as many web pages as possible. Do not return the original web pages.",
                },
                {"role": "user", "content": query},
            ],  # type: ignore
            tools=[{"type": "web_search"}, {"type": "web_extractor"}],  # type: ignore
            temperature=0,
        )

        response = self._dump_response(response)

        # 现在的response应该是一个字典
        self._logger.debug(f"{type(response)}")
        # 如果有error字段且不为空，说明请求失败了
        if "error" in response and response["error"] is not None:
            return json.dumps(response["error"], ensure_ascii=False)
        return self._extract_response_text(response)
