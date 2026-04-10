# 使用阿里百炼平台的resp api，以使用百炼平台额外提供的服务，如网页搜索等
from openai import OpenAI
from typing import Any
import logging
from .AgentBase import AgentBase, get_local_config

# 专用于实现网页搜索功能，虽然比较简单，但可以利用百炼平台的搜索工具，避免自己去调用搜索引擎api的麻烦，同时成本也比较低廉（4元/1000次）
class AliBailian_RESP_Agent(AgentBase):
    def __init__(self):

        cfg = get_local_config("AliBailian-qwen3.6-plus")
        self.model_name = cfg["model"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def set_system_prompt(self, system_prompt: str) -> None:
        raise NotImplementedError(
            "AliBailian_RESP_Agent does not support system prompt."
        )

    def chat(self, messages: str) -> Any:
        raise NotImplementedError("AliBailian_RESP_Agent does not support chat method.")

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

        # 屏蔽 UserWarning: Pydantic serializer warnings:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            response = response.model_dump()  # type: ignore

        # 现在的response应该是一个字典
        logging.debug(f"{type(response)}")
        # 如果有error字段且不为空，说明请求失败了
        if "error" in response and response["error"] is not None:
            return response["error"]
        # 返回output数组的最后一个元素
        if "output" in response and len(response["output"]) > 0:
            return response["output"][-1]['content'][0]['text']
        else:
            raise ValueError(
                "Invalid response format: 'output' field is missing or empty."
            )
