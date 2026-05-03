# pyright: strict
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

try:
    from ..ConfigMgr.configMgr import (
        get_baseurl_key,
        get_key,
        get_provider_for_baseurl,
        get_provider_for_model,
    )
except ImportError:
    from ConfigMgr.configMgr import (
        get_baseurl_key,
        get_key,
        get_provider_for_baseurl,
        get_provider_for_model,
    )
from .AgentBase import AgentBase, Agent_OpenAIChat_API_Backend
from .ProviderChatAgent import (
    AliyunChatAgent,
    DeepSeekChatAgent,
    OpenAICompatibleChatAgent,
)


def build(
    model_name: str,
    base_url: str | None = None,
    *,
    name: str = "Agent",
    api_key: str | None = None,
) -> AgentBase:
    provider: dict[str, Any] | None = None
    if base_url is None:
        try:
            resolved_base_url, resolved_api_key = get_baseurl_key(model_name)
            provider = get_provider_for_model(model_name)
        except ValueError as exc:
            raise ValueError(
                f"Unable to build agent for model '{model_name}': {exc}"
            ) from exc
    else:
        resolved_base_url = base_url
        if api_key is not None:
            resolved_api_key = api_key
        else:
            try:
                resolved_api_key = get_key(base_url)
            except ValueError as exc:
                raise ValueError(
                    "Unable to build agent for model "
                    f"'{model_name}' with base_url '{base_url}': {exc}"
                ) from exc
        provider = get_provider_for_baseurl(base_url)

    agent_cls = _select_agent_class(
        model_name=model_name,
        base_url=resolved_base_url,
        provider=provider,
    )
    return agent_cls(
        name=name,
        base_url=resolved_base_url,
        model_name=model_name,
        api_key=resolved_api_key,
    )


def _select_agent_class(
    *,
    model_name: str,
    base_url: str,
    provider: dict[str, Any] | None,
) -> type[Agent_OpenAIChat_API_Backend]:
    provider_name = _provider_name(provider)
    hostname = (urlparse(base_url).hostname or "").lower()
    normalized_model = model_name.lower()

    provider_or_host_is_deepseek = _matches_any(
        provider_name,
        ("deepseek",),
    ) or _matches_any(
        hostname,
        ("deepseek",),
    )

    provider_or_host_is_aliyun = _matches_any(
        provider_name,
        (
            "alibaba",
            "aliyun",
            "aliyuncs",
            "bailian",
            "dashscope",
            "百炼",
            "通义",
        )
    ) or _matches_any(
        hostname,
        (
            "alibaba",
            "alibabacloud",
            "aliyun",
            "aliyuncs",
            "bailian",
            "dashscope",
        ),
    )

    if provider_or_host_is_aliyun:
        return AliyunChatAgent

    if provider_or_host_is_deepseek or normalized_model.startswith("deepseek"):
        return DeepSeekChatAgent

    if normalized_model.startswith(("qwen", "qwq", "qvq")):
        return AliyunChatAgent

    return OpenAICompatibleChatAgent


def _provider_name(provider: dict[str, Any] | None) -> str:
    if provider is None:
        return ""
    value = provider.get("provider")
    if isinstance(value, str):
        return value.lower()
    return ""


def _matches_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)
