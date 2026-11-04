"""Клиент чат-модели. Ключ и адрес провайдера берутся из окружения."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from catalog.config import get_settings


def reasoning_body(*, thinking: bool) -> dict[str, Any]:
    if thinking:
        return {"enabled": True, "effort": "low", "exclude": False}
    return {"enabled": False, "effort": "none"}


class CatalogChatModel(ChatOpenAI):
    """OpenAI-совместимый клиент, который сохраняет текст рассуждения отдельно от ответа.

    Провайдер возвращает рассуждение рядом с content. Стандартный адаптер
    это поле отбрасывает, поэтому клиент кладет его в additional_kwargs.
    """

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ):
        result = super()._create_chat_result(response, generation_info)
        payload = response if isinstance(response, dict) else response.model_dump()
        try:
            raw_message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return result
        reasoning = _reasoning_text(raw_message)
        if reasoning and result.generations:
            message = result.generations[0].message
            message.additional_kwargs = {
                **message.additional_kwargs,
                "reasoning": reasoning,
            }
        return result


def build_chat_model(*, thinking: bool = False) -> CatalogChatModel:
    settings = get_settings()
    return CatalogChatModel(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        max_tokens=800,
        timeout=90,
        max_retries=1,
        extra_body={"reasoning": reasoning_body(thinking=thinking)},
    )


def message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def message_reasoning(message: AIMessage) -> str | None:
    value = (message.additional_kwargs or {}).get("reasoning")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def message_usage(message: AIMessage) -> dict[str, int | None]:
    usage = message.usage_metadata or {}
    details = usage.get("output_token_details") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": details.get("reasoning"),
    }


def _reasoning_text(raw_message: dict) -> str | None:
    value = raw_message.get("reasoning")
    if value is None:
        value = raw_message.get("reasoning_content")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
