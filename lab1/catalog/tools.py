"""Инструмент поиска. Выборка выполняется только после проверки аргументов."""

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from catalog.schema import ProductQuery
from catalog.search import search


@tool("search_products", args_schema=ProductQuery)
def search_products(
    category: str,
    max_price: int | None = None,
    sort_by: str = "price_asc",
) -> str:
    """Ищет товары в локальном каталоге по категории, верхней цене и порядку сортировки."""

    params = ProductQuery.model_validate(
        {"category": category, "max_price": max_price, "sort_by": sort_by}
    )
    result = search(params)
    return result.message or _names(result)


def resolve_tool_calls(
    message: AIMessage,
    *,
    database: Path | None = None,
) -> dict[str, Any]:
    """Разобрать tool_calls и выполнить поиск, не отправляя результат модели."""

    calls = list(message.tool_calls or [])
    if not calls:
        return {
            "content": _content(message),
            "tool_calls": [],
            "searched": False,
            "returned_to_model": False,
            "outcomes": [],
            "note": "Модель не вызвала инструмент, поиск не запускался.",
        }

    outcomes = []
    for call in calls:
        record = _resolve_one(call, database)
        outcomes.append(record)
    return {
        "content": _content(message),
        "tool_calls": [
            {"name": call.get("name"), "args": call.get("args"), "id": call.get("id")}
            for call in calls
        ],
        "searched": any(item["searched"] for item in outcomes),
        "returned_to_model": False,
        "outcomes": outcomes,
        "note": None,
    }


def _resolve_one(call: dict[str, Any], database: Path | None) -> dict[str, Any]:
    name = call.get("name")
    args = call.get("args")
    if name != "search_products":
        return {
            "name": name,
            "args": args,
            "searched": False,
            "error": "Неизвестный инструмент.",
            "result": None,
        }
    try:
        params = ProductQuery.model_validate(args)
    except ValidationError as exc:
        return {
            "name": name,
            "args": args,
            "searched": False,
            "error": str(exc),
            "result": None,
        }
    result = search(params, database=database)
    return {
        "name": name,
        "args": params.model_dump(mode="json"),
        "searched": True,
        "error": None,
        "result": result.as_dict(),
    }


def _content(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if not content:
        return ""
    chunks: list[str] = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            chunks.append(block["text"])
    return "".join(chunks).strip()


def _names(result) -> str:
    return ", ".join(item.name for item in result.products)
