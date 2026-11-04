"""Один вопрос пользователя через итоговый пайплайн."""

import time
from typing import Any

from catalog.flows.recovery import run_query
from catalog.llm import build_chat_model, message_text, message_usage
from catalog.prompts import build_parser, build_prompt


class _Recorder:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def invoke(self, messages):
        started = time.perf_counter()
        message = self.inner.invoke(messages)
        self.calls.append(
            {
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "content": message_text(message),
                "usage": message_usage(message),
            }
        )
        return message


def answer(query: str) -> dict[str, Any]:
    parser = build_parser()
    prompt = build_prompt(parser)
    rendered = prompt.invoke({"query": query})
    messages = [
        {"type": message.type, "content": message.content}
        for message in rendered.to_messages()
    ]
    recorder = _Recorder(build_chat_model(thinking=False))
    outcome = run_query(query, recorder, retry=True)
    result = outcome.get("result") or {}
    params = result.get("params")
    products = [
        f"{item['name']}, {item['brand']} — {item['price']} ₽, остаток {item['stock']}"
        for item in result.get("products") or []
    ]
    report = {
        "query": query,
        "messages": messages,
        "calls": recorder.calls,
        "params": params,
        "products": products,
        "message": outcome.get("message"),
        "error": None if outcome.get("ok") else outcome.get("message"),
        "journal": outcome.get("journal") or [],
        "attempts": outcome.get("attempts"),
        "searched": outcome.get("searched"),
    }
    report["pipeline"] = build_ask_pipeline(report)
    return report


def build_ask_pipeline(report: dict[str, Any]) -> dict[str, Any]:
    messages = "\n\n".join(
        f"{item['type']}:\n{item['content']}" for item in report.get("messages") or []
    )
    calls = report.get("calls") or []
    journal = report.get("journal") or []
    params = report.get("params")
    products = report.get("products") or []
    error = report.get("error")
    attempts = report.get("attempts") or len(calls) or 1
    elapsed = round(sum(call.get("elapsed_sec") or 0 for call in calls), 3)
    model_trace = "\n\n".join(_call_trace(index, call) for index, call in enumerate(calls, start=1))
    if error and not report.get("searched"):
        search_summary = "не выполнялся"
        search_trace = "Поиск не выполнялся: параметры не прошли разбор."
        search_status = "error"
    elif report.get("message") and not products:
        search_summary = "без выборки"
        search_trace = str(report["message"])
        search_status = "ok"
    else:
        search_summary = f"{len(products)} товаров"
        search_trace = "\n".join(f"- {item}" for item in products) or "Подходящих товаров нет."
        search_status = "ok"
    nodes = [
        _node("prompt", "Промпт", "system и human", messages or "Промпт пуст.", 0, 0, "ok"),
        _node("model", "Модель", f"{elapsed} с", model_trace or "Вызова не было.", 1, 0, "error" if error else "ok"),
        _node(
            "parse",
            "Разбор",
            "ошибка" if error else _params_line(params),
            error or _params_line(params),
            2,
            0,
            "error" if error else "ok",
        ),
        _node(
            "retry",
            "Повтор",
            f"{attempts} из 3",
            f"Попыток {attempts} из 3. Повтор оборачивает только промпт, модель и разбор.\n\n{model_trace}",
            3,
            0,
            "error" if error else "ok",
        ),
        _node("search", "Поиск", search_summary, search_trace, 4, 0, search_status),
        _node("journal", "Журнал", " → ".join(event.get("step", "") for event in journal) or "пусто", _journal_trace(journal), 3, 1, "ok"),
    ]
    edges = [
        ["prompt", "model"],
        ["model", "parse"],
        ["parse", "retry"],
        ["retry", "search"],
        ["parse", "journal"],
    ]
    return {"nodes": nodes, "edges": edges}


def _call_trace(index: int, call: dict[str, Any]) -> str:
    usage = call.get("usage") or {}
    return (
        f"Попытка {index}, {call.get('elapsed_sec')} с, "
        f"вход {usage.get('input_tokens')}, выход {usage.get('output_tokens')}\n"
        f"{call.get('content') or '—'}"
    )


def _journal_trace(journal: list[dict[str, Any]]) -> str:
    if not journal:
        return "Журнал пуст."
    lines = []
    for event in journal:
        payload = {key: value for key, value in event.items() if key != "step"}
        lines.append(event.get("step", "") + (f" {payload}" if payload else ""))
    return "\n".join(lines)


def _params_line(params: dict[str, Any] | None) -> str:
    if not params:
        return "—"
    price = "без потолка" if params.get("max_price") is None else str(params.get("max_price"))
    return f"{params.get('category')}, {price}, {params.get('sort_by')}"


def _node(
    node_id: str,
    title: str,
    summary: str,
    trace: str,
    column: int,
    row: int,
    status: str,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "title": title,
        "summary": summary,
        "trace": trace.strip() + "\n",
        "column": column,
        "row": row,
        "status": status,
    }
