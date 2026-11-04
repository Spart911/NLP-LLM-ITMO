"""Разбор запроса в проверенный объект параметров и сравнение режима рассуждения."""

import time
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from catalog.llm import (
    build_chat_model,
    message_reasoning,
    message_text,
    message_usage,
)
from catalog.prompts import build_parser, build_prompt, build_structured_chain
from catalog.report import write_report
from catalog.schema import ProductQuery

CASES: tuple[dict[str, Any], ...] = (
    {
        "query": "Найди ноутбуки дешевле 70 000 рублей",
        "expect": {"category": "laptop", "max_price": 70000, "sort_by": "price_asc"},
        "focus": "Категория и числовое ограничение",
    },
    {
        "query": "Покажи самые дорогие смартфоны",
        "expect": {
            "category": "smartphone",
            "max_price": None,
            "sort_by": "price_desc",
        },
        "focus": "Сортировка по убыванию цены",
    },
    {
        "query": "Хочу посмотреть планшеты",
        "expect": {"category": "tablet", "max_price": None, "sort_by": "price_asc"},
        "focus": "Запрос без ограничения цены",
    },
    {
        "query": "Покажи ноутбуки до 50 тысяч",
        "expect": {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"},
        "focus": "Нормализация числового выражения",
    },
    {
        "query": "Найди холодильники",
        "expect": {"category": "other", "max_price": None, "sort_by": "price_asc"},
        "focus": "Категория вне каталога",
    },
    {
        "query": "Возьми что-нибудь для конспектов и фильмов, тысяч до тридцати",
        "expect": None,
        "focus": "Категория не названа, потолок цены спрятан в разговорной форме",
        "ambiguity": (
            "Конспекты и фильмы одинаково подходят планшету и ноутбуку, "
            "категория в запросе не названа. «Тысяч до тридцати» задает потолок 30000 рублей."
        ),
    },
    {
        "query": "Нужен нормальный экран в карман, не самый дешевый",
        "expect": None,
        "focus": "Размытая категория и порядок без потолка цены",
        "ambiguity": (
            "«Экран в карман» обычно означает смартфон, тем же словосочетанием "
            "можно назвать маленький планшет. «Не самый дешевый» просит порядок "
            "от дорогих к дешевым и не называет верхнюю цену."
        ),
    },
)


def _describe(parsed: ProductQuery) -> str:
    if parsed.max_price is None:
        price = "потолок цены не задан"
    else:
        price = f"потолок цены {parsed.max_price} рублей"
    order = "сначала дешевые" if parsed.sort_by == "price_asc" else "сначала дорогие"
    return f"Система выбрала категорию {parsed.category}, {price}, порядок: {order}."


def _check(parsed: ProductQuery, expect: dict[str, Any] | None) -> dict[str, Any]:
    actual = parsed.model_dump(mode="json")
    if expect is None:
        return {"ok": None, "mismatches": {}}
    mismatches = {
        field: {"expected": value, "actual": actual[field]}
        for field, value in expect.items()
        if actual[field] != value
    }
    return {"ok": not mismatches, "mismatches": mismatches}


def _thinking_summary(rows: list[dict[str, Any]]) -> str:
    grouped: dict[str, dict[bool, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["query"], {})[row["thinking"]] = row
    lines: list[str] = []
    for query, pair in grouped.items():
        off = pair[False]
        on = pair[True]
        separated = bool(on["reasoning"]) and on["reasoning"] not in (on["content"] or "")
        place = (
            "рассуждение записано отдельным полем и в content не входит"
            if separated
            else "рассуждение осталось внутри content"
        )
        lines.append(
            f"«{query}»: без рассуждения {off['elapsed_sec']} с, "
            f"выходных токенов {off['output_tokens']} "
            f"(из них рассуждение {off['reasoning_tokens']}), "
            f"разбор content: {'успешен' if off['parsed'] else 'ошибка'}. "
            f"С рассуждением {on['elapsed_sec']} с, "
            f"выходных токенов {on['output_tokens']} "
            f"(из них рассуждение {on['reasoning_tokens']}), "
            f"разбор content: {'успешен' if on['parsed'] else 'ошибка'}; {place}."
        )
    lines.append(
        "Парсер читает только content. Текст рассуждения хранится отдельно "
        "и в строку, которую разбирают как JSON, не подмешивается."
    )
    return " ".join(lines)


def _compare_thinking(prompt: ChatPromptTemplate, parser: PydanticOutputParser) -> list[dict]:
    queries = [case["query"] for case in CASES if case.get("ambiguity")]
    rows: list[dict[str, Any]] = []
    for query in queries:
        for thinking in (False, True):
            llm = build_chat_model(thinking=thinking)
            started = time.perf_counter()
            message = llm.invoke(prompt.invoke({"query": query}))
            elapsed = round(time.perf_counter() - started, 3)
            content = message_text(message)
            usage = message_usage(message)
            try:
                parsed = parser.parse(content)
            except Exception as exc:
                parsed_json = None
                error = f"{type(exc).__name__}: {exc}"
            else:
                parsed_json = parsed.model_dump(mode="json")
                error = None
            rows.append(
                {
                    "query": query,
                    "thinking": thinking,
                    "elapsed_sec": elapsed,
                    "content": content,
                    "reasoning": message_reasoning(message),
                    "parsed": parsed_json,
                    "parse_error": error,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage["reasoning_tokens"],
                }
            )
    return rows


def run() -> dict:
    parser = build_parser()
    prompt = build_prompt(parser)
    chain = build_structured_chain(build_chat_model(thinking=False), parser)
    system_text = prompt.invoke({"query": "пример"}).messages[0].content

    cases = []
    for case in CASES:
        started = time.perf_counter()
        try:
            parsed = chain.invoke({"query": case["query"]})
        except Exception as exc:
            cases.append(
                {
                    "query": case["query"],
                    "focus": case["focus"],
                    "elapsed_sec": round(time.perf_counter() - started, 3),
                    "result_type": None,
                    "json": None,
                    "fields": None,
                    "check": {"ok": False, "mismatches": {}},
                    "error": f"{type(exc).__name__}: {exc}",
                    "ambiguity": case.get("ambiguity"),
                    "interpretation": None,
                }
            )
            continue
        if not isinstance(parsed, ProductQuery):
            raise TypeError(f"Цепочка вернула {type(parsed).__name__}, ожидался ProductQuery")
        interpretation = _describe(parsed)
        if note := case.get("ambiguity"):
            interpretation = f"{note} {interpretation}"
        cases.append(
            {
                "query": case["query"],
                "focus": case["focus"],
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "result_type": type(parsed).__name__,
                "json": parsed.model_dump_json(ensure_ascii=False),
                "fields": parsed.model_dump(mode="json"),
                "check": _check(parsed, case["expect"]),
                "error": None,
                "ambiguity": case.get("ambiguity"),
                "interpretation": interpretation,
            }
        )

    thinking = _compare_thinking(prompt, parser)
    report = {
        "format_instructions_in_system_prompt": parser.get_format_instructions() in system_text,
        "cases": cases,
        "thinking": thinking,
        "thinking_summary": _thinking_summary(thinking),
    }
    path = write_report("interpret.json", report)

    print(f"Инструкции формата в системном сообщении: {report['format_instructions_in_system_prompt']}.")
    print()
    for case in cases:
        status = case["check"]["ok"]
        mark = "—" if status is None else ("ok" if status else "расхождение")
        print(f"[{mark}] {case['query']}")
        if case["json"]:
            print(f"  {case['json']}")
        if case["interpretation"]:
            print(f"  {case['interpretation']}")
        if case["error"]:
            print(f"  {case['error']}")
        print()
    print(report["thinking_summary"])
    print()
    print(f"Отчет: {path}")
    return report
