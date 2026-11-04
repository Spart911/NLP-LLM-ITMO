"""Граф шагов прогона для страницы каталога."""

import json
from typing import Any


def build_pipeline(flow_id: str, report: dict[str, Any]) -> dict[str, Any]:
    builder = _BUILDERS.get(flow_id)
    if builder is None:
        raise KeyError(flow_id)
    nodes, edges = builder(report)
    return {"nodes": nodes, "edges": edges}


def _node(
    node_id: str,
    title: str,
    summary: str,
    trace: str,
    *,
    column: int,
    row: int,
    status: str = "ok",
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


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _worst(statuses: list[str]) -> str:
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _classify(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    messages = "\n\n".join(
        f"{item['type']}:\n{item['content']}" for item in report.get("sample_messages") or []
    )
    batch = report.get("batch") or []
    batch_text = "\n\n".join(
        f"[{row.get('language')}] {row.get('query')}\n{row.get('answer')}" for row in batch
    )
    nodes = [
        _node(
            "prompt",
            "Промпт",
            "системное и пользовательское",
            messages or "Сообщений нет.",
            column=0,
            row=0,
        ),
        _node(
            "direct",
            "Прямой вызов",
            str(report.get("direct_type") or "—"),
            f"Тип ответа: {report.get('direct_type')}\n\n{report.get('direct_content') or ''}",
            column=1,
            row=0,
        ),
        _node(
            "chain",
            "Цепочка",
            "строка" if report.get("chain_is_str") else str(report.get("chain_type") or "—"),
            (
                f"Тип: {report.get('chain_type')}\n"
                f"Это строка: {'да' if report.get('chain_is_str') else 'нет'}\n\n"
                f"{report.get('chain_content') or ''}"
            ),
            column=1,
            row=1,
        ),
        _node(
            "batch",
            "Пакет",
            f"{len(batch)} формулировки",
            (
                f"Элемент пакета имеет тип {report.get('batch_item_type')}.\n\n"
                f"{batch_text}"
            ),
            column=2,
            row=1,
        ),
    ]
    edges = [["prompt", "direct"], ["prompt", "chain"], ["chain", "batch"]]
    return nodes, edges


def _interpret(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    cases = report.get("cases") or []
    thinking = report.get("thinking") or []
    failed = [case for case in cases if case.get("error") or (case.get("check") or {}).get("ok") is False]
    model_lines = []
    parsed_lines = []
    check_lines = []
    for case in cases:
        check = case.get("check") or {}
        mark = _check_mark(check.get("ok"), case.get("error"))
        model_lines.append(
            f"[{mark}] {case.get('query')}\n"
            f"{case.get('elapsed_sec')} с, тип {case.get('result_type') or '—'}"
        )
        parsed_lines.append(
            f"[{mark}] {case.get('query')}\n"
            f"{case.get('json') or case.get('error') or '—'}\n"
            f"{case.get('interpretation') or ''}".rstrip()
        )
        mismatch = check.get("mismatches") or {}
        detail = _dump(mismatch) if mismatch else "расхождений нет"
        check_lines.append(f"[{mark}] {case.get('focus')}: {case.get('query')}\n{detail}")
    thinking_lines = []
    thinking_failed = False
    for row in thinking:
        if row.get("parse_error"):
            thinking_failed = True
        mode = "с рассуждением" if row.get("thinking") else "без рассуждения"
        thinking_lines.append(
            f"{mode}, {row.get('elapsed_sec')} с, "
            f"вход {row.get('input_tokens')}, выход {row.get('output_tokens')}, "
            f"рассуждение {row.get('reasoning_tokens')}\n"
            f"запрос: {row.get('query')}\n"
            f"content:\n{row.get('content') or '—'}\n"
            f"reasoning:\n{row.get('reasoning') or '—'}\n"
            f"разбор: {_dump(row.get('parsed')) if row.get('parsed') else row.get('parse_error') or '—'}"
        )
    instructions = "да" if report.get("format_instructions_in_system_prompt") else "нет"
    nodes = [
        _node(
            "prompt",
            "Промпт",
            f"инструкции формата: {instructions}",
            f"Инструкции формата входят в системное сообщение: {instructions}.",
            column=0,
            row=0,
        ),
        _node(
            "model",
            "Модель",
            f"{len(cases)} запросов",
            "\n\n".join(model_lines) or "Запросов нет.",
            column=1,
            row=0,
            status="error" if failed else "ok",
        ),
        _node(
            "parse",
            "Разбор",
            "ProductQuery",
            "\n\n".join(parsed_lines) or "Разбора нет.",
            column=2,
            row=0,
            status="error" if failed else "ok",
        ),
        _node(
            "check",
            "Проверка",
            f"{len(cases) - len(failed)}/{len(cases)} без ошибки",
            "\n\n".join(check_lines) or "Проверок нет.",
            column=3,
            row=0,
            status="error" if failed else "ok",
        ),
        _node(
            "thinking",
            "Рассуждение",
            f"{len(thinking)} ответа",
            (
                (report.get("thinking_summary") or "")
                + "\n\n"
                + "\n\n".join(thinking_lines)
            ).strip(),
            column=2,
            row=1,
            status="error" if thinking_failed else "ok",
        ),
    ]
    edges = [["prompt", "model"], ["model", "parse"], ["parse", "check"], ["model", "thinking"]]
    return nodes, edges


def _lookup(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    pipeline = report.get("pipeline") or []
    plain = report.get("without_database") or []
    tools = report.get("tools") or []
    bad_pipeline = [row for row in pipeline if row.get("error") or row.get("correct") is False]
    tool_lines = []
    search_lines = []
    for row in tools:
        calls = row.get("tool_calls") or []
        tool_lines.append(
            f"{row.get('query')}\n"
            f"content: {row.get('content') or 'пусто'}\n"
            f"tool_calls: {_dump(calls) if calls else 'нет'}\n"
            f"{row.get('note') or ''}".rstrip()
        )
        outcomes = row.get("outcomes") or []
        if not outcomes:
            search_lines.append(f"{row.get('query')}\nпоиск не запускался")
            continue
        blocks = []
        for outcome in outcomes:
            blocks.append(
                f"args: {_dump(outcome.get('args'))}\n"
                f"поиск: {'да' if outcome.get('searched') else 'нет'}\n"
                f"{outcome.get('error') or _dump(outcome.get('result'))}"
            )
        search_lines.append(f"{row.get('query')}\n" + "\n".join(blocks))
    plain_failed = [row for row in plain if row.get("catalog_names_in_answer")]
    nodes = [
        _node(
            "ask",
            "Запрос",
            f"{len(pipeline)} фраз",
            "\n".join(f"- {row.get('query')}" for row in pipeline) or "Запросов нет.",
            column=0,
            row=1,
        ),
        _node(
            "params",
            "Параметры",
            f"{len(pipeline) - len(bad_pipeline)}/{len(pipeline)} сходятся",
            "\n\n".join(
                f"[{'ok' if row.get('correct') else 'расхождение'}] {row.get('query')}\n"
                f"{_dump(row.get('params'))}\n{row.get('error') or ''}".rstrip()
                for row in pipeline
            ),
            column=1,
            row=0,
            status="error" if bad_pipeline else "ok",
        ),
        _node(
            "catalog",
            "Каталог",
            "SQLite",
            "\n\n".join(_catalog_block(row) for row in pipeline) or "Выборки нет.",
            column=2,
            row=0,
            status="error" if bad_pipeline else "ok",
        ),
        _node(
            "plain",
            "Без базы",
            f"{len(plain)} вопроса",
            "\n\n".join(
                f"{row.get('ask')}\n"
                f"названия из каталога в ответе: {', '.join(row.get('catalog_names_in_answer') or []) or 'нет'}\n"
                f"{row.get('model_answer') or ''}"
                for row in plain
            ),
            column=1,
            row=1,
            status="warn" if plain_failed else "ok",
        ),
        _node(
            "tool",
            "Инструмент",
            f"{sum(1 for row in tools if row.get('tool_calls'))}/{len(tools)} с вызовом",
            "\n\n".join(tool_lines) or "Вызовов нет.",
            column=1,
            row=2,
        ),
        _node(
            "tool-search",
            "Поиск",
            "результат модели не возвращался",
            "\n\n".join(search_lines) or "Поиска нет.",
            column=2,
            row=2,
        ),
    ]
    edges = [
        ["ask", "params"],
        ["params", "catalog"],
        ["ask", "plain"],
        ["ask", "tool"],
        ["tool", "tool-search"],
    ]
    return nodes, edges


def _catalog_block(row: dict[str, Any]) -> str:
    products = row.get("products") or []
    lines = [str(row.get("query"))]
    if products:
        lines.extend(f"- {item}" for item in products)
    if row.get("message"):
        lines.append(str(row["message"]))
    if row.get("error"):
        lines.append(str(row["error"]))
    return "\n".join(lines)


def _recovery(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    samples = report.get("errors") or []
    runs = (
        ("Без повтора", report.get("without_retry") or {}),
        ("Повтор, вторая попытка верная", report.get("with_retry") or {}),
        ("Повтор, три неверные попытки", report.get("attempts_exhausted") or {}),
        ("Потолок цены отклонен", report.get("confirmation_declined") or {}),
    )
    answer_lines = []
    parse_lines = []
    retry_lines = []
    check_lines = []
    search_lines = []
    for title, row in runs:
        journal = row.get("journal") or []
        steps = " → ".join(event.get("step", "") for event in journal)
        answer_lines.append(f"{title}\nстарт модели, попыток {row.get('attempts')}")
        parse_lines.append(f"{title}\n{steps}")
        retry_lines.append(f"{title}: попыток {row.get('attempts')}\n{steps}")
        check_lines.append(
            f"{title}: проверка {'пройдена' if row.get('ok') else 'не пройдена'}"
        )
        search_lines.append(
            f"{title}: поиск {'выполнен' if row.get('searched') else 'не выполнялся'}"
            + (f"\n{row.get('message')}" if row.get("message") else "")
            + (f"\n{_dump(row.get('result'))}" if row.get("result") else "")
        )
    sample_text = "\n\n".join(
        f"{row.get('kind')}\n{row.get('error_type')} / {row.get('cause_type')}\n"
        f"{row.get('stage')}\nвход: {row.get('text')}"
        for row in samples
    )
    nodes = [
        _node("answer", "Ответ", "четыре прогона", "\n\n".join(answer_lines), column=0, row=0),
        _node("parse", "Разбор", "журнал шагов", "\n\n".join(parse_lines), column=1, row=0),
        _node("retry", "Повтор", "не больше трех", "\n\n".join(retry_lines), column=2, row=0),
        _node("check", "Проверка", "перед поиском", "\n\n".join(check_lines), column=3, row=0),
        _node("search", "Поиск", _search_summary(runs), "\n\n".join(search_lines), column=4, row=0),
        _node("samples", "Образцы", f"{len(samples)} ошибки", sample_text or "Образцов нет.", column=1, row=1),
    ]
    edges = [
        ["answer", "parse"],
        ["parse", "retry"],
        ["retry", "check"],
        ["check", "search"],
        ["parse", "samples"],
    ]
    return nodes, edges


def _search_summary(runs: tuple[tuple[str, dict[str, Any]], ...]) -> str:
    done = sum(1 for _, row in runs if row.get("searched"))
    return f"выполнен в {done} из {len(runs)}"


def _dialogue(report: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[str]]]:
    dialogues = report.get("dialogues") or []
    nodes = [
        _node(
            "turns",
            "Реплика",
            f"{len(dialogues)} диалога",
            "\n\n".join(_dialogue_queries(item) for item in dialogues) or "Диалогов нет.",
            column=0,
            row=0,
        )
    ]
    edges: list[list[str]] = []
    for index, mode in enumerate(("state", "history")):
        label = "Состояние" if mode == "state" else "История"
        nodes.append(
            _node(
                f"{mode}-context",
                label,
                "последний JSON" if mode == "state" else "все реплики",
                _mode_trace(dialogues, mode, "context"),
                column=1,
                row=index,
            )
        )
        nodes.append(
            _node(
                f"{mode}-model",
                "Модель",
                _token_summary(dialogues, mode),
                _mode_trace(dialogues, mode, "model"),
                column=2,
                row=index,
                status=_mode_status(dialogues, mode),
            )
        )
        nodes.append(
            _node(
                f"{mode}-params",
                "Параметры",
                _score_summary(dialogues, mode),
                _mode_trace(dialogues, mode, "params"),
                column=3,
                row=index,
                status=_mode_status(dialogues, mode),
            )
        )
        nodes.append(
            _node(
                f"{mode}-search",
                "Поиск",
                "после верного разбора",
                _mode_trace(dialogues, mode, "search"),
                column=4,
                row=index,
                status=_mode_status(dialogues, mode),
            )
        )
        edges.append(["turns", f"{mode}-context"])
        edges.append([f"{mode}-context", f"{mode}-model"])
        edges.append([f"{mode}-model", f"{mode}-params"])
        edges.append([f"{mode}-params", f"{mode}-search"])
    return nodes, edges


def _dialogue_queries(dialogue: dict[str, Any]) -> str:
    turns = (dialogue.get("state") or {}).get("turns") or []
    lines = [str(dialogue.get("title"))]
    lines.extend(f"- {turn.get('query')}" for turn in turns)
    return "\n".join(lines)


def _mode_trace(dialogues: list[dict[str, Any]], mode: str, kind: str) -> str:
    blocks = []
    for dialogue in dialogues:
        block = dialogue.get(mode) or {}
        lines = [f"{dialogue.get('title')} / {mode}"]
        if kind == "context":
            kept = "в запрос уходит последний JSON параметров" if mode == "state" else "в запрос уходит вся история реплик"
            lines.append(kept)
            lines.append(f"входных токенов за диалог: {block.get('input_tokens')}")
        for turn in block.get("turns") or []:
            if kind == "model":
                lines.append(
                    f"{turn.get('query')}: вход {turn.get('input_tokens')}, выход {turn.get('output_tokens')}"
                )
            elif kind == "params":
                mark = "ok" if turn.get("correct") else "расхождение"
                lines.append(f"[{mark}] {turn.get('query')}\n{_dump(turn.get('params'))}")
                if turn.get("error"):
                    lines.append(str(turn["error"]))
            elif kind == "search":
                products = turn.get("products") or []
                lines.append(f"{turn.get('query')}\n" + ("\n".join(f"- {item}" for item in products) or "пусто"))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _token_summary(dialogues: list[dict[str, Any]], mode: str) -> str:
    total = sum((item.get(mode) or {}).get("input_tokens") or 0 for item in dialogues)
    return f"{total} входных"


def _score_summary(dialogues: list[dict[str, Any]], mode: str) -> str:
    correct = sum((item.get(mode) or {}).get("correct_turns") or 0 for item in dialogues)
    total = sum(len((item.get(mode) or {}).get("turns") or []) for item in dialogues)
    return f"{correct}/{total}"


def _mode_status(dialogues: list[dict[str, Any]], mode: str) -> str:
    for dialogue in dialogues:
        for turn in (dialogue.get(mode) or {}).get("turns") or []:
            if not turn.get("correct"):
                return "error"
    return "ok"


def _check_mark(ok: bool | None, error: str | None) -> str:
    if error or ok is False:
        return "расхождение"
    if ok is None:
        return "—"
    return "ok"


_BUILDERS = {
    "classify": _classify,
    "interpret": _interpret,
    "lookup": _lookup,
    "recovery": _recovery,
    "dialogue": _dialogue,
}
