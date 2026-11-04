"""Повтор разбора и журнал шагов. Поиск по базе стоит за границей повтора."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

from catalog.journal import Journal
from catalog.llm import message_text
from catalog.prompts import build_parser, build_prompt
from catalog.report import write_report
from catalog.schema import ProductQuery
from catalog.search import search

MAX_ATTEMPTS = 3

INVALID_JSON = "это не json"
VALID_JSON = '{"category": "laptop", "max_price": 70000, "sort_by": "price_asc"}'
UNBOUNDED_JSON = '{"category": "tablet", "max_price": null, "sort_by": "price_asc"}'

ERROR_SAMPLES: tuple[dict[str, str], ...] = (
    {
        "kind": "синтаксически некорректный JSON",
        "text": INVALID_JSON,
        "stage": "Разбор текста в JSON. До проверки полей ProductQuery выполнение не доходит.",
    },
    {
        "kind": "значение вне перечисления",
        "text": '{"category": "fridge", "max_price": 1000, "sort_by": "price_asc"}',
        "stage": "JSON уже получен. Ошибка возникает при проверке поля category моделью ProductQuery.",
    },
    {
        "kind": "нарушение числового ограничения",
        "text": '{"category": "laptop", "max_price": -5, "sort_by": "price_asc"}',
        "stage": "JSON уже получен. Ошибка возникает на ограничении max_price >= 0.",
    },
)

Confirmer = Callable[[ProductQuery], bool]


def analyze_errors() -> list[dict[str, Any]]:
    parser = build_parser()
    rows = []
    for sample in ERROR_SAMPLES:
        try:
            parser.parse(sample["text"])
        except OutputParserException as exc:
            cause = exc.__cause__
            rows.append(
                {
                    "kind": sample["kind"],
                    "text": sample["text"],
                    "stage": sample["stage"],
                    "error_type": type(exc).__name__,
                    "cause_type": type(cause).__name__ if cause else None,
                    "message": str(exc),
                }
            )
        else:
            rows.append(
                {
                    "kind": sample["kind"],
                    "text": sample["text"],
                    "stage": sample["stage"],
                    "error_type": None,
                    "cause_type": None,
                    "message": "Парсер принял ответ, пример не показывает ошибку.",
                }
            )
    return rows


def build_parse_chain(llm, parser, journal: Journal, *, retry: bool):
    prompt = build_prompt(parser)

    def invoke_llm(messages):
        journal.attempts += 1
        journal.add("llm_start", attempt=journal.attempts)
        return llm.invoke(messages)

    def parse_output(message):
        text = message_text(message)
        try:
            parsed = parser.parse(text)
        except OutputParserException as exc:
            cause = exc.__cause__
            journal.add(
                "parse_error",
                attempt=journal.attempts,
                error_type=type(exc).__name__,
                cause_type=type(cause).__name__ if cause else None,
            )
            raise
        journal.add(
            "parsed",
            attempt=journal.attempts,
            params=parsed.model_dump(mode="json"),
        )
        return parsed

    chain = prompt | RunnableLambda(invoke_llm) | RunnableLambda(parse_output)
    if retry:
        return chain.with_retry(
            stop_after_attempt=MAX_ATTEMPTS,
            retry_if_exception_type=(OutputParserException,),
            wait_exponential_jitter=False,
        )
    return chain


def _failure_message(attempts: int, *, retry: bool) -> str:
    if retry and attempts >= MAX_ATTEMPTS:
        return (
            f"Не удалось разобрать параметры запроса за {MAX_ATTEMPTS} попытки. "
            "Поиск по каталогу не выполнялся."
        )
    return "Не удалось разобрать параметры запроса. Поиск по каталогу не выполнялся."


def execute(
    *,
    responses: list[str],
    retry: bool,
    query: str = "Найди ноутбуки дешевле 70 000 рублей",
    confirmer: Confirmer | None = None,
    database: Path | None = None,
) -> dict[str, Any]:
    return run_query(
        query,
        FakeListChatModel(responses=responses),
        retry=retry,
        confirmer=confirmer,
        database=database,
    )


def run_query(
    query: str,
    llm,
    *,
    retry: bool,
    confirmer: Confirmer | None = None,
    database: Path | None = None,
) -> dict[str, Any]:
    """Промпт, модель, разбор с повтором, журнал и поиск после проверки."""

    journal = Journal()
    parser = build_parser()
    chain = build_parse_chain(llm, parser, journal, retry=retry)
    allow = confirmer or (lambda _params: True)
    try:
        params = chain.invoke({"query": query})
    except OutputParserException:
        message = _failure_message(journal.attempts, retry=retry)
        journal.add("stopped", message=message)
        return _report(journal, ok=False, searched=False, message=message, result=None)
    return _after_parse(params, journal, allow, database)


def _after_parse(
    params: ProductQuery,
    journal: Journal,
    confirmer: Confirmer,
    database: Path | None,
) -> dict[str, Any]:
    try:
        checked = ProductQuery.model_validate(params.model_dump())
    except Exception as exc:
        journal.add("validation", ok=False, error=type(exc).__name__)
        message = "Параметры не прошли проверку. Поиск по каталогу не выполнялся."
        journal.add("stopped", message=message)
        return _report(journal, ok=False, searched=False, message=message, result=None)

    journal.add("validation", ok=True, params=checked.model_dump(mode="json"))
    if checked.category != "other" and checked.max_price is None:
        accepted = confirmer(checked)
        journal.add("confirmation", accepted=accepted)
        if not accepted:
            message = "Поиск без ограничения цены отменен."
            journal.add("search_skipped", reason=message)
            return _report(journal, ok=True, searched=False, message=message, result=None)

    result = search(checked, database=database)
    journal.add("search", found=len(result.products), message=result.message)
    return _report(
        journal,
        ok=True,
        searched=True,
        message=result.message,
        result=result.as_dict(),
    )


def _report(
    journal: Journal,
    *,
    ok: bool,
    searched: bool,
    message: str | None,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "attempts": journal.attempts,
        "searched": searched,
        "message": message,
        "result": result,
        "journal": journal.events,
    }


def run() -> dict[str, Any]:
    recovered = execute(responses=[INVALID_JSON, VALID_JSON], retry=True)
    without_retry = execute(responses=[INVALID_JSON, VALID_JSON], retry=False)
    exhausted = execute(responses=[INVALID_JSON, INVALID_JSON, INVALID_JSON], retry=True)
    declined = execute(
        responses=[UNBOUNDED_JSON],
        retry=True,
        query="Хочу посмотреть планшеты",
        confirmer=lambda _params: False,
    )
    report = {
        "errors": analyze_errors(),
        "without_retry": without_retry,
        "with_retry": recovered,
        "attempts_exhausted": exhausted,
        "confirmation_declined": declined,
    }
    path = write_report("recovery.json", report)
    _print(report, path)
    return report


def _print(report: dict[str, Any], path: Path) -> None:
    print("Ошибки разбора")
    for row in report["errors"]:
        print(f"- {row['kind']}: {row['error_type']} / {row['cause_type']}")
        print(f"  {row['stage']}")
    print()
    for title, key in (
        ("Без повтора", "without_retry"),
        ("С повтором, вторая попытка верная", "with_retry"),
        ("С повтором, все три попытки неверные", "attempts_exhausted"),
        ("Поиск без потолка цены отклонен", "confirmation_declined"),
    ):
        row = report[key]
        print(
            f"{title}: попыток {row['attempts']}, "
            f"поиск {'выполнен' if row['searched'] else 'не выполнялся'}"
        )
        if row["message"]:
            print(f"  {row['message']}")
        print("  журнал: " + " -> ".join(event["step"] for event in row["journal"]))
    print()
    print(f"Отчет: {path}")
