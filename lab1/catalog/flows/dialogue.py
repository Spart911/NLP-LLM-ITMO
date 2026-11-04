"""Уточнения диалога. Параметры и история хранятся в программе, не в модели."""

from pathlib import Path
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from catalog.llm import build_chat_model, message_text, message_usage
from catalog.prompts import SYSTEM_PROMPT, build_parser
from catalog.report import write_report
from catalog.schema import ProductQuery
from catalog.search import search

STATE_RULE = (
    "В сообщении пользователя лежит JSON текущих параметров и новая реплика. "
    "Если текущие параметры равны null, собери объект только из реплики. "
    "Иначе верни полный объект: меняй поле только когда реплика этого требует, "
    "остальные поля сохрани."
)
HISTORY_RULE = (
    "В сообщении пользователя лежит история диалога и новая реплика. "
    "Если история пуста, собери объект только из реплики. "
    "Иначе возьми последние параметры из истории и верни полный объект: "
    "меняй поле только когда новая реплика этого требует."
)

DIALOGUES: tuple[dict[str, Any], ...] = (
    {
        "title": "Ноутбуки, порядок, затем смартфоны",
        "turns": (
            {
                "query": "Покажи ноутбуки до 80 тысяч",
                "expect": {
                    "category": "laptop",
                    "max_price": 80000,
                    "sort_by": "price_asc",
                },
            },
            {
                "query": "А сначала дорогие?",
                "expect": {
                    "category": "laptop",
                    "max_price": 80000,
                    "sort_by": "price_desc",
                },
            },
            {
                "query": "А что есть среди смартфонов?",
                "expect": {
                    "category": "smartphone",
                    "max_price": 80000,
                    "sort_by": "price_desc",
                },
            },
        ),
    },
    {
        "title": "Планшеты, потолок цены, затем ноутбуки",
        "turns": (
            {
                "query": "Покажи самые дешевые планшеты",
                "expect": {
                    "category": "tablet",
                    "max_price": None,
                    "sort_by": "price_asc",
                },
            },
            {
                "query": "До 30 тысяч",
                "expect": {
                    "category": "tablet",
                    "max_price": 30000,
                    "sort_by": "price_asc",
                },
            },
            {
                "query": "А ноутбуки?",
                "expect": {
                    "category": "laptop",
                    "max_price": 30000,
                    "sort_by": "price_asc",
                },
            },
        ),
    },
)


class DialogState:
    """Последний удачный запрос и реплики, из которых он получился."""

    def __init__(self) -> None:
        self.params: ProductQuery | None = None
        self.turns: list[dict[str, str]] = []

    def state_json(self) -> str:
        if self.params is None:
            return "null"
        return self.params.model_dump_json(ensure_ascii=False)

    def history_text(self) -> str:
        if not self.turns:
            return "пусто"
        return "\n".join(
            f"Пользователь: {turn['query']}\nПараметры: {turn['params']}"
            for turn in self.turns
        )

    def remember(self, query: str, params: ProductQuery) -> None:
        self.params = params
        self.turns.append(
            {"query": query, "params": params.model_dump_json(ensure_ascii=False)}
        )


def build_followup_prompt(parser, mode: str) -> ChatPromptTemplate:
    if mode == "state":
        system = f"{STATE_RULE}\n{SYSTEM_PROMPT}"
        human = "Текущие параметры: {context}\nНовая реплика: {query}"
    elif mode == "history":
        system = f"{HISTORY_RULE}\n{SYSTEM_PROMPT}"
        human = "История диалога:\n{context}\nНовая реплика: {query}"
    else:
        raise ValueError(f"Неизвестный режим диалога: {mode}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions())


def build_followup_chain(llm, parser, mode: str):
    prompt = build_followup_prompt(parser, mode)

    def parse_message(message) -> dict[str, Any]:
        usage = message_usage(message)
        parsed = parser.parse(message_text(message))
        return {
            "params": parsed,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
        }

    return prompt | llm | RunnableLambda(parse_message)


def run_dialogue(
    llm,
    dialogue: dict[str, Any],
    mode: str,
    *,
    database: Path | None = None,
) -> dict[str, Any]:
    session = DialogState()
    parser = build_parser()
    chain = build_followup_chain(llm, parser, mode)
    rows = []
    for turn in dialogue["turns"]:
        context = session.state_json() if mode == "state" else session.history_text()
        try:
            parsed = chain.invoke({"query": turn["query"], "context": context})
        except OutputParserException as exc:
            rows.append(
                {
                    "query": turn["query"],
                    "params": None,
                    "products": [],
                    "correct": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        params: ProductQuery = parsed["params"]
        found = search(params, database=database)
        actual = params.model_dump(mode="json")
        rows.append(
            {
                "query": turn["query"],
                "params": actual,
                "products": [f"{item.name} — {item.price}" for item in found.products],
                "correct": all(actual[key] == value for key, value in turn["expect"].items()),
                "input_tokens": parsed["input_tokens"],
                "output_tokens": parsed["output_tokens"],
                "error": None,
            }
        )
        session.remember(turn["query"], params)
    return {
        "title": dialogue["title"],
        "mode": mode,
        "turns": rows,
        "input_tokens": sum(row["input_tokens"] or 0 for row in rows),
        "correct_turns": sum(1 for row in rows if row["correct"]),
    }


def compare_dialogues(llm, *, database: Path | None = None) -> list[dict[str, Any]]:
    compared = []
    for dialogue in DIALOGUES:
        state = run_dialogue(llm, dialogue, "state", database=database)
        history = run_dialogue(llm, dialogue, "history", database=database)
        compared.append(
            {
                "title": dialogue["title"],
                "state": state,
                "history": history,
                "input_tokens": {
                    "state": state["input_tokens"],
                    "history": history["input_tokens"],
                },
            }
        )
    return compared


def run() -> dict[str, Any]:
    report = {"dialogues": compare_dialogues(build_chat_model(thinking=False))}
    path = write_report("dialogue.json", report)
    for dialogue in report["dialogues"]:
        print(dialogue["title"])
        for mode in ("state", "history"):
            block = dialogue[mode]
            print(
                f"  {mode}: верных шагов {block['correct_turns']}/"
                f"{len(block['turns'])}, входных токенов {block['input_tokens']}"
            )
            for turn in block["turns"]:
                mark = "ok" if turn["correct"] else "расхождение"
                print(
                    f"    [{mark}] {turn['query']} -> {turn['params']} "
                    f"({turn['input_tokens']} входных)"
                )
        print()
    print(f"Отчет: {path}")
    return report
