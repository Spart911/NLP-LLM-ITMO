"""Поиск по каталогу: цепочка возвращает товары, инструмент только отдает параметры."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from catalog.llm import build_chat_model, message_text
from catalog.prompts import build_parser, build_structured_chain
from catalog.report import write_report
from catalog.schema import ProductQuery
from catalog.search import search
from catalog.tools import resolve_tool_calls, search_products

PIPELINE_CASES: tuple[dict[str, Any], ...] = (
    {
        "query": "Найди ноутбуки дешевле 70 000 рублей",
        "expect": {"category": "laptop", "max_price": 70000, "sort_by": "price_asc"},
    },
    {
        "query": "Покажи самые дорогие смартфоны",
        "expect": {"category": "smartphone", "max_price": None, "sort_by": "price_desc"},
    },
    {
        "query": "Хочу посмотреть планшеты",
        "expect": {"category": "tablet", "max_price": None, "sort_by": "price_asc"},
    },
    {
        "query": "Покажи ноутбуки до 50 тысяч",
        "expect": {"category": "laptop", "max_price": 50000, "sort_by": "price_asc"},
    },
    {
        "query": "Найди холодильники",
        "expect": {"category": "other", "max_price": None, "sort_by": "price_asc"},
    },
    {
        "query": "Возьми что-нибудь для конспектов и фильмов, тысяч до тридцати",
        "expect": None,
    },
    {
        "query": "Нужен нормальный экран в карман, не самый дешевый",
        "expect": None,
    },
    {
        "query": "Есть ли ноутбуки дешевле 20 тысяч",
        "expect": {"category": "laptop", "max_price": 20000, "sort_by": "price_asc"},
    },
    {
        "query": "Покажи смартфоны до 30 тысяч",
        "expect": {"category": "smartphone", "max_price": 30000, "sort_by": "price_asc"},
    },
    {
        "query": "Покажи самые дешевые планшеты",
        "expect": {"category": "tablet", "max_price": None, "sort_by": "price_asc"},
    },
)

NO_DB_CASES: tuple[dict[str, str], ...] = (
    {
        "source": "Найди ноутбуки дешевле 70 000 рублей",
        "ask": "Какие ноутбуки дешевле 70 000 рублей есть в каталоге?",
    },
    {
        "source": "Покажи самые дорогие смартфоны",
        "ask": "Какие самые дорогие смартфоны есть в каталоге?",
    },
    {
        "source": "Есть ли ноутбуки дешевле 20 тысяч",
        "ask": "Какие ноутбуки дешевле 20 000 рублей есть в каталоге?",
    },
)

TOOL_QUERIES: tuple[str, ...] = (
    "Найди ноутбуки дешевле 70 000 рублей",
    "Покажи самые дорогие смартфоны",
    "Хочу посмотреть планшеты",
    "Покажи ноутбуки до 50 тысяч",
    "Найди холодильники",
    "Привет! Как тебя зовут?",
)

TOOL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты интерфейс каталога товаров. Когда пользователь ищет товар, "
            "вызови инструмент search_products и не перечисляй товары сам. "
            "Если запрос не про поиск товара, ответь одной фразой и инструмент не вызывай.",
        ),
        ("human", "{query}"),
    ]
)

NO_DB_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Ответь на вопрос пользователя о товарах."),
        ("human", "{query}"),
    ]
)

APPROACH_COMPARISON = (
    "В текстовой цепочке схема ProductQuery описана в системном промпте "
    "инструкциями формата. Модель пишет JSON в content, парсер достает поля из этого текста, "
    "Pydantic их проверяет, и программа вызывает поиск. "
    "У инструмента схема описана аргументами search_products и передается через bind_tools. "
    "Вызов из ответа разбирает сервер провайдера и кладет его в tool_calls. "
    "Приложение проверяет args через ProductQuery и само выполняет SQL. "
    "Результат поиска модели не отправляется."
)


def build_catalog_chain(llm=None):
    model = llm or build_chat_model(thinking=False)
    return build_structured_chain(model, build_parser()) | RunnableLambda(_search_payload)


def _search_payload(params: ProductQuery) -> dict[str, Any]:
    return search(params).as_dict()


def _products_follow(params: dict[str, Any], products: list[dict[str, Any]]) -> bool:
    if params["category"] == "other":
        return products == []
    categories_ok = all(item["category"] == params["category"] for item in products)
    price_ok = params["max_price"] is None or all(
        item["price"] <= params["max_price"] for item in products
    )
    prices = [item["price"] for item in products]
    ordered = prices == sorted(prices, reverse=params["sort_by"] == "price_desc")
    return categories_ok and price_ok and ordered


def _correct(expect: dict[str, Any] | None, payload: dict[str, Any]) -> bool:
    params = payload["params"]
    if expect is not None and any(params[key] != value for key, value in expect.items()):
        return False
    return _products_follow(params, payload["products"])


def _product_line(item: dict[str, Any]) -> str:
    return (
        f"{item['name']}, {item['brand']} — {item['price']} ₽, "
        f"остаток {item['stock']}"
    )


def _run_pipeline(chain) -> list[dict[str, Any]]:
    rows = []
    for case in PIPELINE_CASES:
        try:
            payload = chain.invoke({"query": case["query"]})
        except Exception as exc:
            rows.append(
                {
                    "query": case["query"],
                    "params": None,
                    "products": [],
                    "message": None,
                    "correct": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "query": case["query"],
                "params": payload["params"],
                "products": [_product_line(item) for item in payload["products"]],
                "message": payload["message"],
                "correct": _correct(case["expect"], payload),
                "error": None,
            }
        )
    return rows


def _run_without_database(llm, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query = {row["query"]: row for row in rows}
    chain = NO_DB_PROMPT | llm | RunnableLambda(message_text)
    comparisons = []
    for case in NO_DB_CASES:
        answer = chain.invoke({"query": case["ask"]})
        catalog = by_query[case["source"]]["products"]
        mentioned = [name for name in catalog if name.split(" — ")[0] in answer]
        comparisons.append(
            {
                "source": case["source"],
                "ask": case["ask"],
                "catalog": catalog,
                "model_answer": answer,
                "catalog_names_in_answer": mentioned,
            }
        )
    return comparisons


def _run_tools(llm) -> list[dict[str, Any]]:
    bound = TOOL_PROMPT | llm.bind_tools([search_products])
    rows = []
    for query in TOOL_QUERIES:
        message = bound.invoke({"query": query})
        resolved = resolve_tool_calls(message)
        resolved["query"] = query
        rows.append(resolved)
    return rows


def run() -> dict[str, Any]:
    llm = build_chat_model(thinking=False)
    pipeline = _run_pipeline(build_catalog_chain(llm))
    without_database = _run_without_database(llm, pipeline)
    tools = _run_tools(llm)
    report = {
        "pipeline": pipeline,
        "without_database": without_database,
        "tools": tools,
        "approach_comparison": APPROACH_COMPARISON,
    }
    path = write_report("lookup.json", report)

    print("Поиск по каталогу")
    for row in pipeline:
        mark = "ok" if row["correct"] else "расхождение"
        print(f"[{mark}] {row['query']}")
        print(f"  параметры: {row['params']}")
        if row["products"]:
            print("  товары: " + "; ".join(row["products"]))
        if row["message"]:
            print(f"  {row['message']}")
        if row["error"]:
            print(f"  {row['error']}")
    print()
    print("Ответ модели без базы")
    for row in without_database:
        print(f"- {row['ask']}")
        print(f"  каталог: {', '.join(row['catalog']) or 'пусто'}")
        print(f"  модель: {row['model_answer'][:400]}")
    print()
    print("Инструмент search_products")
    for row in tools:
        print(f"- {row['query']}")
        print(f"  content: {row['content'] or 'пусто'}")
        print(f"  tool_calls: {row['tool_calls'] or 'нет'}")
        print(f"  поиск выполнен: {row['searched']}; результат модели не возвращался")
    print()
    print(APPROACH_COMPARISON)
    print()
    print(f"Отчет: {path}")
    return report
