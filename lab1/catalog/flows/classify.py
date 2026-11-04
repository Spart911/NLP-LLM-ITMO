"""Короткий текстовый ответ: промпт, модель и строковый разбор."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from catalog.llm import build_chat_model, message_text
from catalog.report import write_report

SAMPLES = (
    {
        "query": "Покажи самые дешевые ноутбуки до 80 тысяч рублей",
        "language": "русский",
    },
    {
        "query": "Нужен смартфон с хорошей камерой",
        "language": "русский",
    },
    {
        "query": "Хочу планшет для рукописных заметок",
        "language": "английский",
    },
)


def category_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты определяешь категорию товара по запросу пользователя. "
                "Ответь коротко на языке: {language}. "
                "Назови категорию своими словами и одно предложение пояснения.",
            ),
            ("human", "{query}"),
        ]
    )


def build_category_chain(llm):
    return category_prompt() | llm | StrOutputParser()


def run() -> dict:
    llm = build_chat_model(thinking=False)
    prompt = category_prompt()
    chain = build_category_chain(llm)
    sample = SAMPLES[0]

    direct = llm.invoke(prompt.invoke(sample))
    chain_text = chain.invoke(sample)
    batched = chain.batch(list(SAMPLES))
    messages = prompt.invoke(sample).to_messages()

    report = {
        "direct_type": type(direct).__name__,
        "chain_type": type(chain_text).__name__,
        "chain_is_str": isinstance(chain_text, str),
        "batch_item_type": type(batched[0]).__name__,
        "sample_messages": [
            {"type": message.type, "content": message.content} for message in messages
        ],
        "direct_content": message_text(direct),
        "chain_content": chain_text,
        "batch": [
            {
                "query": item["query"],
                "language": item["language"],
                "answer": answer,
            }
            for item, answer in zip(SAMPLES, batched, strict=True)
        ],
    }
    path = write_report("classify.json", report)

    print(f"Прямой вызов модели возвращает {report['direct_type']}.")
    print(
        f"Цепочка возвращает {report['chain_type']}. "
        f"Это строка: {report['chain_is_str']}."
    )
    print(f"Пакетный вызов возвращает список значений типа {report['batch_item_type']}.")
    print()
    for row in report["batch"]:
        print(f"[{row['language']}] {row['query']}")
        print(f"  {row['answer']}")
        print()
    print(f"Отчет: {path}")
    return report
