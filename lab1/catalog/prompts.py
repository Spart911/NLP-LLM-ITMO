"""Общий промпт разбора запроса в параметры каталога."""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from catalog.schema import ProductQuery

SYSTEM_PROMPT = """Ты переводишь запрос о товаре в параметры поиска каталога.
Поддерживаемые категории: laptop (ноутбук), smartphone (смартфон, телефон), tablet (планшет).
Любой другой предмет — category other.
max_price — целое число рублей. «50 тысяч» и «50 тыс.» означают 50000, пробелы внутри числа не меняют его величину.
Если верхней границы цены нет, max_price равен null.
sort_by = price_desc, когда просят сначала дорогие или самые дорогие.
sort_by = price_asc, когда просят сначала дешевые или порядок не указан.
Верни только объект по инструкции формата, без пояснения вокруг него.

{format_instructions}"""


def build_parser() -> PydanticOutputParser:
    return PydanticOutputParser(pydantic_object=ProductQuery)


def build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions())


def build_structured_chain(llm, parser: PydanticOutputParser):
    return build_prompt(parser) | llm | parser
