"""Параметры поиска, которые программа может проверить до обращения к каталогу."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["laptop", "smartphone", "tablet", "other"]
SortBy = Literal["price_asc", "price_desc"]


class ProductQuery(BaseModel):
    """Структура запроса к каталогу."""

    category: Category = Field(
        description=(
            "Категория товара: laptop, smartphone или tablet. "
            "Если товар к ним не относится, укажи other."
        ),
    )
    max_price: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Верхняя граница цены в рублях. Целое неотрицательное число. "
            "null, если пользователь цену не ограничил."
        ),
    )
    sort_by: SortBy = Field(
        default="price_asc",
        description=(
            "Сортировка по цене: price_asc — сначала дешевые, "
            "price_desc — сначала дорогие. Если порядок не назван, price_asc."
        ),
    )
