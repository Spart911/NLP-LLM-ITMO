"""Поиск по каталогу. SQL собирает программа, модель его не пишет и не выполняет."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catalog.db import connect
from catalog.schema import ProductQuery

UNSUPPORTED_CATEGORY = (
    "Категория не поддерживается. Поиск по каталогу не выполняется."
)
EMPTY_CATALOG = "Подходящих товаров нет."


@dataclass(frozen=True)
class Product:
    id: int
    name: str
    brand: str
    category: str
    price: int
    stock: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "price": self.price,
            "stock": self.stock,
        }


@dataclass(frozen=True)
class SearchResult:
    """Результат программного поиска.

    При category = other выборка из базы не делается: в каталоге нет такой
    категории, и искать по всем товарам было бы неверно.
    """

    params: ProductQuery
    products: list[Product]
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "params": self.params.model_dump(mode="json"),
            "products": [item.as_dict() for item in self.products],
            "message": self.message,
        }


def compile_search(params: ProductQuery) -> tuple[str, list[Any]] | None:
    """Собрать параметризованный запрос. Для other возвращает None."""

    if params.category == "other":
        return None
    direction = "ASC" if params.sort_by == "price_asc" else "DESC"
    clauses = ["category = ?"]
    values: list[Any] = [params.category]
    if params.max_price is not None:
        clauses.append("price <= ?")
        values.append(params.max_price)
    sql = (
        "SELECT id, name, brand, category, price, stock FROM products "
        f"WHERE {' AND '.join(clauses)} "
        f"ORDER BY price {direction}, id ASC"
    )
    return sql, values


def search(params: ProductQuery, *, database: Path | None = None) -> SearchResult:
    compiled = compile_search(params)
    if compiled is None:
        return SearchResult(params=params, products=[], message=UNSUPPORTED_CATEGORY)
    sql, values = compiled
    connection = connect(database)
    try:
        rows = connection.execute(sql, values).fetchall()
    finally:
        connection.close()
    products = [
        Product(
            id=row["id"],
            name=row["name"],
            brand=row["brand"],
            category=row["category"],
            price=row["price"],
            stock=row["stock"],
        )
        for row in rows
    ]
    message = None if products else EMPTY_CATALOG
    return SearchResult(params=params, products=products, message=message)
