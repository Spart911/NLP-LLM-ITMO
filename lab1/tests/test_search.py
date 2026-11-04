import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from catalog.db import connect
from catalog.schema import ProductQuery
from catalog.search import EMPTY_CATALOG, UNSUPPORTED_CATEGORY, compile_search, search
from catalog.tools import resolve_tool_calls


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.database = Path(self._directory.name) / "products.sqlite"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_seed_covers_three_categories(self) -> None:
        connection = connect(self.database)
        try:
            total = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            counts = dict(
                connection.execute(
                    "SELECT category, COUNT(*) FROM products GROUP BY category"
                ).fetchall()
            )
            columns = [
                row[1] for row in connection.execute("PRAGMA table_info(products)")
            ]
        finally:
            connection.close()
        self.assertEqual(columns, ["id", "name", "brand", "category", "price", "stock"])
        self.assertEqual(total, 24)
        self.assertEqual(counts["laptop"], 8)
        self.assertEqual(counts["smartphone"], 8)
        self.assertEqual(counts["tablet"], 8)

    def test_laptop_price_and_sort(self) -> None:
        result = search(
            ProductQuery(category="laptop", max_price=70000, sort_by="price_asc"),
            database=self.database,
        )
        prices = [item.price for item in result.products]
        self.assertEqual(prices, [45000, 48000, 54000, 62000, 69000])
        self.assertTrue(all(item.brand and item.stock >= 0 for item in result.products))
        self.assertTrue(all(item.category == "laptop" for item in result.products))

    def test_empty_result(self) -> None:
        result = search(
            ProductQuery(category="laptop", max_price=20000),
            database=self.database,
        )
        self.assertEqual(result.products, [])
        self.assertEqual(result.message, EMPTY_CATALOG)

    def test_price_desc(self) -> None:
        result = search(
            ProductQuery(category="smartphone", sort_by="price_desc"),
            database=self.database,
        )
        prices = [item.price for item in result.products]
        self.assertEqual(prices, sorted(prices, reverse=True))
        self.assertEqual(len(prices), 8)

    def test_other_does_not_scan_catalog(self) -> None:
        params = ProductQuery(category="other")
        self.assertIsNone(compile_search(params))
        result = search(params, database=self.database)
        self.assertEqual(result.products, [])
        self.assertEqual(result.message, UNSUPPORTED_CATEGORY)

    def test_sql_binds_values(self) -> None:
        compiled = compile_search(
            ProductQuery(category="tablet", max_price=30000, sort_by="price_asc")
        )
        self.assertIsNotNone(compiled)
        sql, values = compiled
        self.assertEqual(sql.count("?"), 2)
        self.assertEqual(values, ["tablet", 30000])
        self.assertIn("ORDER BY price ASC", sql)

    def test_invalid_category_never_reaches_sql(self) -> None:
        with self.assertRaises(ValidationError):
            ProductQuery(category="laptop' OR 1=1 --")


class ToolCallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.database = Path(self._directory.name) / "products.sqlite"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_missing_tool_call_skips_search(self) -> None:
        message = AIMessage(content="Меня зовут каталог.")
        resolved = resolve_tool_calls(message, database=self.database)
        self.assertFalse(resolved["searched"])
        self.assertEqual(resolved["tool_calls"], [])
        self.assertFalse(resolved["returned_to_model"])

    def test_search_runs_only_after_validation(self) -> None:
        valid = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_products",
                    "args": {
                        "category": "laptop",
                        "max_price": 50000,
                        "sort_by": "price_asc",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        resolved = resolve_tool_calls(valid, database=self.database)
        self.assertTrue(resolved["searched"])
        self.assertEqual(
            resolved["outcomes"][0]["result"]["products"][0]["name"],
            "Acer Aspire 3",
        )
        self.assertFalse(resolved["returned_to_model"])

        invalid = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_products",
                    "args": {"category": "fridge", "sort_by": "price_asc"},
                    "id": "call-2",
                    "type": "tool_call",
                }
            ],
        )
        rejected = resolve_tool_calls(invalid, database=self.database)
        self.assertFalse(rejected["searched"])
        self.assertIsNone(rejected["outcomes"][0]["result"])
        self.assertIsNotNone(rejected["outcomes"][0]["error"])


if __name__ == "__main__":
    unittest.main()
