import unittest

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from catalog.flows.classify import SAMPLES, category_prompt
from catalog.prompts import build_parser, build_prompt
from catalog.schema import ProductQuery


class ProductQueryTests(unittest.TestCase):
    def test_price_and_sort_defaults(self) -> None:
        item = ProductQuery(category="laptop")
        self.assertIsNone(item.max_price)
        self.assertEqual(item.sort_by, "price_asc")

    def test_negative_price_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ProductQuery(category="laptop", max_price=-1)

    def test_unknown_category_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ProductQuery(category="fridge")

    def test_json_roundtrip(self) -> None:
        item = ProductQuery(
            category="smartphone",
            max_price=40000,
            sort_by="price_desc",
        )
        restored = ProductQuery.model_validate_json(item.model_dump_json())
        self.assertEqual(restored, item)

    def test_parser_reads_fenced_json(self) -> None:
        parser = PydanticOutputParser(pydantic_object=ProductQuery)
        text = """```json
{"category": "tablet", "max_price": null, "sort_by": "price_asc"}
```"""
        parsed = parser.parse(text)
        self.assertIsInstance(parsed, ProductQuery)
        self.assertEqual(parsed.category, "tablet")
        self.assertIsNone(parsed.max_price)


class PromptTests(unittest.TestCase):
    def test_category_prompt_uses_system_and_human_variables(self) -> None:
        sample = SAMPLES[0]
        messages = category_prompt().invoke(sample).to_messages()
        self.assertEqual(messages[0].type, "system")
        self.assertEqual(messages[1].type, "human")
        self.assertIn(sample["language"], messages[0].content)
        self.assertEqual(messages[1].content, sample["query"])

    def test_format_instructions_go_into_system_message(self) -> None:
        parser = build_parser()
        messages = build_prompt(parser).invoke({"query": "планшеты"}).to_messages()
        self.assertEqual(messages[0].type, "system")
        self.assertIn(parser.get_format_instructions(), messages[0].content)
        self.assertEqual(messages[1].content, "планшеты")


if __name__ == "__main__":
    unittest.main()
