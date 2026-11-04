import tempfile
import unittest
from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from catalog.flows.dialogue import DialogState, DIALOGUES, build_followup_prompt, run_dialogue
from catalog.prompts import build_parser
from catalog.schema import ProductQuery


class DialogStateTests(unittest.TestCase):
    def test_state_keeps_last_params_and_history_keeps_turns(self) -> None:
        session = DialogState()
        self.assertEqual(session.state_json(), "null")
        self.assertEqual(session.history_text(), "пусто")
        session.remember(
            "Покажи ноутбуки до 80 тысяч",
            ProductQuery(category="laptop", max_price=80000, sort_by="price_asc"),
        )
        self.assertEqual(session.params.max_price, 80000)
        self.assertNotIn("Покажи ноутбуки", session.state_json())
        self.assertIn("Покажи ноутбуки до 80 тысяч", session.history_text())
        self.assertIn("80000", session.history_text())

    def test_prompts_put_context_in_the_human_message(self) -> None:
        parser = build_parser()
        state = build_followup_prompt(parser, "state").invoke(
            {"query": "А сначала дорогие?", "context": '{"max_price":80000}'}
        )
        history = build_followup_prompt(parser, "history").invoke(
            {
                "query": "А сначала дорогие?",
                "context": "Пользователь: Покажи ноутбуки до 80 тысяч",
            }
        )
        state_human = state.to_messages()[1].content
        history_human = history.to_messages()[1].content
        self.assertIn("Текущие параметры", state_human)
        self.assertIn("80000", state_human)
        self.assertNotIn("Покажи ноутбуки", state_human)
        self.assertIn("История диалога", history_human)
        self.assertIn("Покажи ноутбуки до 80 тысяч", history_human)
        self.assertIn(parser.get_format_instructions(), state.to_messages()[0].content)


class FollowupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.database = Path(self._directory.name) / "products.sqlite"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_followup_updates_stored_params_and_searches(self) -> None:
        dialogue = {
            "title": "короткий",
            "turns": DIALOGUES[0]["turns"][:2],
        }
        llm = FakeListChatModel(
            responses=[
                '{"category": "laptop", "max_price": 80000, "sort_by": "price_asc"}',
                '{"category": "laptop", "max_price": 80000, "sort_by": "price_desc"}',
            ]
        )
        report = run_dialogue(llm, dialogue, "state", database=self.database)
        self.assertEqual(report["correct_turns"], 2)
        self.assertEqual(report["turns"][1]["params"]["sort_by"], "price_desc")
        self.assertEqual(report["turns"][1]["products"][0], "Lenovo IdeaPad 5 — 69000")

    def test_invalid_reply_does_not_replace_params(self) -> None:
        dialogue = {
            "title": "сбой",
            "turns": (
                DIALOGUES[0]["turns"][0],
                {"query": "А сначала дорогие?", "expect": DIALOGUES[0]["turns"][1]["expect"]},
            ),
        }
        llm = FakeListChatModel(
            responses=[
                '{"category": "laptop", "max_price": 80000, "sort_by": "price_asc"}',
                "это не json",
            ]
        )
        report = run_dialogue(llm, dialogue, "history", database=self.database)
        self.assertTrue(report["turns"][0]["correct"])
        self.assertFalse(report["turns"][1]["correct"])
        self.assertEqual(report["turns"][1]["products"], [])
        self.assertIsNone(report["turns"][1]["params"])


if __name__ == "__main__":
    unittest.main()
