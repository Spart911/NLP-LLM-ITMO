import unittest

from catalog.ask import build_ask_pipeline


class AskPipelineTests(unittest.TestCase):
    def test_success_includes_retry_and_journal(self) -> None:
        graph = build_ask_pipeline(
            {
                "messages": [{"type": "human", "content": "ноутбуки"}],
                "calls": [
                    {
                        "elapsed_sec": 1.2,
                        "content": '{"category":"laptop"}',
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    }
                ],
                "params": {"category": "laptop", "max_price": 70000, "sort_by": "price_asc"},
                "products": ["Acer Aspire 3, Acer — 45000 ₽, остаток 6"],
                "message": None,
                "error": None,
                "attempts": 1,
                "searched": True,
                "journal": [
                    {"step": "llm_start", "attempt": 1},
                    {"step": "parsed", "attempt": 1},
                    {"step": "validation", "ok": True},
                    {"step": "search", "found": 1},
                ],
            }
        )
        ids = [node["id"] for node in graph["nodes"]]
        self.assertEqual(ids, ["prompt", "model", "parse", "retry", "search", "journal"])
        self.assertEqual(graph["nodes"][4]["status"], "ok")
        self.assertIn("Acer Aspire 3", graph["nodes"][4]["trace"])
        self.assertIn("llm_start", graph["nodes"][5]["trace"])

    def test_parse_error_skips_search(self) -> None:
        graph = build_ask_pipeline(
            {
                "messages": [],
                "calls": [{"elapsed_sec": 0.4, "content": "это не json", "usage": {}}],
                "params": None,
                "products": [],
                "message": "Не удалось разобрать параметры запроса за 3 попытки.",
                "error": "Не удалось разобрать параметры запроса за 3 попытки.",
                "attempts": 3,
                "searched": False,
                "journal": [
                    {"step": "llm_start", "attempt": 1},
                    {"step": "parse_error", "attempt": 1},
                    {"step": "stopped"},
                ],
            }
        )
        search = next(node for node in graph["nodes"] if node["id"] == "search")
        retry = next(node for node in graph["nodes"] if node["id"] == "retry")
        self.assertEqual(search["status"], "error")
        self.assertIn("не выполнялся", search["trace"])
        self.assertEqual(retry["summary"], "3 из 3")
