import tempfile
import unittest
from pathlib import Path

from catalog.flows.recovery import (
    INVALID_JSON,
    VALID_JSON,
    analyze_errors,
    execute,
)


def _steps(report: dict) -> list[str]:
    return [event["step"] for event in report["journal"]]


class ErrorAnalysisTests(unittest.TestCase):
    def test_three_distinct_parse_failures(self) -> None:
        rows = analyze_errors()
        self.assertEqual(len(rows), 3)
        causes = [row["cause_type"] for row in rows]
        self.assertEqual(causes, ["JSONDecodeError", "ValidationError", "ValidationError"])
        self.assertTrue(all(row["error_type"] == "OutputParserException" for row in rows))


class RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.database = Path(self._directory.name) / "products.sqlite"

    def tearDown(self) -> None:
        self._directory.cleanup()

    def test_without_retry_stops_before_search(self) -> None:
        report = execute(
            responses=[INVALID_JSON, VALID_JSON],
            retry=False,
            database=self.database,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["attempts"], 1)
        self.assertFalse(report["searched"])
        self.assertNotIn("search", _steps(report))
        self.assertNotIn("validation", _steps(report))

    def test_retry_searches_once_after_valid_parse(self) -> None:
        report = execute(
            responses=[INVALID_JSON, VALID_JSON],
            retry=True,
            database=self.database,
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["attempts"], 2)
        self.assertTrue(report["searched"])
        steps = _steps(report)
        self.assertEqual(steps.count("llm_start"), 2)
        self.assertEqual(steps.count("search"), 1)
        self.assertLess(steps.index("validation"), steps.index("search"))
        self.assertEqual(report["result"]["params"]["max_price"], 70000)
        self.assertGreater(len(report["result"]["products"]), 1)

    def test_exhausted_attempts_keep_the_database_idle(self) -> None:
        report = execute(
            responses=[INVALID_JSON, INVALID_JSON, INVALID_JSON],
            retry=True,
            database=self.database,
        )
        self.assertFalse(report["ok"])
        self.assertEqual(report["attempts"], 3)
        self.assertFalse(report["searched"])
        self.assertIn("3 попытки", report["message"])
        self.assertEqual(_steps(report).count("parse_error"), 3)
        self.assertNotIn("search", _steps(report))

    def test_unbounded_search_waits_for_confirmation(self) -> None:
        report = execute(
            responses=['{"category": "tablet", "max_price": null, "sort_by": "price_asc"}'],
            retry=True,
            confirmer=lambda _params: False,
            database=self.database,
        )
        self.assertTrue(report["ok"])
        self.assertFalse(report["searched"])
        steps = _steps(report)
        self.assertLess(steps.index("validation"), steps.index("confirmation"))
        self.assertNotIn("search", steps)
        self.assertIn("search_skipped", steps)


if __name__ == "__main__":
    unittest.main()
