import json
import unittest
from pathlib import Path

from catalog.pipeline import build_pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results"


class PipelineTests(unittest.TestCase):
    def test_saved_reports_become_connected_graphs(self) -> None:
        for flow_id in ("classify", "interpret", "lookup", "recovery", "dialogue"):
            report = json.loads((RESULTS / f"{flow_id}.json").read_text(encoding="utf-8"))
            graph = build_pipeline(flow_id, report)
            ids = [node["id"] for node in graph["nodes"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(node["trace"].strip() for node in graph["nodes"]))
            for source, target in graph["edges"]:
                self.assertIn(source, ids)
                self.assertIn(target, ids)

    def test_graphs_differ_between_flows(self) -> None:
        classify = json.loads((RESULTS / "classify.json").read_text(encoding="utf-8"))
        interpret = json.loads((RESULTS / "interpret.json").read_text(encoding="utf-8"))
        left = [node["id"] for node in build_pipeline("classify", classify)["nodes"]]
        right = [node["id"] for node in build_pipeline("interpret", interpret)["nodes"]]
        self.assertNotEqual(left, right)
