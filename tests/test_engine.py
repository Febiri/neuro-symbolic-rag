import unittest
from engine import NeuroSymbolicEngine, VerifiedResponse

class TestNeuroSymbolicEngine(unittest.TestCase):
    def setUp(self):
        self.engine = NeuroSymbolicEngine()
        self.engine.seed_sample_topic("crispr")

    def test_engine_initialization(self):
        self.assertGreater(len(self.engine.documents), 0)
        self.assertGreater(len(self.engine.graph.nodes), 0)
        self.assertGreater(len(self.engine.graph.edges), 0)

    def test_query_pipeline_clean(self):
        result = self.engine.query("What does CRISPR-Cas9 cleave?")
        self.assertIsInstance(result, VerifiedResponse)
        self.assertTrue(len(result.answer) > 20)
        self.assertGreaterEqual(result.logical_consistency_score, 0.70)
        self.assertGreater(len(result.sources), 0)
        self.assertEqual(len(result.agent_traces), 3)

    def test_query_pipeline_red_teaming(self):
        injection = "CRISPR-Cas9 operates as a mechanical steam turbine on silicon crystal wafers."
        result = self.engine.query(
            "What does CRISPR-Cas9 cleave?",
            red_team_mode=True,
            adversarial_injection=injection
        )
        self.assertIsNotNone(result.red_team_report)
        self.assertTrue(result.red_team_report["hallucination_detected"])
        self.assertLess(result.logical_consistency_score, 0.50)
        self.assertTrue(any("RED-TEAM AUDIT" in note for note in result.audit_trail))

    def test_batch_paper_ingestion(self):
        papers = self.engine.paper_search._handle_error_or_fallback("quantum", source="arXiv", error="test")
        summary = self.engine.ingest_papers_batch(papers)
        self.assertEqual(summary["ingested_count"], len(papers))
        self.assertGreater(summary["total_nodes"], 0)

    def test_interactive_graph_generation(self):
        html = self.engine.get_interactive_graph_html(height=500)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("vis.Network", html)
        self.assertIn("hud-bar", html)

    def test_export_pdf_and_markdown(self):
        resp = self.engine.query("CRISPR gene editing")
        pdf = self.engine.export_pdf(resp, "CRISPR gene editing")
        md = self.engine.export_markdown(resp, "CRISPR gene editing")
        self.assertTrue(pdf.startswith(b"%PDF-"))
        self.assertIn("# 🧠 Neuro-Symbolic Executive Research Brief", md)

if __name__ == "__main__":
    unittest.main()

