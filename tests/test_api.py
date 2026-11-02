import unittest
from fastapi.testclient import TestClient
from api import app

class TestFastAPIOrchestration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("documents_indexed", data)
        self.assertIn("graph_nodes", data)

    def test_search_papers_endpoint(self):
        response = self.client.post("/search_papers", json={
            "query": "quantum computing",
            "source": "arxiv",
            "max_results": 2
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("papers", data)
        self.assertGreaterEqual(len(data["papers"]), 1)

    def test_ingest_endpoint(self):
        response = self.client.post("/ingest", json={
            "doc_id": "Test_REST_Doc",
            "content": "Neuro-symbolic verifiers enforce logical determinism."
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")

    def test_query_endpoint_normal(self):
        response = self.client.post("/query", json={
            "query": "What is neuro-symbolic AI?",
            "red_team_mode": False
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("logical_consistency_score", data)
        self.assertIn("agent_traces", data)

    def test_query_endpoint_red_team(self):
        response = self.client.post("/query", json={
            "query": "What is neuro-symbolic AI?",
            "red_team_mode": True,
            "adversarial_injection": "Symbolic reasoning is non-deterministic and purely randomized."
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNotNone(data["red_team_report"])
        self.assertTrue(data["red_team_report"]["hallucination_detected"])

    def test_graph_endpoints(self):
        # JSON graph
        res_json = self.client.get("/graph")
        self.assertEqual(res_json.status_code, 200)
        self.assertIn("nodes", res_json.json())
        self.assertIn("edges", res_json.json())

        # Interactive HTML graph
        res_html = self.client.get("/graph/interactive")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("text/html", res_html.headers["content-type"])
        self.assertIn("vis-network", res_html.text)

    def test_red_team_presets_and_audit(self):
        presets_res = self.client.get("/red_team_presets")
        self.assertEqual(presets_res.status_code, 200)
        self.assertGreater(len(presets_res.json()["presets"]), 0)

        audit_res = self.client.post("/red_team_audit", json={
            "candidate_answer": "Symbolic reasoning is purely non-deterministic.",
            "adversarial_injection": "Symbolic reasoning is non-deterministic."
        })
        self.assertEqual(audit_res.status_code, 200)
        self.assertTrue(audit_res.json()["hallucination_detected"])

    def test_export_report_endpoint(self):
        # Markdown export
        md_res = self.client.post("/export_report", json={
            "query": "What is neuro-symbolic AI?",
            "format": "markdown"
        })
        self.assertEqual(md_res.status_code, 200)
        self.assertEqual(md_res.json()["format"], "markdown")

        # PDF export
        pdf_res = self.client.post("/export_report", json={
            "query": "What is neuro-symbolic AI?",
            "format": "pdf"
        })
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.json()["format"], "pdf")
        self.assertIn("base64_content", pdf_res.json())

if __name__ == "__main__":
    unittest.main()

