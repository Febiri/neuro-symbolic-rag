import unittest
from report_generator import ResearchReportGenerator

class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        self.gen = ResearchReportGenerator()

    def test_pdf_generation_valid_bytes(self):
        pdf_bytes = self.gen.generate_pdf(
            query="What is CRISPR-Cas9?",
            answer="CRISPR-Cas9 enables targeted genome editing in eukaryotic organisms.",
            sources=["Doc_1"],
            logical_score=1.0,
            verification_notes="Pass: 100% of claims verified against knowledge graph.",
            audit_trail=["Pass: All cited sources verified in active registry."],
            papers_meta={"Doc_1": {"title": "CRISPR Review", "source": "PubMed", "published_date": "2024"}},
            triplets=[("CRISPR-Cas9", "enables", "genome editing", "Doc_1")]
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)
        # Check standard PDF magic header
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))

    def test_markdown_generation_syntax(self):
        md_text = self.gen.generate_markdown(
            query="Quantum Circuit Synthesis",
            answer="Quantum circuit synthesis combines variational ansatz optimization with parameterized gates.",
            sources=["arXiv:2401"],
            logical_score=0.95,
            verification_notes="All sources verified.",
            audit_trail=["Pass: Relational checks confirmed."],
            papers_meta={"arXiv:2401": {"title": "Quantum Synthesis", "source": "arXiv", "published_date": "2024"}},
            triplets=[("Quantum circuit", "combines", "variational ansatz", "arXiv:2401")]
        )
        self.assertIn("# 🧠 Neuro-Symbolic Executive Research Brief", md_text)
        self.assertIn("```mermaid", md_text)
        self.assertIn("arXiv:2401", md_text)
        self.assertIn("img.shields.io/badge", md_text)

    def test_portfolio_badge_html(self):
        badge_html = self.gen.get_portfolio_badge_html(
            score=1.0,
            query="CRISPR Gene Editing",
            audit_hash="ABCDEF1234567890"
        )
        self.assertIn("Research Integrity Verified", badge_html)
        self.assertIn("100% Verified", badge_html)
        self.assertIn("#ABCDEF1234567890", badge_html)

if __name__ == "__main__":
    unittest.main()

