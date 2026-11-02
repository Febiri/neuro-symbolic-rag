import unittest
from kg_engine import KnowledgeGraphEngine

class TestKnowledgeGraphEngine(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraphEngine()

    def test_scientific_triplet_extraction(self):
        text = "CRISPR-Cas9 enables targeted genome editing. Engineered SpCas9 nucleases eliminate off-target cleavage."
        triplets = self.kg.extract_triplets(text, "Doc_1")
        self.assertTrue(any(p == "enables" for s, p, o in triplets))
        self.assertTrue(any(p == "eliminate" for s, p, o in triplets))

    def test_document_and_graph_population(self):
        self.kg.add_document(
            "Paper_A",
            "Base editors combine modified Cas9 with cytidine deaminase. Cas9 cleaves pathogenic alleles.",
            {"title": "Base Editing", "source": "PubMed", "published_date": "2024"}
        )
        self.assertIn("Paper_A", self.kg.documents)
        self.assertIn("Paper_A", self.kg.documents_metadata)
        self.assertGreater(len(self.kg.graph.nodes), 0)
        self.assertGreater(len(self.kg.graph.edges), 0)

    def test_cross_paper_bridges(self):
        self.kg.add_document("Paper_1", "Cas9 ribonucleoprotein nanoparticles facilitate targeted gene insertion.")
        self.kg.add_document("Paper_2", "Engineered Cas9 nucleases eliminate off-target cleavage in hematological stem cells.")
        bridges = self.kg.get_cross_paper_bridges()
        # Should detect Cas9 or related bridging concepts
        self.assertIsInstance(bridges, list)

    def test_vis_network_payload(self):
        self.kg.add_document("Doc_Test", "Neuro-symbolic AI combines neural networks with deterministic symbolic logic.")
        data = self.kg.to_vis_network_data()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertGreater(len(data["nodes"]), 0)
        first_node = data["nodes"][0]
        self.assertIn("id", first_node)
        self.assertIn("color", first_node)
        self.assertIn("title", first_node)

    def test_hybrid_search(self):
        self.kg.add_document("Doc_AI", "Neuro-symbolic AI enforces determinism.")
        self.kg.add_document("Doc_Bio", "CRISPR gene editing cleaves double-strand DNA.")
        results = self.kg.hybrid_search("CRISPR gene editing", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "Doc_Bio")

if __name__ == "__main__":
    unittest.main()

