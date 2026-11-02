import unittest
from kg_engine import KnowledgeGraphEngine
from red_teaming import RedTeamAuditor

class TestRedTeaming(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraphEngine()
        self.kg.add_document(
            "Doc_Axiom",
            "Neuro-symbolic AI combines neural networks with symbolic reasoning. Symbolic reasoning is deterministic."
        )
        self.kg.add_document(
            "Doc_CRISPR",
            "CRISPR-Cas9 enables targeted genome editing. SpCas9 cleaves pathogenic alleles in eukaryotic cells."
        )
        self.auditor = RedTeamAuditor(self.kg)

    def test_dynamic_attacks_generated(self):
        attacks = self.auditor.generate_dynamic_attacks_for_graph()
        self.assertGreater(len(attacks), 2)
        self.assertTrue(any(a.category == "Direct Factual Contradiction" for a in attacks))

    def test_direct_determinism_contradiction_intercepted(self):
        candidate_ans = "Symbolic reasoning is non-deterministic and randomized."
        injection = "Symbolic reasoning is non-deterministic."
        report = self.auditor.audit_injection(
            candidate_answer=candidate_ans,
            injection_text=injection,
            sources=["Doc_Axiom"],
            claims=[]
        )
        self.assertTrue(report.hallucination_detected)
        self.assertEqual(report.alert_level, "CRITICAL")
        self.assertLess(report.penalized_score, 0.70)
        self.assertTrue(any("CONTRADICTION" in v for v in report.detected_violations))
        self.assertIn("Autonomous Symbolic Correction", report.remediated_answer)

    def test_unindexed_phantom_entity_intercepted(self):
        candidate_ans = "Quantum synthesis uses the Xenon-Neutrino-Hyperdrive formulated in Dr. Ghost's patent."
        injection = "Xenon-Neutrino-Hyperdrive"
        report = self.auditor.audit_injection(
            candidate_answer=candidate_ans,
            injection_text=injection,
            sources=["Doc_Axiom"],
            claims=[]
        )
        self.assertTrue(report.hallucination_detected)
        self.assertTrue(any("PHANTOM ENTITY DETECTED" in v for v in report.detected_violations))

    def test_topological_relational_disconnection_intercepted(self):
        # Doc_Axiom mentions "neural networks", Doc_CRISPR mentions "pathogenic alleles"
        # They belong to completely disconnected components in the knowledge graph
        injection = "Neural networks directly cleaves pathogenic alleles."
        report = self.auditor.audit_injection(
            candidate_answer="",
            injection_text=injection,
            sources=["Doc_Axiom", "Doc_CRISPR"],
            claims=[]
        )
        self.assertTrue(report.hallucination_detected)
        self.assertTrue(any("RELATIONAL DISCONNECTION" in v or "CONTRADICTION" in v for v in report.detected_violations))

    def test_clean_input_passes(self):
        candidate_ans = "Neuro-symbolic AI combines neural networks with symbolic reasoning."
        report = self.auditor.audit_injection(
            candidate_answer=candidate_ans,
            injection_text="",
            sources=["Doc_Axiom"],
            claims=[]
        )
        self.assertFalse(report.hallucination_detected)
        self.assertEqual(report.alert_level, "CLEAN")
        self.assertEqual(report.penalized_score, 1.0)

if __name__ == "__main__":
    unittest.main()
