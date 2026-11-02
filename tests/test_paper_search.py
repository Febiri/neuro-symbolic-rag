import unittest
from paper_search import PaperSearchEngine, ScientificPaper, sanitize_text

class TestPaperSearch(unittest.TestCase):
    def setUp(self):
        self.engine = PaperSearchEngine(timeout=8)

    def test_sanitize_text_greek_letters(self):
        raw = "Amyloid-\u03b2 peptide with \u03b1-synuclein"
        clean = sanitize_text(raw)
        self.assertIn("beta", clean)
        self.assertIn("alpha", clean)

    def test_handle_error_or_fallback(self):
        papers = self.engine._handle_error_or_fallback("CRISPR gene editing", source="PubMed", error="Simulated Timeout")
        self.assertGreater(len(papers), 0)
        p = papers[0]
        self.assertIn("CRISPR", p.title)
        self.assertEqual(p.source, "PubMed")
        self.assertFalse(p.is_live)

    def test_parse_arxiv_sample(self):
        sample_xml = """<?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2402.12345v1</id>
            <title>Neuro-Symbolic Logic in Large Language Models</title>
            <summary>We show that deterministic constraint checkers eliminate hallucinations.</summary>
            <published>2024-02-15T00:00:00Z</published>
            <author><name>Dr. Alice</name></author>
            <arxiv:primary_category term="cs.AI"/>
          </entry>
        </feed>"""
        papers = self.engine._parse_arxiv_response(sample_xml)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].id, "arXiv:2402.12345v1")
        self.assertIn("Neuro-Symbolic Logic", papers[0].title)
        self.assertEqual(papers[0].authors, ["Dr. Alice"])
        self.assertEqual(papers[0].journal_or_category, "cs.AI")

    def test_parse_pubmed_sample(self):
        sample_xml = """<?xml version="1.0" encoding="utf-8"?>
        <PubmedArticleSet>
          <PubmedArticle>
            <MedlineCitation>
              <PMID>38123456</PMID>
              <Article>
                <ArticleTitle>High-efficiency CRISPR genome editing</ArticleTitle>
                <Abstract>
                  <AbstractText>Engineered Cas9 nucleases cleave specific DNA target sequences.</AbstractText>
                </Abstract>
                <AuthorList>
                  <Author><LastName>Smith</LastName><ForeName>Jane</ForeName></Author>
                </AuthorList>
                <Journal><Title>Nature Biotechnology</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
              </Article>
            </MedlineCitation>
          </PubmedArticle>
        </PubmedArticleSet>"""
        papers = self.engine._parse_pubmed_xml(sample_xml)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].id, "PubMed:38123456")
        self.assertIn("CRISPR genome editing", papers[0].title)
        self.assertIn("Engineered Cas9 nucleases", papers[0].abstract)
        self.assertEqual(papers[0].published_date, "2024")

    def test_search_all_interface(self):
        res = self.engine.search_all("quantum", max_per_source=1)
        self.assertIn("arxiv", res)
        self.assertIn("pubmed", res)
        self.assertIn("all", res)
        self.assertGreater(len(res["all"]), 0)

if __name__ == "__main__":
    unittest.main()
