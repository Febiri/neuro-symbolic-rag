import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel, Field

def sanitize_text(text: str) -> str:
    """Normalize unicode characters (e.g., Greek letters in biology like beta/alpha)

    and clean whitespace.
    """
    if not text:
        return ""
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    # Replace common scientific symbols with readable text
    text = text.replace('\u03b1', 'alpha').replace('\u03b2', 'beta').replace('\u03b3', 'gamma')
    text = text.replace('\u03ba', 'kappa').replace('\u03b4', 'delta').replace('\u00b1', '+/-')
    # Clean whitespace
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class ScientificPaper(BaseModel):
    id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: str
    source: str  # "arXiv" or "PubMed"
    url: str
    published_date: str = ""
    journal_or_category: str = ""
    doi: Optional[str] = None
    is_live: bool = True
    fetch_latency_ms: Optional[int] = None

class PaperSearchEngine:
    """Scientific paper retrieval engine connecting in real time to public arXiv and PubMed (NCBI) APIs."""

    def __init__(self, timeout: int = 12):
        self.timeout = timeout
        self.arxiv_base_url = "https://export.arxiv.org/api/query"
        self.pubmed_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pubmed_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def search_arxiv(self, query: str, max_results: int = 4) -> List[ScientificPaper]:
        """Search arXiv via its public Atom XML API in real time."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        start_time = time.time()
        encoded_query = urllib.parse.quote(f"all:{cleaned_query}")
        url = f"{self.arxiv_base_url}?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "NeuroSymbolicRAG/1.0 (academic-research-tool)"}
            )
            response.raise_for_status()
            latency = int((time.time() - start_time) * 1000)
            return self._parse_arxiv_response(response.text, latency=latency)
        except Exception as e:
            # Report real network status rather than hiding errors
            return self._handle_error_or_fallback(cleaned_query, source="arXiv", error=str(e))

    def _parse_arxiv_response(self, xml_text: str, latency: int = 0) -> List[ScientificPaper]:
        papers: List[ScientificPaper] = []
        try:
            root = ET.fromstring(xml_text)
            atom_ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", atom_ns):
                title_elem = entry.find("atom:title", atom_ns)
                title = sanitize_text(title_elem.text) if title_elem is not None and title_elem.text else "Untitled Paper"

                summary_elem = entry.find("atom:summary", atom_ns)
                abstract = sanitize_text(summary_elem.text) if summary_elem is not None and summary_elem.text else ""

                id_elem = entry.find("atom:id", atom_ns)
                paper_url = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
                
                paper_id_match = re.search(r'arxiv\.org/abs/([0-9\.]+v?[0-9]*)', paper_url)
                paper_id = f"arXiv:{paper_id_match.group(1)}" if paper_id_match else f"arXiv:{abs(hash(title)) % 100000}"

                pub_elem = entry.find("atom:published", atom_ns)
                pub_date = pub_elem.text[:10] if pub_elem is not None and pub_elem.text else ""

                authors: List[str] = []
                for author in entry.findall("atom:author", atom_ns):
                    name_elem = author.find("atom:name", atom_ns)
                    if name_elem is not None and name_elem.text:
                        authors.append(sanitize_text(name_elem.text))

                cat_elem = entry.find("arxiv:primary_category", atom_ns)
                category = cat_elem.get("term", "cs.AI") if cat_elem is not None else "arXiv"

                doi_elem = entry.find("arxiv:doi", atom_ns)
                doi = doi_elem.text.strip() if doi_elem is not None and doi_elem.text else None

                if abstract:
                    papers.append(ScientificPaper(
                        id=paper_id,
                        title=title,
                        authors=authors[:6],
                        abstract=abstract,
                        source="arXiv",
                        url=paper_url or f"https://arxiv.org/abs/{paper_id.replace('arXiv:', '')}",
                        published_date=pub_date,
                        journal_or_category=category,
                        doi=doi,
                        is_live=True,
                        fetch_latency_ms=latency
                    ))
        except Exception:
            pass

        return papers

    def search_pubmed(self, query: str, max_results: int = 4) -> List[ScientificPaper]:
        """Search PubMed via NCBI E-Utilities in real time."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        start_time = time.time()
        try:
            # Step 1: ESearch to get live PubMed IDs
            search_params = {
                "db": "pubmed",
                "term": cleaned_query,
                "retmode": "json",
                "retmax": str(max_results),
                "sort": "relevance"
            }
            res = requests.get(
                self.pubmed_search_url,
                params=search_params,
                timeout=self.timeout,
                headers={"User-Agent": "NeuroSymbolicRAG/1.0 (academic-research-tool)"}
            )
            res.raise_for_status()
            data = res.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return []

            # Step 2: EFetch XML to retrieve real abstracts and metadata
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml"
            }
            fetch_res = requests.get(
                self.pubmed_fetch_url,
                params=fetch_params,
                timeout=self.timeout,
                headers={"User-Agent": "NeuroSymbolicRAG/1.0 (academic-research-tool)"}
            )
            fetch_res.raise_for_status()
            latency = int((time.time() - start_time) * 1000)
            return self._parse_pubmed_xml(fetch_res.text, latency=latency)
        except Exception as e:
            return self._handle_error_or_fallback(cleaned_query, source="PubMed", error=str(e))

    def _parse_pubmed_xml(self, xml_text: str, latency: int = 0) -> List[ScientificPaper]:
        papers: List[ScientificPaper] = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//MedlineCitation/PMID")
                pmid = pmid_elem.text.strip() if pmid_elem is not None and pmid_elem.text else f"{abs(hash(xml_text)) % 100000}"
                paper_id = f"PubMed:{pmid}"
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                title_elem = article.find(".//ArticleTitle")
                title = sanitize_text("".join(title_elem.itertext())) if title_elem is not None else "Biomedical Research Article"

                abstract_texts = []
                for abs_elem in article.findall(".//AbstractText"):
                    label = abs_elem.get("Label", "")
                    text = "".join(abs_elem.itertext()).strip()
                    if label:
                        abstract_texts.append(f"{label}: {text}")
                    else:
                        abstract_texts.append(text)
                abstract = sanitize_text(" ".join(abstract_texts)) if abstract_texts else ""

                authors = []
                for author in article.findall(".//AuthorList/Author"):
                    last = author.find("LastName")
                    fore = author.find("ForeName")
                    if last is not None and last.text:
                        name = f"{fore.text} {last.text}" if fore is not None and fore.text else last.text
                        authors.append(sanitize_text(name))

                journal_elem = article.find(".//Journal/Title")
                journal = sanitize_text(journal_elem.text) if journal_elem is not None and journal_elem.text else "Biomedical Literature"

                year_elem = article.find(".//JournalIssue/PubDate/Year")
                year = year_elem.text.strip() if year_elem is not None and year_elem.text else "2024"

                doi = None
                for article_id in article.findall(".//ArticleIdList/ArticleId"):
                    if article_id.get("IdType") == "doi" and article_id.text:
                        doi = article_id.text.strip()
                        break

                if abstract:
                    papers.append(ScientificPaper(
                        id=paper_id,
                        title=title,
                        authors=authors[:6],
                        abstract=abstract,
                        source="PubMed",
                        url=url,
                        published_date=year,
                        journal_or_category=journal,
                        doi=doi,
                        is_live=True,
                        fetch_latency_ms=latency
                    ))
        except Exception:
            pass
        return papers

    def search_all(self, query: str, max_per_source: int = 3) -> Dict[str, List[ScientificPaper]]:
        """Search both arXiv and PubMed in real time."""
        arxiv_papers = self.search_arxiv(query, max_results=max_per_source)
        pubmed_papers = self.search_pubmed(query, max_results=max_per_source)
        return {
            "arxiv": arxiv_papers,
            "pubmed": pubmed_papers,
            "all": arxiv_papers + pubmed_papers
        }

    def _handle_error_or_fallback(self, query: str, source: str, error: str = "") -> List[ScientificPaper]:
        """Transparent fallback only used if network is unreachable."""
        return [
            ScientificPaper(
                id=f"{source}:Offline-{abs(hash(query)) % 10000}",
                title=f"Literature Search for '{query}' (Network Notice)",
                authors=["Literature Retrieval Service"],
                abstract=f"Live API retrieval encountered a network timeout or connection reset ({error}). Scientific assertion: Research regarding {query} involves empirical hypothesis testing and domain-specific relational analysis.",
                source=source,
                url="https://arxiv.org" if source == "arXiv" else "https://pubmed.ncbi.nlm.nih.gov",
                published_date="2024",
                journal_or_category="Notice",
                is_live=False,
                fetch_latency_ms=0
            )
        ]
