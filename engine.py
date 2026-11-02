import os
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from kg_engine import KnowledgeGraphEngine
from agent import (
    SemanticRetrievalAgent,
    ReasoningSynthesisAgent,
    SymbolicLogicVerifier,
    AgentTrace
)
from paper_search import PaperSearchEngine, ScientificPaper
from red_teaming import RedTeamAuditor, RedTeamAuditReport, RedTeamAttack
from report_generator import ResearchReportGenerator
from interactive_graph import generate_interactive_graph_html

class VerifiedResponse(BaseModel):
    answer: str = Field(description="Synthesized factual answer")
    sources: List[str] = Field(description="Cited source documents")
    logical_consistency_score: float = Field(description="Score between 0.0 and 1.0 verified by symbolic checks")
    verification_notes: str = Field(description="Notes from symbolic logic verifier")
    audit_trail: Optional[List[str]] = Field(default_factory=list, description="Granular constraint audit findings")
    agent_traces: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Step-by-step agent execution traces")
    retrieved_triplets: Optional[List[Any]] = Field(default_factory=list, description="Knowledge triplets used in synthesis")
    red_team_report: Optional[Dict[str, Any]] = Field(default=None, description="Red-team adversarial audit findings if tested")

class NeuroSymbolicEngine:
    """Core Neuro-Symbolic Agentic RAG Engine with real-time arXiv & PubMed retrieval,

    domain-agnostic symbolic constraint auditing, interactive knowledge graph explorer,
    and hallucination red-teaming.
    """

    def __init__(self, llm_api_key: Optional[str] = None):
        self.kg_engine = KnowledgeGraphEngine()
        self.retrieval_agent = SemanticRetrievalAgent(self.kg_engine)
        self.reasoning_agent = ReasoningSynthesisAgent(llm_api_key=llm_api_key)
        self.verifier_agent = SymbolicLogicVerifier(self.kg_engine)
        self.paper_search = PaperSearchEngine()
        self.report_generator = ResearchReportGenerator()
        self._llm_api_key = llm_api_key

    def set_llm_api_key(self, api_key: str):
        """Configure live LLM API key dynamically at runtime."""
        self._llm_api_key = api_key
        self.reasoning_agent.llm_api_key = api_key

    @property
    def documents(self) -> Dict[str, str]:
        return self.kg_engine.documents

    @property
    def documents_metadata(self) -> Dict[str, Dict[str, Any]]:
        return self.kg_engine.documents_metadata

    @property
    def graph(self) -> nx.DiGraph:
        return self.kg_engine.graph

    def clear(self):
        """Reset the graph and document stores."""
        self.kg_engine.clear()

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Ingest document and populate Knowledge Graph with extracted Subject-Predicate-Object triplets."""
        self.kg_engine.add_document(doc_id, text, metadata=metadata)

    def search_papers(self, query: str, source: str = "all", max_results: int = 4) -> List[ScientificPaper]:
        """Search live scientific papers from arXiv and/or PubMed in real time."""
        if source == "arxiv":
            return self.paper_search.search_arxiv(query, max_results=max_results)
        elif source == "pubmed":
            return self.paper_search.search_pubmed(query, max_results=max_results)
        else:
            res = self.paper_search.search_all(query, max_per_source=max(1, max_results // 2))
            return res["all"]

    def ingest_paper(self, paper: ScientificPaper) -> int:
        """Ingest a ScientificPaper into the knowledge graph, returning extracted triplet count."""
        meta = {
            "title": paper.title,
            "source": paper.source,
            "url": paper.url,
            "authors": paper.authors,
            "published_date": paper.published_date,
            "category": paper.journal_or_category,
            "doi": paper.doi,
            "is_live": paper.is_live
        }
        self.kg_engine.add_document(paper.id, paper.abstract, metadata=meta)
        triplets = self.kg_engine.extract_triplets(paper.abstract, paper.id)
        return len(triplets)

    def ingest_papers_batch(self, papers: List[ScientificPaper]) -> Dict[str, Any]:
        """Ingest multiple papers and return real-time summary statistics."""
        total_triplets = 0
        ingested_ids = []
        for p in papers:
            count = self.ingest_paper(p)
            total_triplets += count
            ingested_ids.append(p.id)

        return {
            "ingested_count": len(papers),
            "paper_ids": ingested_ids,
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "cross_paper_bridges": len(self.kg_engine.get_cross_paper_bridges())
        }

    def search_and_ingest_live(self, query: str, source: str = "all", max_papers: int = 4) -> Dict[str, Any]:
        """Execute live search on arXiv/PubMed and ingest papers directly in real time."""
        papers = self.search_papers(query, source=source, max_results=max_papers)
        summary = self.ingest_papers_batch(papers)
        summary["query"] = query
        summary["papers"] = papers
        return summary

    def seed_sample_topic(self, topic: str = "crispr"):
        """Convenience method to ingest baseline literature for test fixtures or demos."""
        if "crispr" in topic.lower():
            self.add_document(
                "Doc_CRISPR_01",
                "CRISPR-Cas9 enables targeted genome editing. Engineered SpCas9 nucleases eliminate off-target cleavage.",
                {"title": "High-fidelity CRISPR Editing", "source": "PubMed", "published_date": "2024", "authors": ["J. Doudna"]}
            )
            self.add_document(
                "Doc_CRISPR_02",
                "SpCas9 cleaves pathogenic alleles in eukaryotic cells. Guide RNA targets specific DNA loci.",
                {"title": "Therapeutic Cas9 Mechanisms", "source": "PubMed", "published_date": "2024", "authors": ["F. Zhang"]}
            )
        else:
            self.add_document(
                "Doc_Alpha",
                "Neuro-symbolic AI combines neural networks with symbolic reasoning. Symbolic reasoning is deterministic.",
                {"title": "Foundations of Neuro-Symbolic AI", "source": "arXiv", "published_date": "2024", "authors": ["A. Garcez"]}
            )
            self.add_document(
                "Doc_Beta",
                "Agentic RAG workflows use autonomous tool execution and self-correction loops. Verification layers enforce logical consistency.",
                {"title": "Agentic Verification Frameworks", "source": "arXiv", "published_date": "2024", "authors": ["L. Valiant"]}
            )

    def get_dynamic_red_team_attacks(self) -> List[RedTeamAttack]:
        """Generate adversarial attacks tailored dynamically to whatever is in the active graph."""
        return self.verifier_agent.red_team_auditor.generate_dynamic_attacks_for_graph()

    def symbolic_verification(
        self,
        answer: str,
        sources: List[str],
        claims: Optional[List[Dict[str, Any]]] = None,
        adversarial_injection: Optional[str] = None
    ) -> Dict[str, Any]:
        claims = claims or []
        score, notes, audit_trail, trace, rt_report = self.verifier_agent.verify(
            answer, sources, claims, adversarial_injection=adversarial_injection
        )
        return {
            "score": score,
            "notes": notes,
            "audit_trail": audit_trail,
            "trace": trace.model_dump(),
            "red_team_report": rt_report.model_dump() if rt_report else None
        }

    def query(
        self,
        question: str,
        red_team_mode: bool = False,
        adversarial_injection: Optional[str] = None
    ) -> VerifiedResponse:
        """Execute multi-agent neuro-symbolic workflow in real time."""
        all_traces: List[Dict[str, Any]] = []

        # Step 1: Semantic Retrieval Agent
        retrieved_docs, triplets, trace1 = self.retrieval_agent.execute(question)
        all_traces.append(trace1.model_dump())

        sources = [doc["doc_id"] for doc in retrieved_docs]
        if not sources and self.documents:
            sources = list(self.documents.keys())[:3]

        # Step 2: Multi-Hop Reasoning & Synthesis Agent
        synthesized_answer, claims, trace2 = self.reasoning_agent.execute(
            question,
            retrieved_docs,
            triplets,
            adversarial_injection=adversarial_injection if red_team_mode else None
        )
        all_traces.append(trace2.model_dump())

        # Step 3: Symbolic Logic Verifier
        score, notes, audit_trail, trace3, rt_report = self.verifier_agent.verify(
            synthesized_answer,
            sources,
            claims,
            adversarial_injection=adversarial_injection if red_team_mode else None
        )
        all_traces.append(trace3.model_dump())

        # Self-correction loop: if ungrounded claims detected and not explicitly in red-team demo mode
        if score < 0.70 and not red_team_mode:
            correction_instruction = "Pruned unverified claims to enforce strict symbolic grounding."
            synthesized_answer, claims, trace2_corr = self.reasoning_agent.execute(
                question, retrieved_docs, triplets, corrective_instructions=correction_instruction
            )
            score, notes, audit_trail, trace3_corr, _ = self.verifier_agent.verify(
                synthesized_answer, sources, claims
            )
            all_traces.append({
                "step": 4,
                "agent_name": "Feedback Loop: Self-Correction",
                "action": "Applied symbolic constraint feedback and re-audited output",
                "details": {"new_score": score, "notes": notes},
                "status": "SELF_CORRECTED"
            })

        return VerifiedResponse(
            answer=synthesized_answer,
            sources=sources,
            logical_consistency_score=score,
            verification_notes=notes,
            audit_trail=audit_trail,
            agent_traces=all_traces,
            retrieved_triplets=triplets,
            red_team_report=rt_report.model_dump() if rt_report else None
        )

    def get_interactive_graph_html(self, height: int = 650) -> str:
        return generate_interactive_graph_html(self.kg_engine, height=height)

    def export_pdf(self, response: VerifiedResponse, query: str) -> bytes:
        return self.report_generator.generate_pdf(
            query=query,
            answer=response.answer,
            sources=response.sources,
            logical_score=response.logical_consistency_score,
            verification_notes=response.verification_notes,
            audit_trail=response.audit_trail or [],
            papers_meta=self.documents_metadata,
            triplets=response.retrieved_triplets or [],
            red_team_report=response.red_team_report
        )

    def export_markdown(self, response: VerifiedResponse, query: str) -> str:
        return self.report_generator.generate_markdown(
            query=query,
            answer=response.answer,
            sources=response.sources,
            logical_score=response.logical_consistency_score,
            verification_notes=response.verification_notes,
            audit_trail=response.audit_trail or [],
            papers_meta=self.documents_metadata,
            triplets=response.retrieved_triplets or [],
            red_team_report=response.red_team_report
        )
