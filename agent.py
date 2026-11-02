import re
import json
from typing import List, Dict, Any, Tuple, Optional
import requests
import networkx as nx
from pydantic import BaseModel, Field
from red_teaming import RedTeamAuditor, RedTeamAuditReport

class AgentTrace(BaseModel):
    step: int
    agent_name: str
    action: str
    details: Dict[str, Any]
    status: str = "COMPLETED"

class AgenticWorkflowResult(BaseModel):
    query: str
    answer: str
    sources: List[str]
    logical_consistency_score: float
    verification_notes: str
    claims: List[Dict[str, Any]]
    retrieved_triplets: List[Tuple[str, str, str]]
    traces: List[AgentTrace]
    self_corrected: bool = False
    red_team_report: Optional[Dict[str, Any]] = None

class SemanticRetrievalAgent:
    """Agent 1: Semantic Retrieval & Knowledge Graph Traversal Agent."""

    def __init__(self, kg_engine):
        self.kg = kg_engine

    def execute(self, query: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]], AgentTrace]:
        search_results = self.kg.hybrid_search(query, top_k=4)
        sources = [res["doc_id"] for res in search_results]

        query_words = {w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower())}
        relevant_triplets: List[Tuple[str, str, str]] = []

        for u, v, data in self.kg.graph.edges(data=True):
            src_doc = data.get("source", "")
            relation = data.get("relation", "")
            # Prioritize edges belonging to retrieved documents or mentioning query words
            if src_doc in sources or any(w in str(u).lower() or w in str(v).lower() for w in query_words):
                relevant_triplets.append((u, relation, v))

        if len(relevant_triplets) < 2 and len(self.kg.graph.edges) > 0:
            for u, v, data in list(self.kg.graph.edges(data=True))[:6]:
                triple = (u, data.get("relation", ""), v)
                if triple not in relevant_triplets:
                    relevant_triplets.append(triple)

        trace = AgentTrace(
            step=1,
            agent_name="Agent 1: Semantic Retrieval & KG Traversal",
            action="Executed hybrid vector/graph retrieval on active scientific corpus",
            details={
                "retrieved_documents": sources,
                "triplets_found": len(relevant_triplets),
                "triplets_sample": [f"({s}) ──[{p}]──> ({o})" for s, p, o in relevant_triplets[:4]]
            }
        )
        return search_results, relevant_triplets, trace

class ReasoningSynthesisAgent:
    """Agent 2: Multi-Hop Reasoning & Synthesis Agent supporting both real-time Live LLM

    (Gemini / OpenAI) synthesis and dynamic extractive-abstractive graph synthesis.
    """

    def __init__(self, llm_api_key: Optional[str] = None, llm_provider: str = "gemini"):
        self.llm_api_key = llm_api_key
        self.llm_provider = llm_provider.lower()

    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call Gemini REST API directly if key is provided."""
        if not self.llm_api_key:
            return None
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.llm_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
        }
        try:
            res = requests.post(url, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass
        return None

    def execute(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        triplets: List[Tuple[str, str, str]],
        corrective_instructions: str = "",
        adversarial_injection: Optional[str] = None
    ) -> Tuple[str, List[Dict[str, Any]], AgentTrace]:
        claims: List[Dict[str, Any]] = []

        if not retrieved_docs and not triplets:
            return (
                f"No verified documentation or Knowledge Graph relations currently indexed for query: '{query}'. Ingest papers to synthesize answers.",
                claims,
                AgentTrace(
                    step=2,
                    agent_name="Agent 2: Reasoning & Multi-Hop Synthesis",
                    action="Evidence retrieval empty",
                    details={"reason": "Knowledge store is empty"},
                    status="FAILED"
                )
            )

        # -----------------------------------------------------------------------------------------
        # Try Live LLM generation first if API key is present
        # -----------------------------------------------------------------------------------------
        llm_response = None
        if self.llm_api_key:
            context_snippets = "\n".join([f"[{d['doc_id']}] {d['content'][:400]}" for d in retrieved_docs[:3]])
            triplets_str = ", ".join([f"({s}) -[{p}]-> ({o})" for s, p, o in triplets[:6]])
            llm_prompt = (
                f"You are a factual scientific reasoning agent. Answer the question using ONLY the provided evidence.\n"
                f"Question: {query}\n"
                f"Ingested Evidence:\n{context_snippets}\n"
                f"Knowledge Graph Relations:\n{triplets_str}\n\n"
                f"Requirements: Write a concise factual synthesis citing the source IDs like [Doc_ID]."
            )
            llm_response = self._call_gemini_api(llm_prompt)

        # -----------------------------------------------------------------------------------------
        # Dynamic Extractive-Abstractive Synthesis (No canned boilerplate strings)
        # -----------------------------------------------------------------------------------------
        if llm_response:
            synthesized_answer = llm_response
            synthesis_mode = "Live LLM (Gemini) Synthesis"
        else:
            synthesis_mode = "Dynamic Graph-Grounded Synthesis"
            # Extract the most relevant sentences from the actual papers matching the query terms
            query_words = {w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower())}
            candidate_sentences = []
            
            for doc in retrieved_docs:
                doc_id = doc["doc_id"]
                content = doc.get("content", "")
                sentences = re.split(r'(?<=[.!?])\s+', content)
                for sent in sentences:
                    sent_clean = sent.strip()
                    if len(sent_clean) > 25:
                        sent_words = set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', sent_clean.lower()))
                        overlap = len(query_words & sent_words)
                        if overlap > 0:
                            candidate_sentences.append((overlap, sent_clean, doc_id))

            candidate_sentences.sort(key=lambda x: x[0], reverse=True)
            selected_passages = []
            seen_text = set()

            for _, sent, doc_id in candidate_sentences[:3]:
                if sent not in seen_text:
                    selected_passages.append(f"{sent} [{doc_id}]")
                    seen_text.add(sent)

            if not selected_passages and retrieved_docs:
                first_doc = retrieved_docs[0]
                first_sent = re.split(r'(?<=[.!?])\s+', first_doc["content"])[0]
                selected_passages.append(f"{first_sent} [{first_doc['doc_id']}]")

            # Stitch extracted facts into a coherent synthesis paragraph
            passage_body = " ".join(selected_passages)
            
            # Incorporate top knowledge graph relational paths
            if triplets:
                top_facts = [f"{s} {p} {o}" for s, p, o in triplets[:3]]
                relations_summary = f" Relational knowledge graph analysis confirms that {', and '.join(top_facts)}."
            else:
                relations_summary = ""

            synthesized_answer = f"{passage_body}{relations_summary}"

        # Register claims for symbolic constraint checking
        for u, rel, v in triplets[:5]:
            claims.append({
                "subject": u,
                "predicate": rel,
                "object": v,
                "statement": f"{u} {rel} {v}."
            })

        # Inject adversarial claim if Red-Teaming mode is active
        if adversarial_injection:
            synthesized_answer = f"{synthesized_answer} However, {adversarial_injection.strip()}"
            claims.append({
                "subject": "Injected_Assertion",
                "predicate": "claims",
                "object": adversarial_injection,
                "statement": adversarial_injection
            })

        if corrective_instructions:
            synthesized_answer += f" [Self-Correction: {corrective_instructions}]"

        trace = AgentTrace(
            step=2,
            agent_name="Agent 2: Reasoning & Multi-Hop Synthesis",
            action=f"Synthesized evidence using {synthesis_mode}",
            details={
                "synthesis_mode": synthesis_mode,
                "claims_audited": len(claims),
                "adversarial_injection_active": bool(adversarial_injection),
                "applied_corrections": bool(corrective_instructions)
            }
        )
        return synthesized_answer, claims, trace

class SymbolicLogicVerifier:
    """Agent 3: Deterministic Programmatic Constraint Checker & Universal Auditor."""

    def __init__(self, kg_engine):
        self.kg = kg_engine
        self.red_team_auditor = RedTeamAuditor(kg_engine)

    def verify(
        self,
        answer: str,
        sources: List[str],
        claims: List[Dict[str, Any]],
        adversarial_injection: Optional[str] = None
    ) -> Tuple[float, str, List[str], AgentTrace, Optional[RedTeamAuditReport]]:
        penalties = 0.0
        audit_notes: List[str] = []

        # Constraint 1: Source Validation
        missing_sources = [s for s in sources if s not in self.kg.documents]
        if missing_sources:
            penalties += 0.40
            audit_notes.append(f"Failed Source Registry Audit: Unknown source(s) {missing_sources}.")
        else:
            audit_notes.append("Pass: All cited sources verified in active scientific registry.")

        # Constraint 2: Non-empty response
        if not answer.strip():
            penalties += 1.0
            audit_notes.append("Failed Completeness Constraint: Empty synthesis.")

        # Constraint 3: Graph Edge Existence & Relational Integrity
        unverified_claims = 0
        for claim in claims:
            subj = claim.get("subject")
            obj = claim.get("object")
            if subj and obj:
                if self.kg.graph.has_edge(subj, obj):
                    pass
                elif self.kg.graph.has_node(subj) and self.kg.graph.has_node(obj) and nx.has_path(self.kg.graph, subj, obj):
                    audit_notes.append(f"Multi-hop Path Verified: '{subj}' -> '{obj}'.")
                elif subj == "Injected_Assertion":
                    unverified_claims += 2
                else:
                    unverified_claims += 1

        if unverified_claims > 0:
            penalties += min(0.35, unverified_claims * 0.12)
            audit_notes.append(f"Relational Verification Notice: {unverified_claims} assertion(s) lack direct edges in the Knowledge Graph.")
        else:
            audit_notes.append("Pass: 100% of atomic claims verified against symbolic graph edges.")

        # Constraint 4: Universal Red-Team Auditor evaluation (Domain-agnostic checks)
        red_team_report: Optional[RedTeamAuditReport] = None
        if adversarial_injection and adversarial_injection.strip():
            red_team_report = self.red_team_auditor.audit_injection(
                candidate_answer=answer,
                injection_text=adversarial_injection.strip(),
                sources=sources,
                claims=claims
            )
            if red_team_report.hallucination_detected:
                penalties += 0.50
                for viol in red_team_report.detected_violations:
                    audit_notes.append(f"RED-TEAM AUDIT: {viol}")

        final_score = max(0.0, min(1.0, 1.0 - penalties))
        summary_notes = " | ".join(audit_notes)

        trace = AgentTrace(
            step=3,
            agent_name="Agent 3: Symbolic Logic Verifier",
            action="Executed deterministic constraint checks & logical auditing",
            details={
                "score": round(final_score, 2),
                "audit_notes": audit_notes,
                "verified": final_score >= 0.70,
                "red_team_flag": red_team_report.hallucination_detected if red_team_report else False
            }
        )
        return final_score, summary_notes, audit_notes, trace, red_team_report
