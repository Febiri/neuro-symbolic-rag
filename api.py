import base64
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from engine import NeuroSymbolicEngine, VerifiedResponse
from paper_search import ScientificPaper
from red_teaming import RedTeamAttack
from config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI Orchestration Layer for Neuro-Symbolic Agentic RAG Engine with arXiv/PubMed search, Red-Teaming, and Interactive Graph Explorer"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# Global engine singleton
engine = NeuroSymbolicEngine()

# Seed default documents
engine.add_document(
    "Doc_Alpha",
    "Neuro-symbolic AI combines neural networks with symbolic reasoning. Symbolic reasoning is deterministic.",
    {"title": "Foundations of Neuro-Symbolic AI", "source": "arXiv", "published_date": "2024", "authors": ["A. Garcez"]}
)
engine.add_document(
    "Doc_Beta",
    "Agentic RAG workflows use autonomous tool execution and self-correction loops. Verification layers enforce logical consistency.",
    {"title": "Agentic Verification Frameworks", "source": "arXiv", "published_date": "2024", "authors": ["L. Valiant"]}
)

class IngestRequest(BaseModel):
    doc_id: str = Field(..., json_schema_extra={"example": "Research_Paper_01"})
    content: str = Field(..., json_schema_extra={"example": "Neuro-symbolic architectures eliminate hallucinations through deterministic constraint checkers."})
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is neuro-symbolic AI?"})
    red_team_mode: bool = Field(default=False, description="Enable hallucination red-teaming mode")
    adversarial_injection: Optional[str] = Field(default=None, json_schema_extra={"example": "Symbolic reasoning is non-deterministic."})

class VerifyRequest(BaseModel):
    answer: str = Field(..., json_schema_extra={"example": "Symbolic reasoning is deterministic."})
    sources: List[str] = Field(..., json_schema_extra={"example": ["Doc_Alpha"]})
    adversarial_injection: Optional[str] = None

class PaperSearchRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "CRISPR gene editing"})
    source: str = Field(default="all", json_schema_extra={"example": "all"}, description="arxiv, pubmed, or all")
    max_results: int = Field(default=4, ge=1, le=20)

class IngestPapersRequest(BaseModel):
    papers: List[ScientificPaper]

class RedTeamAuditRequest(BaseModel):
    candidate_answer: str
    adversarial_injection: str
    sources: Optional[List[str]] = Field(default_factory=list)

class ExportReportRequest(BaseModel):
    query: str
    format: str = Field(default="pdf", json_schema_extra={"example": "pdf"}, description="'pdf' or 'markdown'")
    response_payload: Optional[Dict[str, Any]] = None

@app.get("/")
def get_root():
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documents_indexed": len(engine.documents),
        "graph_nodes": len(engine.graph.nodes),
        "graph_edges": len(engine.graph.edges),
        "cross_paper_bridges": len(engine.kg_engine.get_cross_paper_bridges())
    }

@app.post("/ingest")
def ingest_document(req: IngestRequest):
    if not req.doc_id.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="Document ID and content must not be empty.")
    
    engine.add_document(req.doc_id, req.content, metadata=req.metadata)
    return {
        "status": "success",
        "message": f"Successfully ingested {req.doc_id}",
        "total_documents": len(engine.documents),
        "graph_nodes": len(engine.graph.nodes),
        "graph_edges": len(engine.graph.edges)
    }

@app.post("/search_papers")
def search_papers(req: PaperSearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    papers = engine.search_papers(req.query, source=req.source, max_results=req.max_results)
    return {
        "query": req.query,
        "source": req.source,
        "results_count": len(papers),
        "papers": [p.model_dump() for p in papers]
    }

@app.post("/ingest_papers")
def ingest_papers(req: IngestPapersRequest):
    if not req.papers:
        raise HTTPException(status_code=400, detail="Paper list cannot be empty.")
    summary = engine.ingest_papers_batch(req.papers)
    return {
        "status": "success",
        "message": f"Ingested {len(req.papers)} papers into Knowledge Graph",
        "summary": summary
    }

@app.post("/query", response_model=VerifiedResponse)
def query_pipeline(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return engine.query(
        req.query,
        red_team_mode=req.red_team_mode,
        adversarial_injection=req.adversarial_injection
    )

@app.post("/verify")
def verify_answer(req: VerifyRequest):
    result = engine.symbolic_verification(
        req.answer,
        req.sources,
        adversarial_injection=req.adversarial_injection
    )
    return result

@app.get("/red_team_presets")
def get_red_team_presets():
    attacks = engine.get_dynamic_red_team_attacks()
    return {
        "presets": [a.model_dump() for a in attacks]
    }

@app.post("/red_team_audit")
def red_team_audit(req: RedTeamAuditRequest):
    report = engine.verifier_agent.red_team_auditor.audit_injection(
        candidate_answer=req.candidate_answer,
        injection_text=req.adversarial_injection,
        sources=req.sources or list(engine.documents.keys()),
        claims=[]
    )
    return report.model_dump()

@app.get("/graph")
def get_graph_data():
    nodes = [{"id": n, "label": str(n), "sources": engine.graph.nodes[n].get("sources", [])} for n in engine.graph.nodes]
    edges = [
        {
            "source": u,
            "target": v,
            "relation": data.get("relation", ""),
            "document": data.get("source", "")
        }
        for u, v, data in engine.graph.edges(data=True)
    ]
    bridges = engine.kg_engine.get_cross_paper_bridges()
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "cross_paper_bridges": bridges
    }

@app.get("/graph/interactive", response_class=HTMLResponse)
def get_interactive_graph_canvas(height: int = 700):
    html = engine.get_interactive_graph_html(height=height)
    return HTMLResponse(content=html)

@app.post("/export_report")
def export_report(req: ExportReportRequest):
    # Query engine if payload not provided
    if req.response_payload:
        resp = VerifiedResponse(**req.response_payload)
    else:
        resp = engine.query(req.query)

    if req.format.lower() == "markdown":
        md_text = engine.export_markdown(resp, req.query)
        return {"format": "markdown", "content": md_text}
    else:
        pdf_bytes = engine.export_pdf(resp, req.query)
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        return {
            "format": "pdf",
            "filename": f"Research_Brief_{req.query[:20].replace(' ', '_')}.pdf",
            "base64_content": b64_pdf
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
