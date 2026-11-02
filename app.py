import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
import streamlit.components.v1 as components

from engine import NeuroSymbolicEngine, VerifiedResponse
from red_teaming import RedTeamAttack
from paper_search import ScientificPaper

st.set_page_config(
    page_title="Neuro-Symbolic Agentic RAG Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main {
        background-color: #0B0F19;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(0, 210, 255, 0.25);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        margin-right: 6px;
    }
    
    .live-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid #10B981;
        margin-left: 6px;
    }
    
    .arxiv-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .pubmed-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .red-team-alert {
        background: linear-gradient(135deg, rgba(69, 10, 10, 0.95), rgba(127, 29, 29, 0.85));
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);
    }
    
    .strike-claim {
        text-decoration: line-through;
        color: #FCA5A5;
        background: rgba(0, 0, 0, 0.4);
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    .clean-alert {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9), rgba(20, 83, 45, 0.8));
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State without hardcoded seed documents
if "engine" not in st.session_state:
    st.session_state.engine = NeuroSymbolicEngine()

if "last_query_result" not in st.session_state:
    st.session_state.last_query_result = None

if "last_query_text" not in st.session_state:
    st.session_state.last_query_text = ""

if "search_results" not in st.session_state:
    st.session_state.search_results = []

engine: NeuroSymbolicEngine = st.session_state.engine

# Sidebar: Engine Control Center
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.title("Control Center")
    st.caption("Real-Time Neuro-Symbolic Agentic Engine")

    # API Key Configuration
    with st.expander("🔑 Live Neural LLM (Optional)"):
        st.caption("Provide an API key to switch from dynamic graph synthesis to live neural LLM reasoning:")
        api_key_input = st.text_input("Gemini API Key", type="password", value=engine._llm_api_key or "")
        if st.button("Apply API Key", use_container_width=True):
            engine.set_llm_api_key(api_key_input.strip())
            st.success("API Key updated for live neural synthesis!")

    st.divider()
    st.subheader("📊 Live Telemetry")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Indexed Docs", len(engine.documents))
    with col_s2:
        st.metric("Graph Nodes", len(engine.graph.nodes))
    
    col_s3, col_s4 = st.columns(2)
    with col_s3:
        st.metric("Graph Edges", len(engine.graph.edges))
    with col_s4:
        bridges = engine.kg_engine.get_cross_paper_bridges()
        st.metric("Cross Bridges", len(bridges))

    st.divider()
    st.subheader("🌐 Live One-Click Research Topics")
    st.caption("Triggers REAL-TIME API calls to arXiv / PubMed:")
    
    if st.button("🧬 Live Fetch: CRISPR Cas9", use_container_width=True):
        with st.spinner("Connecting to live arXiv & PubMed APIs..."):
            summary = engine.search_and_ingest_live("CRISPR Cas9 genome editing", max_papers=3)
            st.session_state.last_query_text = "What mechanisms do SpCas9 nucleases use to cleave target DNA?"
            st.session_state.last_query_result = None
            st.success(f"Live ingested {summary['ingested_count']} papers! ({summary['total_nodes']} nodes created)")
            st.rerun()

    if st.button("⚛️ Live Fetch: Quantum Circuit LLM", use_container_width=True):
        with st.spinner("Connecting to live arXiv API..."):
            summary = engine.search_and_ingest_live("quantum circuit synthesis language models", max_papers=3)
            st.session_state.last_query_text = "How are quantum circuit transformations verified?"
            st.session_state.last_query_result = None
            st.success(f"Live ingested {summary['ingested_count']} papers! ({summary['total_nodes']} nodes created)")
            st.rerun()

    if st.button("🔬 Live Fetch: Alzheimer Immunotherapy", use_container_width=True):
        with st.spinner("Connecting to live PubMed API..."):
            summary = engine.search_and_ingest_live("Alzheimer amyloid beta immunotherapy", max_papers=3)
            st.session_state.last_query_text = "How do monoclonal antibodies target amyloid-beta in Alzheimer's?"
            st.session_state.last_query_result = None
            st.success(f"Live ingested {summary['ingested_count']} papers! ({summary['total_nodes']} nodes created)")
            st.rerun()

    st.divider()
    if st.button("🧹 Clear All Knowledge", use_container_width=True):
        engine.clear()
        st.session_state.search_results = []
        st.session_state.last_query_result = None
        st.session_state.last_query_text = ""
        st.success("Knowledge Graph cleared.")
        st.rerun()

    with st.expander("📥 Manual Note Ingestion"):
        doc_id = st.text_input("Document Identifier", value="Lab_Observation_01")
        doc_text = st.text_area(
            "Document Content",
            value="Neuro-symbolic architectures eliminate hallucinations through deterministic constraint checkers.",
            height=90
        )
        if st.button("Ingest Custom Note", use_container_width=True):
            if doc_id.strip() and doc_text.strip():
                engine.add_document(doc_id.strip(), doc_text.strip())
                st.success(f"Ingested '{doc_id}' into Knowledge Graph!")
                st.rerun()

# Top Hero Banner
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; color: #FFFFFF; font-size: 2.1rem; font-weight: 800;">
                🧠 Neuro-Symbolic Agentic RAG
            </h1>
            <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 1.05rem;">
                Real-Time Literature Retrieval &bull; Dynamic Knowledge Graph &bull; Generalized Symbolic Verifier
            </p>
        </div>
        <div style="margin-top: 10px;">
            <span class="source-badge">⚡ Real-Time arXiv API</span>
            <span class="source-badge">⚡ Real-Time PubMed NCBI</span>
            <span class="source-badge">🛡️ Zero-Keyword Red-Team</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tabs = st.tabs([
    "🔬 Live arXiv & PubMed Search",
    "⚡ Agentic Pipeline & Red-Teaming",
    "🕸️ Interactive Knowledge Graph",
    "📑 Research Brief & CV Badge Export",
    "📄 Indexed Corpus Registry",
    "⚖️ Symbolic Verifier Sandbox"
])

# -------------------------------------------------------------------------------------------------
# TAB 1: Live arXiv & PubMed Paper Search Integration
# -------------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🔬 Live Scientific Paper Retrieval & Automatic Graph Ingestion")
    st.markdown("Query the **public arXiv Atom API** and **NCBI PubMed E-Utilities API** in real time. Live abstracts are parsed on-the-fly into relational triplets and integrated into the active Knowledge Graph.")

    col_s_input, col_s_src, col_s_max, col_s_btn = st.columns([4, 2, 1, 2])
    with col_s_input:
        search_query = st.text_input(
            "Scientific Topic / Search Keywords",
            value=st.session_state.last_query_text or "CRISPR gene editing",
            placeholder="Type ANY topic: e.g. CRISPR gene editing, mRNA lipid nanoparticles, Quantum LLMs, etc."
        )
    with col_s_src:
        search_src = st.selectbox("Literature Database", ["All (arXiv + PubMed)", "arXiv Only", "PubMed Only"])
    with col_s_max:
        max_papers = st.number_input("Max Papers", min_value=2, max_value=10, value=4, step=2)
    with col_s_btn:
        st.write("")
        st.write("")
        search_btn = st.button("🔍 Search Live APIs", type="primary", use_container_width=True)

    if search_btn and search_query.strip():
        with st.spinner(f"Sending live request to {search_src} for '{search_query}'..."):
            mapped_src = "arxiv" if search_src == "arXiv Only" else ("pubmed" if search_src == "PubMed Only" else "all")
            papers = engine.search_papers(search_query.strip(), source=mapped_src, max_results=int(max_papers))
            st.session_state.search_results = papers
            if not st.session_state.last_query_text:
                st.session_state.last_query_text = search_query.strip()

    # Display Search Results
    if st.session_state.search_results:
        st.markdown(f"#### 📄 Live API Results ({len(st.session_state.search_results)} Papers Returned)")
        
        col_ingest_all, col_status = st.columns([2, 3])
        with col_ingest_all:
            if st.button("⚡ Ingest All into Knowledge Graph", type="primary", use_container_width=True):
                summary = engine.ingest_papers_batch(st.session_state.search_results)
                st.success(f"Ingested {summary['ingested_count']} papers! Generated {summary['total_nodes']} nodes and {summary['total_edges']} relational edges.")
                st.rerun()
        with col_status:
            live_count = sum(1 for p in st.session_state.search_results if getattr(p, 'is_live', True))
            st.caption(f"Status: **{live_count}/{len(st.session_state.search_results)} fetched live from external APIs**")

        for idx, paper in enumerate(st.session_state.search_results):
            source_badge_html = f"<span class='arxiv-badge'>arXiv</span>" if paper.source == "arXiv" else f"<span class='pubmed-badge'>PubMed</span>"
            latency_badge = f"<span class='live-badge'>LIVE &bull; {paper.fetch_latency_ms}ms</span>" if paper.fetch_latency_ms else "<span class='live-badge'>LIVE</span>"
            
            with st.expander(f"{idx+1}. [{paper.source}] {paper.title} ({paper.published_date})", expanded=(idx < 2)):
                st.markdown(f"**Authors:** {', '.join(paper.authors) if paper.authors else 'Research Group'} &nbsp;|&nbsp; {source_badge_html} {latency_badge} &nbsp;|&nbsp; 🔗 [Open Publication]({paper.url})", unsafe_allow_html=True)
                st.markdown(f"**Abstract:**")
                st.info(paper.abstract)

                triplets = engine.kg_engine.extract_triplets(paper.abstract, paper.id)
                if triplets:
                    st.caption(f"🔬 Extracted {len(triplets)} Relational Triplet(s) on-the-fly:")
                    t_cols = st.columns(min(3, len(triplets)))
                    for i, (s, p, o) in enumerate(triplets[:3]):
                        with t_cols[i % len(t_cols)]:
                            st.code(f"({s})\n──[{p}]──>\n({o})", language="text")

                col_b1, _ = st.columns([1, 4])
                with col_b1:
                    if st.button(f"📥 Ingest Paper #{idx+1}", key=f"ingest_single_{paper.id}"):
                        count = engine.ingest_paper(paper)
                        st.success(f"Ingested '{paper.id}'! Extracted {count} triplets.")
                        st.rerun()
    elif len(engine.documents) == 0:
        st.info("💡 No papers currently loaded. Type any research topic above and click **Search Live APIs**, or use the 1-click pills in the sidebar to fetch live papers immediately.")

# -------------------------------------------------------------------------------------------------
# TAB 2: Multi-Agent Pipeline & Hallucination Red-Teaming Mode
# -------------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("⚡ Autonomous Multi-Agent Query & Hallucination Red-Teaming")
    st.markdown("Execute the 3-agent neuro-symbolic workflow in real time: **Semantic Retrieval** &rarr; **Multi-Hop Synthesis** &rarr; **Deterministic Symbolic Verifier**.")

    # Red-Teaming Security Sandbox
    st.markdown("### 🛡️ Red-Teaming Security Sandbox")
    col_rt_toggle, col_rt_mode = st.columns([1, 3])
    with col_rt_toggle:
        red_team_active = st.toggle("🚨 Hallucination Red-Teaming Mode", value=False)
    
    adversarial_injection = None
    if red_team_active:
        st.markdown("""
        <div style="background: rgba(127, 29, 29, 0.25); border: 1px solid #EF4444; border-radius: 10px; padding: 14px; margin-bottom: 12px;">
            <b style="color: #EF4444; font-size: 14px;">🚨 ADVERSARIAL RED-TEAMING ACTIVE:</b>
            <span style="color: #FCA5A5; font-size: 13px;"> Inject any arbitrary false statement, direct contradiction, or phantom entity. The Symbolic Verifier checks it dynamically using graph topology and semantic negation checks.</span>
        </div>
        """, unsafe_allow_html=True)

        dynamic_attacks = engine.get_dynamic_red_team_attacks()
        attack_options = [a.name for a in dynamic_attacks] + ["Custom Deceptive Injection"]
        
        col_preset, col_custom = st.columns([2, 2])
        with col_preset:
            selected_attack_name = st.selectbox("Select Dynamically Generated Attack", attack_options)
        
        with col_custom:
            if selected_attack_name == "Custom Deceptive Injection":
                adversarial_injection = st.text_input(
                    "Custom False Claim to Inject (Any statement on any topic)",
                    value="The active mechanism operates as a mechanical steam turbine on silicon crystal wafers."
                )
            else:
                attack_obj = next(a for a in dynamic_attacks if a.name == selected_attack_name)
                adversarial_injection = attack_obj.injection_text
                st.caption(f"**Category:** `{attack_obj.category}` &bull; **Target:** `{attack_obj.target_concept}`")
                st.info(f"**Simulated Deception:** \"{adversarial_injection}\"")

    st.divider()

    # Query Input
    col_q, col_q_btn = st.columns([4, 1])
    with col_q:
        user_query = st.text_input(
            "Research Query:",
            value=st.session_state.last_query_text or "What are the primary findings and mechanisms demonstrated in the literature?",
            label_visibility="collapsed"
        )
    with col_q_btn:
        run_query_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

    if run_query_btn and user_query.strip():
        if len(engine.documents) == 0:
            st.warning("Knowledge Graph is currently empty. Please search and ingest papers in Tab 1 first, or use a 1-click pill in the sidebar.")
        else:
            st.session_state.last_query_text = user_query
            with st.spinner("Orchestrating Semantic Retrieval, Multi-Hop Synthesis, and Deterministic Logic Audit..."):
                result = engine.query(
                    user_query.strip(),
                    red_team_mode=red_team_active,
                    adversarial_injection=adversarial_injection if red_team_active else None
                )
                st.session_state.last_query_result = result

    # Display Query Results
    if st.session_state.last_query_result:
        res: VerifiedResponse = st.session_state.last_query_result

        # Check Red-Teaming Report
        if res.red_team_report and res.red_team_report.get("is_active"):
            rt_report = res.red_team_report
            if rt_report.get("hallucination_detected"):
                violations_html = "".join([f"<li><b>Violation:</b> {v}</li>" for v in rt_report.get("detected_violations", [])])
                ground_truth_html = "".join([f"<li>{gt}</li>" for gt in rt_report.get("ground_truth_counter_evidence", [])])
                
                st.markdown(f"""
                <div class="red-team-alert">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                        <span style="font-size: 28px;">🚨</span>
                        <div>
                            <h3 style="margin: 0; color: #FFFFFF; font-size: 1.25rem;">
                                CRITICAL HALLUCINATION DETECTED: SYMBOLIC AUDIT TRIGGERED
                            </h3>
                            <p style="margin: 2px 0 0 0; color: #FCA5A5; font-size: 0.9rem;">
                                The deterministic Symbolic Verifier dynamically caught the deception through constraint checks.
                            </p>
                        </div>
                    </div>
                    <div style="background: rgba(0, 0, 0, 0.4); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                        <b style="color: #F87171;">Intercepted Deceptive Assertion:</b><br/>
                        <span class="strike-claim">"{rt_report.get('attack_applied', '')}"</span>
                    </div>
                    <div style="color: #FECACA; font-size: 0.9rem; margin-bottom: 10px;">
                        <ul style="margin: 0; padding-left: 20px;">
                            {violations_html}
                        </ul>
                    </div>
                    <div style="background: rgba(6, 78, 59, 0.5); border: 1px solid #10B981; border-radius: 8px; padding: 10px 14px;">
                        <b style="color: #34D399;">Verified Ground Truth from Knowledge Graph:</b>
                        <ul style="margin: 4px 0 0 0; padding-left: 20px; color: #E2E8F0; font-size: 0.88rem;">
                            {ground_truth_html}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="clean-alert">
                    <b style="color: #34D399;">🛡️ RED-TEAM AUDIT CLEAN:</b>
                    <span style="color: #E2E8F0;"> No ungrounded adversarial claims or logic contradictions detected.</span>
                </div>
                """, unsafe_allow_html=True)

        col_ans, col_score = st.columns([3, 2])

        with col_ans:
            st.markdown("#### 📝 Dynamic Multi-Hop Synthesized Answer")
            st.info(res.answer)

            st.markdown("#### 📚 Verified Scientific Sources Cited")
            if res.sources:
                for src in res.sources:
                    meta = engine.documents_metadata.get(src, {})
                    url = meta.get("url", "")
                    if url:
                        st.markdown(f"<a href='{url}' target='_blank' style='text-decoration: none;'><span class='source-badge'>📄 {src} &nearr;</span></a>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='source-badge'>📄 {src}</span>", unsafe_allow_html=True)
            else:
                st.caption("No external sources cited.")

        with col_score:
            st.markdown("#### ⚖️ Deterministic Symbolic Audit")
            score_pct = int(res.logical_consistency_score * 100)
            score_color = "#10B981" if score_pct >= 80 else ("#F59E0B" if score_pct >= 50 else "#EF4444")
            status_text = "VERIFIED FACTUAL CONSISTENCY" if score_pct >= 80 else ("PARTIAL GROUNDING" if score_pct >= 50 else "HALLUCINATION DETECTED")

            st.markdown(
                f"""
                <div style="background: rgba(17, 24, 39, 0.85); border: 2px solid {score_color}; border-radius: 12px; padding: 18px; text-align: center;">
                    <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em;">
                        Logical Consistency Score
                    </div>
                    <div style="color: {score_color}; font-size: 3.2rem; font-weight: 800; margin: 8px 0;">
                        {score_pct}%
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.85rem; font-weight: 600;">
                        {status_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("##### Constraint Checklist Findings:")
            if res.audit_trail:
                for item in res.audit_trail:
                    if "Pass" in item or "Verified" in item:
                        st.markdown(f"✅ `{item}`")
                    elif "RED-TEAM" in item:
                        st.markdown(f"🚨 `{item}`")
                    else:
                        st.markdown(f"⚠️ `{item}`")

        # Agent Execution Traces
        st.divider()
        st.markdown("#### 🔍 Real-Time Multi-Agent Execution Traces")
        if res.agent_traces:
            for trace in res.agent_traces:
                agent_title = trace.get("agent_name", "Agent")
                step_num = trace.get("step", 0)
                action = trace.get("action", "")
                details = trace.get("details", {})
                status = trace.get("status", "COMPLETED")

                with st.expander(f"Step {step_num}: {agent_title} &mdash; [{status}]", expanded=(step_num in [1, 3])):
                    st.markdown(f"**Action Executed:** `{action}`")
                    st.json(details)

# -------------------------------------------------------------------------------------------------
# TAB 3: Visual Interactive Knowledge Graph Explorer
# -------------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🕸️ Visual Interactive Knowledge Graph Explorer")
    st.markdown("Explore concepts and cross-paper connections on a live **Vis.js Network** canvas. **Click on any node** to inspect connected papers and relations.")

    graph_html = engine.get_interactive_graph_html(height=620)
    components.html(graph_html, height=640, scrolling=False)

    st.divider()

    col_c1, col_c2 = st.columns([1, 1])
    with col_c1:
        st.markdown("##### 🌉 Cross-Paper Bridging Concepts")
        bridges = engine.kg_engine.get_cross_paper_bridges()
        if bridges:
            bridge_data = []
            for b in bridges:
                bridge_data.append({
                    "Concept Entity": b["concept"],
                    "Connected Papers": ", ".join(b["connected_papers"]),
                    "Degree": b["degree"]
                })
            st.dataframe(pd.DataFrame(bridge_data), use_container_width=True, height=260)
        else:
            st.info("No cross-paper bridges detected yet. Ingest multiple papers from arXiv/PubMed to see concepts connect across literature.")

    with col_c2:
        st.markdown("##### 📜 Extracted Triplet Assertions")
        triplets_data = []
        for u, v, data in engine.graph.edges(data=True):
            triplets_data.append({
                "Subject": u,
                "Predicate": data.get("relation", ""),
                "Object": v,
                "Paper Source": data.get("source", "")
            })
        if triplets_data:
            st.dataframe(pd.DataFrame(triplets_data), use_container_width=True, height=260)
        else:
            st.info("Knowledge Graph is empty.")

    with st.expander("🖼️ Export Publication-Grade Matplotlib Figure (300 DPI)"):
        fig = engine.kg_engine.render_graph_figure(max_nodes=35)
        st.pyplot(fig)

# -------------------------------------------------------------------------------------------------
# TAB 4: Export to Research Brief & CV Portfolio Badge
# -------------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📑 Export to Research Brief & CV Portfolio Certification Badge")
    st.markdown("Generate publication-grade downloadable research summaries and verifiable portfolio badges.")

    cur_result: VerifiedResponse = st.session_state.last_query_result
    cur_query: str = st.session_state.last_query_text

    if not cur_result:
        st.warning("Please run a query in the **Agentic Pipeline** tab first to generate your research brief.")
    else:
        cur_date_str = datetime.now().strftime("%Y%m%d_%H%M")
        audit_hash = hashlib.sha256(f"{cur_query}{cur_result.answer}".encode()).hexdigest()[:16].upper()

        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            st.markdown("#### 📄 Executive Summary PDF")
            st.markdown("Download a formatted multi-page PDF executive brief containing literature citations, knowledge graph triplet matrix, and symbolic logic audit seals.")
            
            pdf_bytes = engine.export_pdf(cur_result, cur_query)
            st.download_button(
                label="📥 Download Executive Summary PDF",
                data=pdf_bytes,
                file_name=f"Research_Brief_{cur_date_str}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

        with col_exp2:
            st.markdown("#### 📝 Polished Markdown Brief")
            st.markdown("Download a GitHub Flavored Markdown research brief with Mermaid graph syntax and structured reference tables.")
            
            md_content = engine.export_markdown(cur_result, cur_query)
            st.download_button(
                label="📥 Download Markdown Brief (.md)",
                data=md_content,
                file_name=f"Research_Brief_{cur_date_str}.md",
                mime="text/markdown",
                use_container_width=True
            )

        st.divider()

        st.markdown("#### 🏅 CV & GitHub Portfolio Certification Badges")
        col_b_preview, col_b_code = st.columns([1, 1])

        with col_b_preview:
            st.markdown("##### Verifiable Research Integrity Card:")
            badge_card_html = engine.report_generator.get_portfolio_badge_html(
                score=cur_result.logical_consistency_score,
                query=cur_query,
                audit_hash=audit_hash
            )
            st.markdown(badge_card_html, unsafe_allow_html=True)

        with col_b_code:
            st.markdown("##### Markdown Badge Snippets for GitHub README:")
            score_pct = int(cur_result.logical_consistency_score * 100)
            badge_code = f"""[![Neuro-Symbolic Verified](https://img.shields.io/badge/Neuro--Symbolic%20Verification-{score_pct}%25%20Consistent-00D2FF?style=for-the-badge&logo=probot)](https://github.com/yourusername/neuro-symbolic-rag)
[![Corpus](https://img.shields.io/badge/Corpus-arXiv%20%7C%20PubMed-10B981?style=for-the-badge&logo=arxiv)](https://github.com/yourusername/neuro-symbolic-rag)
[![Audit Seal](https://img.shields.io/badge/Audit%20Seal-%23{audit_hash}-8B5CF6?style=for-the-badge)](https://github.com/yourusername/neuro-symbolic-rag)"""
            st.code(badge_code, language="markdown")

        with st.expander("👁️ Live Preview: Generated Markdown Research Brief"):
            st.markdown(md_content)

# -------------------------------------------------------------------------------------------------
# TAB 5: Ingested Documents Repository
# -------------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("📄 Indexed Scientific Literature Repository")
    if engine.documents:
        st.markdown(f"**Total Documents Registered:** `{len(engine.documents)}`")
        for doc_name, content in engine.documents.items():
            meta = engine.documents_metadata.get(doc_name, {})
            title = meta.get("title", doc_name)
            source_platform = meta.get("source", "Manual")
            url = meta.get("url", "")
            authors = ", ".join(meta.get("authors", [])) if meta.get("authors") else "Research Group"
            year = meta.get("published_date", "2024")

            with st.expander(f"📄 [{source_platform}] {title} ({year})", expanded=False):
                if url:
                    st.markdown(f"🔗 **Publication Link:** [{url}]({url})")
                st.markdown(f"**Authors:** {authors} &nbsp;|&nbsp; **Doc ID:** `{doc_name}`")
                st.text_area(f"Full Content [{doc_name}]", value=content, height=130, disabled=True)
                
                triplets = engine.kg_engine.extract_triplets(content, doc_name)
                st.caption(f"Extracted {len(triplets)} triplet(s) from this document:")
                for s, p, o in triplets:
                    st.code(f"({s}) ──[{p}]──> ({o})", language="text")
    else:
        st.warning("No documents currently loaded. Search arXiv / PubMed in Tab 1 or trigger a live fetch in the sidebar.")

# -------------------------------------------------------------------------------------------------
# TAB 6: Symbolic Verifier Sandbox
# -------------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("⚖️ Interactive Symbolic Constraint Sandbox")
    st.caption("Audit arbitrary claims, inverted axioms, or simulated hallucinations against deterministic constraint checkers.")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        test_answer = st.text_area(
            "Candidate Answer to Audit",
            value="CRISPR-Cas9 enables targeted genome editing in eukaryotic organisms. SpCas9 variants eliminate off-target cleavage.",
            height=130
        )
    with col_v2:
        available_sources = list(engine.documents.keys())
        selected_sources = st.multiselect(
            "Cited Source References",
            options=available_sources + ["Ghost_Paper_Volume_Omega"],
            default=available_sources[:2] if available_sources else []
        )
        adversarial_test = st.text_input("Optional Adversarial Injection to Audit:", value="")

    if st.button("Audit Candidate Answer", type="primary"):
        audit_res = engine.symbolic_verification(
            test_answer,
            selected_sources,
            adversarial_injection=adversarial_test if adversarial_test.strip() else None
        )
        st.markdown("#### Audit Outcome")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.metric("Consistency Score", f"{audit_res['score'] * 100:.0f}%")
        with col_r2:
            st.info(audit_res["notes"])
            if audit_res.get("audit_trail"):
                for trail in audit_res["audit_trail"]:
                    if "Pass" in trail or "Verified" in trail:
                        st.markdown(f"✅ `{trail}`")
                    else:
                        st.markdown(f"⚠️ `{trail}`")
