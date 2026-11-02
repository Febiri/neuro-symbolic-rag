import re
import math
from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
import matplotlib.pyplot as plt
import io

class KnowledgeGraphEngine:
    """Module B: Vector/Lexical Hybrid Index & In-Memory NetworkX Knowledge Graph

    with cross-paper relational tracking and interactive visualization support.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.documents: Dict[str, str] = {}
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        self.doc_tokens: Dict[str, List[str]] = {}
        
        # Enhanced scientific and relational predicates (supporting singular and plural verb inflections)
        self.predicates_pattern = re.compile(
            r'\b(is a|is an|is|are|combines?|enforces?|uses?|requires?|provides?|executes?|eliminates?|'
            r'consists? of|contains?|generates?|solves?|enables?|targets?|inhibits?|activates?|treats?|'
            r'expresses?|regulates?|encodes?|cleaves?|improves?|outperforms?|demonstrates?|replaces?|'
            r'reduces?|synthesizes?|enhances?|modifies?|delivers?|accelerates?|verifies?|proves?|binds?|'
            r'facilitates?|mediates?|induces?|prevents?|attenuates?|modulates?|interacts? with|causes?|'
            r'mitigates?|conducts?|exhibits?|displays?|yields?|forms?|preserves?|suppresses?|blocks?)\b',
            re.IGNORECASE
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer."""
        return re.findall(r'\b[a-zA-Z0-9_\-]+\b', text.lower())

    def _extract_pos_triplet(self, sentence: str) -> Optional[Tuple[str, str, str]]:
        """Fallback dynamic POS extraction for arbitrary scientific literature sentences."""
        try:
            from nltk import pos_tag
            words = re.findall(r'\b[a-zA-Z0-9_\-]+\b', sentence)
            if len(words) < 3:
                return None
            tagged = pos_tag(words)
            # Find primary active verbs
            verb_indices = [i for i, (w, t) in enumerate(tagged) if t.startswith('VB') and w.lower() not in {'be', 'been', 'is', 'are', 'was', 'were'}]
            if not verb_indices:
                return None
            v_idx = verb_indices[0]
            if v_idx == 0 or v_idx >= len(words) - 1:
                return None
            
            subject = ' '.join(words[:v_idx]).strip()
            predicate = words[v_idx].lower()
            obj = ' '.join(words[v_idx+1:]).strip()
            # Clean prepositional phrases from object
            obj = re.split(r'[,;]|\bby\b|\bthrough\b|\bwith\b|\busing\b|\bin\b|\bat\b', obj)[0].strip()

            if 2 < len(subject) < 55 and 2 < len(obj) < 55 and subject.lower() != obj.lower():
                return (subject, predicate, obj)
        except Exception:
            pass
        return None

    def clear(self):
        """Reset the graph and document store."""
        self.graph.clear()
        self.documents.clear()
        self.documents_metadata.clear()
        self.doc_tokens.clear()

    def extract_triplets(self, text: str, source_id: str) -> List[Tuple[str, str, str]]:
        """Extract Subject-Predicate-Object triplets using hybrid pattern matching and POS tagging."""
        triplets: List[Tuple[str, str, str]] = []
        sentences = re.split(r'[.\n;]+', text)

        for sent in sentences:
            sent_clean = sent.strip()
            if not sent_clean:
                continue

            match = self.predicates_pattern.search(sent_clean)
            if match:
                predicate = match.group(0).lower().strip()
                start_idx = match.start()
                end_idx = match.end()

                subject = sent_clean[:start_idx].strip()
                obj = sent_clean[end_idx:].strip()

                # Clean trailing clauses or punctuations
                subject = re.sub(r'^(the|a|an|in this paper,|we show that|our findings indicate that)\s+', '', subject, flags=re.IGNORECASE).strip()
                obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE).strip()
                obj = re.split(r'[,;]|\bby\b|\bthrough\b|\bwith\b|\busing\b|\bin\b|\bat\b', obj)[0].strip()

                if 2 < len(subject) < 55 and 2 < len(obj) < 55:
                    subject = subject.strip(" ,.-()")
                    obj = obj.strip(" ,.-()")
                    if subject and obj and subject.lower() != obj.lower():
                        triplets.append((subject, predicate, obj))
            else:
                # Dynamic POS fallback for arbitrary verbs
                pos_res = self._extract_pos_triplet(sent_clean)
                if pos_res:
                    s, p, o = pos_res
                    triplets.append((s, p, o))

        return triplets

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Index document content and update the knowledge graph with extracted triplets."""
        self.documents[doc_id] = text
        self.documents_metadata[doc_id] = metadata or {
            "title": doc_id,
            "source": "Manual Ingestion",
            "url": "",
            "authors": [],
            "published_date": "2024"
        }
        self.doc_tokens[doc_id] = self._tokenize(text)

        triplets = self.extract_triplets(text, doc_id)
        for subj, pred, obj in triplets:
            # Add or update nodes with source tracking
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, label=subj, type="concept", sources=[doc_id])
            else:
                sources = self.graph.nodes[subj].get("sources", [])
                if doc_id not in sources:
                    sources.append(doc_id)
                    self.graph.nodes[subj]["sources"] = sources

            if not self.graph.has_node(obj):
                self.graph.add_node(obj, label=obj, type="concept", sources=[doc_id])
            else:
                sources = self.graph.nodes[obj].get("sources", [])
                if doc_id not in sources:
                    sources.append(doc_id)
                    self.graph.nodes[obj]["sources"] = sources

            self.graph.add_edge(subj, obj, relation=pred, source=doc_id)

    def hybrid_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Perform lexical-semantic BM25-style relevance scoring with graph node matching."""
        q_tokens = set(self._tokenize(query))
        if not q_tokens or not self.documents:
            return []

        scores: List[Tuple[str, float]] = []
        n_docs = len(self.documents)

        # Calculate BM25/TF-IDF approximation
        for doc_id, tokens in self.doc_tokens.items():
            if not tokens:
                continue
            token_counts = {}
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1

            doc_len = len(tokens)
            score = 0.0

            for qt in q_tokens:
                tf = token_counts.get(qt, 0) / doc_len
                df = sum(1 for d_toks in self.doc_tokens.values() if qt in d_toks)
                idf = math.log((n_docs + 1) / (df + 1)) + 1.0
                score += tf * idf

            # Graph relevance boost: check if query tokens appear in graph nodes sourced from this doc
            graph_boost = 0.0
            for u, v, data in self.graph.edges(data=True):
                if data.get("source") == doc_id:
                    for qt in q_tokens:
                        if qt in u.lower() or qt in v.lower():
                            graph_boost += 0.25

            scores.append((doc_id, score + graph_boost))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_id, score in scores[:top_k]:
            results.append({
                "doc_id": doc_id,
                "score": round(score, 4),
                "content": self.documents[doc_id],
                "metadata": self.documents_metadata.get(doc_id, {})
            })
        return results

    def get_cross_paper_bridges(self) -> List[Dict[str, Any]]:
        """Identify concept nodes that connect across multiple papers."""
        bridges = []
        for node in self.graph.nodes():
            sources = self.graph.nodes[node].get("sources", [])
            # Also inspect incoming and outgoing edge sources
            edge_sources = set()
            for _, _, data in self.graph.out_edges(node, data=True):
                edge_sources.add(data.get("source", ""))
            for _, _, data in self.graph.in_edges(node, data=True):
                edge_sources.add(data.get("source", ""))
            all_sources = list(set(sources) | edge_sources - {""})

            if len(all_sources) > 1:
                bridges.append({
                    "concept": node,
                    "connected_papers": all_sources,
                    "degree": self.graph.degree(node)
                })
        bridges.sort(key=lambda x: len(x["connected_papers"]) * 10 + x["degree"], reverse=True)
        return bridges

    def to_vis_network_data(self) -> Dict[str, Any]:
        """Generate interactive Vis.js Network data payload with high-contrast scientific styling."""
        vis_nodes = []
        vis_edges = []

        if len(self.graph.nodes) == 0:
            return {"nodes": [], "edges": []}

        # Calculate degrees for node scaling
        degrees = dict(self.graph.degree())
        max_deg = max(degrees.values()) if degrees else 1

        # Palette for papers and concepts
        paper_palette = ["#F59E0B", "#EC4899", "#8B5CF6", "#10B981", "#3B82F6", "#F97316"]
        unique_sources = list(self.documents.keys())
        source_color_map = {src: paper_palette[i % len(paper_palette)] for i, src in enumerate(unique_sources)}

        for node in self.graph.nodes():
            deg = degrees.get(node, 1)
            node_sources = self.graph.nodes[node].get("sources", [])
            
            # Determine color and hub status
            is_hub = deg >= 3
            if len(node_sources) > 1:
                # Multi-paper cross-bridge concept
                color = {
                    "background": "#00E5FF",
                    "border": "#FFFFFF",
                    "highlight": {"background": "#00B4D8", "border": "#FFFFFF"}
                }
                group = "Cross-Paper Hub"
                size = min(35, 18 + (deg * 4))
            elif node_sources and node_sources[0] in source_color_map:
                bg = source_color_map[node_sources[0]]
                color = {
                    "background": bg,
                    "border": "#FFFFFF",
                    "highlight": {"background": "#E2E8F0", "border": bg}
                }
                group = node_sources[0]
                size = min(30, 14 + (deg * 3))
            else:
                color = {
                    "background": "#4FD1C5",
                    "border": "#2D3748",
                    "highlight": {"background": "#81E6D9", "border": "#FFFFFF"}
                }
                group = "Concept"
                size = 16

            # Construct clean tooltip
            sources_str = ", ".join(node_sources) if node_sources else "Indexed"
            tooltip_html = (
                f"<div style='font-family: Inter, sans-serif; padding: 6px 10px; max-width: 250px;'>"
                f"<b style='color: #00D2FF; font-size: 13px;'>{node}</b><br/>"
                f"<span style='color: #CBD5E0; font-size: 11px;'>Connectivity: {deg} connection(s)</span><br/>"
                f"<span style='color: #A0AEC0; font-size: 10px;'>Sources: {sources_str}</span>"
                f"</div>"
            )

            vis_nodes.append({
                "id": node,
                "label": node if len(node) <= 22 else node[:20] + "...",
                "title": tooltip_html,
                "size": size,
                "color": color,
                "font": {"color": "#FFFFFF", "size": 12, "face": "Inter, system-ui, sans-serif"},
                "shape": "dot",
                "shadow": {"enabled": is_hub, "color": "rgba(0, 229, 255, 0.4)", "size": 10},
                "group": group
            })

        for u, v, data in self.graph.edges(data=True):
            relation = data.get("relation", "relates to")
            src_doc = data.get("source", "")
            edge_tooltip = f"<b>{u}</b> &mdash;[<i>{relation}</i>]&rarr; <b>{v}</b><br/><small>Paper: {src_doc}</small>"

            vis_edges.append({
                "from": u,
                "to": v,
                "label": relation,
                "title": edge_tooltip,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.75}},
                "color": {"color": "#4A5568", "highlight": "#00D2FF", "hover": "#81E6D9"},
                "font": {"color": "#A0AEC0", "size": 10, "align": "middle", "strokeWidth": 0},
                "smooth": {"type": "curvedCW", "roundness": 0.15}
            })

        return {"nodes": vis_nodes, "edges": vis_edges}

    def render_graph_figure(self, max_nodes: int = 35) -> plt.Figure:
        """Render a publication-grade NetworkX graph visualization using Matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor('#0B0F19')
        ax.set_facecolor('#0B0F19')

        if len(self.graph.nodes) == 0:
            ax.text(
                0.5, 0.5, "Knowledge Graph is Empty.\nSearch and ingest scientific papers or add documents.",
                ha="center", va="center", color="#A0AEC0", fontsize=12
            )
            ax.axis('off')
            return fig

        if len(self.graph.nodes) > max_nodes:
            sampled_nodes = list(self.graph.nodes)[:max_nodes]
            G_sub = self.graph.subgraph(sampled_nodes)
        else:
            G_sub = self.graph

        pos = nx.spring_layout(G_sub, seed=42, k=0.8)

        nx.draw_networkx_nodes(
            G_sub, pos, ax=ax,
            node_color="#00D2FF",
            node_size=1600,
            alpha=0.88,
            edgecolors="#FFFFFF",
            linewidths=1.5
        )

        nx.draw_networkx_edges(
            G_sub, pos, ax=ax,
            arrowstyle="->",
            arrowsize=16,
            edge_color="#4FD1C5",
            width=1.5,
            alpha=0.7,
            connectionstyle="arc3,rad=0.1"
        )

        labels = {n: (str(n)[:18] + '...' if len(str(n)) > 18 else str(n)) for n in G_sub.nodes()}
        nx.draw_networkx_labels(
            G_sub, pos, labels=labels, ax=ax,
            font_size=8,
            font_color="#0B0F19",
            font_weight="bold",
            font_family="sans-serif"
        )

        edge_labels = nx.get_edge_attributes(G_sub, 'relation')
        formatted_edge_labels = {k: v[:15] for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            G_sub, pos, edge_labels=formatted_edge_labels, ax=ax,
            font_size=7,
            font_color="#E2E8F0",
            bbox=dict(boxstyle="round,pad=0.2", fc="#1A202C", ec="#4A5568", alpha=0.9)
        )

        ax.axis('off')
        plt.tight_layout()
        return fig
