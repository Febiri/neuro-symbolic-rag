import re
from typing import List, Dict, Any, Optional, Set, Tuple
import networkx as nx
from pydantic import BaseModel, Field

# Domain-agnostic semantic polarity and antonym mappings
POLAR_ANTONYM_PAIRS = [
    ("deterministic", "non-deterministic"),
    ("deterministic", "randomized"),
    ("deterministic", "stochastic"),
    ("enables", "prevents"),
    ("activates", "inhibits"),
    ("increases", "decreases"),
    ("promotes", "suppresses"),
    ("accelerates", "decelerates"),
    ("cleaves", "ligates"),
    ("eliminates", "generates"),
    ("eliminates", "produces"),
    ("therapeutic", "toxic"),
    ("viable", "inviable"),
    ("accurate", "erroneous"),
    ("ordered", "disordered"),
    ("coherent", "decoherent"),
    ("verified", "unverified"),
    ("grounded", "ungrounded"),
]

NEGATION_CUES = {
    "not", "never", "no", "cannot", "does not", "did not", "will not",
    "fails to", "without", "unable to", "neither", "nor", "opposite of"
}

GENERIC_FUNCTION_WORDS = {
    "this", "that", "with", "from", "have", "were", "been", "which", "about",
    "other", "their", "into", "operate", "operates", "rely", "relies", "based",
    "according", "system", "method", "protocol", "purely", "directly", "merely",
    "solely", "inherently", "randomly", "actually", "typically", "often", "always",
    "cleave", "cleaves", "combine", "combines", "using", "states", "shows", "claims"
}

class RedTeamAttack(BaseModel):
    attack_id: str
    name: str
    category: str
    injection_text: str
    target_concept: str
    rationale: str

class RedTeamAuditReport(BaseModel):
    is_active: bool = False
    hallucination_detected: bool = False
    alert_level: str = "CLEAN"  # "CLEAN", "WARNING", "CRITICAL"
    attack_applied: Optional[str] = None
    original_score: float = 1.0
    penalized_score: float = 1.0
    detected_violations: List[str] = Field(default_factory=list)
    hallucinatory_claims: List[str] = Field(default_factory=list)
    ground_truth_counter_evidence: List[str] = Field(default_factory=list)
    remediation_status: str = "NOT_TRIGGERED"
    remediated_answer: Optional[str] = None

class RedTeamAuditor:
    """Universal, domain-agnostic Symbolic Logic Auditor that detects hallucinations,

    unindexed phantom entities, topological disconnections, and semantic contradictions
    in real time without hardcoded domain keywords.
    """

    def __init__(self, kg_engine):
        self.kg = kg_engine

    def _extract_terms(self, text: str) -> Set[str]:
        """Extract multi-character informative tokens, ignoring generic function words."""
        words = re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', text.lower())
        return {w for w in words if w not in GENERIC_FUNCTION_WORDS}

    def _find_matching_graph_nodes(self, text: str) -> List[str]:
        """Find nodes in the active graph whose label or core entity is mentioned in the text."""
        lower_text = text.lower()
        matched = []
        for node in self.kg.graph.nodes():
            node_str = str(node).lower().strip()
            if len(node_str) >= 3:
                # Direct substring match
                if node_str in lower_text:
                    matched.append(node)
                # Reverse match: text mentions the primary entity part
                elif len(node_str) > 6 and any(sub in lower_text for sub in [node_str.split(' in ')[0], node_str.split(' of ')[0]]):
                    matched.append(node)
        return list(set(matched))

    def audit_injection(
        self,
        candidate_answer: str,
        injection_text: str,
        sources: List[str],
        claims: Optional[List[Dict[str, Any]]] = None
    ) -> RedTeamAuditReport:
        """Dynamically evaluate candidate answers and injected statements against the active Knowledge Graph."""
        violations: List[str] = []
        hallucinated_claims: List[str] = []
        ground_truth: List[str] = []
        penalty = 0.0

        has_injection = bool(injection_text and injection_text.strip())
        target_text = f"{candidate_answer} {injection_text}".strip()
        lower_target = target_text.lower()
        active_injection = injection_text.strip() if has_injection else ""

        # -----------------------------------------------------------------------------------------
        # Check 1: Dynamic Source Registry Audit (Detect Forged / Unknown Documents)
        # -----------------------------------------------------------------------------------------
        cited_in_text = re.findall(r'\[([a-zA-Z0-9_\-:]+)\]', target_text)
        all_cited = list(set(sources + cited_in_text))
        for src in all_cited:
            if src not in self.kg.documents and not any(src.lower() in k.lower() for k in self.kg.documents.keys()):
                if any(tag in src.lower() for tag in ["doc", "paper", "arxiv", "pubmed", "secret", "omega"]):
                    violations.append(
                        f"SOURCE REGISTRY VIOLATION [Rule S1 - Citation Forgery]: "
                        f"Cited document '{src}' is absent from the indexed literature repository."
                    )
                    hallucinated_claims.append(f"Source citation '{src}'")
                    penalty += 0.45

        # -----------------------------------------------------------------------------------------
        # Check 2: Dynamic Phantom Entity Detection (Zero Grounding in Document Registry)
        # -----------------------------------------------------------------------------------------
        if active_injection:
            inj_terms = self._extract_terms(active_injection)
            corpus_text = " ".join(self.kg.documents.values()).lower()
            corpus_nodes = " ".join([str(n).lower() for n in self.kg.graph.nodes()])
            
            # Check if any term is part of known antonym pairs or negations (which are contradictions, not phantom terms)
            known_antonyms = {neg for pos, neg in POLAR_ANTONYM_PAIRS} | {pos for pos, neg in POLAR_ANTONYM_PAIRS} | NEGATION_CUES

            phantom_terms = []
            for term in inj_terms:
                if len(term) >= 4 and term not in corpus_text and term not in corpus_nodes and term not in known_antonyms:
                    phantom_terms.append(term)

            if len(phantom_terms) >= 1 and len(self.kg.documents) > 0:
                violations.append(
                    f"PHANTOM ENTITY DETECTED [Rule E1 - Zero Corpus Grounding]: "
                    f"Assertion references unverified terminology ({', '.join(phantom_terms[:3])}) "
                    f"with 0 occurrences across all {len(self.kg.documents)} indexed research papers."
                )
                hallucinated_claims.append(f"Unindexed terms: {', '.join(phantom_terms[:3])}")
                penalty += 0.50

        # -----------------------------------------------------------------------------------------
        # Check 3: Dynamic Factual Contradiction & Antonym Inversion Detection
        # -----------------------------------------------------------------------------------------
        for u, v, data in self.kg.graph.edges(data=True):
            u_str = str(u).lower()
            v_str = str(v).lower()
            rel = str(data.get("relation", "")).lower()

            # Check if subject u is present in the target text
            if u_str in lower_target:
                sentences = re.split(r'[.\n;]+', target_text)
                for sent in sentences:
                    sent_lower = sent.lower()
                    if u_str in sent_lower:
                        # Case A: Object v is also in sentence
                        if v_str in sent_lower:
                            has_negation = any(re.search(rf'\b{re.escape(cue)}\b', sent_lower) for cue in NEGATION_CUES)
                            if has_negation and not any(re.search(rf'\b{re.escape(cue)}\b', rel) for cue in NEGATION_CUES):
                                violations.append(
                                    f"LOGICAL CONTRADICTION [Rule C1 - Negation Inversion]: "
                                    f"Statement asserts negation between '{u}' and '{v}', "
                                    f"violating verified Knowledge Graph edge: '({u}) --[{rel}]--> ({v})'."
                                )
                                hallucinated_claims.append(sent.strip())
                                ground_truth.append(f"Verified Fact: ({u}) {rel} ({v})")
                                penalty += 0.55

                        # Case B: Object v has an antonym in the sentence (e.g. deterministic vs non-deterministic)
                        for pos, neg in POLAR_ANTONYM_PAIRS:
                            if pos in v_str and re.search(rf'\b{re.escape(neg)}\b', sent_lower):
                                violations.append(
                                    f"FACTUAL CONTRADICTION [Rule C2 - Semantic Antonym Inversion]: "
                                    f"Claim asserts '{neg}' regarding '{u}', "
                                    f"contradicting verified Knowledge Graph relation: '({u}) --[{rel}]--> ({v})'."
                                )
                                hallucinated_claims.append(sent.strip())
                                ground_truth.append(f"Verified Fact: ({u}) {rel} ({v})")
                                penalty += 0.55
                            elif neg in v_str and re.search(rf'\b{re.escape(pos)}\b', sent_lower):
                                violations.append(
                                    f"FACTUAL CONTRADICTION [Rule C2 - Semantic Antonym Inversion]: "
                                    f"Claim asserts '{pos}' regarding '{u}', "
                                    f"contradicting verified relation: '({u}) --[{rel}]--> ({v})'."
                                )
                                hallucinated_claims.append(sent.strip())
                                ground_truth.append(f"Verified Fact: ({u}) {rel} ({v})")
                                penalty += 0.55

                        # Case C: Relation has an antonym in the sentence (e.g. enables vs prevents, activates vs inhibits)
                        for pos, neg in POLAR_ANTONYM_PAIRS:
                            if pos in rel and re.search(rf'\b{re.escape(neg)}\b', sent_lower):
                                violations.append(
                                    f"FACTUAL CONTRADICTION [Rule C3 - Predicate Inversion]: "
                                    f"Claim asserts '{neg}' between '{u}' and related entities, "
                                    f"contradicting verified relation: '({u}) --[{rel}]--> ({v})'."
                                )
                                hallucinated_claims.append(sent.strip())
                                ground_truth.append(f"Verified Fact: ({u}) {rel} ({v})")
                                penalty += 0.55

        # -----------------------------------------------------------------------------------------
        # Check 4: Dynamic Topological Reachability & Disconnection Audit
        # -----------------------------------------------------------------------------------------
        if has_injection and active_injection:
            matched_nodes = self._find_matching_graph_nodes(active_injection)
            if len(matched_nodes) >= 2:
                node_pairs = [(matched_nodes[i], matched_nodes[j]) for i in range(len(matched_nodes)) for j in range(i + 1, len(matched_nodes))]
                for na, nb in node_pairs:
                    if not self.kg.graph.has_edge(na, nb) and not self.kg.graph.has_edge(nb, na):
                        has_path_ab = nx.has_path(self.kg.graph, na, nb) if self.kg.graph.has_node(na) and self.kg.graph.has_node(nb) else False
                        has_path_ba = nx.has_path(self.kg.graph, nb, na) if self.kg.graph.has_node(na) and self.kg.graph.has_node(nb) else False
                        
                        if not has_path_ab and not has_path_ba:
                            violations.append(
                                f"RELATIONAL DISCONNECTION [Rule R1 - Topological Isolation]: "
                                f"Injected assertion connects '{na}' and '{nb}', which belong to disconnected components in the Knowledge Graph with 0 relational path."
                            )
                            hallucinated_claims.append(f"Disconnected assertion: '{na}' <-> '{nb}'")
                            penalty += 0.45

        # Compile ground truth evidence if empty
        if violations and not ground_truth:
            sample_edges = list(self.kg.graph.edges(data=True))[:3]
            for u, v, data in sample_edges:
                ground_truth.append(f"Verified Fact: ({u}) {data.get('relation','relates to')} ({v})")

        penalized_score = max(0.0, round(1.0 - penalty, 2))
        detected = len(violations) > 0
        has_critical = any("CONTRADICTION" in v or "PHANTOM" in v for v in violations) or penalized_score <= 0.50
        alert_level = "CRITICAL" if (detected and has_critical) else ("WARNING" if detected else "CLEAN")

        # Autonomous Dynamic Remediation
        remediated_text = candidate_answer
        if detected:
            if active_injection and active_injection in remediated_text:
                remediated_text = remediated_text.replace(active_injection, "")
            for hc in hallucinated_claims:
                remediated_text = re.sub(re.escape(hc), "", remediated_text, flags=re.IGNORECASE)
            
            clean_evidence = "; ".join(ground_truth[:2]) if ground_truth else "Grounding strictly enforced against verified Knowledge Graph."
            remediated_text = f"{remediated_text.strip()} [Autonomous Symbolic Correction: Intercepted ungrounded claim. Ground truth: {clean_evidence}]"

        return RedTeamAuditReport(
            is_active=has_injection or detected,
            hallucination_detected=detected,
            alert_level=alert_level,
            attack_applied=active_injection if has_injection else None,
            original_score=1.0,
            penalized_score=penalized_score,
            detected_violations=violations,
            hallucinatory_claims=hallucinated_claims,
            ground_truth_counter_evidence=ground_truth,
            remediation_status="REMEDIATED" if detected else "VERIFIED_CLEAN",
            remediated_answer=remediated_text if detected else candidate_answer
        )

    def generate_dynamic_attacks_for_graph(self) -> List[RedTeamAttack]:
        """Dynamically generate adversarial attack options tailored to WHATEVER papers

        and concepts are currently loaded in the Knowledge Graph.
        """
        attacks: List[RedTeamAttack] = []

        # Attack 1: Dynamic Axiom Inversion from real graph edge
        edges = list(self.kg.graph.edges(data=True))
        if edges:
            u, v, data = edges[0]
            rel = data.get("relation", "is")
            neg_rel = f"does not {rel}" if not rel.startswith("is") else f"{rel} not"
            attacks.append(RedTeamAttack(
                attack_id="dyn_inversion",
                name=f"Factual Contradiction: Invert '{u}' -> '{v}'",
                category="Direct Factual Contradiction",
                injection_text=f"{u} {neg_rel} {v}.",
                target_concept=str(u),
                rationale=f"Directly inverts ground-truth Knowledge Graph edge '({u}) --[{rel}]--> ({v})'."
            ))

        # Attack 2: Disconnected components attack
        if len(self.kg.graph.nodes) >= 2:
            nodes = list(self.kg.graph.nodes)
            n1 = nodes[0]
            n2 = nodes[-1]
            if not self.kg.graph.has_edge(n1, n2):
                attacks.append(RedTeamAttack(
                    attack_id="dyn_disconnect",
                    name=f"Relational Hijack: Link Disconnected Concepts",
                    category="Relational Disconnection",
                    injection_text=f"{n1} directly mechanizes and eliminates {n2}.",
                    target_concept=str(n1),
                    rationale=f"Forces an ungrounded connection between topologically isolated nodes '{n1}' and '{n2}'."
                ))

        # Attack 3: Phantom Entity Injection
        sample_node = list(self.kg.graph.nodes)[0] if self.kg.graph.nodes else "The active mechanism"
        attacks.append(RedTeamAttack(
            attack_id="dyn_phantom",
            name="Phantom Entity: Synthetic Unindexed Term",
            category="Phantom Entity Fabrication",
            injection_text=f"{sample_node} is powered by the hypothetical Krypton-Chronoflux-Resonator.",
            target_concept=str(sample_node),
            rationale="Introduces a completely unindexed term with 0 grounding in the research corpus."
        ))

        # Attack 4: Forged Citation
        attacks.append(RedTeamAttack(
            attack_id="dyn_citation",
            name="Citation Forgery: Fabricated Source Tag",
            category="Citation Forgery",
            injection_text="As conclusively established in [Paper_Unregistered_Omega_999].",
            target_concept="Source Registry",
            rationale="Cites an unregistered document ID to test source validation constraints."
        ))

        return attacks
