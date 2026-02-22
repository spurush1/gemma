"""
ICD-10 Context Graph Module

Primary backend: Neo4j (persistent graph database)
Fallback backend: NetworkX (in-memory, used when Neo4j is unavailable)

The public API is identical regardless of which backend is active.
On startup, the module checks Neo4j availability; if reachable it seeds
the graph (idempotent) and delegates all queries to Cypher. If Neo4j is
not running, it transparently falls back to the in-memory NetworkX graph.
"""

import re
import logging
import networkx as nx
import simple_icd_10_cm as cm
from typing import Optional
from icd10_data import (
    ICD10_DATA, get_all_codes,
    get_real_excludes, get_real_children, get_real_ancestors,
    _infer_category, _build_code_entry
)

logger = logging.getLogger("context_graph")

# Attempt to import Neo4j backend
try:
    import graph_db as _gdb
    _NEO4J_IMPORT_OK = True
except ImportError:
    _NEO4J_IMPORT_OK = False
    logger.warning("graph_db module not found. Neo4j backend disabled.")


class ICD10ContextGraph:
    """
    ICD-10 Knowledge Graph.
    Uses Neo4j as primary store (persistent, Cypher queries).
    Falls back to NetworkX (in-memory) when Neo4j is unavailable.
    """

    def __init__(self):
        # Determine backend
        self._use_neo4j = False
        if _NEO4J_IMPORT_OK:
            try:
                if _gdb.is_neo4j_available():
                    _gdb.ensure_graph_seeded()
                    self._use_neo4j = True
                    logger.info("Context graph backend: Neo4j (persistent)")
                else:
                    logger.info("Context graph backend: NetworkX (Neo4j unreachable)")
            except Exception as e:
                logger.warning(f"Neo4j init failed ({e}). Falling back to NetworkX.")

        # Always build in-memory NetworkX graph as fallback / mirror
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """
        Build the knowledge graph from real CMS ICD-10-CM data.

        Node types:
          - icd10_code  : every code in the augmented clinical subset
          - symptom     : clinical symptom (curated)
          - image_finding: radiology finding (curated)
          - disease     : category-level grouping

        Edge sources:
          - symptom/finding → code  : curated clinical augmentation
          - parent_child edges      : REAL CMS hierarchy via simple_icd_10_cm
          - excludes edges          : REAL CMS excludes1 rules via simple_icd_10_cm
          - ancestor chain edges    : REAL CMS multi-level ancestry
        """
        icd_data = get_all_codes()   # augmented subset (clinical data)

        for code, data in icd_data.items():

            # -----------------------------------------------
            # 1. icd10_code node — attrs from real CMS data
            # -----------------------------------------------
            self.graph.add_node(
                code,
                node_type="icd10_code",
                description=data["description"],      # real CMS description
                category=data["category"],
                severity=data["severity"],
                avg_reimbursement_usd=data["avg_reimbursement_usd"],
                is_billable=data.get("is_billable", True),
            )

            # -----------------------------------------------
            # 2. Symptom nodes + edges (curated augmentation)
            # -----------------------------------------------
            for symptom in data.get("symptoms", []):
                symptom_node = f"symptom:{symptom}"
                if not self.graph.has_node(symptom_node):
                    self.graph.add_node(symptom_node, node_type="symptom", name=symptom)
                confidence = self._estimate_symptom_confidence(symptom, code, icd_data)
                self.graph.add_edge(
                    symptom_node, code,
                    edge_type="symptom_to_code", confidence=confidence
                )

            # -----------------------------------------------
            # 3. Image finding nodes + edges (curated)
            # -----------------------------------------------
            for finding in data.get("image_findings", []):
                finding_node = f"finding:{finding}"
                if not self.graph.has_node(finding_node):
                    self.graph.add_node(finding_node, node_type="image_finding", name=finding)
                confidence = self._estimate_finding_confidence(finding, code)
                self.graph.add_edge(
                    finding_node, code,
                    edge_type="finding_to_code", confidence=confidence
                )

            # -----------------------------------------------
            # 4. Disease (category) node + edge
            # -----------------------------------------------
            category = data["category"]
            disease_node = f"disease:{category}"
            if not self.graph.has_node(disease_node):
                self.graph.add_node(disease_node, node_type="disease", name=category)
            self.graph.add_edge(disease_node, code, edge_type="disease_to_code", confidence=1.0)

            # -----------------------------------------------
            # 5. REAL CMS parent-child hierarchy edges
            #    Walk the full ancestor chain from CMS, not
            #    just the immediate parent from our hand-coded data.
            # -----------------------------------------------
            real_ancestors = data.get("ancestors", [])  # already fetched from CMS
            for ancestor in real_ancestors:
                # Ensure ancestor node exists
                if not self.graph.has_node(ancestor):
                    if cm.is_valid_item(ancestor):
                        self.graph.add_node(
                            ancestor,
                            node_type="icd10_code",
                            description=cm.get_description(ancestor),
                            category=_infer_category(ancestor),
                            severity="unknown",
                            avg_reimbursement_usd=0,
                            is_billable=cm.is_leaf(ancestor),
                        )
                    else:
                        self.graph.add_node(
                            ancestor,
                            node_type="icd10_block",
                            description=ancestor,
                            category=_infer_category(code),
                        )

            # Add direct parent → code edge from real CMS
            real_parent = data.get("parent_code")
            if real_parent and real_parent != code:
                if not self.graph.has_node(real_parent):
                    self.graph.add_node(
                        real_parent,
                        node_type="icd10_code",
                        description=cm.get_description(real_parent) if cm.is_valid_item(real_parent) else real_parent,
                        category=_infer_category(real_parent),
                        severity="unknown",
                        avg_reimbursement_usd=0,
                    )
                self.graph.add_edge(real_parent, code, edge_type="parent_child")

            # Also add real CMS children of this code (siblings in the subset)
            real_children = data.get("children", [])
            for child_code in real_children:
                if child_code in icd_data:
                    self.graph.add_edge(code, child_code, edge_type="parent_child")

            # -----------------------------------------------
            # 6. REAL CMS excludes1 edges
            #    These come directly from the official CMS tabular,
            #    not our hand-coded list.
            # -----------------------------------------------
            real_excludes = data.get("excludes1", [])   # from CMS via simple_icd_10_cm
            for excl_text in real_excludes:
                # excludes1 entries are text descriptions, not always bare codes.
                # Extract the code portion (e.g. "J18.1" from "J18.1 Lobar pneumonia")
                excl_code = excl_text.split()[0] if excl_text else ""
                if excl_code and cm.is_valid_item(excl_code) and excl_code != code:
                    if not self.graph.has_node(excl_code):
                        self.graph.add_node(
                            excl_code,
                            node_type="icd10_code",
                            description=cm.get_description(excl_code),
                            category=_infer_category(excl_code),
                            severity="unknown",
                            avg_reimbursement_usd=0,
                        )
                    self.graph.add_edge(code, excl_code, edge_type="excludes")
                    self.graph.add_edge(excl_code, code, edge_type="excludes")

    def _estimate_symptom_confidence(
        self,
        symptom: str,
        code: str,
        icd_data: dict
    ) -> float:
        """
        Estimate symptom-to-code confidence based on how many codes
        share this symptom (more unique = higher confidence).
        """
        codes_with_symptom = sum(
            1 for c, d in icd_data.items()
            if symptom in d.get("symptoms", [])
        )
        if codes_with_symptom == 0:
            return 0.5
        # Rarer symptom = higher confidence for this specific code
        base_confidence = 1.0 / codes_with_symptom
        return min(0.95, max(0.1, base_confidence * 3))

    def _estimate_finding_confidence(self, finding: str, code: str) -> float:
        """
        Estimate image finding confidence based on clinical specificity.
        Positive findings (consolidation, cardiomegaly) have higher confidence
        than normal/negative findings.
        """
        high_confidence_findings = {
            "lobar_consolidation", "bilateral_infiltrates", "cardiomegaly",
            "pleural_effusion", "pneumothorax", "air_bronchograms",
            "kerley_b_lines", "pulmonary_edema", "hyperinflation",
            "ground_glass_opacities", "free_air_under_diaphragm"
        }
        low_confidence_findings = {
            "normal_chest", "clear_lung_fields", "no_consolidation",
            "no_acute_findings", "no_free_air", "no_relevant_findings"
        }
        if finding in high_confidence_findings:
            return 0.85
        elif finding in low_confidence_findings:
            return 0.25
        else:
            return 0.55

    @property
    def backend(self) -> str:
        """Returns the active backend: 'neo4j' or 'networkx'."""
        return "neo4j" if self._use_neo4j else "networkx"

    # =========================================================
    # PUBLIC API METHODS
    # Each method routes to Neo4j (Cypher) or NetworkX based on
    # which backend is active.
    # =========================================================

    def get_candidate_codes(
        self,
        symptoms: list,
        image_findings: list
    ) -> list:
        """
        Returns ranked list of candidate ICD-10 codes.
        Neo4j: Cypher MATCH traversal with SUM of confidence scores.
        NetworkX: in-memory graph traversal.
        """
        if self._use_neo4j:
            try:
                return _gdb.neo4j_get_candidate_codes(symptoms, image_findings)
            except Exception as e:
                logger.warning(f"Neo4j get_candidate_codes failed ({e}). Using NetworkX.")

        # --- NetworkX fallback ---
        code_scores = {}  # type: dict

        for symptom in symptoms:
            symptom_node = f"symptom:{symptom}"
            if self.graph.has_node(symptom_node):
                for neighbor in self.graph.successors(symptom_node):
                    if self.graph.nodes[neighbor].get("node_type") == "icd10_code":
                        edge_data = self.graph.edges[symptom_node, neighbor]
                        confidence = edge_data.get("confidence", 0.5)
                        code_scores[neighbor] = code_scores.get(neighbor, 0) + confidence

        for finding in image_findings:
            finding_normalized = finding.lower().replace(" ", "_").replace("-", "_")
            finding_node = f"finding:{finding_normalized}"
            if self.graph.has_node(finding_node):
                for neighbor in self.graph.successors(finding_node):
                    if self.graph.nodes[neighbor].get("node_type") == "icd10_code":
                        edge_data = self.graph.edges[finding_node, neighbor]
                        confidence = edge_data.get("confidence", 0.5)
                        code_scores[neighbor] = code_scores.get(neighbor, 0) + (confidence * 1.5)

        if not code_scores:
            # Fallback: return all codes sorted by reimbursement for demonstration
            all_codes = [
                code for code, attrs in self.graph.nodes(data=True)
                if attrs.get("node_type") == "icd10_code"
            ]
            return [
                {
                    "code": c,
                    "description": self.graph.nodes[c].get("description", ""),
                    "category": self.graph.nodes[c].get("category", ""),
                    "score": 0.1,
                    "confidence": 0.1
                }
                for c in all_codes[:10]
            ]

        # Sort by score descending
        sorted_codes = sorted(code_scores.items(), key=lambda x: x[1], reverse=True)

        # Normalize scores to 0-1 range
        max_score = sorted_codes[0][1] if sorted_codes else 1
        results = []
        for code, score in sorted_codes[:15]:  # Top 15 candidates
            node_attrs = self.graph.nodes[code]
            normalized_confidence = min(1.0, score / max_score)
            results.append({
                "code": code,
                "description": node_attrs.get("description", ""),
                "category": node_attrs.get("category", ""),
                "severity": node_attrs.get("severity", ""),
                "avg_reimbursement_usd": node_attrs.get("avg_reimbursement_usd", 0),
                "score": round(score, 3),
                "confidence": round(normalized_confidence, 3)
            })

        return results

    def get_exclusions(self, code: str) -> list:
        """Returns codes that cannot be used with this code (CMS Excludes1)."""
        if self._use_neo4j:
            try:
                return _gdb.neo4j_get_exclusions(code)
            except Exception as e:
                logger.warning(f"Neo4j get_exclusions failed ({e}). Using NetworkX.")

        # --- NetworkX fallback ---
        if not self.graph.has_node(code):
            return []
        exclusions = []
        for neighbor in self.graph.successors(code):
            edge_data = self.graph.edges[code, neighbor]
            if edge_data.get("edge_type") == "excludes":
                node_attrs = self.graph.nodes[neighbor]
                if node_attrs.get("node_type") == "icd10_code":
                    exclusions.append({
                        "code": neighbor,
                        "description": node_attrs.get("description", ""),
                        "reason": f"{code} and {neighbor} cannot be coded together"
                    })
        return exclusions

    def get_related_codes(self, code: str) -> dict:
        """
        Returns parent, children, siblings, and ancestors.
        Neo4j: Cypher PARENT_OF traversal.
        Fallback: real CMS simple_icd_10_cm calls.
        """
        if self._use_neo4j:
            try:
                return _gdb.neo4j_get_related_codes(code)
            except Exception as e:
                logger.warning(f"Neo4j get_related_codes failed ({e}). Using CMS library.")

        # --- CMS library fallback ---
        if not cm.is_valid_item(code):
            return {"parent": None, "children": [], "siblings": [], "ancestors": []}

        # Real parent from CMS
        real_parent_code = cm.get_parent(code)
        parent = None
        if real_parent_code and cm.is_valid_item(real_parent_code):
            parent = {
                "code": real_parent_code,
                "description": cm.get_description(real_parent_code)
            }

        # Real children from CMS
        children = []
        try:
            for child in cm.get_children(code):
                children.append({
                    "code": child,
                    "description": cm.get_description(child)
                })
        except ValueError:
            pass

        # Real siblings — other children of the same parent from CMS
        siblings = []
        if real_parent_code:
            try:
                for sibling in cm.get_children(real_parent_code):
                    if sibling != code:
                        siblings.append({
                            "code": sibling,
                            "description": cm.get_description(sibling)
                        })
            except ValueError:
                pass

        # Full ancestor chain from CMS
        ancestors = [
            {"code": a, "description": cm.get_description(a) if cm.is_valid_item(a) else a}
            for a in cm.get_ancestors(code)
        ]

        return {
            "parent": parent,
            "children": children,
            "siblings": siblings,
            "ancestors": ancestors
        }

    @staticmethod
    def _extract_codes_from_excludes_text(entries: list) -> set:
        """
        CMS stores excludes1 as natural language with codes embedded in
        parentheses: e.g. 'influenza (J09.X2, J10.1)' or 'infection NOS (J22)'.
        Extracts all bare code strings using regex.
        """
        found = set()
        # Matches ICD-10-CM codes: letter + digits + optional dot + alphanumeric
        pattern = re.compile(r'\b([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b')
        for text in entries:
            found.update(pattern.findall(text))
        return found

    def _get_inherited_excludes(self, code: str) -> set:
        """
        CMS excludes1 rules apply at category/block level and are inherited
        by all descendant codes. Walk up the ancestor chain and collect all
        excludes1 codes from the real CMS tabular.
        """
        all_excl = set()
        if not cm.is_valid_item(code):
            return all_excl
        # Include the code itself + all its ancestors
        items_to_check = [code] + cm.get_ancestors(code)
        for item in items_to_check:
            try:
                raw = cm.get_excludes1(item)
                all_excl.update(self._extract_codes_from_excludes_text(raw))
            except ValueError:
                pass
        return all_excl

    def validate_code_combination(self, codes: list) -> dict:
        """
        Checks if multiple codes can be used together.
        Neo4j: Cypher EXCLUDES1 + PARENT_OF query.
        Fallback: inherited CMS excludes1 via simple_icd_10_cm with regex parsing.
        """
        if self._use_neo4j:
            try:
                return _gdb.neo4j_validate_code_combination(codes)
            except Exception as e:
                logger.warning(f"Neo4j validate_code_combination failed ({e}). Using CMS library.")

        # --- CMS library fallback ---
        conflicts = []

        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                code1 = codes[i]
                code2 = codes[j]

                if cm.is_valid_item(code1) and cm.is_valid_item(code2):
                    # Collect excludes1 for both codes + all their ancestors
                    excl1_codes = self._get_inherited_excludes(code1)
                    excl2_codes = self._get_inherited_excludes(code2)

                    if code2 in excl1_codes:
                        conflicts.append({
                            "code1": code1,
                            "code2": code2,
                            "reason": f"CMS Excludes1 (inherited): {code1} cannot be coded with {code2}"
                        })
                    elif code1 in excl2_codes:
                        conflicts.append({
                            "code1": code2,
                            "code2": code1,
                            "reason": f"CMS Excludes1 (inherited): {code2} cannot be coded with {code1}"
                        })
                    elif cm.is_ancestor(code1, code2) or cm.is_ancestor(code2, code1):
                        conflicts.append({
                            "code1": code1,
                            "code2": code2,
                            "reason": "Hierarchy conflict: one code is an ancestor of the other (redundant coding)"
                        })
                else:
                    # Fall back to graph edges for non-CMS codes
                    if self.graph.has_edge(code1, code2):
                        edge = self.graph.edges[code1, code2]
                        if edge.get("edge_type") == "excludes":
                            conflicts.append({
                                "code1": code1,
                                "code2": code2,
                                "reason": f"Excludes rule: {code1} cannot be coded with {code2}"
                            })

        if conflicts:
            reasons = "; ".join(c["reason"] for c in conflicts)
            return {"valid": False, "conflicts": conflicts, "reason": reasons}

        return {"valid": True, "conflicts": [], "reason": "All codes can be used together"}

    def get_severity_path(self, code: str) -> dict:
        """
        Returns all codes in same category ordered by severity.
        Neo4j: ORDER BY CASE severity Cypher query.
        Fallback: NetworkX node attribute scan.
        """
        if self._use_neo4j:
            try:
                return _gdb.neo4j_get_severity_path(code)
            except Exception as e:
                logger.warning(f"Neo4j get_severity_path failed ({e}). Using NetworkX.")

        # --- NetworkX fallback ---
        if not self.graph.has_node(code):
            return {"codes": [], "current_index": -1}
        current_attrs = self.graph.nodes[code]
        category = current_attrs.get("category")
        severity_order = {"mild": 1, "moderate": 2, "severe": 3}
        category_codes = []
        for node, attrs in self.graph.nodes(data=True):
            if (attrs.get("node_type") == "icd10_code" and attrs.get("category") == category):
                category_codes.append({
                    "code": node,
                    "description": attrs.get("description", ""),
                    "severity": attrs.get("severity", "mild"),
                    "severity_rank": severity_order.get(attrs.get("severity", "mild"), 1),
                    "avg_reimbursement_usd": attrs.get("avg_reimbursement_usd", 0)
                })
        category_codes.sort(key=lambda x: x["severity_rank"])
        current_index = next((i for i, c in enumerate(category_codes) if c["code"] == code), -1)
        return {"codes": category_codes, "current_code": code,
                "current_index": current_index, "category": category}

    def calculate_financial_gap(self, code1: str, code2: str) -> int:
        """
        Returns reimbursement difference (code1 - code2).
        Neo4j: property lookup via Cypher.
        Fallback: NetworkX node attribute lookup.
        """
        if self._use_neo4j:
            try:
                return _gdb.neo4j_calculate_financial_gap(code1, code2)
            except Exception as e:
                logger.warning(f"Neo4j calculate_financial_gap failed ({e}). Using NetworkX.")

        # --- NetworkX fallback ---
        if not self.graph.has_node(code1) or not self.graph.has_node(code2):
            return 0
        reimb1 = self.graph.nodes[code1].get("avg_reimbursement_usd", 0)
        reimb2 = self.graph.nodes[code2].get("avg_reimbursement_usd", 0)
        return reimb1 - reimb2

    def get_code_info(self, code: str) -> Optional[dict]:
        """Returns full node attributes for a given code."""
        if self._use_neo4j:
            try:
                info = _gdb.neo4j_get_code_info(code)
                if info:
                    info["related"] = self.get_related_codes(code)
                    info["exclusions"] = self.get_exclusions(code)
                    return info
            except Exception as e:
                logger.warning(f"Neo4j get_code_info failed ({e}). Using NetworkX.")

        # --- NetworkX fallback ---
        if not self.graph.has_node(code):
            return None
        attrs = self.graph.nodes[code]
        if attrs.get("node_type") != "icd10_code":
            return None
        return {
            "code": code,
            "description": attrs.get("description", ""),
            "category": attrs.get("category", ""),
            "severity": attrs.get("severity", ""),
            "avg_reimbursement_usd": attrs.get("avg_reimbursement_usd", 0),
            "related": self.get_related_codes(code),
            "exclusions": self.get_exclusions(code)
        }

    def visualize_subgraph(self, code: str, depth: int = 1):
        """
        Prints the neighborhood of a given ICD-10 code for debugging.
        Shows all directly connected nodes within `depth` hops.
        """
        if not self.graph.has_node(code):
            print(f"Code {code} not found in graph.")
            return

        print(f"\n{'='*60}")
        print(f"Subgraph for: {code}")
        attrs = self.graph.nodes[code]
        print(f"Description: {attrs.get('description', 'N/A')}")
        print(f"Category: {attrs.get('category', 'N/A')}")
        print(f"Severity: {attrs.get('severity', 'N/A')}")
        print(f"Avg Reimbursement: ${attrs.get('avg_reimbursement_usd', 0):,}")
        print(f"{'='*60}")

        print("\nIncoming connections (→ this code):")
        for pred in self.graph.predecessors(code):
            edge = self.graph.edges[pred, code]
            edge_type = edge.get("edge_type", "unknown")
            confidence = edge.get("confidence", "")
            conf_str = f" [confidence: {confidence:.2f}]" if confidence else ""
            pred_type = self.graph.nodes[pred].get("node_type", "?")
            print(f"  [{pred_type}] {pred} --({edge_type}){conf_str}--> {code}")

        print("\nOutgoing connections (this code →):")
        for succ in self.graph.successors(code):
            edge = self.graph.edges[code, succ]
            edge_type = edge.get("edge_type", "unknown")
            succ_type = self.graph.nodes[succ].get("node_type", "?")
            print(f"  {code} --({edge_type})--> [{succ_type}] {succ}")

        print(f"\nActive backend: {self.backend.upper()}")
        print(f"NetworkX nodes: {self.graph.number_of_nodes()}")
        print(f"NetworkX edges: {self.graph.number_of_edges()}")
        if self._use_neo4j:
            try:
                stats = _gdb.get_graph_stats()
                print(f"Neo4j stats: {stats}")
            except Exception:
                pass


# Module-level singleton instance
_graph_instance: Optional[ICD10ContextGraph] = None


def get_graph() -> ICD10ContextGraph:
    """Returns the singleton context graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = ICD10ContextGraph()
    return _graph_instance


# Convenience module-level functions
def get_candidate_codes(symptoms: list, image_findings: list) -> list:
    return get_graph().get_candidate_codes(symptoms, image_findings)


def get_exclusions(code: str) -> list:
    return get_graph().get_exclusions(code)


def get_related_codes(code: str) -> dict:
    return get_graph().get_related_codes(code)


def validate_code_combination(codes: list) -> dict:
    return get_graph().validate_code_combination(codes)


def get_severity_path(code: str) -> dict:
    return get_graph().get_severity_path(code)


def calculate_financial_gap(code1: str, code2: str) -> int:
    return get_graph().calculate_financial_gap(code1, code2)


def visualize_subgraph(code: str):
    return get_graph().visualize_subgraph(code)


if __name__ == "__main__":
    print("Building ICD-10 Context Graph...")
    g = ICD10ContextGraph()
    print(f"Graph built: {g.graph.number_of_nodes()} nodes, {g.graph.number_of_edges()} edges")

    print("\n--- Testing get_candidate_codes ---")
    candidates = g.get_candidate_codes(
        symptoms=["fever", "productive_cough", "dyspnea"],
        image_findings=["lobar_consolidation", "air_bronchograms"]
    )
    print("Top 5 candidate codes:")
    for c in candidates[:5]:
        print(f"  {c['code']}: {c['description']} (confidence: {c['confidence']})")

    print("\n--- Testing get_exclusions ---")
    exclusions = g.get_exclusions("J18.1")
    print(f"Exclusions for J18.1: {[e['code'] for e in exclusions]}")

    print("\n--- Testing validate_code_combination ---")
    result = g.validate_code_combination(["J06.9", "J18.1"])
    print(f"J06.9 + J18.1 valid: {result['valid']}, reason: {result['reason']}")

    print("\n--- Testing calculate_financial_gap ---")
    gap = g.calculate_financial_gap("J18.1", "J06.9")
    print(f"Financial gap J18.1 vs J06.9: ${gap:,}")

    print("\n--- Visualizing subgraph for J18.1 ---")
    g.visualize_subgraph("J18.1")
