"""
Neo4j Graph Database Module
Persists the ICD-10 knowledge graph in Neo4j.

Node labels:
  :ICD10Code     — billable codes + clinically augmented codes
  :ICD10Block    — non-billable parent blocks (J18, J09-J18, chapter 10)
  :Symptom       — clinical symptom (curated augmentation)
  :ImageFinding  — radiology finding (curated augmentation)
  :Disease       — category grouping (respiratory, cardiovascular, etc.)

Relationship types:
  (:Symptom)-[:INDICATES {confidence}]->(:ICD10Code)
  (:ImageFinding)-[:SUGGESTS {confidence}]->(:ICD10Code)
  (:ICD10Code|ICD10Block)-[:PARENT_OF]->(:ICD10Code)
  (:ICD10Code)-[:EXCLUDES1]->(:ICD10Code)
  (:Disease)-[:INCLUDES]->(:ICD10Code)

Data sources:
  - Node attributes + symptoms + image_findings: icd10_data.CLINICAL_AUGMENTATION
  - Real descriptions + hierarchy + excludes:    simple_icd_10_cm (CMS FY2024)
"""

import os
import re
import logging
from typing import Optional

import simple_icd_10_cm as cm
from neo4j import GraphDatabase
from dotenv import load_dotenv

from icd10_data import ICD10_DATA, CLINICAL_AUGMENTATION, _infer_category

load_dotenv()

logger = logging.getLogger("graph_db")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "medgemma123")


# =========================================================
# DRIVER SINGLETON
# =========================================================

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info(f"Neo4j driver connected to {NEO4J_URI}")
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def is_neo4j_available() -> bool:
    """Returns True if Neo4j is reachable."""
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return True
    except Exception as e:
        logger.warning(f"Neo4j not available: {e}")
        return False


# =========================================================
# SCHEMA SETUP (constraints + indexes)
# =========================================================

def setup_schema(session):
    """Create uniqueness constraints and indexes."""
    schema_statements = [
        "CREATE CONSTRAINT icd10code_code IF NOT EXISTS FOR (n:ICD10Code) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT icd10block_code IF NOT EXISTS FOR (n:ICD10Block) REQUIRE n.code IS UNIQUE",
        "CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT finding_name IF NOT EXISTS FOR (n:ImageFinding) REQUIRE n.name IS UNIQUE",
        "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
        "CREATE INDEX icd10code_category IF NOT EXISTS FOR (n:ICD10Code) ON (n.category)",
        "CREATE INDEX icd10code_severity IF NOT EXISTS FOR (n:ICD10Code) ON (n.severity)",
    ]
    for stmt in schema_statements:
        try:
            session.run(stmt)
        except Exception as e:
            logger.debug(f"Schema statement skipped (may already exist): {e}")
    logger.info("Neo4j schema constraints and indexes ensured.")


# =========================================================
# HELPER — extract ICD codes from CMS excludes1 text
# =========================================================

_CODE_PATTERN = re.compile(r'\b([A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b')


def _extract_codes_from_excludes_text(entries: list) -> set:
    """
    CMS excludes1 entries are natural language with codes in parentheses:
      'influenza (J09.X2, J10.1)' → {'J09.X2', 'J10.1'}
      'infection NOS (J22)'       → {'J22'}
    """
    found = set()
    for text in entries:
        found.update(_CODE_PATTERN.findall(text))
    return found


def _get_inherited_excludes_codes(code: str) -> set:
    """Walk up the CMS ancestor chain and collect all excludes1 codes."""
    all_excl = set()
    items = [code] + (cm.get_ancestors(code) if cm.is_valid_item(code) else [])
    for item in items:
        try:
            raw = cm.get_excludes1(item)
            all_excl.update(_extract_codes_from_excludes_text(raw))
        except ValueError:
            pass
    return all_excl


# =========================================================
# GRAPH SEEDING
# =========================================================

def seed_graph(session):
    """
    Load the full ICD-10 knowledge graph into Neo4j.
    Uses MERGE so re-running is idempotent (safe to call multiple times).
    """
    logger.info("Seeding Neo4j graph from CMS ICD-10-CM data...")

    # -------------------------------------------------------
    # 1. Create/update :ICD10Code nodes for augmented codes
    # -------------------------------------------------------
    logger.info("  Creating ICD10Code nodes...")
    for code, data in ICD10_DATA.items():
        session.run("""
            MERGE (c:ICD10Code {code: $code})
            SET c.description       = $description,
                c.category          = $category,
                c.severity          = $severity,
                c.avg_reimbursement_usd = $reimbursement,
                c.is_billable       = $is_billable,
                c.has_clinical_data = true
        """, {
            "code": code,
            "description": data["description"],
            "category": data["category"],
            "severity": data["severity"],
            "reimbursement": data["avg_reimbursement_usd"],
            "is_billable": data.get("is_billable", True),
        })

    logger.info(f"  Created {len(ICD10_DATA)} ICD10Code nodes.")

    # -------------------------------------------------------
    # 2. Create :Symptom nodes + :INDICATES relationships
    # -------------------------------------------------------
    logger.info("  Creating Symptom nodes and INDICATES edges...")
    symptom_count = 0
    for code, data in ICD10_DATA.items():
        for symptom in data.get("symptoms", []):
            # Count how many augmented codes share this symptom (for confidence)
            codes_with_symptom = sum(
                1 for d in ICD10_DATA.values()
                if symptom in d.get("symptoms", [])
            )
            confidence = min(0.95, max(0.1, 3.0 / max(codes_with_symptom, 1)))

            session.run("""
                MERGE (s:Symptom {name: $symptom})
                WITH s
                MATCH (c:ICD10Code {code: $code})
                MERGE (s)-[r:INDICATES]->(c)
                SET r.confidence = $confidence
            """, {"symptom": symptom, "code": code, "confidence": confidence})
            symptom_count += 1
    logger.info(f"  Created {symptom_count} INDICATES relationships.")

    # -------------------------------------------------------
    # 3. Create :ImageFinding nodes + :SUGGESTS relationships
    # -------------------------------------------------------
    logger.info("  Creating ImageFinding nodes and SUGGESTS edges...")
    HIGH_CONF = {
        "lobar_consolidation", "bilateral_infiltrates", "cardiomegaly",
        "pleural_effusion", "pneumothorax", "air_bronchograms",
        "kerley_b_lines", "pulmonary_edema", "hyperinflation",
        "ground_glass_opacities", "free_air_under_diaphragm"
    }
    LOW_CONF = {
        "normal_chest", "clear_lung_fields", "no_consolidation",
        "no_acute_findings", "no_free_air", "no_relevant_findings"
    }
    finding_count = 0
    for code, data in ICD10_DATA.items():
        for finding in data.get("image_findings", []):
            confidence = 0.85 if finding in HIGH_CONF else (0.25 if finding in LOW_CONF else 0.55)
            session.run("""
                MERGE (f:ImageFinding {name: $finding})
                WITH f
                MATCH (c:ICD10Code {code: $code})
                MERGE (f)-[r:SUGGESTS]->(c)
                SET r.confidence = $confidence
            """, {"finding": finding, "code": code, "confidence": confidence})
            finding_count += 1
    logger.info(f"  Created {finding_count} SUGGESTS relationships.")

    # -------------------------------------------------------
    # 4. Create :Disease (category) nodes + :INCLUDES edges
    # -------------------------------------------------------
    logger.info("  Creating Disease category nodes...")
    for code, data in ICD10_DATA.items():
        category = data["category"]
        session.run("""
            MERGE (d:Disease {name: $category})
            WITH d
            MATCH (c:ICD10Code {code: $code})
            MERGE (d)-[:INCLUDES]->(c)
        """, {"category": category, "code": code})

    # -------------------------------------------------------
    # 5. Real CMS hierarchy — PARENT_OF edges
    #    Walk up full ancestor chain from CMS for each code.
    # -------------------------------------------------------
    logger.info("  Creating real CMS PARENT_OF hierarchy edges...")
    hierarchy_count = 0
    for code in ICD10_DATA:
        if not cm.is_valid_item(code):
            continue

        real_parent = cm.get_parent(code)
        if not real_parent:
            continue

        # Decide node label: billable codes → :ICD10Code, blocks → :ICD10Block
        parent_is_leaf = cm.is_leaf(real_parent) if cm.is_valid_item(real_parent) else False
        parent_desc = cm.get_description(real_parent) if cm.is_valid_item(real_parent) else real_parent
        parent_label = "ICD10Code" if parent_is_leaf else "ICD10Block"

        session.run(f"""
            MERGE (p:{parent_label} {{code: $parent}})
            SET p.description = $desc,
                p.category    = $category,
                p.is_billable = $is_leaf
            WITH p
            MATCH (c:ICD10Code {{code: $code}})
            MERGE (p)-[:PARENT_OF]->(c)
        """, {
            "parent": real_parent,
            "desc": parent_desc,
            "category": _infer_category(real_parent),
            "is_leaf": parent_is_leaf,
            "code": code,
        })
        hierarchy_count += 1

        # Also connect ancestors in chain so full path exists
        ancestors = cm.get_ancestors(code)
        for i, ancestor in enumerate(ancestors):
            child_in_chain = code if i == 0 else ancestors[i - 1]
            child_is_leaf = cm.is_leaf(child_in_chain) if cm.is_valid_item(child_in_chain) else False
            child_label = "ICD10Code" if child_is_leaf else "ICD10Block"
            anc_is_leaf = cm.is_leaf(ancestor) if cm.is_valid_item(ancestor) else False
            anc_label = "ICD10Code" if anc_is_leaf else "ICD10Block"

            if ancestor != child_in_chain:
                session.run(f"""
                    MERGE (a:{anc_label} {{code: $ancestor}})
                    SET a.description = $adesc,
                        a.category    = $acat
                    WITH a
                    MERGE (ch:{child_label} {{code: $child}})
                    SET ch.description = $cdesc
                    WITH a, ch
                    MERGE (a)-[:PARENT_OF]->(ch)
                """, {
                    "ancestor": ancestor,
                    "adesc": cm.get_description(ancestor) if cm.is_valid_item(ancestor) else ancestor,
                    "acat": _infer_category(ancestor),
                    "child": child_in_chain,
                    "cdesc": cm.get_description(child_in_chain) if cm.is_valid_item(child_in_chain) else child_in_chain,
                })
                hierarchy_count += 1

    logger.info(f"  Created {hierarchy_count} PARENT_OF relationships from real CMS.")

    # -------------------------------------------------------
    # 6. Real CMS EXCLUDES1 edges
    #    Parse inherited excludes from ancestor chain.
    # -------------------------------------------------------
    logger.info("  Creating real CMS EXCLUDES1 edges...")
    excl_count = 0
    for code in ICD10_DATA:
        excluded_codes = _get_inherited_excludes_codes(code)
        for excl_code in excluded_codes:
            if excl_code == code:
                continue
            if not cm.is_valid_item(excl_code):
                continue
            # Only link if excluded code also has a node (in our subset or created as ancestor)
            excl_desc = cm.get_description(excl_code)
            excl_label = "ICD10Code" if cm.is_leaf(excl_code) else "ICD10Block"
            session.run(f"""
                MATCH (c:ICD10Code {{code: $code}})
                MERGE (e:{excl_label} {{code: $excl}})
                SET e.description = $desc,
                    e.category    = $cat
                MERGE (c)-[:EXCLUDES1]->(e)
            """, {
                "code": code,
                "excl": excl_code,
                "desc": excl_desc,
                "cat": _infer_category(excl_code),
            })
            excl_count += 1

    logger.info(f"  Created {excl_count} EXCLUDES1 relationships from real CMS.")
    logger.info("Neo4j seeding complete.")


def ensure_graph_seeded():
    """
    Check if the graph has already been seeded.
    If not, seed it. Safe to call on every startup.
    """
    driver = get_driver()
    with driver.session() as session:
        setup_schema(session)
        result = session.run("MATCH (c:ICD10Code) RETURN count(c) AS n")
        count = result.single()["n"]
        if count == 0:
            logger.info("Neo4j graph is empty. Seeding now...")
            seed_graph(session)
        else:
            logger.info(f"Neo4j graph already seeded ({count} ICD10Code nodes found). Skipping.")


# =========================================================
# CYPHER QUERY FUNCTIONS
# =========================================================

def neo4j_get_candidate_codes(symptoms: list, image_findings: list) -> list:
    """
    Traverse graph from Symptom + ImageFinding nodes to ICD10Code nodes.
    Returns ranked list by combined confidence score.
    Image findings are weighted 1.5x (more objective than symptoms).
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            // Score from symptoms
            OPTIONAL MATCH (s:Symptom)-[r:INDICATES]->(c:ICD10Code)
            WHERE s.name IN $symptoms
            WITH c, COALESCE(SUM(r.confidence), 0) AS symptom_score

            // Score from image findings (weighted 1.5x)
            OPTIONAL MATCH (f:ImageFinding)-[r2:SUGGESTS]->(c)
            WHERE f.name IN $findings
            WITH c, symptom_score, COALESCE(SUM(r2.confidence * 1.5), 0) AS finding_score

            WHERE c IS NOT NULL
            WITH c, (symptom_score + finding_score) AS total_score
            ORDER BY total_score DESC
            LIMIT 15

            RETURN c.code          AS code,
                   c.description   AS description,
                   c.category      AS category,
                   c.severity      AS severity,
                   c.avg_reimbursement_usd AS avg_reimbursement_usd,
                   total_score     AS score
        """, {"symptoms": symptoms, "findings": image_findings})

        rows = result.data()
        if not rows:
            return []

        max_score = rows[0]["score"] if rows[0]["score"] else 1
        return [
            {
                "code": r["code"],
                "description": r["description"] or "",
                "category": r["category"] or "",
                "severity": r["severity"] or "",
                "avg_reimbursement_usd": r["avg_reimbursement_usd"] or 0,
                "score": round(float(r["score"]), 3),
                "confidence": round(min(1.0, float(r["score"]) / max(max_score, 0.001)), 3),
            }
            for r in rows
        ]


def neo4j_get_exclusions(code: str) -> list:
    """
    Returns codes that cannot be used with this code
    (direct EXCLUDES1 relationships stored in Neo4j from real CMS data).
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c {code: $code})-[:EXCLUDES1]->(e)
            RETURN e.code AS code, e.description AS description
        """, {"code": code})
        return [
            {
                "code": r["code"],
                "description": r["description"] or "",
                "reason": f"{code} and {r['code']} cannot be coded together (CMS Excludes1)"
            }
            for r in result.data()
        ]


def neo4j_get_related_codes(code: str) -> dict:
    """
    Returns parent, children, siblings, and ancestors from the real CMS
    hierarchy stored in Neo4j as PARENT_OF edges.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            // Parent
            OPTIONAL MATCH (parent)-[:PARENT_OF]->(c {code: $code})
            WITH collect({code: parent.code, description: parent.description})[0] AS parent_node

            // Children
            OPTIONAL MATCH (c2 {code: $code})-[:PARENT_OF]->(child)
            WITH parent_node, collect({code: child.code, description: child.description}) AS children

            // Siblings (other children of the same parent)
            OPTIONAL MATCH (parent2)-[:PARENT_OF]->(c3 {code: $code})
            OPTIONAL MATCH (parent2)-[:PARENT_OF]->(sibling)
            WHERE sibling.code <> $code
            WITH parent_node, children,
                 collect({code: sibling.code, description: sibling.description}) AS siblings

            // Full ancestor chain
            OPTIONAL MATCH path = (ancestor)-[:PARENT_OF*]->(leaf {code: $code})
            WITH parent_node, children, siblings,
                 [n IN nodes(path) WHERE n.code <> $code |
                  {code: n.code, description: n.description}] AS ancestors

            RETURN parent_node AS parent,
                   children,
                   siblings,
                   COALESCE(ancestors, []) AS ancestors
        """, {"code": code})

        row = result.single()
        if not row:
            return {"parent": None, "children": [], "siblings": [], "ancestors": []}

        # Clean out None entries that come from unmatched OPTIONAL MATCHes
        def clean(lst):
            return [x for x in lst if x and x.get("code")]

        parent = row["parent"] if (row["parent"] and row["parent"].get("code")) else None
        return {
            "parent": parent,
            "children": clean(row["children"]),
            "siblings": clean(row["siblings"]),
            "ancestors": clean(row["ancestors"]),
        }


def neo4j_validate_code_combination(codes: list) -> dict:
    """
    Checks if a list of codes can be used together by querying EXCLUDES1
    and PARENT_OF relationships in Neo4j.
    """
    if len(codes) < 2:
        return {"valid": True, "conflicts": [], "reason": "Only one code provided"}

    driver = get_driver()
    conflicts = []

    with driver.session() as session:
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                code1, code2 = codes[i], codes[j]

                # Check EXCLUDES1 in both directions
                result = session.run("""
                    OPTIONAL MATCH (a {code: $c1})-[:EXCLUDES1]->(b {code: $c2})
                    OPTIONAL MATCH (b2 {code: $c2})-[:EXCLUDES1]->(a2 {code: $c1})
                    OPTIONAL MATCH (p)-[:PARENT_OF]->(ch {code: $c2})
                        WHERE p.code = $c1
                    OPTIONAL MATCH (p2)-[:PARENT_OF]->(ch2 {code: $c1})
                        WHERE p2.code = $c2
                    RETURN
                        a IS NOT NULL  AS c1_excludes_c2,
                        b2 IS NOT NULL AS c2_excludes_c1,
                        p IS NOT NULL  AS c1_is_parent_of_c2,
                        p2 IS NOT NULL AS c2_is_parent_of_c1
                """, {"c1": code1, "c2": code2})

                row = result.single()
                if row:
                    if row["c1_excludes_c2"]:
                        conflicts.append({
                            "code1": code1, "code2": code2,
                            "reason": f"CMS Excludes1: {code1} cannot be coded with {code2}"
                        })
                    elif row["c2_excludes_c1"]:
                        conflicts.append({
                            "code1": code2, "code2": code1,
                            "reason": f"CMS Excludes1: {code2} cannot be coded with {code1}"
                        })
                    elif row["c1_is_parent_of_c2"] or row["c2_is_parent_of_c1"]:
                        conflicts.append({
                            "code1": code1, "code2": code2,
                            "reason": f"Hierarchy conflict: one code is an ancestor of the other"
                        })

    if conflicts:
        return {
            "valid": False,
            "conflicts": conflicts,
            "reason": "; ".join(c["reason"] for c in conflicts)
        }
    return {"valid": True, "conflicts": [], "reason": "All codes can be used together"}


def neo4j_get_severity_path(code: str) -> dict:
    """
    Returns all codes in the same category ordered by severity.
    Used to detect upcoding/downcoding paths.
    """
    driver = get_driver()
    with driver.session() as session:
        # Get this code's category
        cat_result = session.run(
            "MATCH (c:ICD10Code {code: $code}) RETURN c.category AS category",
            {"code": code}
        )
        row = cat_result.single()
        if not row:
            return {"codes": [], "current_index": -1}
        category = row["category"]

        severity_order = {"mild": 1, "moderate": 2, "severe": 3}

        result = session.run("""
            MATCH (c:ICD10Code)
            WHERE c.category = $category AND c.has_clinical_data = true
            RETURN c.code AS code,
                   c.description AS description,
                   c.severity AS severity,
                   c.avg_reimbursement_usd AS avg_reimbursement_usd
            ORDER BY
              CASE c.severity
                WHEN 'mild'     THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'severe'   THEN 3
                ELSE 4
              END
        """, {"category": category})

        codes = [
            {
                "code": r["code"],
                "description": r["description"] or "",
                "severity": r["severity"] or "",
                "severity_rank": severity_order.get(r["severity"], 4),
                "avg_reimbursement_usd": r["avg_reimbursement_usd"] or 0,
            }
            for r in result.data()
        ]

        current_index = next((i for i, c in enumerate(codes) if c["code"] == code), -1)
        return {"codes": codes, "current_code": code, "current_index": current_index, "category": category}


def neo4j_calculate_financial_gap(code1: str, code2: str) -> int:
    """Returns reimbursement difference between two codes (code1 - code2)."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            OPTIONAL MATCH (c1:ICD10Code {code: $code1})
            OPTIONAL MATCH (c2:ICD10Code {code: $code2})
            RETURN
                COALESCE(c1.avg_reimbursement_usd, 0) AS r1,
                COALESCE(c2.avg_reimbursement_usd, 0) AS r2
        """, {"code1": code1, "code2": code2})
        row = result.single()
        if not row:
            return 0
        return int(row["r1"]) - int(row["r2"])


def neo4j_get_code_info(code: str) -> Optional[dict]:
    """Returns full node attributes for a given ICD10Code from Neo4j."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:ICD10Code {code: $code})
            RETURN c.code AS code,
                   c.description AS description,
                   c.category AS category,
                   c.severity AS severity,
                   c.avg_reimbursement_usd AS avg_reimbursement_usd,
                   c.is_billable AS is_billable,
                   c.has_clinical_data AS has_clinical_data
        """, {"code": code})
        row = result.single()
        if not row:
            return None
        return dict(row)


def get_graph_stats() -> dict:
    """Returns counts of all node types and relationship types in Neo4j."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run("""
            CALL apoc.meta.stats() YIELD labels, relTypesCount
            RETURN labels, relTypesCount
        """)
        try:
            row = result.single()
            return {"labels": dict(row["labels"]), "relationships": dict(row["relTypesCount"])}
        except Exception:
            # apoc not available — fall back to manual counts
            stats = {}
            for label in ["ICD10Code", "ICD10Block", "Symptom", "ImageFinding", "Disease"]:
                r = session.run(f"MATCH (n:{label}) RETURN count(n) AS n")
                stats[label] = r.single()["n"]
            for rel in ["INDICATES", "SUGGESTS", "PARENT_OF", "EXCLUDES1", "INCLUDES"]:
                r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")
                stats[rel] = r.single()["n"]
            return stats


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    print("Connecting to Neo4j...")
    if not is_neo4j_available():
        print("Neo4j not available. Start it with: docker-compose up neo4j")
        exit(1)

    print("Seeding graph...")
    ensure_graph_seeded()

    print("\nGraph stats:")
    stats = get_graph_stats()
    print(json.dumps(stats, indent=2))

    print("\nCandidate codes for fever + lobar_consolidation:")
    candidates = neo4j_get_candidate_codes(
        symptoms=["fever", "productive_cough", "dyspnea"],
        image_findings=["lobar_consolidation", "air_bronchograms"]
    )
    for c in candidates[:5]:
        print(f"  {c['code']}: {c['description']} (confidence={c['confidence']})")

    print("\nRelated codes for J18.1:")
    related = neo4j_get_related_codes("J18.1")
    print(f"  parent: {related['parent']}")
    print(f"  siblings: {[s['code'] for s in related['siblings']]}")

    print("\nValidate J06.9 + J22:")
    v = neo4j_validate_code_combination(["J06.9", "J22"])
    print(f"  valid={v['valid']}, reason={v['reason']}")

    print("\nFinancial gap J18.1 vs J06.9:")
    gap = neo4j_calculate_financial_gap("J18.1", "J06.9")
    print(f"  ${gap:,}")
