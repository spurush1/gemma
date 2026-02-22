"""
Causal Rules Seeder
===================
Automatically generates causal_rules.yaml entries for new ICD-10 codes using:
  1. MedGemma (primary) — prompts the model for visual requirements
  2. UMLS API (optional) — enriches with SNOMED morphology data
  3. Merge strategy — where both agree, auto-validates the entry

Usage:
  # Seed specific codes
  python scripts/seed_causal_rules.py --codes J15.1 J15.6 J84.10

  # Seed from a file (one code per line)
  python scripts/seed_causal_rules.py --codes-file scripts/target_codes.txt

  # Seed top-N high-reimbursement codes not yet in rules file
  python scripts/seed_causal_rules.py --top 50

  # Dry run — print what would be added without writing
  python scripts/seed_causal_rules.py --codes J15.1 --dry-run

  # Use UMLS API enrichment (requires UMLS_API_KEY in .env)
  python scripts/seed_causal_rules.py --codes J15.1 --umls

Output:
  Appends new entries to backend/causal_rules.yaml
  Existing entries are NEVER overwritten (safe to re-run)

Confidence tiers in output:
  source: medgemma_generated, validated: false
    → -0.15 penalty at runtime until a clinician sets validated: true
  source: umls_snomed+medgemma, validated: false (both agree)
    → same penalty but higher inherent reliability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
RULES_FILE  = BACKEND_DIR / "causal_rules.yaml"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

MEDGEMMA_API_KEY  = os.getenv("MEDGEMMA_API_KEY", "")
MEDGEMMA_ENDPOINT = os.getenv("MEDGEMMA_ENDPOINT",
    "https://router.huggingface.co/featherless-ai/v1/chat/completions")
MEDGEMMA_MODEL    = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
UMLS_API_KEY      = os.getenv("UMLS_API_KEY", "")

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

import httpx
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("seed_causal_rules")


# ── MEDGEMMA SEEDING ──────────────────────────────────────────────────────────

SEED_SYSTEM = (
    "You are a clinical coding expert building a fraud-detection knowledge base. "
    "Return compact single-line JSON only. No extra text."
)

SEED_USER = """ICD-10 code {code} — {description}

For this medical condition, return a single-line JSON object:
{{"visual_required_any":["finding1","finding2"],"visual_required_all":["finding3"],"visual_supporting":["finding4"],"fraud_differential":"X00.0","minimum_causal_score":0.6,"voice_keywords":["symptom1","symptom2"],"anatomical_location":"organ/region","causal_necessity":"one sentence clinical why"}}

Rules:
- visual_required_any: chest X-ray findings where ANY ONE confirms this diagnosis (OR logic)
- visual_required_all: findings ALL of which must be present together (AND logic) — use [] if none
- visual_supporting: findings that support but are not alone sufficient — use [] if none
- fraud_differential: the single ICD-10 code most commonly miscoded instead of this one
- minimum_causal_score: 0.8 for high-reimbursement rare codes, 0.6 for common ones, 0.3 for non-imaging
- voice_keywords: symptoms a patient would describe in their own words (underscore_format)
- Use underscore_format for all finding names

Be conservative. Only include findings with strong radiological evidence."""


def call_medgemma(code: str, description: str) -> Optional[dict]:
    """Call MedGemma to generate causal requirements for a code."""
    if not MEDGEMMA_API_KEY or not MEDGEMMA_ENDPOINT:
        logger.error("MEDGEMMA_API_KEY or MEDGEMMA_ENDPOINT not set in .env")
        return None

    prompt = SEED_USER.format(code=code, description=description)
    payload = {
        "model": MEDGEMMA_MODEL,
        "messages": [
            {"role": "system", "content": SEED_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {MEDGEMMA_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, 4):
        try:
            with httpx.Client(timeout=45) as client:
                resp = client.post(MEDGEMMA_ENDPOINT, headers=headers, json=payload)
            if resp.status_code == 200:
                content = resp.json()["choices"][0]["message"]["content"].strip()
                # Extract JSON from response
                start = content.find("{")
                end   = content.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(content[start:end])
                logger.warning(f"  [{code}] Could not parse JSON from response: {content[:120]}")
                return None
            else:
                logger.warning(f"  [{code}] Attempt {attempt}/3 — HTTP {resp.status_code}")
                time.sleep(3)
        except Exception as e:
            logger.warning(f"  [{code}] Attempt {attempt}/3 — {e}")
            time.sleep(3)

    return None


# ── UMLS ENRICHMENT ───────────────────────────────────────────────────────────

def umls_get_snomed_findings(icd_code: str) -> list:
    """
    Query UMLS to get SNOMED CT associated morphology for an ICD-10 code.
    Returns a list of finding strings in underscore_format.
    Requires UMLS_API_KEY in .env — free to obtain from https://uts.nlm.nih.gov
    """
    if not UMLS_API_KEY:
        return []

    try:
        # Step 1: Get CUI from ICD-10 code
        url = f"{UMLS_BASE}/search/current?string={icd_code}&sabs=ICD10CM&returnIdType=concept&apiKey={UMLS_API_KEY}"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        results = data.get("result", {}).get("results", [])
        if not results:
            return []
        cui = results[0]["ui"]

        # Step 2: Get SNOMED relations for the CUI
        url2 = f"{UMLS_BASE}/content/current/CUI/{cui}/relations?sabs=SNOMEDCT_US&apiKey={UMLS_API_KEY}"
        with urllib.request.urlopen(url2, timeout=10) as r:
            data2 = json.loads(r.read())

        findings = []
        for rel in data2.get("result", []):
            rel_label = rel.get("relationLabel", "").lower()
            related   = rel.get("relatedIdName", "").lower()
            # Map SNOMED relation types to visual finding vocabulary
            if rel_label in ("has_associated_morphology", "finding_site", "causative_agent"):
                normalized = related.replace(" ", "_").replace("-", "_")[:40]
                if normalized:
                    findings.append(normalized)

        return findings[:6]  # cap to avoid noise

    except Exception as e:
        logger.debug(f"  UMLS lookup failed for {icd_code}: {e}")
        return []


# ── MERGE STRATEGY ────────────────────────────────────────────────────────────

def merge_sources(mg_result: dict, umls_findings: list) -> tuple:
    """
    Merge MedGemma output with UMLS findings.
    Returns (merged_dict, source_label).

    Agreement logic:
      - Any UMLS finding that also appears in MedGemma's lists → auto-validated overlap
      - Unique UMLS findings → added to visual_supporting with lower weight
    """
    source = "medgemma_generated"

    if umls_findings:
        mg_any = set(mg_result.get("visual_required_any", []))
        mg_all = set(mg_result.get("visual_required_all", []))
        mg_sup = set(mg_result.get("visual_supporting", []))

        agreed     = [f for f in umls_findings if f in mg_any | mg_all]
        umls_extra = [f for f in umls_findings if f not in mg_any | mg_all | mg_sup]

        if agreed:
            source = "umls_snomed+medgemma"
        if umls_extra:
            mg_result["visual_supporting"] = list(mg_sup | set(umls_extra[:3]))

    return mg_result, source


# ── ICD-10 DESCRIPTION LOOKUP ─────────────────────────────────────────────────

def get_description(code: str) -> str:
    """Look up ICD-10 description from simple_icd_10_cm or CMS data."""
    try:
        import simple_icd_10_cm as cm
        if cm.is_valid_item(code):
            return cm.get_description(code)
    except Exception:
        pass
    try:
        from icd10_data import ICD10_DATA
        entry = ICD10_DATA.get(code.upper(), {})
        return entry.get("description", code)
    except Exception:
        pass
    return code


def get_top_unfilled_codes(n: int, existing_codes: set) -> list:
    """
    Return top-N ICD-10 codes by reimbursement that don't yet have causal rules.
    Uses CLINICAL_AUGMENTATION reimbursement as proxy for fraud risk.
    """
    try:
        from icd10_data import CLINICAL_AUGMENTATION
        ranked = sorted(
            CLINICAL_AUGMENTATION.items(),
            key=lambda x: x[1].get("avg_reimbursement_usd", 0),
            reverse=True
        )
        return [
            (code, get_description(code))
            for code, _ in ranked
            if code.upper() not in existing_codes
        ][:n]
    except Exception:
        return []


# ── YAML HELPERS ──────────────────────────────────────────────────────────────

def load_existing_rules() -> dict:
    if not RULES_FILE.exists():
        return {}
    with open(RULES_FILE, "r") as f:
        return yaml.safe_load(f) or {}


def append_rule(code: str, rule_dict: dict, dry_run: bool = False):
    """Append a new rule to causal_rules.yaml without touching existing entries."""
    lines = [
        f"\n{code}:",
        f'  description: "{rule_dict.get("description", "")}"',
        f'  anatomical_location: "{rule_dict.get("anatomical_location", "")}"',
        f'  causal_necessity: "{rule_dict.get("causal_necessity", "").strip()}"',
        f'  visual_required_any: {json.dumps(rule_dict.get("visual_required_any", []))}',
        f'  visual_required_all: {json.dumps(rule_dict.get("visual_required_all", []))}',
        f'  visual_supporting: {json.dumps(rule_dict.get("visual_supporting", []))}',
        f'  alternative_diagnoses: []',
        f'  fraud_differential: "{rule_dict.get("fraud_differential", "")}"',
        f'  minimum_causal_score: {rule_dict.get("minimum_causal_score", 0.6)}',
        f'  voice_keywords: {json.dumps(rule_dict.get("voice_keywords", []))}',
        f'  source: {rule_dict.get("source", "medgemma_generated")}',
        f'  validated: false',
        "",
    ]
    block = "\n".join(lines)

    if dry_run:
        print(f"\n[DRY RUN] Would append to {RULES_FILE}:")
        print(block)
        return

    with open(RULES_FILE, "a") as f:
        f.write(block)
    logger.info(f"  ✓ Appended {code} to {RULES_FILE}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seed causal_rules.yaml with new ICD-10 codes")
    parser.add_argument("--codes",      nargs="+", help="ICD-10 codes to seed (e.g. J15.1 J15.6)")
    parser.add_argument("--codes-file", help="File with one code per line")
    parser.add_argument("--top",        type=int,  help="Seed top-N unfilled high-reimbursement codes")
    parser.add_argument("--umls",       action="store_true", help="Enrich with UMLS/SNOMED data")
    parser.add_argument("--dry-run",    action="store_true", help="Print output without writing")
    args = parser.parse_args()

    # Build target code list
    target_codes: list[tuple[str, str]] = []

    if args.codes:
        for code in args.codes:
            target_codes.append((code.upper(), get_description(code.upper())))

    if args.codes_file:
        with open(args.codes_file) as f:
            for line in f:
                code = line.strip().upper()
                if code:
                    target_codes.append((code, get_description(code)))

    existing = load_existing_rules()
    existing_codes = {k.upper() for k in existing}

    if args.top:
        target_codes.extend(get_top_unfilled_codes(args.top, existing_codes))

    if not target_codes:
        parser.print_help()
        return

    # Deduplicate and skip already-seeded codes
    seen = set()
    to_seed = []
    for code, desc in target_codes:
        if code in existing_codes:
            logger.info(f"Skipping {code} — already in causal_rules.yaml")
            continue
        if code not in seen:
            seen.add(code)
            to_seed.append((code, desc))

    if not to_seed:
        logger.info("All target codes already seeded. Nothing to do.")
        return

    print(f"\n{'='*60}")
    print(f"Seeding {len(to_seed)} codes into causal_rules.yaml")
    print(f"UMLS enrichment: {'enabled' if args.umls else 'disabled'}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")
    print(f"{'='*60}\n")

    succeeded = 0
    failed    = []

    for i, (code, description) in enumerate(to_seed, 1):
        print(f"[{i}/{len(to_seed)}] {code} — {description[:60]}")

        # Step 1: MedGemma seeding
        mg_result = call_medgemma(code, description)
        if not mg_result:
            logger.warning(f"  ✗ MedGemma failed for {code} — skipping")
            failed.append(code)
            continue

        mg_result["description"] = description

        # Step 2: Optional UMLS enrichment
        umls_findings = []
        if args.umls:
            umls_findings = umls_get_snomed_findings(code)
            if umls_findings:
                logger.info(f"  UMLS found {len(umls_findings)} findings: {umls_findings}")

        # Step 3: Merge
        merged, source = merge_sources(mg_result, umls_findings)
        merged["source"] = source

        # Step 4: Write
        append_rule(code, merged, dry_run=args.dry_run)
        succeeded += 1
        time.sleep(0.5)   # polite rate limit

    print(f"\n{'='*60}")
    print(f"Done. Succeeded: {succeeded}/{len(to_seed)}")
    if failed:
        print(f"Failed: {failed}")
    if not args.dry_run and succeeded > 0:
        print(f"\nNew entries added to {RULES_FILE}")
        print("Set 'validated: true' after clinical review.")
        print("Restart backend to load new rules: docker-compose restart backend")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
