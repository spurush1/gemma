"""
Open-i (Indiana University) Chest X-ray Downloader
====================================================
Downloads real chest X-ray images from the NLM Open-i public dataset.
Maps radiology findings (Problems field) → ICD-10 codes.

Source: https://openi.nlm.nih.gov  (3,955 de-identified hospital X-rays)
Images: 512×512 PNG, free to download without login.

Usage:
    python tests/download_nih_dataset.py
    python tests/download_nih_dataset.py --skip-images       # metadata only
    python tests/download_nih_dataset.py --per-condition 5   # 5 images each
    python tests/download_nih_dataset.py --count 50          # 50 total

Output:
    tests/sample_dataset/nih_images/     PNG files
    tests/sample_dataset/nih_cases.json  case metadata for validate_100_cases.py
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

OPENI_BASE    = "https://openi.nlm.nih.gov"
OPENI_API     = f"{OPENI_BASE}/api/search"
OPENI_IMG     = f"{OPENI_BASE}"        # image paths are relative, prepend base

# Conditions to collect and their ICD-10 mappings
CONDITIONS = {
    "cardiomegaly":  {"icd10": "I51.7",  "label": "Cardiomegaly",    "keywords": ["cardiomegaly"]},
    "pneumonia":     {"icd10": "J18.9",  "label": "Pneumonia",       "keywords": ["pneumonia"]},
    "consolidation": {"icd10": "J18.1",  "label": "Consolidation",   "keywords": ["consolidation"]},
    "pneumothorax":  {"icd10": "J93.1",  "label": "Pneumothorax",    "keywords": ["pneumothorax"]},
    "atelectasis":   {"icd10": "J98.11", "label": "Atelectasis",     "keywords": ["atelectasis", "pulmonary atelectasis"]},
    "effusion":      {"icd10": "J90",    "label": "Pleural Effusion","keywords": ["effusion", "pleural effusion"]},
    "edema":         {"icd10": "I50.1",  "label": "Pulmonary Edema", "keywords": ["edema", "pulmonary edema", "vascular congestion"]},
    "normal":        {"icd10": "Z03.89", "label": "No Finding",      "keywords": ["normal"]},
}
DEFAULT_PER_CONDITION = 8    # 8 × 8 conditions = 64 real CXR images
PAGE_SIZE             = 100  # records per API call
RATE_LIMIT_SECS       = 0.15 # polite delay between image downloads


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def api_fetch(page_start: int, count: int = PAGE_SIZE) -> dict:
    url = f"{OPENI_API}?q=chest&m={page_start}&n={count}&coll=cxr"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  [API WARN] page {page_start}: {e}")
        return {"list": [], "total": 0}


def matches_condition(problems_str: str, keywords: list) -> bool:
    p = problems_str.lower()
    return any(kw in p for kw in keywords)


def is_normal_only(problems_str: str) -> bool:
    """True only when the record has no pathological finding."""
    p = problems_str.lower().strip()
    return p in ("normal", "") or p.startswith("normal;") or p.endswith(";normal")


def download_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    except Exception as e:
        print(f"  [DL WARN] {url}: {e}")
        return b""


def build_case(record: dict, icd10: str, label: str,
               image_url: str, local_path: str) -> dict:
    problems  = record.get("Problems", "").strip()
    uid       = record.get("uid", "")
    img_file  = record.get("imgLarge", "").split("/")[-1]
    impression = record.get("impression", "").strip()

    return {
        "case_id":                  f"OPENI_{uid}_{img_file.replace('.png','')}",
        "source":                   "Open-i Indiana University CXR",
        "fraud_type":               "ground_truth_label",
        "clinical_note":            impression or f"Chest X-ray. Findings: {problems}.",
        "existing_code":            icd10,
        "expected_correct_code":    icd10,
        "expected_fraud_type":      "none",
        "image_finding_description": problems,
        "nih_label":                label,
        "image_url":                image_url,
        "image_path":               local_path or None,
        "expected_causal_score_range": [0.5, 1.0],
    }


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Open-i CXR dataset")
    parser.add_argument("--output-dir",    default="tests/sample_dataset/nih_images")
    parser.add_argument("--count",         type=int, default=0,
                        help="Total images to collect (0 = per-condition × conditions)")
    parser.add_argument("--per-condition", type=int, default=DEFAULT_PER_CONDITION)
    parser.add_argument("--skip-images",   action="store_true",
                        help="Collect metadata only, no image downloads")
    args = parser.parse_args()

    per_cond   = args.per_condition
    output_dir = Path(args.output_dir)
    output_json = Path("tests/sample_dataset/nih_cases.json")

    print("=" * 60)
    print("Open-i Indiana University CXR Downloader")
    print(f"Target: {per_cond} images × {len(CONDITIONS)} conditions = "
          f"{per_cond * len(CONDITIONS)} total")
    print("=" * 60)

    if not args.skip_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Page through Open-i to collect records per condition ──
    print("\n[1/3] Scanning Open-i catalog...")
    buckets  = {cond: [] for cond in CONDITIONS}
    filled   = 0
    page     = 1
    total    = None

    while filled < len(CONDITIONS) * per_cond:
        data   = api_fetch(page, PAGE_SIZE)
        if total is None:
            total = data.get("total", 0)
            print(f"  Catalog has {total:,} records — scanning in pages of {PAGE_SIZE}")

        records = data.get("list", [])
        if not records:
            break

        for rec in records:
            problems = rec.get("Problems", "")
            img_path = rec.get("imgLarge", "")
            if not img_path:
                continue

            for cond, cfg in CONDITIONS.items():
                if len(buckets[cond]) >= per_cond:
                    continue
                if cond == "normal":
                    if is_normal_only(problems):
                        buckets[cond].append(rec)
                else:
                    if matches_condition(problems, cfg["keywords"]):
                        buckets[cond].append(rec)

        filled = sum(len(v) for v in buckets.values())
        target = len(CONDITIONS) * per_cond
        page  += PAGE_SIZE
        print(f"  Scanned {min(page-1, total):,}/{total:,} records | "
              f"collected {filled}/{target}", end="\r")

        if page > total:
            break
        time.sleep(0.05)  # polite

    print()
    print("\n  Condition breakdown:")
    for cond, recs in buckets.items():
        cfg = CONDITIONS[cond]
        print(f"    {cfg['label']:25s} → {cfg['icd10']}: {len(recs)} cases")

    # ── Step 2: Download images ──
    all_cases = []
    total_dl  = sum(len(v) for v in buckets.values())
    done      = 0

    print(f"\n[2/3] {'Downloading' if not args.skip_images else 'Recording'} "
          f"{total_dl} images...")

    for cond, recs in buckets.items():
        cfg = CONDITIONS[cond]
        for rec in recs:
            done += 1
            img_rel   = rec.get("imgLarge", "")
            img_url   = f"{OPENI_IMG}{img_rel}"
            filename  = img_rel.split("/")[-1]
            local_path = ""

            if not args.skip_images:
                local_file = output_dir / filename
                if local_file.exists():
                    local_path = str(local_file)
                    print(f"  [{done}/{total_dl}] {filename} (cached)")
                else:
                    print(f"  [{done}/{total_dl}] {filename}...", end=" ", flush=True)
                    img_bytes = download_bytes(img_url)
                    if img_bytes:
                        local_file.write_bytes(img_bytes)
                        local_path = str(local_file)
                        print(f"{len(img_bytes)//1024}KB")
                    else:
                        print("FAILED")
                    time.sleep(RATE_LIMIT_SECS)
            else:
                print(f"  [{done}/{total_dl}] {filename} (skip-images mode)")

            case = build_case(rec, cfg["icd10"], cfg["label"], img_url, local_path)
            all_cases.append(case)

    # ── Step 3: Save JSON ──
    print(f"\n[3/3] Saving {len(all_cases)} cases → {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(all_cases, f, indent=2)

    downloaded = sum(1 for c in all_cases if c.get("image_path"))
    print(f"\n{'='*60}")
    print(f"Done!  {len(all_cases)} cases saved.")
    if not args.skip_images:
        print(f"Images: {downloaded}/{len(all_cases)} saved to {output_dir}/")
    print(f"\nNext:")
    print(f"  python tests/validate_100_cases.py "
          f"--dataset tests/sample_dataset/nih_cases.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
