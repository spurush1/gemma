"""
100-Case Validation Runner
============================
Runs all cases in sample_dataset/cases.json through /analyze and /causal-audit,
then computes accuracy, precision, recall, F1 + financial/litigation metrics.

Usage:
    # Start backend first:
    cd backend && uvicorn app:app --port 8000 &

    # Run validation:
    python tests/validate_100_cases.py [--dataset tests/sample_dataset/cases.json]
                                        [--endpoint http://localhost:8000]
                                        [--output tests/validation_results.json]
                                        [--causal-only]
                                        [--limit N]

Output:
    tests/validation_results.json — full results per case
    Console summary with accuracy/precision/recall/F1 + financial + litigation metrics
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_DATASET = "tests/sample_dataset/cases.json"
DEFAULT_OUTPUT = "tests/validation_results.json"
REQUEST_TIMEOUT = 300  # seconds per case (LLM pipeline takes 3+ API calls × ~60s each)
RETRY_DELAY = 2


# =========================================================
# API HELPERS
# =========================================================

def call_analyze(endpoint: str, case: dict) -> Optional[dict]:
    """Call /analyze endpoint for a case."""
    try:
        resp = requests.post(
            f"{endpoint}/analyze",
            data={
                "clinical_note": case["clinical_note"],
                "existing_code": case["existing_code"],
            },
            timeout=REQUEST_TIMEOUT
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"    [WARN] /analyze returned {resp.status_code}: {resp.text[:100]}")
        return None
    except Exception as e:
        print(f"    [ERROR] /analyze failed: {e}")
        return None


def call_causal_audit(endpoint: str, case: dict) -> Optional[dict]:
    """Call /causal-audit endpoint, attaching X-ray image when available."""
    try:
        image_path = case.get("image_path")
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as img_f:
                resp = requests.post(
                    f"{endpoint}/causal-audit",
                    data={
                        "clinical_note": case["clinical_note"],
                        "existing_code": case["existing_code"],
                    },
                    files={"image": (Path(image_path).name, img_f, "image/png")},
                    timeout=REQUEST_TIMEOUT
                )
        else:
            resp = requests.post(
                f"{endpoint}/causal-audit",
                data={
                    "clinical_note": case["clinical_note"],
                    "existing_code": case["existing_code"],
                },
                timeout=REQUEST_TIMEOUT
            )
        if resp.status_code == 200:
            return resp.json()
        print(f"    [WARN] /causal-audit returned {resp.status_code}: {resp.text[:100]}")
        return None
    except Exception as e:
        print(f"    [ERROR] /causal-audit failed: {e}")
        return None


# =========================================================
# METRICS COMPUTATION
# =========================================================

def is_fraud(fraud_type: str) -> bool:
    """True if the case is a fraud (downcoding/upcoding/unrelated)."""
    return fraud_type.lower() not in ("none", "correct", "no_fraud", "")


def analyze_result_fraud_detected(analyze_result: Optional[dict]) -> bool:
    """Extract fraud detection from /analyze response."""
    if not analyze_result:
        return False
    fraud_risk = analyze_result.get("fraud_risk", "none").lower()
    return fraud_risk not in ("none", "")


def causal_result_fraud_detected(causal_result: Optional[dict]) -> bool:
    """Extract fraud detection from /causal-audit response."""
    if not causal_result:
        return False
    verdict = causal_result.get("final_verdict", {})
    return bool(verdict.get("fraud_detected", False))


def compute_binary_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Compute precision, recall, F1, accuracy for binary classification."""
    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": total
    }


# =========================================================
# FINANCIAL & LITIGATION AGGREGATION
# =========================================================

def aggregate_financial_metrics(results: list[dict]) -> dict:
    """Compute aggregate financial and litigation metrics across all cases."""
    total_upcoding_exposure = 0
    upcoding_cases = 0
    total_downcoding_gap = 0
    downcoding_cases = 0
    litigation_scores = []
    critical_litigation = 0
    high_litigation = 0

    for r in results:
        fraud_type = r.get("fraud_type", "none")
        causal = r.get("causal_audit_result") or {}

        upcoding_impact = causal.get("upcoding_financial_impact_usd", 0)
        if upcoding_impact > 0:
            total_upcoding_exposure += upcoding_impact
            upcoding_cases += 1

        lit_risk = causal.get("litigation_risk", {})
        lit_score = lit_risk.get("score", 0)
        lit_label = lit_risk.get("label", "none")
        applies = lit_risk.get("applies", False)

        if fraud_type == "downcoding" and applies:
            litigation_scores.append(lit_score)
            downcoding_cases += 1
            if lit_label == "critical":
                critical_litigation += 1
            elif lit_label == "high":
                high_litigation += 1

    avg_lit_score = sum(litigation_scores) / len(litigation_scores) if litigation_scores else 0.0

    return {
        "total_upcoding_financial_exposure_usd": total_upcoding_exposure,
        "upcoding_cases_detected": upcoding_cases,
        "avg_upcoding_exposure_per_case_usd": (
            total_upcoding_exposure // upcoding_cases if upcoding_cases > 0 else 0
        ),
        "avg_litigation_risk_score_downcoding": round(avg_lit_score, 1),
        "critical_litigation_cases": critical_litigation,
        "high_litigation_cases": high_litigation,
        "downcoding_cases_with_litigation_risk": len(litigation_scores),
    }


# =========================================================
# FRAUD TYPE BREAKDOWN
# =========================================================

def per_type_breakdown(results: list[dict], source: str = "causal") -> dict:
    """Compute metrics broken down by fraud type."""
    types = ["downcoding", "upcoding", "correct", "unrelated"]
    breakdown = {}

    for ft in types:
        subset = [r for r in results if r.get("fraud_type", "").lower() == ft]
        if not subset:
            continue

        gt = [is_fraud(r["fraud_type"]) for r in subset]
        if source == "causal":
            pred = [causal_result_fraud_detected(r.get("causal_audit_result")) for r in subset]
        else:
            pred = [analyze_result_fraud_detected(r.get("analyze_result")) for r in subset]

        metrics = compute_binary_metrics(pred, gt)
        breakdown[ft] = {
            "count": len(subset),
            "fraud_detected_count": sum(pred),
            "accuracy": round(metrics["accuracy"] * 100, 1),
            "precision": round(metrics["precision"] * 100, 1),
            "recall": round(metrics["recall"] * 100, 1),
            "f1": round(metrics["f1"], 3),
        }

    return breakdown


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Validate 100 cases through ICD-10 system")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--causal-only", action="store_true",
                        help="Only run /causal-audit (skip /analyze)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of cases (for quick testing)")
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)

    with open(dataset_path) as f:
        cases = json.load(f)

    if args.limit:
        cases = cases[:args.limit]

    print("=" * 60)
    print("ICD-10 MedGemma — 100-Case Validation")
    print("=" * 60)
    print(f"Dataset:     {dataset_path} ({len(cases)} cases)")
    print(f"Backend:     {args.endpoint}")
    print(f"Causal only: {args.causal_only}")

    # Check backend health
    try:
        health = requests.get(f"{args.endpoint}/health", timeout=10).json()
        print(f"Backend:     ✓ {health.get('graph_backend', 'unknown')} graph, "
              f"{health.get('total_icd10_codes', '?')} ICD codes")
    except Exception as e:
        print(f"\nERROR: Cannot reach backend at {args.endpoint}: {e}")
        print("Start backend: cd backend && uvicorn app:app --port 8000")
        sys.exit(1)

    print()

    # Run all cases
    results = []
    start_total = time.time()

    for i, case in enumerate(cases):
        case_id = case.get("case_id", f"CASE_{i+1:03d}")
        fraud_type = case.get("fraud_type", "unknown")
        print(f"[{i+1:3d}/{len(cases)}] {case_id} ({fraud_type})", end=" ... ", flush=True)

        result = {
            "case_id": case_id,
            "fraud_type": fraud_type,
            "existing_code": case.get("existing_code"),
            "expected_correct_code": case.get("expected_correct_code"),
            "expected_fraud_type": case.get("expected_fraud_type"),
            "analyze_result": None,
            "causal_audit_result": None,
            "analyze_fraud_detected": False,
            "causal_fraud_detected": False,
            "analyze_correct": None,
            "causal_correct": None,
        }

        # Run /analyze
        if not args.causal_only:
            analyze_res = call_analyze(args.endpoint, case)
            result["analyze_result"] = analyze_res
            result["analyze_fraud_detected"] = analyze_result_fraud_detected(analyze_res)
            result["analyze_correct"] = (
                result["analyze_fraud_detected"] == is_fraud(fraud_type)
            )

        # Run /causal-audit
        causal_res = call_causal_audit(args.endpoint, case)
        result["causal_audit_result"] = causal_res
        result["causal_fraud_detected"] = causal_result_fraud_detected(causal_res)
        result["causal_correct"] = (
            result["causal_fraud_detected"] == is_fraud(fraud_type)
        )

        # Print per-case result
        causal_ok = "✓" if result["causal_correct"] else "✗"
        analyze_ok = ("✓" if result["analyze_correct"] else "✗") if not args.causal_only else "-"
        print(f"analyze={analyze_ok} causal={causal_ok}", end="")

        if causal_res:
            score = causal_res.get("causal_score")
            verdict = causal_res.get("final_verdict", {})
            risk = verdict.get("overall_risk", "?")
            score_str = f"{score:.2f}" if score is not None else "N/A"
            print(f" | score={score_str} risk={risk}", end="")

        print()
        results.append(result)
        time.sleep(0.2)  # Small delay between cases

    total_time = time.time() - start_total

    # -------------------------------------------------------
    # COMPUTE METRICS
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("CAUSAL AUDIT VALIDATION RESULTS")
    print("=" * 60)

    gt_labels = [is_fraud(r["fraud_type"]) for r in results]
    causal_preds = [r["causal_fraud_detected"] for r in results]

    causal_metrics = compute_binary_metrics(causal_preds, gt_labels)
    print(f"\nTotal Cases:        {len(results)}")
    print(f"Runtime:            {total_time:.1f}s ({total_time/len(results):.1f}s/case)")
    print(f"\n--- Causal Audit (/causal-audit) ---")
    print(f"Accuracy:           {causal_metrics['accuracy']*100:.1f}%")
    print(f"Precision:          {causal_metrics['precision']*100:.1f}%")
    print(f"Recall:             {causal_metrics['recall']*100:.1f}%")
    print(f"F1 Score:           {causal_metrics['f1']:.3f}")
    print(f"TP={causal_metrics['tp']} FP={causal_metrics['fp']} "
          f"FN={causal_metrics['fn']} TN={causal_metrics['tn']}")

    if not args.causal_only:
        analyze_preds = [r["analyze_fraud_detected"] for r in results]
        analyze_metrics = compute_binary_metrics(analyze_preds, gt_labels)
        print(f"\n--- Baseline /analyze ---")
        print(f"Accuracy:           {analyze_metrics['accuracy']*100:.1f}%")
        print(f"Precision:          {analyze_metrics['precision']*100:.1f}%")
        print(f"Recall:             {analyze_metrics['recall']*100:.1f}%")
        print(f"F1 Score:           {analyze_metrics['f1']:.3f}")

        causal_f1 = causal_metrics["f1"]
        analyze_f1 = analyze_metrics["f1"]
        improvement = (causal_f1 - analyze_f1) / max(analyze_f1, 0.001) * 100
        print(f"\nCausal layer F1 improvement: {improvement:+.1f}% over baseline")

    # Per-fraud-type breakdown
    print(f"\n--- Per-Type Breakdown (Causal Audit) ---")
    breakdown = per_type_breakdown(results, source="causal")
    header = f"{'Type':12s} {'Count':6s} {'Detected':9s} {'Acc':6s} {'Prec':6s} {'Rec':6s} {'F1':6s}"
    print(header)
    print("-" * len(header))
    for ft, m in breakdown.items():
        print(f"{ft:12s} {m['count']:6d} {m['fraud_detected_count']:9d} "
              f"{m['accuracy']:5.1f}% {m['precision']:5.1f}% "
              f"{m['recall']:5.1f}% {m['f1']:5.3f}")

    # Financial & litigation metrics
    fin_metrics = aggregate_financial_metrics(results)
    print(f"\n--- Financial Impact ---")
    print(f"Total Upcoding Exposure:   ${fin_metrics['total_upcoding_financial_exposure_usd']:>10,}")
    print(f"Upcoding Cases Detected:   {fin_metrics['upcoding_cases_detected']}")
    if fin_metrics['upcoding_cases_detected'] > 0:
        print(f"Avg Per Upcoding Case:     ${fin_metrics['avg_upcoding_exposure_per_case_usd']:>10,}")

    print(f"\n--- Litigation Risk (Downcoding) ---")
    print(f"Cases with Litigation Risk:{fin_metrics['downcoding_cases_with_litigation_risk']}")
    print(f"Avg Risk Score:            {fin_metrics['avg_litigation_risk_score_downcoding']}/100")
    print(f"Critical Risk Cases:       {fin_metrics['critical_litigation_cases']}")
    print(f"High Risk Cases:           {fin_metrics['high_litigation_cases']}")

    # Save results
    output = {
        "summary": {
            "total_cases": len(results),
            "runtime_seconds": round(total_time, 1),
            "causal_metrics": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in causal_metrics.items()},
            "per_type_breakdown": breakdown,
            "financial_metrics": fin_metrics,
        },
        "cases": results,
    }

    if not args.causal_only:
        output["summary"]["analyze_metrics"] = {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in analyze_metrics.items()
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ Full results saved to {output_path}")

    # Print worst misses
    failures = [r for r in results if not r.get("causal_correct", True)]
    if failures:
        print(f"\n--- Misclassified Cases ({len(failures)}) ---")
        for r in failures[:10]:
            detected = "FRAUD" if r["causal_fraud_detected"] else "CLEAN"
            actual = "FRAUD" if is_fraud(r["fraud_type"]) else "CLEAN"
            print(f"  {r['case_id']:12s} ({r['fraud_type']:12s}): "
                  f"predicted={detected}, actual={actual}, "
                  f"code={r['existing_code']}")

    causal_correct_count = sum(1 for r in results if r.get("causal_correct", False))
    print(f"\nFinal: {causal_correct_count}/{len(results)} cases correctly classified "
          f"({causal_correct_count/len(results)*100:.1f}% accuracy)")


if __name__ == "__main__":
    main()
