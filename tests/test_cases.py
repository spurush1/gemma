"""
ICD-10 MedGemma Validation System — Test Cases
5 realistic test scenarios demonstrating the system's capabilities.
"""

import json
import requests
from typing import Optional

API_BASE = "http://localhost:8000"

# =========================================================
# TEST CASE DEFINITIONS
# =========================================================

TEST_CASES = [
    {
        "case_id": "CASE_01",
        "description": "Downcoding — Revenue Loss",
        "scenario": "downcoding",
        "clinical_note": (
            "Patient is a 62-year-old male presenting with 4 days of high-grade fever "
            "(39.8°C), rigors, productive cough with rusty-brown sputum, right-sided "
            "pleuritic chest pain that worsens with inspiration, and progressive dyspnea. "
            "SpO2 85% on room air, requiring 4L/min supplemental O2. Auscultation reveals "
            "decreased breath sounds, dullness to percussion, and bronchial breathing over "
            "the right lower lobe. WBC 18,400 with left shift. CRP 214. Chest X-ray "
            "demonstrates dense right lower lobe lobar consolidation with air bronchograms. "
            "Blood cultures drawn. Patient admitted to inpatient unit for IV antibiotics."
        ),
        "image_finding_description": (
            "PA chest X-ray shows dense homogeneous opacification of the right lower lobe "
            "with visible air bronchograms. No pleural effusion. No cardiomegaly. "
            "Left lung is clear. Findings are classic for bacterial lobar pneumonia."
        ),
        "existing_code": "J06.9",
        "existing_description": "Acute upper respiratory infection, unspecified (Common Cold)",
        "expected_correct_code": "J18.1",
        "expected_correct_description": "Lobar pneumonia, unspecified organism",
        "expected_mismatch": True,
        "expected_fraud_risk": "high",
        "expected_risk_type": "downcoding",
        "existing_reimbursement": 180,
        "correct_reimbursement": 2400,
        "expected_financial_impact_usd": 2220,
        "clinical_rationale": (
            "Radiographic lobar consolidation with air bronchograms + systemic sepsis "
            "response (high fever, elevated WBC, left shift) is diagnostic of lobar "
            "pneumonia (J18.1). Coding as J06.9 (Common Cold) represents a severe "
            "downcode with $2,220 revenue loss."
        )
    },
    {
        "case_id": "CASE_02",
        "description": "Upcoding — Billing Fraud",
        "scenario": "upcoding",
        "clinical_note": (
            "Patient is a 28-year-old female with known asthma presenting for routine "
            "follow-up. She reports mild wheezing approximately once per week that "
            "resolves spontaneously. No nocturnal symptoms. No recent emergency visits. "
            "SpO2 99% on room air. Lungs clear to auscultation. Peak flow 95% predicted. "
            "Well-controlled on low-dose inhaled corticosteroid. No systemic steroids "
            "required in past 12 months. Patient is exercising regularly and working full "
            "time without activity limitation."
        ),
        "image_finding_description": (
            "PA chest X-ray shows clear lung fields bilaterally. No infiltrates, no "
            "consolidation, no hyperinflation. Normal diaphragms. Normal cardiac size. "
            "No acute abnormality identified. Findings consistent with well-controlled asthma."
        ),
        "existing_code": "J45.50",
        "existing_description": "Severe persistent asthma, uncomplicated",
        "expected_correct_code": "J45.20",
        "expected_correct_description": "Mild intermittent asthma, uncomplicated",
        "expected_mismatch": True,
        "expected_fraud_risk": "high",
        "expected_risk_type": "upcoding",
        "existing_reimbursement": 3100,
        "correct_reimbursement": 890,
        "expected_financial_impact_usd": 2210,
        "clinical_rationale": (
            "Symptoms <1/week, no nighttime symptoms, 99% SpO2, clear chest X-ray, and "
            "100% peak flow meets criteria for mild intermittent asthma (J45.20). "
            "Billing as severe persistent asthma (J45.50) represents $2,210 overbilling."
        )
    },
    {
        "case_id": "CASE_03",
        "description": "Correct Coding — No Fraud",
        "scenario": "correct",
        "clinical_note": (
            "Patient is a 71-year-old female with history of heart failure presenting with "
            "3-day worsening dyspnea on exertion and bilateral ankle swelling. She reports "
            "orthopnea requiring 3 pillows and one episode of paroxysmal nocturnal dyspnea. "
            "Weight gain of 4kg over 2 weeks. JVP elevated at 10cm. Bilateral pitting edema "
            "to mid-shin. Auscultation reveals bibasal crackles and S3 gallop. SpO2 94% "
            "on room air. BNP 890 pg/mL. Echo shows EF 35%, dilated LV. CXR shows "
            "cardiomegaly, bilateral pleural effusions, and pulmonary vascular congestion "
            "with Kerley B lines."
        ),
        "image_finding_description": (
            "PA chest X-ray demonstrates cardiomegaly with cardiothoracic ratio 0.62. "
            "Bilateral pleural effusions with blunting of costophrenic angles. Perihilar "
            "haziness and vascular redistribution. Kerley B lines visible at lung bases. "
            "Findings are consistent with congestive heart failure."
        ),
        "existing_code": "I50.1",
        "existing_description": "Left ventricular failure, unspecified",
        "expected_correct_code": "I50.1",
        "expected_correct_description": "Left ventricular failure, unspecified",
        "expected_mismatch": False,
        "expected_fraud_risk": "none",
        "expected_risk_type": "none",
        "existing_reimbursement": 12000,
        "correct_reimbursement": 12000,
        "expected_financial_impact_usd": 0,
        "clinical_rationale": (
            "Clinical presentation (orthopnea, PND, edema, S3, elevated BNP) combined "
            "with CXR findings (cardiomegaly, Kerley B lines, pleural effusions) perfectly "
            "match I50.1 (Left ventricular failure). No coding error. No fraud."
        )
    },
    {
        "case_id": "CASE_04",
        "description": "Unrelated Code — Wrong Category",
        "scenario": "unrelated",
        "clinical_note": (
            "Patient is a 68-year-old male with 2-week history of progressive exertional "
            "dyspnea, bilateral leg edema, and fatigue. No fever. No cough. No pleuritic "
            "chest pain. History of coronary artery disease and hypertension. JVP elevated. "
            "Bilateral pitting edema to knees. Auscultation: bibasal crackles. Heart: "
            "displaced apex, S3 gallop. SpO2 93%. Echo: EF 30%, global hypokinesis. "
            "ECG: LBBB. CXR: marked cardiomegaly, pulmonary edema pattern."
        ),
        "image_finding_description": (
            "PA chest X-ray shows markedly enlarged cardiac silhouette with cardiothoracic "
            "ratio 0.68. Bilateral perihilar haziness and vascular congestion. Kerley B "
            "lines. Bilateral pleural effusions. No focal consolidation or infiltrate. "
            "Findings indicate cardiomegaly with congestive heart failure — NOT pneumonia."
        ),
        "existing_code": "J18.9",
        "existing_description": "Unspecified pneumonia",
        "expected_correct_code": "I50.9",
        "expected_correct_description": "Heart failure, unspecified",
        "expected_mismatch": True,
        "expected_fraud_risk": "high",
        "expected_risk_type": "unrelated",
        "existing_reimbursement": 2000,
        "correct_reimbursement": 9500,
        "expected_financial_impact_usd": 7500,
        "clinical_rationale": (
            "Patient has classic heart failure: cardiomegaly on CXR, Kerley B lines, S3, "
            "EF 30%, elevated JVP. No fever, no cough, no consolidation on CXR. "
            "Coding as J18.9 (Pneumonia) is completely wrong category. Correct code "
            "I50.9 (Heart Failure). Financial gap: $7,500."
        )
    },
    {
        "case_id": "CASE_05",
        "description": "Missing Image — Text Only Input",
        "scenario": "text_only",
        "clinical_note": (
            "Patient is a 45-year-old female presenting with 5-day history of epigastric "
            "pain, nausea, and one episode of vomiting. Pain is burning in nature, worse "
            "before meals, relieved partially by antacids. She reports dark stools over "
            "the past 2 days. Takes NSAIDs regularly for arthritis. H. pylori breath "
            "test positive. Hemoglobin 9.8 g/dL (baseline 13.2). Hematocrit 30%. "
            "BP 105/68 (normally 120/80). Resting tachycardia at 102 bpm."
        ),
        "image_finding_description": None,  # No image
        "existing_code": "K25.9",
        "existing_description": "Gastric ulcer, unspecified, without hemorrhage or perforation",
        "expected_correct_code": "K25.0",
        "expected_correct_description": "Gastric ulcer, acute with hemorrhage",
        "expected_mismatch": True,
        "expected_fraud_risk": "medium",
        "expected_risk_type": "downcoding",
        "existing_reimbursement": 1200,
        "correct_reimbursement": 8500,
        "expected_financial_impact_usd": 7300,
        "clinical_rationale": (
            "Melena (dark stools) + hemoglobin drop of 3.4 g/dL + tachycardia + hypotension "
            "in NSAID user with H. pylori indicates acute hemorrhagic gastric ulcer (K25.0). "
            "K25.9 excludes hemorrhage — significant downcode. Confidence is reduced without "
            "imaging (no endoscopy or CXR to confirm), demonstrating text-only mode."
        )
    }
]


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_separator(char="=", width=70):
    print(char * width)


def print_result_row(case_id: str, description: str, status: str,
                     expected_code: str, got_code: str,
                     expected_fraud: str, got_fraud: str,
                     expected_impact: int, got_impact: int):
    status_icon = "PASS" if status == "pass" else "FAIL"
    print(f"[{status_icon}] {case_id}: {description}")
    print(f"       Code: expected={expected_code}, got={got_code}")
    print(f"       Fraud: expected={expected_fraud}, got={got_fraud}")
    print(f"       Impact: expected=${expected_impact:,}, got=${got_impact:,}")


def call_analyze(case: dict) -> Optional[dict]:
    """Call the /analyze endpoint for a given test case."""
    try:
        data = {
            "clinical_note": case["clinical_note"],
            "existing_code": case["existing_code"]
        }

        # POST as multipart form (no image for case 5)
        resp = requests.post(
            f"{API_BASE}/analyze",
            data=data,
            timeout=60
        )

        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  ERROR: HTTP {resp.status_code} - {resp.text[:200]}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to {API_BASE}. Is the backend running?")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# =========================================================
# MAIN TEST RUNNER
# =========================================================

def run_all_tests():
    """
    Runs all 5 test cases against the live backend API
    and prints a formatted results summary.
    """
    print_separator("=")
    print("ICD-10 MedGemma Validation System — Test Suite")
    print_separator("=")
    print()

    results = []
    total = len(TEST_CASES)
    passed = 0
    total_impact_detected = 0

    for i, case in enumerate(TEST_CASES, 1):
        print(f"Running {case['case_id']}: {case['description']}")
        print(f"  Clinical scenario: {case['scenario']}")
        print(f"  Submitted code: {case['existing_code']} → Expected: {case['expected_correct_code']}")

        response = call_analyze(case)

        if response is None:
            result = {
                "case_id": case["case_id"],
                "status": "error",
                "error": "API call failed"
            }
            results.append(result)
            print(f"  STATUS: ERROR\n")
            continue

        # Evaluate results
        suggested = response.get("suggested_code", "")
        fraud_risk = response.get("fraud_risk", "none")
        financial_impact = int(response.get("financial_impact_usd", 0))
        mismatch = bool(response.get("mismatch_detected", False))
        confidence = float(response.get("confidence", 0))

        # Pass criteria
        code_match = suggested == case["expected_correct_code"]
        fraud_match = fraud_risk == case["expected_fraud_risk"]
        mismatch_match = mismatch == case["expected_mismatch"]

        # For text-only (case 5), lower confidence is expected
        if case["scenario"] == "text_only":
            confidence_ok = confidence < 0.9  # Should be lower without image
        else:
            confidence_ok = True

        status = "pass" if (code_match and mismatch_match) else "fail"
        if status == "pass":
            passed += 1
        total_impact_detected += financial_impact

        result = {
            "case_id": case["case_id"],
            "description": case["description"],
            "scenario": case["scenario"],
            "status": status,
            "expected_code": case["expected_correct_code"],
            "suggested_code": suggested,
            "code_match": code_match,
            "expected_fraud": case["expected_fraud_risk"],
            "fraud_detected": fraud_risk,
            "fraud_match": fraud_match,
            "expected_mismatch": case["expected_mismatch"],
            "mismatch_detected": mismatch,
            "mismatch_match": mismatch_match,
            "confidence": confidence,
            "confidence_ok": confidence_ok,
            "expected_financial_impact": case["expected_financial_impact_usd"],
            "detected_financial_impact": financial_impact,
            "reasoning": response.get("reasoning", ""),
            "audit_recommendation": response.get("audit_recommendation", "")
        }
        results.append(result)

        status_icon = "PASS" if status == "pass" else "FAIL"
        print(f"  Suggested code: {suggested} {'(correct)' if code_match else '(WRONG)'}")
        print(f"  Confidence: {confidence:.0%}")
        print(f"  Mismatch detected: {mismatch} (expected: {case['expected_mismatch']})")
        print(f"  Fraud risk: {fraud_risk} (expected: {case['expected_fraud_risk']})")
        print(f"  Financial impact: ${financial_impact:,} (expected: ${case['expected_financial_impact_usd']:,})")
        print(f"  STATUS: {status_icon}")
        print()

    # -------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------
    print_separator("=")
    print("RESULTS SUMMARY")
    print_separator("=")
    print(f"Total Cases Tested:        {total}")
    print(f"Passed:                    {passed}/{total} ({passed/total*100:.0f}%)")
    print()

    # By scenario
    for scenario in ["downcoding", "upcoding", "correct", "unrelated", "text_only"]:
        scenario_results = [r for r in results if r.get("scenario") == scenario]
        if scenario_results:
            s_passed = sum(1 for r in scenario_results if r.get("status") == "pass")
            label = scenario.replace("_", " ").title()
            print(f"  {label:<25} {s_passed}/{len(scenario_results)}")

    print()
    print(f"Total Financial Impact Detected:  ${total_impact_detected:,}")
    print_separator("=")

    # Save detailed results
    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({"summary": {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "total_financial_impact_detected": total_impact_detected
        }, "cases": results}, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    # Save failed cases
    failed = [r for r in results if r.get("status") != "pass"]
    if failed:
        with open("failed_cases.json", "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failed cases saved to: failed_cases.json")

    return results


# =========================================================
# INDIVIDUAL TEST CASE DISPLAY
# =========================================================

def print_test_cases():
    """Print all test cases in readable format (no API call needed)."""
    for case in TEST_CASES:
        print_separator("=")
        print(f"{case['case_id']}: {case['description']}")
        print_separator("-")
        print(f"Scenario:    {case['scenario']}")
        print(f"Submitted:   {case['existing_code']} ({case['existing_description']})")
        print(f"Correct:     {case['expected_correct_code']} ({case['expected_correct_description']})")
        print(f"Mismatch:    {case['expected_mismatch']}")
        print(f"Fraud Risk:  {case['expected_fraud_risk']} ({case['expected_risk_type']})")
        print(f"Impact:      ${case['expected_financial_impact_usd']:,}")
        print(f"\nClinical Note (excerpt):")
        print(f"  {case['clinical_note'][:200]}...")
        if case['image_finding_description']:
            print(f"\nImage Findings:")
            print(f"  {case['image_finding_description'][:150]}...")
        else:
            print(f"\nImage: None (text-only test)")
        print(f"\nRationale:")
        print(f"  {case['clinical_rationale']}")
        print()


if __name__ == "__main__":
    import sys

    if "--print-cases" in sys.argv:
        print_test_cases()
    else:
        run_all_tests()
