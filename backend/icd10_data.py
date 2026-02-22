"""
ICD-10 Data Module
Loads real ICD-10-CM codes from the official CMS dataset via the
simple_icd_10_cm library (98,505 codes, FY2024).

Clinical augmentation (symptoms, image_findings, severity, reimbursement)
is provided for the medically-focused subset used in graph traversal and
fraud detection. All other codes are available for lookup and validation.
"""

import simple_icd_10_cm as cm

# =========================================================
# CLINICAL AUGMENTATION
# Symptoms, image findings, severity, and reimbursement data
# for codes relevant to multimodal coding validation.
# These fields are NOT in the CMS dataset and require clinical curation.
# =========================================================

CLINICAL_AUGMENTATION = {
    # --- RESPIRATORY ---
    "J06.9": {
        "symptoms": [
            "runny_nose", "sore_throat", "mild_cough", "low_grade_fever",
            "nasal_congestion", "sneezing", "malaise"
        ],
        "image_findings": ["normal_chest", "clear_lung_fields", "no_consolidation"],
        "severity": "mild",
        "avg_reimbursement_usd": 180
    },
    "J09.X9": {
        "symptoms": ["fever", "cough", "myalgia", "headache", "fatigue"],
        "image_findings": ["bilateral_infiltrates", "patchy_airspace_disease"],
        "severity": "severe",
        "avg_reimbursement_usd": 4200
    },
    "J12.9": {
        "symptoms": ["fever", "cough", "dyspnea", "fatigue"],
        "image_findings": ["bilateral_infiltrates", "ground_glass_opacities"],
        "severity": "moderate",
        "avg_reimbursement_usd": 1900
    },
    "J13": {
        "symptoms": ["fever", "productive_cough", "pleuritic_chest_pain", "dyspnea", "rigors"],
        "image_findings": ["lobar_consolidation", "air_bronchograms"],
        "severity": "severe",
        "avg_reimbursement_usd": 2600
    },
    "J18.0": {
        "symptoms": [
            "fever", "productive_cough", "dyspnea", "chest_pain",
            "tachycardia", "tachypnea", "crackles_on_auscultation"
        ],
        "image_findings": [
            "patchy_infiltrates", "bilateral_airspace_disease",
            "peribronchial_thickening", "diffuse_haziness"
        ],
        "severity": "moderate",
        "avg_reimbursement_usd": 2100
    },
    "J18.1": {
        "symptoms": [
            "high_fever", "productive_cough", "pleuritic_chest_pain",
            "dyspnea", "tachycardia", "tachypnea", "hypoxia",
            "crackles_on_auscultation", "dullness_to_percussion"
        ],
        "image_findings": [
            "lobar_consolidation", "air_bronchograms", "dense_opacity",
            "bilateral_infiltrates", "complete_lobe_opacification"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 2400
    },
    "J18.9": {
        "symptoms": ["fever", "cough", "dyspnea", "chest_pain", "tachycardia"],
        "image_findings": ["pulmonary_infiltrates", "consolidation", "airspace_disease"],
        "severity": "moderate",
        "avg_reimbursement_usd": 2000
    },
    "J20.9": {
        "symptoms": [
            "productive_cough", "low_grade_fever", "wheezing",
            "chest_tightness", "fatigue", "sputum_production"
        ],
        "image_findings": [
            "peribronchial_thickening", "mild_hyperinflation",
            "no_consolidation", "increased_bronchovascular_markings"
        ],
        "severity": "mild",
        "avg_reimbursement_usd": 320
    },
    "J22": {
        "symptoms": ["cough", "fever", "dyspnea", "chest_tightness", "wheezing"],
        "image_findings": ["mild_infiltrates", "peribronchial_changes", "no_focal_consolidation"],
        "severity": "mild",
        "avg_reimbursement_usd": 450
    },
    "J44.1": {
        "symptoms": [
            "worsening_dyspnea", "increased_sputum_production",
            "change_in_sputum_color", "cough", "wheezing", "hypoxia"
        ],
        "image_findings": [
            "hyperinflation", "flattened_diaphragms", "bullae",
            "increased_bronchovascular_markings"
        ],
        "severity": "moderate",
        "avg_reimbursement_usd": 3800
    },
    "J45.20": {
        "symptoms": [
            "intermittent_wheezing", "mild_dyspnea", "cough",
            "chest_tightness", "nocturnal_symptoms_less_than_twice_monthly"
        ],
        "image_findings": [
            "clear_lung_fields", "mild_hyperinflation",
            "no_infiltrates", "normal_diaphragms"
        ],
        "severity": "mild",
        "avg_reimbursement_usd": 890
    },
    "J45.50": {
        "symptoms": [
            "severe_dyspnea", "continuous_wheezing", "frequent_exacerbations",
            "nighttime_symptoms", "limited_activity", "hypoxia",
            "accessory_muscle_use", "pulsus_paradoxus"
        ],
        "image_findings": [
            "hyperinflation", "flattened_diaphragms", "increased_ap_diameter",
            "air_trapping", "peribronchial_cuffing"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 3100
    },
    "J90": {
        "symptoms": [
            "dyspnea", "pleuritic_chest_pain", "cough",
            "decreased_breath_sounds", "dullness_to_percussion"
        ],
        "image_findings": [
            "pleural_effusion", "blunting_of_costophrenic_angle",
            "meniscus_sign", "fluid_layering"
        ],
        "severity": "moderate",
        "avg_reimbursement_usd": 1800
    },
    "J93.1": {
        "symptoms": [
            "sudden_chest_pain", "dyspnea", "decreased_breath_sounds",
            "tachycardia", "hypoxia", "tracheal_deviation"
        ],
        "image_findings": [
            "pneumothorax", "absent_lung_markings",
            "visceral_pleural_line", "tracheal_deviation"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 4200
    },
    "J96.0": {
        "symptoms": [
            "severe_dyspnea", "hypoxia", "cyanosis", "altered_mental_status",
            "accessory_muscle_use", "tachypnea", "diaphoresis"
        ],
        "image_findings": [
            "bilateral_infiltrates", "diffuse_airspace_disease",
            "ground_glass_opacities", "pulmonary_edema_pattern"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 8500
    },
    "J96.1": {
        "symptoms": [
            "chronic_dyspnea", "hypoxia", "polycythemia",
            "barrel_chest", "pursed_lip_breathing", "digital_clubbing"
        ],
        "image_findings": [
            "hyperinflation", "bullae", "flattened_diaphragms",
            "increased_retrosternal_space", "cardiomegaly"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 6200
    },

    # --- CARDIOVASCULAR ---
    "I10": {
        "symptoms": ["headache", "dizziness", "palpitations", "epistaxis", "often_asymptomatic"],
        "image_findings": ["normal_chest", "mild_cardiomegaly", "aortic_prominence"],
        "severity": "mild",
        "avg_reimbursement_usd": 280
    },
    "I21.0": {
        "symptoms": [
            "severe_chest_pain", "diaphoresis", "nausea", "vomiting",
            "radiation_to_left_arm", "dyspnea", "syncope", "hypotension"
        ],
        "image_findings": [
            "cardiomegaly", "pulmonary_edema", "vascular_congestion"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 28000
    },
    "I21.1": {
        "symptoms": [
            "severe_chest_pain", "diaphoresis", "nausea", "bradycardia",
            "epigastric_pain", "hypotension", "radiation_to_jaw"
        ],
        "image_findings": ["cardiomegaly", "pulmonary_vascular_congestion"],
        "severity": "severe",
        "avg_reimbursement_usd": 26000
    },
    "I21.9": {
        "symptoms": ["chest_pain", "diaphoresis", "dyspnea", "nausea", "palpitations"],
        "image_findings": ["cardiomegaly", "pulmonary_congestion"],
        "severity": "severe",
        "avg_reimbursement_usd": 22000
    },
    "I25.10": {
        "symptoms": ["exertional_chest_pain", "dyspnea_on_exertion", "fatigue", "palpitations"],
        "image_findings": ["coronary_artery_calcifications", "mild_cardiomegaly"],
        "severity": "moderate",
        "avg_reimbursement_usd": 4800
    },
    "I48.0": {
        "symptoms": ["palpitations", "irregular_heartbeat", "dyspnea", "fatigue", "dizziness"],
        "image_findings": ["mild_cardiomegaly", "enlarged_left_atrium"],
        "severity": "moderate",
        "avg_reimbursement_usd": 5200
    },
    # I50.0 is not valid in current CMS ICD-10-CM — replaced by I50.1/I50.2/I50.3/I50.4
    # Using I50.1 (Left ventricular failure) as the severe CHF representative
    "I50.1": {
        "symptoms": [
            "dyspnea_on_exertion", "orthopnea", "paroxysmal_nocturnal_dyspnea",
            "bilateral_leg_edema", "fatigue", "weight_gain", "jugular_venous_distension"
        ],
        "image_findings": [
            "cardiomegaly", "pulmonary_edema", "bilateral_pleural_effusions",
            "vascular_redistribution", "kerley_b_lines", "perihilar_haziness"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 12000
    },
    "I50.9": {
        "symptoms": [
            "dyspnea", "fatigue", "leg_edema", "orthopnea",
            "reduced_exercise_tolerance", "peripheral_edema"
        ],
        "image_findings": [
            "cardiomegaly", "pulmonary_vascular_congestion",
            "pleural_effusions", "enlarged_cardiac_silhouette"
        ],
        "severity": "moderate",
        "avg_reimbursement_usd": 9500
    },
    "I51.7": {
        "symptoms": ["dyspnea", "fatigue", "peripheral_edema", "palpitations"],
        "image_findings": [
            "cardiomegaly", "enlarged_cardiac_silhouette",
            "cardiothoracic_ratio_greater_than_0.5"
        ],
        "severity": "moderate",
        "avg_reimbursement_usd": 3600
    },
    "I63.9": {
        "symptoms": [
            "sudden_weakness", "facial_droop", "speech_difficulty",
            "vision_changes", "severe_headache", "confusion"
        ],
        "image_findings": ["normal_chest", "atrial_fibrillation_pattern"],
        "severity": "severe",
        "avg_reimbursement_usd": 18000
    },

    # --- GASTROINTESTINAL ---
    "K25.0": {
        "symptoms": [
            "hematemesis", "melena", "epigastric_pain", "nausea",
            "hypotension", "tachycardia", "pallor"
        ],
        "image_findings": ["normal_chest", "no_free_air"],
        "severity": "severe",
        "avg_reimbursement_usd": 8500
    },
    "K25.9": {
        "symptoms": ["epigastric_pain", "nausea", "bloating", "early_satiety", "anorexia"],
        "image_findings": ["normal_chest", "no_acute_findings"],
        "severity": "mild",
        "avg_reimbursement_usd": 1200
    },
    "K35.2": {
        "symptoms": [
            "severe_abdominal_pain", "fever", "nausea", "vomiting",
            "rigidity", "rebound_tenderness", "guarding"
        ],
        "image_findings": [
            "free_air_possible", "elevated_diaphragm",
            "basilar_atelectasis", "reactive_pleural_effusion"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 14000
    },
    "K35.3": {
        "symptoms": [
            "right_lower_quadrant_pain", "fever", "nausea", "vomiting",
            "mcburneys_point_tenderness"
        ],
        "image_findings": ["normal_chest", "elevated_right_hemidiaphragm"],
        "severity": "moderate",
        "avg_reimbursement_usd": 9800
    },
    "K57.30": {
        "symptoms": ["left_lower_quadrant_pain", "bloating", "constipation", "diarrhea"],
        "image_findings": ["normal_chest", "no_free_air"],
        "severity": "mild",
        "avg_reimbursement_usd": 950
    },
    "K70.30": {
        "symptoms": [
            "jaundice", "fatigue", "spider_angiomata", "palmar_erythema",
            "caput_medusae", "asterixis"
        ],
        "image_findings": [
            "elevated_right_hemidiaphragm", "small_right_pleural_effusion"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 7200
    },
    "K85.9": {
        "symptoms": [
            "severe_epigastric_pain", "radiation_to_back", "nausea",
            "vomiting", "fever", "tachycardia", "guarding"
        ],
        "image_findings": [
            "left_pleural_effusion", "elevated_left_hemidiaphragm",
            "basilar_atelectasis"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 9500
    },
    "K92.1": {
        "symptoms": [
            "black_tarry_stools", "weakness", "dizziness", "pallor",
            "tachycardia", "hypotension_if_severe"
        ],
        "image_findings": ["normal_chest", "no_acute_pulmonary_findings"],
        "severity": "moderate",
        "avg_reimbursement_usd": 2800
    },

    # --- INFECTIOUS / OTHER ---
    "A41.9": {
        "symptoms": [
            "fever", "tachycardia", "tachypnea", "hypotension",
            "altered_mental_status", "decreased_urine_output"
        ],
        "image_findings": [
            "bilateral_infiltrates", "pulmonary_edema", "ards_pattern"
        ],
        "severity": "severe",
        "avg_reimbursement_usd": 22000
    },
    "E11.9": {
        "symptoms": ["polyuria", "polydipsia", "fatigue", "blurred_vision"],
        "image_findings": ["normal_chest", "no_acute_findings"],
        "severity": "mild",
        "avg_reimbursement_usd": 380
    },
    "N18.3": {
        "symptoms": ["fatigue", "decreased_urine_output", "peripheral_edema", "hypertension"],
        "image_findings": ["pulmonary_edema_possible", "cardiomegaly_possible"],
        "severity": "moderate",
        "avg_reimbursement_usd": 2200
    },
}


# =========================================================
# CORE FUNCTIONS
# =========================================================

def _infer_category(code: str) -> str:
    """Infer clinical category from ICD-10-CM chapter letter."""
    if not code:
        return "other"
    prefix = code[0].upper()
    return {
        "A": "infectious", "B": "infectious",
        "C": "neoplasm", "D": "neoplasm",
        "E": "endocrine",
        "F": "psychiatric",
        "G": "neurological",
        "H": "sensory",
        "I": "cardiovascular",
        "J": "respiratory",
        "K": "gastrointestinal",
        "L": "dermatological",
        "M": "musculoskeletal",
        "N": "renal",
        "O": "obstetric",
        "P": "perinatal",
        "Q": "congenital",
        "R": "symptoms_signs",
        "S": "injury", "T": "injury",
        "V": "external", "W": "external", "X": "external", "Y": "external",
        "Z": "factors_influencing_health",
    }.get(prefix, "other")


def _build_code_entry(code: str) -> dict:
    """
    Build a full code entry from real CMS data + optional clinical augmentation.
    """
    description = cm.get_description(code)
    parent = cm.get_parent(code)
    ancestors = cm.get_ancestors(code)
    children = cm.get_children(code) if not cm.is_leaf(code) else []
    excludes1 = cm.get_excludes1(code)   # Real CMS excludes — cannot code together
    excludes2 = cm.get_excludes2(code)   # Real CMS excludes — not included here
    inclusion_terms = cm.get_inclusion_term(code)
    is_billable = cm.is_leaf(code)

    aug = CLINICAL_AUGMENTATION.get(code, {})

    return {
        "code": code,
        "description": description,
        "category": _infer_category(code),
        "parent_code": parent,
        "children": children,
        "ancestors": ancestors,
        "excludes1": excludes1,           # REAL from CMS
        "excludes2": excludes2,           # REAL from CMS
        "inclusion_terms": inclusion_terms,
        "is_billable": is_billable,
        "symptoms": aug.get("symptoms", []),
        "image_findings": aug.get("image_findings", []),
        "severity": aug.get("severity", "unknown"),
        "avg_reimbursement_usd": aug.get("avg_reimbursement_usd", 0),
        "has_clinical_data": code in CLINICAL_AUGMENTATION,
    }


# Pre-build entries for all clinically-augmented codes
ICD10_DATA: dict = {}
for _code in CLINICAL_AUGMENTATION:
    if cm.is_valid_item(_code):
        ICD10_DATA[_code] = _build_code_entry(_code)


# =========================================================
# PUBLIC API
# =========================================================

def get_all_codes() -> dict:
    """Returns augmented subset (codes with clinical data)."""
    return ICD10_DATA


def get_code(code: str) -> "dict | None":
    """
    Returns full entry for any valid ICD-10-CM code.
    Enriched with clinical data if available, otherwise basic CMS data.
    """
    if not cm.is_valid_item(code):
        return None
    if code in ICD10_DATA:
        return ICD10_DATA[code]
    return _build_code_entry(code)


def get_codes_by_category(category: str) -> dict:
    """Returns augmented codes in a given category."""
    return {
        code: data for code, data in ICD10_DATA.items()
        if data.get("category") == category
    }


def search_codes(query: str) -> list:
    """
    Search ALL 98,505 real CMS codes by description keyword.
    Returns up to 50 matches.
    """
    query_lower = query.lower()
    results = []
    for code in cm.get_all_codes(with_dots=True):
        desc = cm.get_description(code)
        if desc and query_lower in desc.lower():
            aug = CLINICAL_AUGMENTATION.get(code, {})
            results.append({
                "code": code,
                "description": desc,
                "category": _infer_category(code),
                "is_billable": cm.is_leaf(code),
                "has_clinical_data": code in CLINICAL_AUGMENTATION,
                "severity": aug.get("severity", ""),
                "avg_reimbursement_usd": aug.get("avg_reimbursement_usd", 0),
            })
        if len(results) >= 50:
            break
    return results


def is_valid_code(code: str) -> bool:
    """Returns True if code exists in the real CMS ICD-10-CM dataset."""
    return cm.is_valid_item(code)


def get_real_excludes(code: str) -> list:
    """Returns real excludes1 from CMS (cannot code together)."""
    if not cm.is_valid_item(code):
        return []
    return cm.get_excludes1(code)


def get_real_children(code: str) -> list:
    """Returns real children from CMS hierarchy."""
    if not cm.is_valid_item(code):
        return []
    try:
        return cm.get_children(code)
    except ValueError:
        return []


def get_real_ancestors(code: str) -> list:
    """Returns real ancestor chain from CMS hierarchy."""
    if not cm.is_valid_item(code):
        return []
    return cm.get_ancestors(code)


if __name__ == "__main__":
    print(f"Total CMS ICD-10-CM codes:       {len(cm.get_all_codes()):,}")
    print(f"Billable (leaf) codes:            {sum(1 for c in cm.get_all_codes() if cm.is_leaf(c)):,}")
    print(f"Codes with clinical augmentation: {len(ICD10_DATA)}")

    print("\nSample — J18.1 (built from real CMS data):")
    import json
    print(json.dumps(ICD10_DATA["J18.1"], indent=2))

    print("\nReal CMS hierarchy for J18:")
    for child in cm.get_children("J18"):
        print(f"  {child}: {cm.get_description(child)}")

    print("\nSearch 'lobar pneumonia' across all 98k CMS codes:")
    for r in search_codes("lobar pneumonia"):
        print(f"  {r['code']}: {r['description']}")
