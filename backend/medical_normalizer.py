"""
Medical Terminology Normalizer
===============================
Normalizes free-form radiological findings to standardized knowledge-engine terms.

Purpose: Fix MedGemma's terminology inconsistency (Problem 2 false positives).
  "right_sided_pleural_effusion" → "pleural_effusion"
  "bilateral_infiltrates"        → "consolidation"
  "possible_cardiomegaly"        → "cardiomegaly"
  "opacity_left_lung"            → "consolidation"

Based on RadLex (Radiology Lexicon) synonym relationships and
clinical terminology equivalences observed in MedGemma output analysis.

Both original AND normalized forms are returned so the LLM evaluator
gets maximum context for semantic matching.
"""

# =========================================================
# COMPREHENSIVE RADLEX SYNONYM MAP
# Maps free-form MedGemma output → standardized KE terms
# =========================================================

RADLEX_SYNONYM_MAP = {

    # --- Pleural effusion variants ---
    "right_sided_pleural_effusion":     "pleural_effusion",
    "left_sided_pleural_effusion":      "pleural_effusion",
    "bilateral_pleural_effusion":       "pleural_effusion",
    "bilateral_pleural_effusions":      "pleural_effusion",
    "right_pleural_effusion":           "pleural_effusion",
    "left_pleural_effusion":            "pleural_effusion",
    "small_pleural_effusion":           "pleural_effusion",
    "moderate_pleural_effusion":        "pleural_effusion",
    "large_pleural_effusion":           "pleural_effusion",
    "pleural_fluid":                    "pleural_effusion",
    "fluid_in_pleural_space":           "pleural_effusion",
    "right_pleural_fluid":              "pleural_effusion",
    "left_pleural_fluid":               "pleural_effusion",
    "layering_pleural_fluid":           "pleural_effusion",
    "subpulmonic_effusion":             "pleural_effusion",
    "costophrenic_angle_blunting":      "costophrenic_blunting",
    "blunted_costophrenic_angle":       "costophrenic_blunting",
    "blunting_of_costophrenic":         "costophrenic_blunting",
    "blunting_of_costophrenic_angle":   "costophrenic_blunting",
    "costophrenic_blunting":            "costophrenic_blunting",
    "costophrenic_angle_obliteration":  "costophrenic_blunting",

    # --- Atelectasis variants ---
    "right_sided_atelectasis":          "atelectasis",
    "left_sided_atelectasis":           "atelectasis",
    "bibasilar_atelectasis":            "atelectasis",
    "bilateral_atelectasis":            "atelectasis",
    "basilar_atelectasis":              "atelectasis",
    "linear_atelectasis":               "atelectasis",
    "discoid_atelectasis":              "atelectasis",
    "plate_like_atelectasis":           "atelectasis",
    "plate_atelectasis":                "atelectasis",
    "subsegmental_atelectasis":         "atelectasis",
    "subsegmental_opacity":             "atelectasis",
    "volume_loss_atelectasis":          "atelectasis",
    "right_lower_lobe_atelectasis":     "atelectasis",
    "left_lower_lobe_atelectasis":      "atelectasis",
    "right_upper_lobe_atelectasis":     "atelectasis",
    "left_upper_lobe_atelectasis":      "atelectasis",
    "collapse":                         "atelectasis",
    "right_lower_lobe_collapse":        "atelectasis",
    "left_lower_lobe_collapse":         "atelectasis",
    "lobar_collapse":                   "atelectasis",

    # --- Cardiomegaly variants ---
    "possible_cardiomegaly":                "cardiomegaly",
    "mild_cardiomegaly":                    "cardiomegaly",
    "moderate_cardiomegaly":               "cardiomegaly",
    "severe_cardiomegaly":                  "cardiomegaly",
    "enlarged_cardiac_silhouette":          "cardiomegaly",
    "prominent_cardiac_silhouette":         "cardiomegaly",
    "large_cardiac_silhouette":             "cardiomegaly",
    "heart_enlargement":                    "cardiomegaly",
    "enlarged_heart":                       "cardiomegaly",
    "large_heart":                          "cardiomegaly",
    "increased_cardiothoracic_ratio":       "cardiomegaly",
    "borderline_cardiomegaly":              "cardiomegaly",
    "borderline_enlarged_heart":            "cardiomegaly",
    "mildly_enlarged_cardiac_silhouette":   "cardiomegaly",
    "globular_cardiomegaly":                "cardiomegaly",
    "globular_heart":                       "cardiomegaly",

    # --- Pneumothorax variants ---
    "right_sided_pneumothorax":         "pneumothorax",
    "left_sided_pneumothorax":          "pneumothorax",
    "right_pneumothorax":               "pneumothorax",
    "left_pneumothorax":                "pneumothorax",
    "tension_pneumothorax":             "pneumothorax",
    "visceral_pleural_line":            "pneumothorax",
    "pleural_line_visible":             "pneumothorax",
    "absent_lung_markings":             "pneumothorax",
    "hyperlucency_hemithorax":          "pneumothorax",
    "small_pneumothorax":               "pneumothorax",
    "apical_pneumothorax":              "pneumothorax",
    "right_apical_pneumothorax":        "pneumothorax",
    "left_apical_pneumothorax":         "pneumothorax",

    # --- Consolidation / Infiltrates ---
    "bilateral_infiltrates":            "consolidation",
    "bilateral_opacities":              "consolidation",
    "bilateral_airspace_opacities":     "consolidation",
    "opacity_left_lung":                "consolidation",
    "opacity_right_lung":               "consolidation",
    "air_space_opacity":                "consolidation",
    "airspace_opacity":                 "consolidation",
    "alveolar_opacity":                 "consolidation",
    "haziness":                         "consolidation",
    "lung_haziness":                    "consolidation",
    "perihilar_haziness":               "pulmonary_edema",
    "air_space_disease":                "consolidation",
    "airspace_disease":                 "consolidation",
    "infiltrate":                       "consolidation",
    "infiltrates":                      "consolidation",
    "pulmonary_infiltrates":            "consolidation",
    "patchy_opacity":                   "consolidation",
    "focal_opacity":                    "consolidation",
    "patchy_consolidation":             "consolidation",
    "multifocal_consolidation":         "consolidation",
    "multifocal_infiltrates":           "consolidation",
    "unilateral_opacity":               "consolidation",
    "right_lung_opacity":               "consolidation",
    "left_lung_opacity":                "consolidation",
    "dense_opacity":                    "consolidation",
    "alveolar_infiltrate":              "consolidation",

    # --- Lobar consolidation (more specific) ---
    "lower_lobe_consolidation":             "lobar_consolidation",
    "right_lower_lobe_consolidation":       "lobar_consolidation",
    "left_lower_lobe_consolidation":        "lobar_consolidation",
    "upper_lobe_consolidation":             "lobar_consolidation",
    "right_upper_lobe_consolidation":       "lobar_consolidation",
    "left_upper_lobe_consolidation":        "lobar_consolidation",
    "lobar_opacity":                        "lobar_consolidation",
    "segmental_consolidation":              "lobar_consolidation",
    "right_lower_lobe_opacity":             "lobar_consolidation",
    "lower_lobe_opacity":                   "lobar_consolidation",
    "lobar_infiltrate":                     "lobar_consolidation",
    "lobar_pneumonia_pattern":              "lobar_consolidation",
    "dense_lobar_consolidation":            "lobar_consolidation",
    "right_middle_lobe_consolidation":      "lobar_consolidation",

    # --- Ground glass opacity ---
    "ground_glass":                     "ground_glass_opacity",
    "gg_opacity":                       "ground_glass_opacity",
    "ground_glass_changes":             "ground_glass_opacity",
    "hazy_opacity":                     "ground_glass_opacity",
    "reticular_opacity":                "ground_glass_opacity",
    "reticular_pattern":                "ground_glass_opacity",
    "interstitial_opacity":             "ground_glass_opacity",
    "interstitial_infiltrates":         "ground_glass_opacity",
    "bilateral_interstitial_infiltrates": "ground_glass_opacity",
    "interstitial_marking":             "ground_glass_opacity",

    # --- Pulmonary edema ---
    "pulmonary_vascular_congestion":    "pulmonary_edema",
    "vascular_congestion":              "pulmonary_edema",
    "cephalization":                    "pulmonary_edema",
    "vascular_redistribution":          "pulmonary_edema",
    "increased_vascular_markings":      "pulmonary_edema",
    "fluid_overload":                   "pulmonary_edema",
    "interstitial_edema":               "pulmonary_edema",
    "alveolar_edema":                   "pulmonary_edema",
    "cardiogenic_edema":                "pulmonary_edema",
    "pulmonary_congestion":             "pulmonary_edema",
    "increased_perihilar_markings":     "pulmonary_edema",
    "bat_wing_pattern":                 "pulmonary_edema",
    "butterfly_pattern":                "pulmonary_edema",

    # --- Kerley B lines ---
    "kerley_lines":                     "kerley_b_lines",
    "kerley_b_lines_present":           "kerley_b_lines",
    "septal_lines":                     "kerley_b_lines",
    "interstitial_lines":               "kerley_b_lines",
    "horizontal_lines_lung_base":       "kerley_b_lines",

    # --- Hyperinflation / COPD ---
    "hyperexpansion":                   "hyperinflation",
    "overdistension":                   "hyperinflation",
    "lung_hyperinflation":              "hyperinflation",
    "increased_lung_volume":            "hyperinflation",
    "increased_ap_diameter":            "hyperinflation",
    "barrel_chest":                     "hyperinflation",
    "air_trapping_pattern":             "air_trapping",
    "bilateral_hyperinflation":         "hyperinflation",
    "bilateral_air_trapping":           "air_trapping",
    "flattening_of_diaphragms":         "flattened_diaphragms",
    "depressed_diaphragms":             "flattened_diaphragms",
    "low_flat_diaphragms":              "flattened_diaphragms",

    # --- Normal/clear findings (Z03.89, J06.9, etc.) ---
    "no_acute_disease":                         "no_acute_findings",
    "no_acute_chest_disease":                   "no_acute_findings",
    "no_acute_cardiopulmonary_findings":        "no_acute_findings",
    "no_acute_cardiopulmonary_disease":         "no_acute_findings",
    "no_acute_intrathoracic_process":           "no_acute_findings",
    "no_acute_process":                         "no_acute_findings",
    "no_acute_pulmonary_disease":               "no_acute_findings",
    "no_significant_abnormality":               "no_acute_findings",
    "unremarkable_chest":                       "normal_chest",
    "normal_radiograph":                        "normal_chest",
    "normal_chest_radiograph":                  "normal_chest",
    "normal_chest_xray":                        "normal_chest",
    "no_abnormality":                           "normal_chest",
    "clear_lungs":                              "clear_lung_fields",
    "clear_bilateral_lungs":                    "clear_lung_fields",
    "lungs_clear":                              "clear_lung_fields",
    "no_infiltrates":                           "clear_lung_fields",
    "no_consolidation":                         "clear_lung_fields",
    "no_opacities":                             "clear_lung_fields",
    "normal_cardiac_size":                      "normal_heart_size",
    "normal_heart":                             "normal_heart_size",
    "no_cardiomegaly":                          "normal_heart_size",
    "normal_cardiac_silhouette":                "normal_heart_size",

    # --- Rib fractures ---
    "rib_fractures":                    "multiple_rib_fractures",
    "multiple_rib_fracture":            "multiple_rib_fractures",
    "bilateral_rib_fractures":          "multiple_rib_fractures",
    "right_rib_fractures":              "multiple_rib_fractures",
    "left_rib_fractures":               "multiple_rib_fractures",
    "single_rib_fracture":              "rib_fracture_visible",
    "cortical_break":                   "cortical_disruption",
    "rib_cortical_disruption":          "cortical_disruption",

    # --- Air bronchograms ---
    "air_bronchogram":                  "air_bronchograms",
    "air_bronchogram_sign":             "air_bronchograms",
    "bronchograms":                     "air_bronchograms",
    "visible_bronchograms":             "air_bronchograms",
}

# Prefixes to strip if full-term lookup in RADLEX_SYNONYM_MAP fails
_STRIP_PREFIXES = [
    "right_sided_", "left_sided_", "bilateral_",
    "possible_", "probable_", "suspected_", "likely_",
    "mild_", "moderate_", "severe_", "chronic_", "acute_",
    "right_", "left_", "bibasilar_", "basilar_",
    "upper_lobe_", "lower_lobe_", "middle_lobe_",
    "right_lower_lobe_", "left_lower_lobe_",
    "right_upper_lobe_", "left_upper_lobe_",
    "right_middle_lobe_",
    "new_", "old_", "known_", "stable_", "apparent_",
    "increased_", "decreased_", "worsening_", "improving_",
    "interstitial_", "alveolar_", "subsegmental_",
    "linear_", "discoid_", "plate_like_", "borderline_",
    "focal_", "diffuse_", "patchy_", "multifocal_",
    "perihilar_", "peribronchovascular_",
    "small_", "large_", "dense_", "subtle_", "faint_",
    "bilateral_lower_",
]


def normalize_finding(f: str) -> str:
    """
    Normalize a single finding to standardized knowledge-engine vocabulary.

    Steps:
      1. Clean and lowercase
      2. Check RADLEX_SYNONYM_MAP for exact match
      3. Strip common prefixes and retry exact map lookup
      4. Return stripped form if no map match (shorter is more general)
    """
    f = f.lower().strip().replace(" ", "_").replace("-", "_")

    # Step 1: direct synonym lookup
    if f in RADLEX_SYNONYM_MAP:
        return RADLEX_SYNONYM_MAP[f]

    # Step 2: strip prefixes and retry
    for prefix in _STRIP_PREFIXES:
        if f.startswith(prefix):
            stripped = f[len(prefix):]
            if stripped in RADLEX_SYNONYM_MAP:
                return RADLEX_SYNONYM_MAP[stripped]
            if stripped:
                return stripped   # return stripped even if not in map (more general term)

    return f


def normalize_findings(findings: list) -> list:
    """
    Normalize a list of findings.

    Both original AND normalized forms are retained so the LLM evaluator
    gets maximum context (original term = provenance, normalized = clean match).
    Returns deduplicated sorted list.
    """
    expanded: set = set()
    for f in findings:
        clean = f.lower().strip().replace(" ", "_").replace("-", "_")
        expanded.add(clean)                 # keep original
        norm = normalize_finding(clean)
        if norm and norm != clean:
            expanded.add(norm)              # add normalized form

    expanded.discard("")
    return sorted(expanded)
