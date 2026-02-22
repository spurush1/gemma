"""
Medical Knowledge Engine — Causal Requirements for ICD-10 Codes
================================================================
Provides CausalRequirement definitions for each code:
  - What visual features are required (AND/OR logic)
  - Anatomical location
  - Clinical "why" (causal necessity)
  - Fraud differential (most common misuse code)
  - Litigation risk scoring for downcoding cases
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Add backend dir to path if running standalone
sys.path.insert(0, os.path.dirname(__file__))

try:
    from icd10_data import ICD10_DATA, CLINICAL_AUGMENTATION
except ImportError:
    ICD10_DATA = {}
    CLINICAL_AUGMENTATION = {}

# YAML rules file — lives next to this module
_RULES_FILE = Path(__file__).parent / "causal_rules.yaml"

# Confidence penalty applied when validated=false (auto-generated rules)
# Prevents auto-generated rules from over-flagging fraud
_UNVALIDATED_PENALTY = 0.15

# Hedge prefixes that should be stripped for finding/symptom matching
# "possible_cardiomegaly" → also try "cardiomegaly"
# Used only in evaluate_text_symptoms() for symptom normalization.
# evaluate_findings() now delegates to MedGemma for semantic matching
# so these prefixes are no longer needed there.
_HEDGE_PREFIXES = (
    "possible_", "suspected_", "probable_", "likely_",
    "mild_", "moderate_", "severe_", "bilateral_", "diffuse_",
)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class CausalRequirement:
    code: str
    description: str
    anatomical_location: str
    causal_necessity: str                     # clinical "why"
    visual_required_any: list                 # OR — at least one must be present
    visual_required_all: list                 # AND — ALL must be present
    visual_supporting: list                   # boosts score, not gating
    alternative_diagnoses: list               # sibling codes for counterfactual
    fraud_differential: str                   # most common misuse code
    minimum_causal_score: float = 0.6        # threshold below which fraud is flagged
    voice_keywords: list = field(default_factory=list)  # symptom words patient would say
    # Provenance — set by YAML loader, determines runtime confidence tier
    source: str = "expert_validated"          # expert_validated | medgemma_generated | umls_snomed
    validated: bool = True                    # false = auto-generated, confidence penalty applied


# =========================================================
# MAIN CLASS
# =========================================================

class MedicalKnowledgeGraph:
    """
    Symbol-Neural Hybrid — ground-truth causal requirements for ICD-10 codes.
    Provides AND/OR visual feature matching, counterfactual prompts,
    and litigation risk scoring.
    """

    def __init__(self):
        self._requirements: dict = {}
        self._build()

    def _build(self):
        """
        Load causal requirements from causal_rules.yaml.
        No per-code Python needed — adding a code = adding YAML entry + restart.
        Falls back to empty graph if file is missing (system still runs,
        evaluate_findings returns code_known=False for all codes).
        """
        if not _RULES_FILE.exists():
            import logging
            logging.getLogger("knowledge_engine").warning(
                f"causal_rules.yaml not found at {_RULES_FILE}. "
                "No causal requirements loaded. Run scripts/seed_causal_rules.py."
            )
            return

        with open(_RULES_FILE, "r") as f:
            rules = yaml.safe_load(f) or {}

        loaded = 0
        for code, cfg in rules.items():
            if not isinstance(cfg, dict):
                continue
            self._add(CausalRequirement(
                code=str(code),
                description=cfg.get("description", ""),
                anatomical_location=cfg.get("anatomical_location", ""),
                causal_necessity=cfg.get("causal_necessity", ""),
                visual_required_any=cfg.get("visual_required_any", []),
                visual_required_all=cfg.get("visual_required_all", []),
                visual_supporting=cfg.get("visual_supporting", []),
                alternative_diagnoses=cfg.get("alternative_diagnoses", []),
                fraud_differential=cfg.get("fraud_differential", ""),
                minimum_causal_score=float(cfg.get("minimum_causal_score", 0.6)),
                voice_keywords=cfg.get("voice_keywords", []),
                source=cfg.get("source", "expert_validated"),
                validated=bool(cfg.get("validated", True)),
            ))
            loaded += 1

        import logging
        logging.getLogger("knowledge_engine").info(
            f"Loaded {loaded} causal rules from {_RULES_FILE}"
        )

    def _build_legacy(self):
        """LEGACY — kept for reference only. Replaced by YAML loader above."""
        # ------------------------------------------------------------------
        # RESPIRATORY (legacy hardcoded — DO NOT USE, kept for git history)
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="J18.1",
            description="Lobar pneumonia, unspecified organism",
            anatomical_location="lung — right or left lower lobe",
            causal_necessity=(
                "Lobar consolidation or air bronchograms are radiographic hallmarks "
                "of bacterial lobar pneumonia; without them the diagnosis cannot be "
                "radiographically confirmed."
            ),
            visual_required_any=["lobar_consolidation", "air_bronchograms"],
            visual_required_all=[],
            visual_supporting=["pleural_effusion", "dense_opacification"],
            alternative_diagnoses=["J18.9", "J06.9", "J12.9"],
            fraud_differential="J06.9",
            voice_keywords=["fever", "cough", "chest_pain", "shortness_of_breath", "sputum"],
        ))

        self._add(CausalRequirement(
            code="J18.9",
            description="Unspecified pneumonia",
            anatomical_location="lung — bilateral or unspecified",
            causal_necessity=(
                "Pulmonary infiltrates, consolidation, or diffuse airspace disease "
                "must be present to code pneumonia."
            ),
            visual_required_any=["pulmonary_infiltrates", "consolidation", "airspace_disease",
                                  "lobar_consolidation"],
            visual_required_all=[],
            visual_supporting=["fever_clinical", "bilateral_opacities"],
            alternative_diagnoses=["J18.1", "J06.9", "J22"],
            fraud_differential="J06.9",
            voice_keywords=["fever", "cough", "shortness_of_breath"],
        ))

        self._add(CausalRequirement(
            code="J06.9",
            description="Acute upper respiratory infection, unspecified",
            anatomical_location="upper airway",
            causal_necessity=(
                "Clear lung fields or absence of pulmonary consolidation is required "
                "to code upper respiratory infection. Any lower tract finding rules "
                "this code out."
            ),
            visual_required_any=["normal_chest", "clear_lung_fields", "no_acute_findings"],
            visual_required_all=[],
            visual_supporting=["clear_sinuses"],
            alternative_diagnoses=["J18.9", "J18.1", "J20.9"],
            fraud_differential="J18.1",
            minimum_causal_score=0.6,
            voice_keywords=["runny_nose", "sore_throat", "mild_cough", "cold"],
        ))

        self._add(CausalRequirement(
            code="J45.20",
            description="Mild intermittent asthma, uncomplicated",
            anatomical_location="bronchial airways — bilateral",
            causal_necessity=(
                "Clear lung fields with at most mild hyperinflation; symptoms <2 days/week, "
                "no nocturnal symptoms, and peak flow ≥80% predicted are required for "
                "mild intermittent classification."
            ),
            visual_required_any=["clear_lung_fields", "normal_diaphragms", "mild_hyperinflation",
                                  "no_acute_findings"],
            visual_required_all=[],
            visual_supporting=["normal_cardiac_size"],
            alternative_diagnoses=["J45.30", "J45.40", "J45.50"],
            fraud_differential="J45.50",
            voice_keywords=["occasional_wheeze", "mild_shortness_of_breath"],
        ))

        self._add(CausalRequirement(
            code="J45.50",
            description="Severe persistent asthma, uncomplicated",
            anatomical_location="bronchial airways — bilateral with air trapping",
            causal_necessity=(
                "Radiographic hyperinflation with flattened diaphragms and air trapping "
                "is required. Clinically: daily symptoms, frequent nocturnal episodes, "
                "and peak flow <60% predicted must be documented."
            ),
            visual_required_any=["air_trapping", "flattened_diaphragms", "severe_hyperinflation"],
            visual_required_all=["hyperinflation"],
            visual_supporting=["peribronchial_cuffing", "increased_lung_markings"],
            alternative_diagnoses=["J45.20", "J45.30", "J45.40"],
            fraud_differential="J45.20",
            voice_keywords=["constant_wheeze", "cant_breathe", "daily_symptoms",
                            "nighttime_symptoms"],
        ))

        self._add(CausalRequirement(
            code="J20.9",
            description="Acute bronchitis, unspecified",
            anatomical_location="bronchial tree",
            causal_necessity=(
                "Acute bronchitis shows bronchial wall thickening or peribronchial "
                "infiltrates without lobar consolidation."
            ),
            visual_required_any=["bronchial_thickening", "peribronchial_infiltrates",
                                  "normal_chest"],
            visual_required_all=[],
            visual_supporting=["increased_bronchial_markings"],
            alternative_diagnoses=["J18.9", "J06.9", "J22"],
            fraud_differential="J18.1",
            voice_keywords=["cough", "mucus", "wheezing"],
        ))

        self._add(CausalRequirement(
            code="J22",
            description="Unspecified acute lower respiratory infection",
            anatomical_location="lower respiratory tract",
            causal_necessity=(
                "Lower respiratory involvement (infiltrates, bronchial changes) without "
                "specific lobar consolidation pattern."
            ),
            visual_required_any=["lower_lobe_infiltrates", "bronchial_thickening",
                                  "perihilar_haziness"],
            visual_required_all=[],
            visual_supporting=["mild_airspace_disease"],
            alternative_diagnoses=["J18.9", "J20.9"],
            fraud_differential="J18.1",
            voice_keywords=["cough", "fever", "shortness_of_breath"],
        ))

        self._add(CausalRequirement(
            code="J90",
            description="Pleural effusion",
            anatomical_location="pleural space",
            causal_necessity=(
                "Blunting of costophrenic angle or layering fluid on lateral decubitus "
                "view must be visible radiographically."
            ),
            visual_required_any=["pleural_effusion", "costophrenic_blunting",
                                  "layering_fluid"],
            visual_required_all=[],
            visual_supporting=["mediastinal_shift"],
            alternative_diagnoses=["J94.8", "I50.1"],
            fraud_differential="J18.9",
            voice_keywords=["shortness_of_breath", "chest_heaviness"],
        ))

        self._add(CausalRequirement(
            code="J93.1",
            description="Spontaneous tension pneumothorax",
            anatomical_location="pleural space — unilateral",
            causal_necessity=(
                "Visceral pleural line with absent lung markings peripheral to it, "
                "and contralateral mediastinal shift are diagnostic."
            ),
            visual_required_any=["visceral_pleural_line", "absent_lung_markings",
                                  "pneumothorax"],
            visual_required_all=[],
            visual_supporting=["mediastinal_shift", "tracheal_deviation"],
            alternative_diagnoses=["J93.0", "J94.8"],
            fraud_differential="J18.9",
            voice_keywords=["sudden_chest_pain", "cant_breathe", "stabbing_pain"],
        ))

        self._add(CausalRequirement(
            code="J96.0",
            description="Acute respiratory failure with hypoxia",
            anatomical_location="lungs — bilateral",
            causal_necessity=(
                "Bilateral pulmonary infiltrates (ARDS pattern) with documented hypoxia "
                "PaO2/FiO2 <300 or SpO2 <90% are required."
            ),
            visual_required_any=["bilateral_infiltrates", "ards_pattern",
                                  "diffuse_airspace_disease"],
            visual_required_all=[],
            visual_supporting=["bilateral_pleural_effusions", "cardiomegaly"],
            alternative_diagnoses=["J96.1", "J18.9"],
            fraud_differential="J22",
            voice_keywords=["cant_breathe_at_rest", "oxygen_required"],
        ))

        self._add(CausalRequirement(
            code="J96.1",
            description="Chronic respiratory failure with hypoxia",
            anatomical_location="lungs — bilateral",
            causal_necessity=(
                "Chronic hypoxia on home oxygen, with hyperinflation or fibrotic "
                "changes on CXR, documented over months."
            ),
            visual_required_any=["hyperinflation", "pulmonary_fibrosis",
                                  "increased_lung_markings"],
            visual_required_all=[],
            visual_supporting=["flattened_diaphragms", "barrel_chest"],
            alternative_diagnoses=["J96.0", "J44.1"],
            fraud_differential="J22",
            voice_keywords=["always_short_of_breath", "home_oxygen"],
        ))

        self._add(CausalRequirement(
            code="J44.1",
            description="COPD with acute exacerbation",
            anatomical_location="lungs — bilateral with hyperinflation",
            causal_necessity=(
                "Known COPD with radiographic hyperinflation and flattened diaphragms; "
                "acute exacerbation documented clinically."
            ),
            visual_required_any=["hyperinflation", "flattened_diaphragms", "air_trapping"],
            visual_required_all=[],
            visual_supporting=["increased_ap_diameter", "bullae"],
            alternative_diagnoses=["J44.0", "J18.9"],
            fraud_differential="J18.9",
            voice_keywords=["copd", "worse_breathing", "increased_secretions"],
        ))

        # ------------------------------------------------------------------
        # CARDIOVASCULAR
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="I50.1",
            description="Left ventricular failure, unspecified",
            anatomical_location="heart and lungs — bilateral vascular congestion",
            causal_necessity=(
                "Cardiomegaly with Kerley B lines and vascular redistribution is the "
                "radiographic triad of left ventricular failure. Without cardiomegaly, "
                "this code cannot be supported radiographically."
            ),
            visual_required_any=["kerley_b_lines", "pulmonary_edema", "pleural_effusions"],
            visual_required_all=["cardiomegaly"],
            visual_supporting=["vascular_redistribution", "perihilar_haziness"],
            alternative_diagnoses=["I50.9", "I50.32", "I50.40"],
            fraud_differential="J18.1",
            voice_keywords=["cant_sleep_lying_flat", "leg_swelling", "shortness_of_breath",
                            "wake_up_gasping"],
        ))

        self._add(CausalRequirement(
            code="I50.9",
            description="Heart failure, unspecified",
            anatomical_location="heart — bilateral",
            causal_necessity=(
                "Cardiomegaly or enlarged cardiac silhouette must be present. "
                "Some vascular congestion is expected."
            ),
            visual_required_any=["cardiomegaly", "enlarged_cardiac_silhouette",
                                  "pulmonary_vascular_congestion"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions", "kerley_b_lines"],
            alternative_diagnoses=["I50.1", "I50.9", "I51.7"],
            fraud_differential="J18.9",
            voice_keywords=["swollen_legs", "fatigue", "shortness_of_breath"],
        ))

        self._add(CausalRequirement(
            code="I21.0",
            description="ST elevation MI of anterior wall",
            anatomical_location="heart — anterior wall",
            causal_necessity=(
                "ST elevation in precordial leads on ECG is mandatory. CXR may show "
                "cardiomegaly or pulmonary edema in large MI."
            ),
            visual_required_any=["cardiomegaly", "pulmonary_edema", "normal_chest"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions"],
            alternative_diagnoses=["I21.1", "I21.9", "I25.10"],
            fraud_differential="I25.10",
            voice_keywords=["chest_pain", "crushing_pressure", "left_arm_pain",
                            "jaw_pain", "sweating"],
        ))

        self._add(CausalRequirement(
            code="I21.9",
            description="Acute myocardial infarction, unspecified",
            anatomical_location="heart",
            causal_necessity=(
                "Elevated troponin and ECG changes required. CXR may show pulmonary "
                "edema in severe cases."
            ),
            visual_required_any=["normal_chest", "cardiomegaly", "pulmonary_edema"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions"],
            alternative_diagnoses=["I21.0", "I21.1", "I25.10"],
            fraud_differential="I25.10",
            voice_keywords=["chest_pain", "heart_attack"],
        ))

        self._add(CausalRequirement(
            code="I25.10",
            description="Atherosclerotic heart disease",
            anatomical_location="coronary arteries",
            causal_necessity=(
                "Coronary artery disease documented by angiography or imaging. "
                "CXR may be normal or show cardiomegaly."
            ),
            visual_required_any=["normal_chest", "cardiomegaly", "aortic_calcification"],
            visual_required_all=[],
            visual_supporting=["aortic_elongation"],
            alternative_diagnoses=["I21.9", "I20.0"],
            fraud_differential="I21.9",
            voice_keywords=["chest_tightness_exertion", "known_heart_disease"],
        ))

        self._add(CausalRequirement(
            code="I10",
            description="Essential (primary) hypertension",
            anatomical_location="cardiovascular system",
            causal_necessity=(
                "Blood pressure ≥130/80 mmHg on multiple readings. CXR may show "
                "cardiomegaly from hypertensive heart disease."
            ),
            visual_required_any=["normal_chest", "cardiomegaly", "aortic_elongation"],
            visual_required_all=[],
            visual_supporting=["aortic_knob_prominence"],
            alternative_diagnoses=["I11.9", "I12.9"],
            fraud_differential="I50.9",
            minimum_causal_score=0.3,
            voice_keywords=["high_blood_pressure", "headache"],
        ))

        self._add(CausalRequirement(
            code="I48.0",
            description="Paroxysmal atrial fibrillation",
            anatomical_location="heart — atria",
            causal_necessity=(
                "Irregular rhythm on ECG with absent P waves. CXR may show enlarged "
                "left atrium or cardiomegaly."
            ),
            visual_required_any=["normal_chest", "cardiomegaly", "left_atrial_enlargement"],
            visual_required_all=[],
            visual_supporting=["pulmonary_vascular_congestion"],
            alternative_diagnoses=["I48.2", "I49.9"],
            fraud_differential="I50.9",
            voice_keywords=["palpitations", "irregular_heartbeat", "dizzy"],
        ))

        self._add(CausalRequirement(
            code="I51.7",
            description="Cardiomegaly",
            anatomical_location="heart — global enlargement",
            causal_necessity=(
                "Cardiothoracic ratio >0.5 on PA chest X-ray is required for the "
                "radiographic diagnosis of cardiomegaly."
            ),
            visual_required_any=["cardiomegaly", "enlarged_cardiac_silhouette"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions"],
            alternative_diagnoses=["I50.9", "I42.0"],
            fraud_differential="I50.9",
            voice_keywords=["enlarged_heart", "shortness_of_breath"],
        ))

        self._add(CausalRequirement(
            code="I63.9",
            description="Cerebral infarction, unspecified",
            anatomical_location="brain (note: CXR not primary for stroke)",
            causal_necessity=(
                "Brain CT/MRI showing ischemic lesion is the gold standard. "
                "CXR may show aspiration pneumonia as complication."
            ),
            visual_required_any=["normal_chest", "aspiration_pattern", "cardiomegaly"],
            visual_required_all=[],
            visual_supporting=["atrial_fibrillation_evidence"],
            alternative_diagnoses=["I64", "G45.9"],
            fraud_differential="G45.9",
            minimum_causal_score=0.3,
            voice_keywords=["weakness_one_side", "cant_speak", "face_drooping"],
        ))

        # ------------------------------------------------------------------
        # MUSCULOSKELETAL (Rib Fractures — classic fraud pair)
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="S22.3",
            description="Fracture of one rib",
            anatomical_location="rib cage — single rib",
            causal_necessity=(
                "Cortical disruption of a single rib must be visible on CXR or CT. "
                "Without radiographic confirmation, this code lacks imaging support."
            ),
            visual_required_any=["rib_fracture_visible", "cortical_disruption",
                                  "single_rib_fracture"],
            visual_required_all=[],
            visual_supporting=["pleural_effusion", "pneumothorax"],
            alternative_diagnoses=["S22.4", "S22.5"],
            fraud_differential="S22.4",
            voice_keywords=["rib_pain", "cant_take_deep_breath", "chest_injury",
                            "fell_on_chest"],
        ))

        self._add(CausalRequirement(
            code="S22.4",
            description="Multiple fractures of ribs",
            anatomical_location="rib cage — multiple ribs bilateral or unilateral",
            causal_necessity=(
                "Multiple cortical disruptions of two or more ribs, or flail chest "
                "pattern, must be radiographically confirmed. Coding multiple fractures "
                "for a single rib fracture is upcoding."
            ),
            visual_required_any=["multiple_rib_fractures", "flail_chest_pattern",
                                  "bilateral_rib_fractures"],
            visual_required_all=[],
            visual_supporting=["pneumothorax", "hemothorax", "pulmonary_contusion"],
            alternative_diagnoses=["S22.3", "S22.5"],
            fraud_differential="S22.3",
            voice_keywords=["multiple_rib_pain", "trauma", "motor_vehicle_accident",
                            "flail_chest", "cant_breathe_after_trauma"],
        ))

        # ------------------------------------------------------------------
        # GASTROINTESTINAL
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="K25.0",
            description="Gastric ulcer, acute with hemorrhage",
            anatomical_location="stomach",
            causal_necessity=(
                "Hematemesis or melena with confirmed drop in hemoglobin or endoscopic "
                "finding of active bleeding. CXR may show free air under diaphragm if "
                "perforation occurs."
            ),
            visual_required_any=["free_air_under_diaphragm", "normal_chest",
                                  "no_acute_chest_findings"],
            visual_required_all=[],
            visual_supporting=["pneumoperitoneum"],
            alternative_diagnoses=["K25.9", "K92.1", "K26.0"],
            fraud_differential="K25.9",
            voice_keywords=["vomiting_blood", "black_stool", "stomach_pain",
                            "melena"],
        ))

        self._add(CausalRequirement(
            code="K25.9",
            description="Gastric ulcer, unspecified, without hemorrhage",
            anatomical_location="stomach",
            causal_necessity=(
                "Endoscopic or imaging evidence of gastric ulcer without active bleeding "
                "or perforation. Normal CXR (no free air)."
            ),
            visual_required_any=["normal_chest", "no_free_air", "no_acute_chest_findings"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["K25.0", "K25.1"],
            fraud_differential="K25.0",
            minimum_causal_score=0.4,
            voice_keywords=["stomach_pain", "burning_stomach", "antacids_help"],
        ))

        self._add(CausalRequirement(
            code="K35.2",
            description="Acute appendicitis with generalized peritonitis",
            anatomical_location="appendix — right lower quadrant",
            causal_necessity=(
                "CT abdomen showing perforated appendix with free peritoneal air or "
                "fluid. CXR may show free air under diaphragm."
            ),
            visual_required_any=["free_air_under_diaphragm", "normal_chest"],
            visual_required_all=[],
            visual_supporting=["pneumoperitoneum"],
            alternative_diagnoses=["K35.3", "K37"],
            fraud_differential="K35.3",
            voice_keywords=["severe_right_lower_abdominal_pain", "fever",
                            "rigid_abdomen"],
        ))

        self._add(CausalRequirement(
            code="K57.30",
            description="Diverticulosis of large intestine without hemorrhage",
            anatomical_location="colon",
            causal_necessity=(
                "CT or colonoscopy evidence of diverticula without active bleeding. "
                "Normal CXR typically."
            ),
            visual_required_any=["normal_chest", "no_acute_findings"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["K57.31", "K57.32"],
            fraud_differential="K57.32",
            minimum_causal_score=0.3,
            voice_keywords=["abdominal_cramps", "change_in_bowel"],
        ))

        self._add(CausalRequirement(
            code="K70.30",
            description="Alcoholic cirrhosis without ascites",
            anatomical_location="liver",
            causal_necessity=(
                "Imaging evidence of liver cirrhosis (nodular liver, portal hypertension) "
                "without ascites on imaging."
            ),
            visual_required_any=["normal_chest", "elevated_right_hemidiaphragm",
                                  "no_acute_chest_findings"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["K70.31", "K74.60"],
            fraud_differential="K70.31",
            minimum_causal_score=0.3,
            voice_keywords=["liver_disease", "alcohol_history", "jaundice"],
        ))

        self._add(CausalRequirement(
            code="K85.9",
            description="Acute pancreatitis, unspecified",
            anatomical_location="pancreas",
            causal_necessity=(
                "Elevated lipase/amylase with CT evidence of pancreatic inflammation. "
                "CXR may show left-sided pleural effusion."
            ),
            visual_required_any=["left_pleural_effusion", "normal_chest",
                                  "left_sided_effusion"],
            visual_required_all=[],
            visual_supporting=["elevated_left_hemidiaphragm"],
            alternative_diagnoses=["K85.0", "K85.1"],
            fraud_differential="K85.0",
            voice_keywords=["severe_epigastric_pain", "radiating_back",
                            "nausea_vomiting"],
        ))

        self._add(CausalRequirement(
            code="K92.1",
            description="Melena",
            anatomical_location="gastrointestinal tract",
            causal_necessity=(
                "Black tarry stools documented clinically. Normal CXR unless "
                "complication. Hemoglobin drop typically present."
            ),
            visual_required_any=["normal_chest", "no_acute_chest_findings"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["K25.0", "K26.0", "K92.0"],
            fraud_differential="K25.0",
            minimum_causal_score=0.3,
            voice_keywords=["black_stool", "dark_stool"],
        ))

        # ------------------------------------------------------------------
        # INFECTIONS / SEPSIS
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="A41.9",
            description="Sepsis, unspecified organism",
            anatomical_location="systemic — bloodstream",
            causal_necessity=(
                "Documented infection source with organ dysfunction (SOFA score ≥2). "
                "CXR may show pneumonia source or pulmonary edema from septic shock."
            ),
            visual_required_any=["pulmonary_infiltrates", "bilateral_infiltrates",
                                  "normal_chest", "cardiomegaly"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions"],
            alternative_diagnoses=["A41.0", "A41.1"],
            fraud_differential="J18.9",
            voice_keywords=["very_sick", "confusion", "high_fever", "low_blood_pressure"],
        ))

        # ------------------------------------------------------------------
        # METABOLIC / ENDOCRINE
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            anatomical_location="systemic — endocrine",
            causal_necessity=(
                "HbA1c ≥6.5% or fasting glucose ≥126 mg/dL documented. "
                "CXR typically normal."
            ),
            visual_required_any=["normal_chest", "no_acute_findings"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["E11.65", "E11.40"],
            fraud_differential="E11.65",
            minimum_causal_score=0.3,
            voice_keywords=["diabetes", "high_blood_sugar", "insulin"],
        ))

        # ------------------------------------------------------------------
        # RENAL
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="N18.3",
            description="Chronic kidney disease, stage 3",
            anatomical_location="kidneys — bilateral",
            causal_necessity=(
                "GFR 30–59 mL/min documented on lab. CXR may show cardiomegaly "
                "from renal hypertension."
            ),
            visual_required_any=["normal_chest", "cardiomegaly", "pulmonary_edema"],
            visual_required_all=[],
            visual_supporting=["pleural_effusions"],
            alternative_diagnoses=["N18.2", "N18.4"],
            fraud_differential="N18.4",
            minimum_causal_score=0.3,
            voice_keywords=["kidney_disease", "low_gfr", "fatigue_ankles"],
        ))

        # ------------------------------------------------------------------
        # INFLUENZA
        # ------------------------------------------------------------------
        self._add(CausalRequirement(
            code="J09.X9",
            description="Influenza due to identified novel influenza A virus",
            anatomical_location="upper and lower respiratory tract",
            causal_necessity=(
                "Positive influenza rapid test or PCR, with systemic symptoms. "
                "CXR may be normal or show bilateral viral pneumonia pattern."
            ),
            visual_required_any=["normal_chest", "bilateral_interstitial_infiltrates",
                                  "viral_pneumonia_pattern"],
            visual_required_all=[],
            visual_supporting=["perihilar_infiltrates"],
            alternative_diagnoses=["J10.1", "J18.9"],
            fraud_differential="J18.9",
            voice_keywords=["flu", "influenza", "body_aches", "high_fever"],
        ))

        self._add(CausalRequirement(
            code="J12.9",
            description="Viral pneumonia, unspecified",
            anatomical_location="lungs — bilateral patchy",
            causal_necessity=(
                "Bilateral interstitial or reticular infiltrates on CXR consistent "
                "with viral pneumonia; bacterial consolidation absent."
            ),
            visual_required_any=["bilateral_interstitial_infiltrates",
                                  "reticular_pattern", "ground_glass_opacity"],
            visual_required_all=[],
            visual_supporting=["perihilar_distribution"],
            alternative_diagnoses=["J18.9", "J18.1"],
            fraud_differential="J18.1",
            voice_keywords=["viral_infection", "covid", "dry_cough"],
        ))

        self._add(CausalRequirement(
            code="J13",
            description="Pneumonia due to Streptococcus pneumoniae",
            anatomical_location="lung — lobar or segmental",
            causal_necessity=(
                "Lobar or segmental consolidation with sputum or blood culture positive "
                "for S. pneumoniae."
            ),
            visual_required_any=["lobar_consolidation", "segmental_consolidation",
                                  "air_bronchograms"],
            visual_required_all=[],
            visual_supporting=["pleural_effusion"],
            alternative_diagnoses=["J18.1", "J18.9"],
            fraud_differential="J06.9",
            voice_keywords=["lobar_pneumonia", "strep_pneumonia", "fever_cough"],
        ))

        self._add(CausalRequirement(
            code="J18.0",
            description="Bronchopneumonia, unspecified organism",
            anatomical_location="lung — bilateral patchy/multifocal",
            causal_necessity=(
                "Multifocal patchy consolidation in a bilateral distribution "
                "rather than lobar pattern."
            ),
            visual_required_any=["patchy_consolidation", "multifocal_infiltrates",
                                  "bilateral_lower_lobe_infiltrates"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["J18.1", "J18.9"],
            fraud_differential="J06.9",
            voice_keywords=["pneumonia", "fever", "bilateral_infection"],
        ))

        # Normal / No Finding (used for false billing)
        self._add(CausalRequirement(
            code="Z03.89",
            description="Encounter for observation for other suspected diseases",
            anatomical_location="systemic observation",
            causal_necessity=(
                "Normal findings or ruled-out condition. No active disease confirmed. "
                "Used for monitoring without diagnosis."
            ),
            visual_required_any=["normal_chest", "no_acute_findings", "clear_lung_fields"],
            visual_required_all=[],
            visual_supporting=[],
            alternative_diagnoses=["J06.9"],
            fraud_differential="J18.1",
            minimum_causal_score=0.5,
            voice_keywords=["feeling_fine", "just_checking", "normal"],
        ))

    def _add(self, req: CausalRequirement):
        """Add a CausalRequirement to the graph."""
        # Fill description from ICD10_DATA if not set
        if not req.description and req.code in ICD10_DATA:
            req.description = ICD10_DATA[req.code].get("description", req.code)
        self._requirements[req.code.upper()] = req

    # =========================================================
    # PUBLIC API
    # =========================================================

    def get_requirements(self, code: str) -> Optional[CausalRequirement]:
        """Return CausalRequirement for a given code, or None if not defined."""
        return self._requirements.get(code.upper())

    def list_codes(self) -> list:
        """Return all codes that have causal requirements defined."""
        return list(self._requirements.keys())

    def evaluate_findings(self, code: str, visual_findings: list) -> dict:
        """
        Core scoring method.
        Compares extracted visual_findings against CausalRequirement.

        Scoring algorithm:
          AND score  (0.5 weight): all required_all features present?
          OR score   (0.4 weight): at least one required_any feature present?
          Supporting (0.1 weight): fraction of supporting features found
          fraud_flagged = causal_score < effective_threshold

        Confidence tiering:
          validated=True  → effective_threshold = minimum_causal_score (strict)
          validated=False → effective_threshold = minimum_causal_score - 0.15
                            (lenient — auto-generated rules don't over-flag fraud)
        """
        req = self.get_requirements(code)
        if req is None:
            return {
                "causal_score": 0.5,
                "all_satisfied": True,
                "any_satisfied": True,
                "found_features": [],
                "missing_features": [],
                "supporting_found": [],
                "fraud_flagged": False,
                "causal_necessity": "No causal rule defined for this code.",
                "code_known": False,
                "rule_source": "none",
                "rule_validated": False,
                "verdict_label": "No rule — cannot assess",
            }

        # Normalize findings to lowercase_underscore
        raw_normalized = {f.lower().replace(" ", "_").replace("-", "_")
                          for f in visual_findings}

        # If no visual findings at all → insufficient evidence, do not flag fraud
        if not raw_normalized:
            return {
                "causal_score": None,
                "all_satisfied": False,
                "any_satisfied": False,
                "found_features": [],
                "missing_features": [],
                "supporting_found": [],
                "fraud_flagged": False,
                "causal_necessity": req.causal_necessity,
                "code_known": True,
                "rule_source": req.source,
                "rule_validated": req.validated,
                "verdict_label": "No image evidence — cannot assess",
            }

        # ── "All-normal" filter ─────────────────────────────────────────
        # MedGemma sometimes reads pathological images as all-normal findings
        # (e.g., "normal_heart_size, clear_lungs" for a cardiomegaly case).
        # All-normal findings carry no pathological signal — they cannot confirm
        # OR meaningfully contradict a code.  Treat as "no useful image evidence"
        # and fall through to text evaluation.
        _NORMAL_PREFIXES = (
            "normal_", "clear_", "unremarkable", "no_", "within_normal",
            "no_acute", "no_focal", "no_evidence", "no_cardiopulmonary",
        )
        is_all_normal = all(
            any(f.startswith(p) for p in _NORMAL_PREFIXES) or f in {
                "normal", "clear", "unremarkable", "bilateral_symmetry",
                "symmetric_lungs", "well_aerated", "well_expanded",
            }
            for f in raw_normalized
        )
        if is_all_normal:
            import logging
            logging.getLogger("knowledge_engine").info(
                f"[{code}] All findings are normal/clear — treating as "
                "uninformative image (MedGemma may not have detected pathology)."
            )
            return {
                "causal_score": None,
                "all_satisfied": False,
                "any_satisfied": False,
                "found_features": list(raw_normalized),
                "missing_features": [],
                "supporting_found": [],
                "fraud_flagged": False,
                "causal_necessity": req.causal_necessity,
                "code_known": True,
                "rule_source": req.source,
                "rule_validated": req.validated,
                "verdict_label": "All-normal image findings — insufficient evidence",
            }

        # ── Tier 1: RadLex normalization ───────────────────────────────────
        # Expand raw findings with standardized synonyms before LLM evaluation.
        # "right_sided_pleural_effusion" → also adds "pleural_effusion"
        # "bilateral_infiltrates"        → also adds "consolidation"
        # "possible_cardiomegaly"        → also adds "cardiomegaly"
        # Both original AND normalized forms are passed to the LLM so it has
        # maximum context for semantic matching.
        try:
            from medical_normalizer import normalize_findings as _norm_findings
            expanded_findings = set(_norm_findings(list(raw_normalized)))
        except ImportError:
            expanded_findings = raw_normalized

        # ── LLM semantic evaluation ────────────────────────────────────────
        # Ask MedGemma whether the observed findings satisfy the code's
        # causal requirements.  This replaces brittle keyword/prefix matching
        # with genuine medical language understanding:
        #   "right_pneumothorax" → satisfies "pneumothorax" ✓
        #   "bibasilar_airspace_disease" → satisfies "airspace_disease" ✓
        #   "enlarged_cardiac_silhouette" → satisfies "cardiomegaly" ✓
        from medgemma_client import evaluate_causal_match

        match = evaluate_causal_match(
            code=code,
            description=req.description,
            causal_necessity=req.causal_necessity,
            required_any=req.visual_required_any,
            required_all=req.visual_required_all,
            visual_findings=sorted(expanded_findings),
        )

        # satisfied=None means API error → conservative: don't flag fraud
        if match["satisfied"] is None:
            return {
                "causal_score": None,
                "all_satisfied": False,
                "any_satisfied": False,
                "found_features": [],
                "missing_features": [],
                "supporting_found": [],
                "fraud_flagged": False,
                "needs_human_review": False,
                "causal_necessity": req.causal_necessity,
                "code_known": True,
                "rule_source": req.source,
                "rule_validated": req.validated,
                "verdict_label": "API error — cannot assess",
            }

        causal_score = round(match["confidence"], 3)

        # ── Tier 3: Human review zone ──────────────────────────────────────
        # Only auto-flag fraud if the LLM says "contradicted" AND is highly
        # confident (≥ HUMAN_REVIEW_THRESHOLD).
        # Borderline cases (LLM says "contradicted" but confidence is low)
        # go to human review instead of being auto-flagged.
        # This prevents over-flagging on ambiguous or incomplete image findings.
        _HUMAN_REVIEW_THRESHOLD = float(
            os.getenv("HUMAN_REVIEW_THRESHOLD", "0.75")
        )
        contradicted = not match["satisfied"]
        needs_human_review = contradicted and (causal_score < _HUMAN_REVIEW_THRESHOLD)
        fraud_flagged = contradicted and (causal_score >= _HUMAN_REVIEW_THRESHOLD)

        effective_threshold = req.minimum_causal_score
        if not req.validated:
            effective_threshold = max(0.0, req.minimum_causal_score - _UNVALIDATED_PENALTY)

        if not req.validated:
            verdict_label = "Possible fraud — rule under clinical review"
        elif fraud_flagged:
            verdict_label = f"Fraud flag — {match['evidence']}"
        elif needs_human_review:
            verdict_label = f"Needs human review (borderline) — {match['evidence']}"
        else:
            verdict_label = f"Code supported — {match['evidence']}"

        missing_features = [f"[MISSING] {m}" for m in match.get("missing", [])]

        return {
            "causal_score": causal_score,
            "all_satisfied": match["satisfied"],
            "any_satisfied": match["satisfied"],
            "found_features": sorted(expanded_findings),
            "missing_features": missing_features,
            "supporting_found": [],
            "fraud_flagged": fraud_flagged,
            "needs_human_review": needs_human_review,
            "causal_necessity": req.causal_necessity,
            "code_known": True,
            "fraud_differential": req.fraud_differential,
            "threshold": req.minimum_causal_score,
            "effective_threshold": effective_threshold,
            "rule_source": req.source,
            "rule_validated": req.validated,
            "verdict_label": verdict_label,
        }

    def evaluate_text_symptoms(self, code: str, symptoms: list) -> dict:
        """
        Text-based causal check used when no X-ray image is available.
        Compares clinical note symptoms against the code's voice_keywords.

        A code's voice_keywords describe what the PATIENT would say if they truly had
        this condition. If the note's symptoms don't match the billed code's keywords,
        this is a strong signal of miscoding.

        Returns the same dict structure as evaluate_findings() for drop-in use.
        """
        _NO_RULE = {
            "causal_score": None,
            "all_satisfied": True,
            "any_satisfied": True,
            "found_features": [],
            "missing_features": [],
            "supporting_found": [],
            "fraud_flagged": False,
            "causal_necessity": "No causal rule defined for this code.",
            "code_known": False,
            "rule_source": "none",
            "rule_validated": False,
            "verdict_label": "No rule — cannot assess (text mode)",
        }

        req = self.get_requirements(code)
        if req is None:
            return _NO_RULE

        if not req.voice_keywords:
            return {**_NO_RULE, "code_known": True, "verdict_label": "No voice_keywords — cannot assess (text mode)"}

        # No symptoms extracted — insufficient evidence
        if not symptoms:
            return {
                "causal_score": None,
                "all_satisfied": False,
                "any_satisfied": False,
                "found_features": [],
                "missing_features": [],
                "supporting_found": [],
                "fraud_flagged": False,
                "causal_necessity": req.causal_necessity,
                "code_known": True,
                "rule_source": req.source,
                "rule_validated": req.validated,
                "verdict_label": "No symptoms extracted — cannot assess (text mode)",
            }

        # Normalize symptoms + strip hedge prefixes (same as evaluate_findings)
        raw_norm = {s.lower().replace(" ", "_").replace("-", "_") for s in symptoms}
        expanded = set(raw_norm)
        for s in raw_norm:
            for prefix in _HEDGE_PREFIXES:
                if s.startswith(prefix):
                    expanded.add(s[len(prefix):])

        # Combined keyword set: voice_keywords ∪ visual_required_any ∪ visual_required_all
        # Mixing both vocabularies handles:
        #   - Patient language → voice_keywords (e.g., "cant_breathe_after_trauma")
        #   - Clinical language → visual_required (e.g., "multiple_rib_fractures")
        combined_keywords = list(set(
            req.voice_keywords
            + req.visual_required_any
            + req.visual_required_all
        ))

        # Strict subset soft-match: ALL words in the *symptom* must appear in the keyword.
        # e.g. "fever" (1 word) → subset of {"mild","fever"} = True  ✓
        #      "chest_pain"   → subset of {"normal","chest"}  = False ✗ (spurious)
        def _soft_match(symptom: str, keyword: str) -> bool:
            s_words = {w for w in symptom.split("_") if len(w) >= 4}
            k_words = {w for w in keyword.split("_") if len(w) >= 4}
            return bool(s_words) and bool(k_words) and s_words.issubset(k_words)

        def _score_against(keywords: list) -> tuple[float, list]:
            matched_kw = []
            for kw in keywords:
                if kw in expanded:
                    matched_kw.append(kw)
                elif any(_soft_match(sym, kw) for sym in expanded):
                    matched_kw.append(kw)
            matched_kw = list(set(matched_kw))
            total = len(keywords) if keywords else 1
            return round(len(matched_kw) / total, 3), matched_kw

        text_score, matched = _score_against(combined_keywords)

        # --- Comparative scoring against the fraud differential ---
        # If the differential code's keywords match the symptoms BETTER than the
        # submitted code, it's likely that a different (potentially correct) code
        # should have been used → fraud signal.
        _COMP_MARGIN = 0.05   # differential must beat submitted score by at least this
        diff_code = req.fraud_differential
        diff_req = self._requirements.get(diff_code.upper()) if diff_code else None

        if diff_req:
            diff_kw = list(set(
                diff_req.voice_keywords
                + diff_req.visual_required_any
                + diff_req.visual_required_all
            ))
            diff_score, _ = _score_against(diff_kw)
            fraud_flagged = diff_score > text_score + _COMP_MARGIN
            effective_threshold = _COMP_MARGIN   # informational only
        else:
            # No differential available — use lenient absolute threshold
            diff_score = None
            effective_threshold = 0.9 / max(len(combined_keywords), 10)
            fraud_flagged = text_score < effective_threshold

        if fraud_flagged:
            if diff_req:
                verdict_label = (
                    f"Fraud flag (text): symptoms fit {diff_code} "
                    f"(score {diff_score:.2f}) better than submitted {code} "
                    f"(score {text_score:.2f})"
                )
            else:
                verdict_label = "Fraud flag (text): symptoms don't align with billed code"
            missing = [kw for kw in req.voice_keywords if kw not in expanded]
            missing_features = [f"[EXPECTED SYMPTOM] {kw}" for kw in missing[:3]]
        else:
            verdict_label = "Text-based check passed — symptoms consistent with code"
            missing_features = []

        return {
            "causal_score": text_score,
            "all_satisfied": bool(matched),
            "any_satisfied": bool(matched),
            "found_features": matched,
            "missing_features": missing_features,
            "supporting_found": [],
            "fraud_flagged": fraud_flagged,
            "causal_necessity": req.causal_necessity,
            "code_known": True,
            "fraud_differential": req.fraud_differential,
            "threshold": req.minimum_causal_score,
            "effective_threshold": effective_threshold,
            "rule_source": req.source,
            "rule_validated": req.validated,
            "verdict_label": verdict_label,
        }

    def get_counterfactual_prompt(
        self,
        code: str,
        alternative_code: str,
        clinical_note: str,
        visual_findings: Optional[list] = None
    ) -> str:
        """
        Build a counterfactual prompt for the MedGemma call.
        Forces causal reasoning: "If patient had X instead of Y, is Y still necessary?"
        """
        req = self.get_requirements(code)
        alt_req = self.get_requirements(alternative_code)

        current_desc = req.description if req else code
        alt_desc = alt_req.description if alt_req else alternative_code
        causal_why = req.causal_necessity if req else f"clinical requirements for {code}"

        findings_text = ""
        if visual_findings:
            findings_text = f"\nVisual findings on imaging: {', '.join(visual_findings)}."

        return (
            f"If this patient had '{alt_desc}' ({alternative_code}) instead of "
            f"'{current_desc}' ({code}), would '{code}' still be medically necessary?\n"
            f"Consider: {causal_why}\n"
            f"Clinical note: {clinical_note[:400]}"
            f"{findings_text}\n"
            f"Answer YES or NO and explain in one sentence why or why not."
        )

    def compute_litigation_risk(
        self,
        submitted_code: str,
        correct_code: str,
        financial_gap: int,
        body_system: str = "other"
    ) -> dict:
        """
        Compute downcoding litigation risk score (0-100).

        Downcoding exposes providers to:
        - RAC (Recovery Audit Contractor) scrutiny
        - Patient harm liability if severity was underestimated
        - False Claims Act exposure for systematic patterns

        Returns: {score, label, factors}
        """
        if financial_gap >= 0:
            # Not downcoding — no litigation risk from downcoding
            return {
                "score": 0,
                "label": "none",
                "factors": [],
                "applies": False,
            }

        abs_gap = abs(financial_gap)
        factors = []
        score = 0

        # Base score from financial gap (capped at 40 points)
        base = min(40, abs_gap // 100)
        score += base
        if abs_gap > 1000:
            factors.append(f"Financial gap of ${abs_gap:,} indicates systematic downcode")

        # Severity penalty
        submitted_data = ICD10_DATA.get(submitted_code, {})
        correct_data = ICD10_DATA.get(correct_code, {})
        severity_order = {"mild": 1, "moderate": 2, "severe": 3}
        sub_sev = severity_order.get(submitted_data.get("severity", "mild"), 1)
        cor_sev = severity_order.get(correct_data.get("severity", "moderate"), 2)
        sev_diff = cor_sev - sub_sev

        if sev_diff >= 2:
            score += 30
            factors.append(
                f"Severity downgrade from '{correct_data.get('severity','severe')}' "
                f"to '{submitted_data.get('severity','mild')}' — patient may have "
                f"received inadequate treatment authorization"
            )
        elif sev_diff == 1:
            score += 15
            factors.append(
                f"Severity understated by one level — treatment may have been limited"
            )

        # Patient harm penalty by body system
        system_lower = body_system.lower() if body_system else "other"
        if system_lower in ("cardiovascular", "respiratory"):
            score += 20
            factors.append(
                f"Downcoding a {system_lower} condition is life-threatening — "
                f"delayed treatment for {correct_code} is a malpractice exposure"
            )
        elif system_lower == "gastrointestinal":
            score += 10
            factors.append(
                f"GI downcoding (e.g., missing hemorrhage) can result in patient harm"
            )
        else:
            score += 5

        # RAC single-case threshold
        if abs_gap > 5000:
            score += 10
            factors.append(
                f"Single-case gap of ${abs_gap:,} exceeds $5,000 RAC audit threshold"
            )

        score = min(100, score)

        if score <= 25:
            label = "low"
        elif score <= 50:
            label = "medium"
        elif score <= 75:
            label = "high"
        else:
            label = "critical"

        return {
            "score": score,
            "label": label,
            "factors": factors,
            "applies": True,
            "rac_flag": abs_gap > 5000,
        }

    def compute_upcoding_exposure(
        self,
        submitted_code: str,
        correct_code: str,
        financial_gap: int
    ) -> dict:
        """
        Compute upcoding financial exposure.
        Returns amount overbilled and payer impact analysis.
        """
        if financial_gap <= 0:
            return {
                "upcoding_detected": False,
                "overbilling_usd": 0,
                "payer_impact": "none",
            }

        submitted_data = ICD10_DATA.get(submitted_code, {})
        correct_data = ICD10_DATA.get(correct_code, {})

        return {
            "upcoding_detected": True,
            "overbilling_usd": financial_gap,
            "submitted_reimbursement": submitted_data.get("avg_reimbursement_usd", 0),
            "correct_reimbursement": correct_data.get("avg_reimbursement_usd", 0),
            "payer_impact": (
                "high" if financial_gap > 2000 else
                "medium" if financial_gap > 500 else
                "low"
            ),
        }


# =========================================================
# MODULE-LEVEL SINGLETON
# =========================================================

_kg_instance: Optional[MedicalKnowledgeGraph] = None


def get_knowledge_graph() -> MedicalKnowledgeGraph:
    """Return the shared MedicalKnowledgeGraph instance (built once)."""
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = MedicalKnowledgeGraph()
    return _kg_instance


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":
    import json
    kg = get_knowledge_graph()
    print(f"Knowledge graph loaded: {len(kg.list_codes())} codes defined")

    # Test 1: J18.1 with correct findings
    result = kg.evaluate_findings("J18.1", ["lobar_consolidation", "air_bronchograms"])
    print(f"\nJ18.1 with correct findings:")
    print(f"  causal_score: {result['causal_score']} | fraud_flagged: {result['fraud_flagged']}")

    # Test 2: J18.1 with NO findings (fraud case)
    result2 = kg.evaluate_findings("J18.1", ["normal_chest", "clear_lung_fields"])
    print(f"\nJ18.1 with clear chest (fraud):")
    print(f"  causal_score: {result2['causal_score']} | fraud_flagged: {result2['fraud_flagged']}")
    print(f"  missing: {result2['missing_features']}")

    # Test 3: S22.4 with only a single rib fracture (upcoding)
    result3 = kg.evaluate_findings("S22.4", ["single_rib_fracture"])
    print(f"\nS22.4 with only single_rib_fracture (should flag):")
    print(f"  causal_score: {result3['causal_score']} | fraud_flagged: {result3['fraud_flagged']}")

    # Test 4: Litigation risk
    risk = kg.compute_litigation_risk("J06.9", "J18.1", -2220, "respiratory")
    print(f"\nLitigation risk for J06.9→J18.1 gap $2220:")
    print(json.dumps(risk, indent=2))

    # Test 5: Counterfactual prompt
    prompt = kg.get_counterfactual_prompt("S22.4", "S22.3", "Patient presents with rib pain")
    print(f"\nCounterfactual prompt (S22.4 vs S22.3):")
    print(prompt)
