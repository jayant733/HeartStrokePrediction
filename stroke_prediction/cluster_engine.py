"""
Unsupervised Patient Segmentation & Phenotyping Engine
======================================================
Flow:
  Patient Features → Preprocessing Pipeline (StandardScaler + OneHotEncoder)
      → Unsupervised Clustering (K-Means / 4 Clinical Phenotypes)
      → Resolve Patient Cluster & Clinical Archetype
      → Retrieve Cluster-Specific Targeted Prescriptions & Diagnostic Protocols
      → Combine with Supervised Risk Classification for Complete Clinical Guidance

Clinical Phenotypes:
  Cluster 0: Senior Cardio-Vascular & Arterial Stiffness Cohort (High Risk)
  Cluster 1: Workplace Stress, Nicotine & Cardiometabolic Strain Cohort (High Risk)
  Cluster 2: Pediatric & Young Juvenile Resilient Cohort (Low Risk)
  Cluster 3: Mid-Life Non-Smoker Metabolic Watch Cohort (Moderate Risk)

Disclaimer:
  All prescription information is educational and advisory clinical guidance.
  It does NOT replace the clinical judgment of a licensed physician.
"""

import os
import pickle
import pandas as pd
from typing import Dict, Any, List

from stroke_prediction.data_processing import get_model_path, pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CLINICAL ARCHETYPES & TARGETED PRESCRIPTIONS KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_PROFILES: Dict[int, Dict[str, Any]] = {
    0: {
        "cluster_id": 0,
        "name": "Senior Cardio-Vascular & Arterial Stiffness Cohort",
        "tag": "Geriatric Vascular Phenotype",
        "color": "#D32F2F",
        "archetype": (
            "Senior patients (mean age ~60, range 45–82) presenting with elevated vascular aging, "
            "highest rate of chronic hypertension (19.2%), cardiac history, and peak historical stroke incidence (28.4%). "
            "Characterized by arterial stiffness, impaired vascular compliance, and microvascular cerebral strain."
        ),
        "population_share": "27.4%",
        "historical_stroke_rate": "28.4%",
        "benchmarks": {
            "mean_age": 59.7,
            "hypertension_rate": "19.2%",
            "heart_disease_rate": "9.6%",
            "mean_glucose": "113.7 mg/dL",
            "mean_bmi": "29.7 kg/m²",
            "predominant_work": "Self-employed & Public sector",
        },
        "primary_risk_drivers": [
            "Advanced neurovascular and systemic arterial remodeling",
            "Chronic essential hypertension and elevated pulse pressure",
            "Carotid and coronary atherosclerotic plaque burden",
        ],
        "targeted_prescriptions": [
            {
                "drug_class": "Angiotensin II Receptor Blocker (ARB) / ACE Inhibitor",
                "examples": "Telmisartan 40–80 mg daily OR Lisinopril 10–20 mg daily",
                "rationale": "First-line blood pressure regulation; attenuates arterial stiffness, confers renal protection, and reduces secondary stroke recurrence.",
                "monitoring": "Serum creatinine and potassium baseline, repeated at 2 weeks.",
            },
            {
                "drug_class": "Dihydropyridine Calcium Channel Blocker (CCB)",
                "examples": "Amlodipine 5–10 mg daily",
                "rationale": "Potent peripheral arterial vasodilation; synergistic with ARBs for sustained 24-hour hemodynamic stability in geriatric patients.",
                "monitoring": "Evaluate for peripheral pedal edema and orthostatic vitals.",
            },
            {
                "drug_class": "Moderate-to-High Intensity Statin Therapy",
                "examples": "Atorvastatin 20–40 mg daily OR Rosuvastatin 10–20 mg daily",
                "rationale": "Plaque stabilization, anti-inflammatory vascular remodeling, and regression of carotid intima-media thickness.",
                "monitoring": "Baseline ALT/AST and fasting lipid panel at 8–12 weeks.",
            },
            {
                "drug_class": "Antiplatelet Agent (Physician Evaluated)",
                "examples": "Low-dose Aspirin (81–100 mg daily) OR Clopidogrel (75 mg daily)",
                "rationale": "Inhibits platelet aggregation in narrowed cranial and systemic vessels. Strict individual risk-benefit assessment for bleeding risk.",
                "monitoring": "Monitor for gastrointestinal bleeding signs; co-prescribe PPI if history of peptic ulcer.",
            },
        ],
        "diagnostic_workup": [
            "Carotid Doppler Ultrasonography (assess carotid stenosis and luminal flow velocities)",
            "24-Hour Ambulatory Blood Pressure Monitoring (ABPM) to detect nocturnal non-dipping patterns",
            "12-Lead Electrocardiogram (ECG) and Echocardiography (screen for left ventricular hypertrophy and paroxysmal Atrial Fibrillation)",
            "Comprehensive Renal Panel (eGFR, serum creatinine, urine albumin-to-creatinine ratio)",
        ],
        "targeted_lifestyle": [
            "Strict sodium restriction (< 1,500 mg/day DASH dietary pattern)",
            "Fall-prevention balance therapy (Tai Chi, supervised stability exercises)",
            "Avoid intense isometric straining or sudden postural shifts",
        ],
    },
    1: {
        "cluster_id": 1,
        "name": "Workplace Stress, Nicotine & Cardiometabolic Strain Cohort",
        "tag": "Nicotine-Metabolic Stress Phenotype",
        "color": "#E65100",
        "archetype": (
            "Working-age professionals (mean age ~53, 100% private sector) exhibiting intense tobacco exposure "
            "(75% active or former smokers), high Class I obesity (mean BMI 31.5), and cardiac comorbidities (12.5%). "
            "High historical stroke rate (26.8%) driven by chronic endothelial inflammation and metabolic dysregulation."
        ),
        "population_share": "31.7%",
        "historical_stroke_rate": "26.8%",
        "benchmarks": {
            "mean_age": 53.1,
            "hypertension_rate": "11.7%",
            "heart_disease_rate": "12.5%",
            "mean_glucose": "113.6 mg/dL",
            "mean_bmi": "31.5 kg/m²",
            "predominant_work": "Private Enterprise Workforce",
        },
        "primary_risk_drivers": [
            "Endothelial oxidative damage and prothrombotic state caused by nicotine",
            "Elevated visceral adiposity and early metabolic syndrome",
            "High-tempo corporate stress, sedentary sitting hours, and sympathetic overdrive",
        ],
        "targeted_prescriptions": [
            {
                "drug_class": "Smoking Cessation Pharmacotherapy",
                "examples": "Varenicline (Chantix) 0.5 mg daily titrated to 1 mg BID OR Bupropion SR 150 mg BID",
                "rationale": "High-efficacy smoking cessation; acts on alpha-4 beta-2 nicotinic receptors, neutralizing withdrawal craving and reducing relapse.",
                "monitoring": "Screen for neuropsychiatric symptoms or mood alterations during titration.",
            },
            {
                "drug_class": "Nicotine Replacement Therapy (NRT) Combination",
                "examples": "Nicotine Transdermal Patch (Step 1: 21 mg/24h) + Nicotine Gum/Lozenge (2–4 mg PRN)",
                "rationale": "Provides steady basal nicotine levels combined with fast-acting relief for acute breakthrough cravings.",
                "monitoring": "Step down patch dosage every 4 weeks (21 mg → 14 mg → 7 mg).",
            },
            {
                "drug_class": "Cardiometabolic Beta-1 Selective / Vasodilating Blocker",
                "examples": "Nebivolol 5 mg daily OR Metoprolol Succinate 25–50 mg daily",
                "rationale": "Controls stress-induced sympathetic tachycardia, blunt resting heart rate, and limits catecholamine-mediated vascular spasm.",
                "monitoring": "Monitor resting heart rate (target 60–70 bpm) and bronchial symptoms.",
            },
            {
                "drug_class": "Cardiovascular Statin & Triglyceride Optimizer",
                "examples": "Rosuvastatin 10–20 mg daily + Icosapent Ethyl (Purified EPA) 2 g BID",
                "rationale": "Addresses elevated triglycerides and remnant cholesterol accelerated by tobacco use, lowering ischemic stroke hazards.",
                "monitoring": "Fasting lipid panel at 12 weeks.",
            },
        ],
        "diagnostic_workup": [
            "Coronary Artery Calcium (CAC) CT Scoring (early subclinical atherosclerosis detection)",
            "Spirometry / Pulmonary Function Testing (screen for tobacco-related early COPD/hypoxia)",
            "Fasting Insulin & HOMA-IR Index (quantify metabolic insulin resistance)",
            "Treadmill Exercise Stress Test (evaluate coronary perfusion under exertion)",
        ],
        "targeted_lifestyle": [
            "Mandatory workplace ergonomic protocol: sit-to-stand transitions every 45 minutes",
            "Structured high-frequency low-impact aerobic cardio (30 min daily brisk walking/cycling)",
            "Anti-inflammatory Mediterranean diet high in omega-3 and polyphenol antioxidants",
        ],
    },
    2: {
        "cluster_id": 2,
        "name": "Pediatric & Young Juvenile Resilient Cohort",
        "tag": "Juvenile Low-Risk Phenotype",
        "color": "#2E7D32",
        "archetype": (
            "Children, adolescents, and young youth (mean age 13.5) with pristine neurovascular integrity: "
            "0.0% hypertension, minimal cardiac anomalies (0.8%), and lowest baseline stroke incidence (3.1%). "
            "Displays high physiological vascular elasticity and low atherosclerotic predisposition."
        ),
        "population_share": "15.2%",
        "historical_stroke_rate": "3.1%",
        "benchmarks": {
            "mean_age": 13.5,
            "hypertension_rate": "0.0%",
            "heart_disease_rate": "0.8%",
            "mean_glucose": "97.5 mg/dL",
            "mean_bmi": "23.0 kg/m²",
            "predominant_work": "School / Children / Youth",
        },
        "primary_risk_drivers": [
            "No conventional lifestyle vascular risk factors identified",
            "Congenital cardiac malformations or hereditary thrombophilias only if symptomatic",
        ],
        "targeted_prescriptions": [
            {
                "drug_class": "Pharmaceuticals Contraindicated",
                "examples": "No prescription medications required or indicated",
                "rationale": "Prescription pharmacological intervention is clinically contraindicated in healthy pediatric/adolescent profiles.",
                "monitoring": "Routine yearly developmental physical examinations only.",
            },
        ],
        "diagnostic_workup": [
            "Annual Routine Pediatric / Adolescent Well-Child Examination",
            "Standard Growth & BMI Percentile Charting",
            "Cardiovascular family history screening for premature early-onset thrombosis",
        ],
        "targeted_lifestyle": [
            "Encourage at least 60 minutes of daily active physical play and team sports",
            "Limit recreational screen time to < 2 hours per day",
            "Encourage balanced whole-food nutrition, strictly limiting sweetened carbonated beverages",
        ],
    },
    3: {
        "cluster_id": 3,
        "name": "Mid-Life Non-Smoker Metabolic Watch Cohort",
        "tag": "Metabolic Pre-Clinical Watch Phenotype",
        "color": "#1976D2",
        "archetype": (
            "Middle-aged adults (mean age ~46) with zero smoking history (100% never smoked), "
            "presenting with emergent metabolic adiposity (mean BMI 30.9) and early pre-hypertension (12.6%). "
            "Moderate historical stroke rate of 14.4%, primarily driven by early insulin resistance and visceral adiposity."
        ),
        "population_share": "25.7%",
        "historical_stroke_rate": "14.4%",
        "benchmarks": {
            "mean_age": 46.3,
            "hypertension_rate": "12.6%",
            "heart_disease_rate": "6.5%",
            "mean_glucose": "111.5 mg/dL",
            "mean_bmi": "30.9 kg/m²",
            "predominant_work": "Corporate & Mixed Work",
        },
        "primary_risk_drivers": [
            "Early insulin resistance and gradual glycemic creeping (pre-diabetes)",
            "Visceral adiposity accelerating low-grade systemic inflammation",
            "Gradual mid-life decline in physical conditioning and metabolic expenditure",
        ],
        "targeted_prescriptions": [
            {
                "drug_class": "Biguanide / Insulin Sensitizer (If Glycemia > 100 mg/dL)",
                "examples": "Metformin 500 mg daily with evening meal (titrate up to 1,000 mg as needed)",
                "rationale": "Enhances peripheral glucose utilization in skeletal muscle, suppresses hepatic gluconeogenesis, and protects endothelial function.",
                "monitoring": "Fasting glucose, HbA1c, and eGFR every 6 months.",
            },
            {
                "drug_class": "Endothelial & Lipid Support Nutraceuticals",
                "examples": "Purified Omega-3 Fatty Acids (EPA/DHA 2,000 mg daily) + CoQ10 (Ubiquinol 100 mg daily)",
                "rationale": "Reduces vascular inflammation, improves lipid sub-fractions, and provides cardioprotective cellular energy support.",
                "monitoring": "Lipid panel check every 6 months.",
            },
            {
                "drug_class": "Mild Thiazide-like Diuretic (If BP persistently ≥ 135/85 mmHg)",
                "examples": "Chlorthalidone 12.5 mg daily OR Indapamide 1.5 mg SR daily",
                "rationale": "Gentle long-acting volume and vascular resistance modulation for early metabolic pre-hypertension.",
                "monitoring": "Serum electrolytes (sodium, potassium) at 4 weeks.",
            },
        ],
        "diagnostic_workup": [
            "Fasting Blood Glucose & Glycated Hemoglobin (HbA1c) every 6 months",
            "Comprehensive Metabolic Panel (CMP) + Liver Function (screen for NAFLD/fatty liver)",
            "Advanced Lipid Profile (ApoB, LDL particle count, Triglyceride-to-HDL ratio)",
            "Home Blood Pressure Log (periodic 7-day morning/evening logging)",
        ],
        "targeted_lifestyle": [
            "DASH / Low-Glycemic Index dietary plan rich in legumes, leafy vegetables, and lean protein",
            "Hypertrophy / Resistance strength training 3x/week to maximize muscle glucose disposal",
            "Target 10,000 steps daily or 150 minutes of zone-2 aerobic conditioning",
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER & CLUSTER INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

_KMEANS_MODEL = None


def get_clustering_model():
    """Load and cache the trained KMeans clustering model."""
    global _KMEANS_MODEL
    if _KMEANS_MODEL is None:
        model_path = get_model_path("kmeans_clustering.pickle")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                _KMEANS_MODEL = pickle.load(f)
        else:
            raise FileNotFoundError(f"KMeans model not found at {model_path}. Train it first.")
    return _KMEANS_MODEL


def predict_patient_cluster(patient_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Predict the unsupervised cluster/phenotype for a patient dataframe.

    Parameters:
        patient_df (pd.DataFrame): Raw patient dataframe with inference columns:
                                   gender, age, hypertension, heart_disease, ever_married,
                                   work_type, Residence_type, avg_glucose_level, bmi, smoking_status

    Returns:
        dict containing:
            - cluster_id (int)
            - name (str)
            - tag (str)
            - color (str)
            - archetype (str)
            - population_share (str)
            - historical_stroke_rate (str)
            - benchmarks (dict)
            - primary_risk_drivers (list)
            - targeted_prescriptions (list of dicts)
            - diagnostic_workup (list)
            - targeted_lifestyle (list)
    """
    patient_df = patient_df.copy()
    if "id" not in patient_df.columns:
        patient_df["id"] = 0
    processed_df = pipeline(patient_df)
    model = get_clustering_model()
    cluster_idx = int(model.predict(processed_df)[0])

    profile = CLUSTER_PROFILES.get(cluster_idx, CLUSTER_PROFILES[0]).copy()
    return profile


def get_all_cluster_summaries() -> List[Dict[str, Any]]:
    """Return high-level summary of all discovered clusters for UI display."""
    summaries = []
    for cid, prof in CLUSTER_PROFILES.items():
        summaries.append({
            "cluster_id": cid,
            "name": prof["name"],
            "tag": prof["tag"],
            "color": prof["color"],
            "population_share": prof["population_share"],
            "historical_stroke_rate": prof["historical_stroke_rate"],
            "archetype": prof["archetype"],
        })
    return summaries
