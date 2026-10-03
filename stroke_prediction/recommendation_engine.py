"""
Stroke Risk Recommendation Engine
===================================
Flow:
  Patient Data → Stroke Prediction (0 / 1) → Risk Profile Builder
      → Rule-Based Recommendations (Lifestyle, Diet, Medication Advice, Monitoring)
      → Guardrail Filter → Final Recommendations Output

Disclaimer:
  This system provides informational guidance ONLY.
  It does NOT replace a licensed medical professional's diagnosis or prescription.
"""

# ─────────────────────────────────────────────────
# MEDICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────────

LIFESTYLE_RECOMMENDATIONS = {
    "smoking": {
        "smokes": [
            "🚭 Quit smoking immediately — smoking doubles stroke risk.",
            "Seek nicotine replacement therapy (NRT) or counselling support.",
            "Avoid second-hand smoke environments.",
        ],
        "formerly smoked": [
            "✅ Continue avoiding tobacco products.",
            "Maintain current smoke-free lifestyle.",
        ],
        "Unknown": [
            "ℹ️ Clarify your smoking history with your doctor.",
        ],
        "never smoked": [
            "✅ Non-smoking status is a key protective factor — keep it up!",
        ],
    },
    "hypertension": {
        "yes": [
            "📏 Monitor blood pressure daily. Target: < 130/80 mmHg.",
            "Take antihypertensive medications as prescribed without skipping.",
            "Reduce sodium intake to < 2.3g/day (< 1 tsp salt/day).",
            "Practice stress-reduction techniques: meditation, yoga, breathing exercises.",
        ],
        "no": [
            "✅ No hypertension detected. Maintain healthy BP through regular exercise.",
        ],
    },
    "heart_disease": {
        "yes": [
            "❤️ Regular cardiac check-ups every 3–6 months are strongly recommended.",
            "Follow a heart-healthy diet (Mediterranean diet).",
            "Anticoagulant therapy (e.g., Warfarin, Apixaban) may be prescribed by your doctor.",
            "Avoid strenuous physical activities without medical clearance.",
        ],
        "no": [
            "✅ No heart disease detected. Protect your heart with regular moderate exercise.",
        ],
    },
    "bmi": {
        "underweight": [
            "⚖️ BMI is below normal. Work with a nutritionist to increase caloric intake healthily.",
        ],
        "normal": [
            "✅ Healthy BMI. Maintain weight through balanced diet and regular activity.",
        ],
        "overweight": [
            "⚠️ Aim to reduce BMI to healthy range (18.5–24.9). Target 5–10% weight loss.",
            "Engage in at least 150 min/week of moderate aerobic exercise.",
        ],
        "obese": [
            "🚨 Obesity significantly elevates stroke risk. Consult a bariatric specialist.",
            "Adopt caloric-deficit diet with medical supervision.",
            "Target 30 min/day of low-impact exercise (walking, swimming).",
        ],
    },
    "glucose": {
        "normal": [
            "✅ Glucose levels within normal range. Continue a low-sugar diet.",
        ],
        "pre_diabetic": [
            "⚠️ Borderline glucose. Reduce refined carbohydrates and added sugars.",
            "Increase dietary fiber intake (whole grains, legumes, vegetables).",
            "Consider fasting glucose monitoring every 3 months.",
        ],
        "diabetic": [
            "🩺 High glucose indicates possible diabetes — consult your endocrinologist.",
            "Monitor blood sugar daily. Target fasting glucose: 80–130 mg/dL.",
            "Adhere to prescribed hypoglycaemic medications (e.g., Metformin).",
            "Follow a diabetic-friendly meal plan (low GI foods).",
        ],
    },
}

DIETARY_RECOMMENDATIONS = [
    "🥗 Follow a Mediterranean-style diet: olive oil, fish, vegetables, whole grains.",
    "🧂 Limit sodium to < 2.3g/day to control blood pressure.",
    "🍬 Reduce refined sugars and processed carbohydrates.",
    "🐟 Eat omega-3 rich foods: salmon, sardines, flaxseed, walnuts.",
    "🥦 Increase potassium intake: bananas, sweet potatoes, spinach, avocados.",
    "🚫 Avoid trans fats and saturated fats (fast food, margarine).",
    "💧 Stay hydrated: minimum 8 glasses of water per day.",
    "🍷 Limit alcohol to ≤ 1 drink/day (women) or ≤ 2 drinks/day (men).",
]

EXERCISE_RECOMMENDATIONS = [
    "🏃 Aim for ≥ 150 minutes of moderate-intensity aerobic activity per week.",
    "🧘 Include yoga or tai chi for stress reduction and balance.",
    "🚶 Take daily 30-minute brisk walks if gym exercise isn't feasible.",
    "💪 Add 2 days/week of strength training to improve metabolic health.",
    "🩺 Always consult your doctor before starting any new exercise regime.",
]

MEDICATION_ADVISORY = {
    "high_risk": [
        "💊 Your doctor may consider antiplatelet therapy (e.g., Aspirin, Clopidogrel).",
        "💊 Statin therapy (e.g., Atorvastatin, Rosuvastatin) may be prescribed for cholesterol.",
        "💊 Antihypertensives (e.g., Amlodipine, Lisinopril) if blood pressure is elevated.",
        "⚠️ Do NOT self-medicate. All medications must be prescribed by a licensed physician.",
        "📅 Schedule an urgent consultation with your neurologist or cardiologist.",
    ],
    "low_risk": [
        "✅ No immediate medication indicated based on current risk profile.",
        "💊 Continue any currently prescribed medications without interruption.",
        "📅 Schedule a routine check-up within 6 months.",
    ],
}

MONITORING_PLAN = {
    "high_risk": [
        "📏 Blood pressure: measure twice daily, log readings.",
        "🩸 Blood glucose: fasting test every morning.",
        "🏥 Full cardiovascular panel (Cholesterol, ECG) every 3 months.",
        "🧠 Neurological assessment every 6 months.",
        "📲 Use a health tracking app or wearable to log vitals daily.",
    ],
    "low_risk": [
        "📏 Blood pressure: monthly check.",
        "🩸 Annual blood glucose and lipid panel.",
        "🏥 General health check-up once a year.",
    ],
}

EMERGENCY_SIGNS = [
    "🚨 CALL EMERGENCY SERVICES IMMEDIATELY if you notice:",
    "   • Sudden numbness or weakness in face, arm, or leg (especially on one side)",
    "   • Sudden confusion or trouble speaking/understanding speech",
    "   • Sudden vision trouble in one or both eyes",
    "   • Sudden severe headache with no known cause",
    "   • Sudden dizziness, loss of balance or coordination",
    "   (Remember the FAST test: Face drooping, Arm weakness, Speech difficulty, Time to call 911)",
]


# ─────────────────────────────────────────────────
# HELPER: RISK PROFILE BUILDER
# ─────────────────────────────────────────────────

def _classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    else:
        return "obese"


def _classify_glucose(glucose: float) -> str:
    if glucose < 100:
        return "normal"
    elif glucose < 126:
        return "pre_diabetic"
    else:
        return "diabetic"


def _normalise_smoking(raw: str) -> str:
    raw = raw.lower().strip()
    if "never" in raw:
        return "never smoked"
    elif "formerly" in raw or "former" in raw:
        return "formerly smoked"
    elif "smokes" in raw or raw == "smokes":
        return "smokes"
    else:
        return "Unknown"


# ─────────────────────────────────────────────────
# MAIN RECOMMENDATION FUNCTION
# ─────────────────────────────────────────────────

import pandas as pd
from stroke_prediction.cluster_engine import predict_patient_cluster


def generate_recommendations(
    prediction: int,
    age: float,
    bmi: float,
    avg_glucose_level: float,
    hypertension: int,
    heart_disease: int,
    smoking_status: str,
    gender: str = "Male",
    ever_married: str = "Yes",
    work_type: str = "Private",
    residence_type: str = "Urban",
) -> dict:
    """
    Core Hybrid Recommendation Engine.

    Flow:
        1. Stroke Prediction Result (0/1) from Supervised Model
        2. Unsupervised Patient Phenotyping (K-Means Clustering into 4 Clinical Archetypes)
        3. Retrieve Cluster-Targeted Prescriptions & Diagnostic Workup
        4. Assemble Lifestyle + Dietary + Rule-based Medication + Monitoring recommendations
        5. Apply Clinical Safety Guardrail (disclaimer & emergency FAST protocol)
        6. Return complete clinical decision support report

    Parameters:
        prediction (int): 1 = High Stroke Risk, 0 = Low Stroke Risk
        age (float): Patient age
        bmi (float): Body Mass Index
        avg_glucose_level (float): Average blood glucose (mg/dL)
        hypertension (int): 1 = Yes, 0 = No
        heart_disease (int): 1 = Yes, 0 = No
        smoking_status (str): Smoking status string
        gender (str): Gender ('Male', 'Female', 'Other')
        ever_married (str): Marital status ('Yes', 'No')
        work_type (str): Work type ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
        residence_type (str): Living environment ('Urban', 'Rural')

    Returns:
        dict with keys: risk_level, risk_profile, cluster_profile,
                        lifestyle_recommendations, dietary_recommendations,
                        exercise_recommendations, medication_advisory,
                        targeted_prescriptions, diagnostic_workup,
                        monitoring_plan, emergency_signs, disclaimer
    """
    risk_level = "HIGH RISK" if prediction == 1 else "LOW RISK"
    risk_key = "high_risk" if prediction == 1 else "low_risk"

    bmi_class = _classify_bmi(bmi)
    glucose_class = _classify_glucose(avg_glucose_level)
    smoking_key = _normalise_smoking(smoking_status)
    hyp_key = "yes" if hypertension == 1 else "no"
    hd_key = "yes" if heart_disease == 1 else "no"

    # Assemble lifestyle tips
    lifestyle_tips = []
    lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS["smoking"].get(smoking_key, []))
    lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS["hypertension"].get(hyp_key, []))
    lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS["heart_disease"].get(hd_key, []))
    lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS["bmi"].get(bmi_class, []))
    lifestyle_tips.extend(LIFESTYLE_RECOMMENDATIONS["glucose"].get(glucose_class, []))

    # Age-specific additions
    age_tips = []
    if age >= 65:
        age_tips.append("👴 At age 65+, fall-prevention exercises are essential to avoid head injuries.")
        age_tips.append("🧠 Schedule cognitive function assessments annually.")

    # Unsupervised Patient Phenotyping
    patient_df = pd.DataFrame([{
        "id": 0,
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": smoking_status,
    }])

    try:
        cluster_profile = predict_patient_cluster(patient_df)
    except Exception as e:
        cluster_profile = {
            "cluster_id": 0,
            "name": "General Clinical Cohort",
            "tag": "Standard Profile",
            "color": "#757575",
            "archetype": "Default clinical assessment profile.",
            "population_share": "N/A",
            "historical_stroke_rate": "N/A",
            "benchmarks": {},
            "primary_risk_drivers": ["Standard clinical factors"],
            "targeted_prescriptions": [],
            "diagnostic_workup": ["Annual physical check-up"],
            "targeted_lifestyle": ["Maintain balanced diet and regular exercise"],
        }

    return {
        "risk_level": risk_level,
        "risk_profile": {
            "bmi_category": bmi_class,
            "glucose_category": glucose_class,
            "smoking_category": smoking_key,
            "hypertension": "Yes" if hypertension else "No",
            "heart_disease": "Yes" if heart_disease else "No",
        },
        "cluster_profile": cluster_profile,
        "lifestyle_recommendations": lifestyle_tips + age_tips,
        "dietary_recommendations": DIETARY_RECOMMENDATIONS,
        "exercise_recommendations": EXERCISE_RECOMMENDATIONS,
        "medication_advisory": MEDICATION_ADVISORY[risk_key],
        "targeted_prescriptions": cluster_profile.get("targeted_prescriptions", []),
        "diagnostic_workup": cluster_profile.get("diagnostic_workup", []),
        "monitoring_plan": MONITORING_PLAN[risk_key],
        "emergency_signs": EMERGENCY_SIGNS,
        "disclaimer": (
            "⚕️ MEDICAL DISCLAIMER: These recommendations and prescription guidelines are for educational and "
            "clinical decision-support purposes only. They do NOT constitute an automatic prescription. "
            "All pharmacological regimens must be prescribed and monitored by a licensed healthcare professional."
        ),
    }

