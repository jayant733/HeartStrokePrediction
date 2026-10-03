import pandas as pd
import numpy as np
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

import web_services as ws
import streamlit as st

# Make Predictions Services


def get_prediction() -> int:
    """_summary_
    Takes users Input from User Interface returns a Singular Prediction
    Returns:
        _type_: _description_
    """
    features = input_details_to_json()
    prediction = ws.get_prediction(features)
    if prediction is not None:
        return prediction
    else:
        st.error("An error occurred while getting the prediction!")


def get_recommendations_for_patient() -> dict:
    """Call /recommend endpoint and return full recommendations dict."""
    features = input_details_to_json()
    result = ws.get_recommendations(features)
    if result is not None:
        return result
    else:
        st.error("Could not retrieve recommendations from the server.")
        return {}


def display_recommendations(rec: dict):
    """Render a full hybrid recommendation & cluster phenotyping report in Streamlit."""
    recs = rec.get("recommendations", {})
    risk_level = recs.get("risk_level", "HIGH RISK" if rec.get("prediction") == 1 else "LOW RISK")
    cluster = rec.get("cluster_info") or recs.get("cluster_profile", {})

    risk_color = "#ff4b4b" if risk_level == "HIGH RISK" else "#21c354"
    cluster_color = cluster.get("color", "#1976D2")

    # Dual Status Header: Supervised Risk + Unsupervised Phenotype
    col_risk, col_cluster = st.columns([1, 1])
    with col_risk:
        st.markdown(
            f"""
            <div style='background:{risk_color};padding:14px;border-radius:10px;text-align:center;'>
              <span style='color:white;font-size:12px;font-weight:bold;letter-spacing:1px;text-transform:uppercase;'>Supervised Risk Level</span>
              <h2 style='color:white;margin:4px 0 0 0;font-size:24px;'>🧠 {risk_level}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cluster:
        cohort_name = cluster.get("tag", f"Cohort {cluster.get('cluster_id', 0)}")
        st.markdown(
            f"""
            <div style='background:{cluster_color};padding:14px;border-radius:10px;text-align:center;'>
              <span style='color:white;font-size:12px;font-weight:bold;letter-spacing:1px;text-transform:uppercase;'>Unsupervised Phenotype</span>
              <h2 style='color:white;margin:4px 0 0 0;font-size:22px;'>🧬 {cohort_name}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Unsupervised Segmentation & Patient Phenotype
    if cluster:
        st.markdown("### 🧬 Unsupervised Patient Segmentation & Phenotype")
        st.info(
            f"**Patient Archetype: {cluster.get('name', '')}**\n\n"
            f"{cluster.get('archetype', '')}"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cohort Stroke Rate", cluster.get("historical_stroke_rate", "N/A"))
        c2.metric("Cohort Population", cluster.get("population_share", "N/A"))
        benchmarks = cluster.get("benchmarks", {})
        c3.metric("Mean Age (Cohort)", f"{benchmarks.get('mean_age', '-')} yrs")
        c4.metric("Mean BMI (Cohort)", benchmarks.get("mean_bmi", "-"))

        drivers = cluster.get("primary_risk_drivers", [])
        if drivers:
            with st.expander("🔍 Primary Phenotype Risk Drivers in this Cluster", expanded=True):
                for driver in drivers:
                    st.markdown(f"- ⚠️ **{driver}**")

    # 2. Targeted Medical Prescriptions (Cluster-Driven)
    prescriptions = recs.get("targeted_prescriptions", cluster.get("targeted_prescriptions", []))
    if prescriptions:
        st.markdown("### 💊 Targeted Pharmacotherapy & Clinical Prescriptions")
        st.caption(
            "Pharmacological classes targeted to the patient's specific unsupervised phenotype and risk factors. "
            "Requires licensed physician evaluation before administration."
        )
        for rx in prescriptions:
            with st.expander(f"🩺 {rx.get('drug_class', 'Prescription Protocol')}", expanded=True):
                st.markdown(f"**Recommended Regimen / Examples:** `{rx.get('examples', '-')}`")
                st.markdown(f"**Pharmacological Rationale:** {rx.get('rationale', '-')}")
                if "monitoring" in rx:
                    st.markdown(f"**Safety & Lab Monitoring:** 🧪 *{rx.get('monitoring')}*")

    # 3. Diagnostic Workup Roadmap
    workup = recs.get("diagnostic_workup", cluster.get("diagnostic_workup", []))
    if workup:
        with st.expander("🔬 Recommended Clinical & Diagnostic Workup", expanded=risk_level == "HIGH RISK"):
            for test in workup:
                st.markdown(f"- 📋 {test}")

    # 3b. Cohort Statistical Validation (Z-Test & P-Value)
    c_id = cluster.get("cluster_id")
    if c_id is not None:
        with st.expander("📊 Cohort Statistical Validation (Z-Test & P-Value)", expanded=False):
            st.caption(
                "Two-sample Z-Test evaluating whether this patient's unsupervised clinical cohort "
                "exhibits a statistically significant difference in stroke incidence compared to other cohorts."
            )
            battery = ws.get_cluster_statistical_battery()
            z_tests = battery.get("cluster_proportions_z_tests", [])
            cohort_z = next((zt for zt in z_tests if f"Cohort {c_id}" in zt.get("cohort_1", "")), None)
            if cohort_z:
                col_z1, col_z2, col_z3 = st.columns(3)
                col_z1.metric("Z-Score", f"{cohort_z.get('z_statistic'):+.3f}")
                col_z2.metric("P-Value", f"{cohort_z.get('p_value'):.4e}")
                col_z3.metric("Decision (α=0.05)", "Reject H0" if cohort_z.get("p_value", 1) < 0.05 else "Fail to Reject")
                st.markdown(f"**Hypothesis Test:** `{cohort_z.get('test_name')}`")
                st.markdown(f"**Null Hypothesis ($H_0$):** {cohort_z.get('null_hypothesis')}")
                st.markdown(f"**Statistical Decision:** `{cohort_z.get('decision')}`")
                st.markdown(f"**Clinical Interpretation:** {cohort_z.get('interpretation')}")
                ci = cohort_z.get("ci_95", [0, 0])
                st.markdown(f"**95% Confidence Interval for Difference:** `[{ci[0]:.2%}, {ci[1]:.2%}]`")

    # 4. Patient Vitals Risk Profile
    profile = recs.get("risk_profile", {})
    if profile:
        with st.expander("📋 Individual Vitals Profile"):
            p1, p2, p3 = st.columns(3)
            p1.metric("BMI Category", profile.get("bmi_category", "-").capitalize())
            p2.metric("Glucose Level", profile.get("glucose_category", "-").replace("_", " ").capitalize())
            p3.metric("Smoking", profile.get("smoking_category", "-").capitalize())
            p1.metric("Hypertension", profile.get("hypertension", "-"))
            p2.metric("Heart Disease", profile.get("heart_disease", "-"))

    # 5. Lifestyle, Diet, Exercise
    with st.expander("🏃 Lifestyle Interventions"):
        for tip in recs.get("lifestyle_recommendations", []):
            st.markdown(f"- {tip}")

    with st.expander("🥗 Dietary Recommendations"):
        for tip in recs.get("dietary_recommendations", []):
            st.markdown(f"- {tip}")

    with st.expander("💪 Exercise Guidelines"):
        for tip in recs.get("exercise_recommendations", []):
            st.markdown(f"- {tip}")

    with st.expander("📏 Routine Monitoring Plan"):
        for tip in recs.get("monitoring_plan", []):
            st.markdown(f"- {tip}")

    # 6. Emergency Signs & Medical Disclaimer
    st.markdown("---")
    st.error("\n".join(recs.get("emergency_signs", [])))
    st.warning(recs.get("disclaimer", ""))



def get_prediction_document(filename: str, data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
     Takes File Input from User Interface returns a Prediction Dataframe
    Args:
        data (pd.DataFrame): _description_
    """
    predictions_file, sucess = ws.get_prediction_document(filename, data)
    if not sucess:
        st.error(predictions_file)
        return pd.DataFrame([])
    elif predictions_file is not None:
        return predictions_file
    else:
        st.error("An error occurred while getting the file's predictions !")
        return predictions_file


def input_details_to_json() -> dict:
    """_summary_
    Takes Useer input and converts it to json format
    Returns:
        _type_: _description_
    """
    myform_json = {"record": {"id": 0,
                              "file_name": "-",
                              "doctor_first_name": doctor_first_name if len(doctor_first_name.strip()) else "N/A",
                              "doctor_last_name": doctor_last_name if len(doctor_last_name.strip()) else "N/A"},
                   "patient": {"id": 0,
                               "firstname": first_name,
                               "lastname": last_name,
                               "gender": gender,
                               "age": age,
                               "hypertension": 1 if hypertension == "yes" else 0,
                               "heart_disease": 1 if heart_disease == "yes" else 0,
                               "ever_married": ever_married,
                               "work_type": work_type,
                               "Residence_type": residence_type,
                               "avg_glucose_level": avg_glucose_level,
                               "bmi": bmi,
                               "smoking_status": smoking_status
                               }
                   }
    return myform_json


# Data Frame Stylers


def data_frame_style_color_neg(val):
    """_summary_
    Pandas Dataframe Styler
    Args:
        val (_type_): _description_

    Returns:
        _type_: _description_
    """
    color = 'red' if type(val) == str and val == "Risk of Stroke" else 'green'
    return 'color: %s' % color


def float_format(val):
    """_summary_
    Pandas Dataframe Styler
    Args:
        val (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "{:.2f}".format(val)


def data_frame_style_display(data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Pandas Dataframe Styler
    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    if "prediction" in data.columns:
        data["prediction"] = data["prediction"].apply(
            lambda x: "Risk of Stroke" if x == 1 or x == "Risk of Stroke" else "Normal")
    if "age" in data.columns:
        data["age"] = data["age"].apply(lambda x: int(x) if pd.notnull(x) else 0)
    if "hypertension" in data.columns:
        data["hypertension"] = data["hypertension"].apply(
            lambda x: "Yes" if x == 1 or x == "Yes" else "No")
    if "heart_disease" in data.columns:
        data["heart_disease"] = data["heart_disease"].apply(
            lambda x: "Yes" if x == 1 or x == "Yes" else "No")
    if "bmi" in data.columns:
        data["bmi"] = data["bmi"].map(float_format)
    if "avg_glucose_level" in data.columns:
        data["avg_glucose_level"] = data["avg_glucose_level"].map(float_format)
    data.drop('record_id', axis=1, inplace=True, errors='ignore')
    data.drop('id', axis=1, inplace=True, errors='ignore')
    data.drop('cluster_id', axis=1, inplace=True, errors='ignore')
    if "prediction" in data.columns:
        st.dataframe(data.style.applymap(
            data_frame_style_color_neg, subset=['prediction']))
    else:
        st.dataframe(data)


# Data Base Services


def search_patient_by_fullname() -> pd.DataFrame:
    """_summary_
    Search for a patient by fullname
    Returns:
        _type_: _description_
    """
    search_results = ws.search_patient_by_fullname(search_patient_first_name,
                                                   search_patient_last_name)
    if search_results is not None:
        return search_results
    else:
        st.error("An error occurred while searching for Patient(s) Record(s)!")


def search_patient_by_window_period() -> pd.DataFrame:
    """_summary_
    Search patient records by Window Period
    Returns:
        pd.DataFrame: _description_
    """
    search_results = ws.search_patient_by_window_period(search_patient_from_date,
                                                        search_patient_to_date)
    if search_results is not None:
        return search_results
    else:
        st.error("An error occurred while searching for Patient(s) Record(s)!")


def search_patients_file_by_date() -> pd.DataFrame:
    """_summary_
    Search File Patients records by Date Created On
    Returns:
        _type_: _description_
    """
    search_results = ws.search_patients_file_by_date(search_file_name,
                                                     search_created_on)

    if search_results is not None:
        return search_results
    else:
        st.error("An error occurred while searching for Patient(s) Record(s)!")


# Form Validations

def validate_search_input_details() -> bool:
    """_summary_
    Validate Inputs from User Interface
    Returns:
        _type_: _description_
    """
    if option == 'Per Patient':
        if len(search_patient_first_name.strip()) == 0 and\
                len(search_patient_last_name.strip()) == 0:
            st.warning("First Name or Last Name is required to Search!")
            return False
    elif option == 'Window Period':
        if search_patient_from_date > search_patient_to_date:
            st.warning(
                "From Date should be strictly greater than To Date to Search!")
            return False
    elif option == 'Per file':
        if len(search_file_name.strip()) == 0:
            st.warning("File name is required to Search!")
            return False
        pattern = re.compile(r'^[a-zA-Z0-9_]+$')
        if not pattern.match(search_file_name):
            st.warning(
                "File name is not a match to the accepted format to Search!")
            return False
    return True


def validate_patient_input_details() -> bool:
    """_summary_
    Validate Patient Inputs from User Interface
    Returns:
        _type_: _description_
    """
    if len(first_name.strip()) == 0:
        st.warning("First Name is required to Add a Patient!")
        return False
    elif len(last_name.strip()) == 0:
        st.warning("Last Name is required to Add a Patient!")
        return False
    return True


# Web Interface Section
st.title("Heart Stroke Prediction")

# CSS Changes for Side Bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 450px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 450px;
        margin-left: -450px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Singular Prediction Page Section
with st.sidebar.expander("Single Predictions"):
    with st.form(key='my_form', clear_on_submit=True):
        st.title("Patient Form")
        gender_list = np.array(["Male", "Female"])
        yes_no = np.array(["Yes", "No"])
        work_type_lit = np.array(
            ['Private', 'Self employed', 'Govt job', 'children', 'Never worked'])
        residence_type_list = np.array(['Urban', 'Rural'])
        first_name = st.text_input(label='First Name')
        last_name = st.text_input(label='Last Name')
        gender = st.radio("Select your Gender", gender_list)
        age = st.number_input(label='Age', min_value=0, step=1, max_value=150)
        hypertension = st.radio(
            "Did you had Hypertension in the past ?", yes_no)
        heart_disease = st.radio(
            "Did you had Heart Problem in the past ?", yes_no)
        ever_married = st.radio("Have you ever been married ?", yes_no)
        work_type = st.radio("What is your Work Type ?", work_type_lit)
        residence_type = st.radio(
            "What is your Living Environment?", residence_type_list)
        avg_glucose_level = st.number_input(
            label='Enter your Average Glucose Level', min_value=0.0, step=0.1)
        bmi = st.number_input(
            label='Enter your Body Mass Index (bmi)', min_value=0.0, step=0., format="%.2f")
        smoking_status = st.radio(
            "Do you Smoke ?", ('smoked', 'formerly smoked', 'Unknown'))
        doctor_first_name = st.text_input(label='Dc. First Name')
        doctor_last_name = st.text_input(label='Dc. Last Name')
        submit_button = st.form_submit_button(label='Predict')

if submit_button:
    if validate_patient_input_details():
        result = get_recommendations_for_patient()
        prediciton = result.get("prediction") if result else None
        if prediciton is not None:
            message = (
                f"{first_name} {last_name} — You are at RISK of a Stroke! ⚠️"
                if prediciton == 1
                else f"{first_name} {last_name} — You are currently at LOW risk. ✅"
            )
            message_color = "red" if prediciton == 1 else "green"
            st.markdown(
                f"<h3 style='text-align:left;color:{message_color}'>{message}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown("## 📊 Personalised Health Recommendations")
            display_recommendations(result)

# File  Prediction Page Section
with st.sidebar.expander("Upload File for Predictions"):
    with st.form(key="predictions", clear_on_submit=True) as form:
        st.title("Select a File to Generate Predictions")
        uploaded_files = st.file_uploader("Choose a CSV file", type={"csv"})
        predict_button = st.form_submit_button("Submit")
        if uploaded_files:
            filename = uploaded_files.name

if predict_button:
    if uploaded_files:
        data = pd.read_csv(uploaded_files)
        data = get_prediction_document(filename, data)
        if data.shape[0] > 0:
            data_frame_style_display(data)
            st.success("Uploaded Successfully!")
    else:
        st.warning("Please Upload a CSV File")

# Unsupervised Clusters Explorer Section
with st.sidebar.expander("🧬 Explore Patient Clinical Clusters & Prescriptions"):
    st.markdown("### 🧬 Unsupervised Patient Phenotypes")
    st.caption("Patients are automatically grouped into 4 distinct clinical phenotypes via K-Means clustering before targeted prescriptions are generated.")
    clusters = ws.get_cluster_profiles()
    for c in clusters:
        with st.expander(f"Cohort {c.get('cluster_id')}: {c.get('tag')}"):
            st.markdown(f"#### {c.get('name')}")
            col_sh1, col_sh2 = st.columns(2)
            col_sh1.metric("Population Share", c.get('population_share', 'N/A'))
            col_sh2.metric("Historical Stroke Rate", c.get('historical_stroke_rate', 'N/A'))
            st.info(f"**Clinical Archetype:**\n\n{c.get('archetype', '')}")

            drivers = c.get('primary_risk_drivers', [])
            if drivers:
                st.markdown("**🔍 Primary Risk Drivers:**")
                for d in drivers:
                    st.markdown(f"- ⚠️ {d}")

            rx_list = c.get('targeted_prescriptions', [])
            if rx_list:
                st.markdown("**💊 Cluster-Targeted Pharmacotherapy:**")
                for rx in rx_list:
                    st.markdown(f"- **{rx.get('drug_class')}:** `{rx.get('examples')}`")
                    st.caption(f"_{rx.get('rationale')}_")


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Hypothesis Testing & Audit Logs Section
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("🔬 Statistical Hypothesis Testing & Audit Logs", expanded=False):
    st.markdown("### 🔬 Statistical Hypothesis Testing & Audit Trail")
    st.caption(
        "Rigorous statistical validation of patient cohorts, biomarker variance, "
        "and data drift using Z-Test, P-Value, ANOVA F-Test, and Log Transformation."
    )

    stat_tab1, stat_tab2, stat_tab3 = st.tabs(["📊 Battery", "🧪 Test Lab", "📜 Audit Log"])

    with stat_tab1:
        st.markdown("#### 1. Cohort Proportion Z-Tests (Stroke Disparity)")
        st.caption("Evaluates if each cluster's stroke rate differs significantly from the rest of the cohort.")
        battery = ws.get_cluster_statistical_battery()
        z_tests = battery.get("cluster_proportions_z_tests", [])
        if z_tests:
            z_rows = []
            for zt in z_tests:
                z_rows.append({
                    "Cohort": zt.get("cohort_1", "").split("(")[0].strip(),
                    "Cohort Rate": f"{zt.get('prop1', 0):.1%}",
                    "Rest Rate": f"{zt.get('prop2', 0):.1%}",
                    "Z-Score": f"{zt.get('z_statistic', 0):+.3f}",
                    "P-Value": f"{zt.get('p_value', 1):.4e}",
                    "Decision (α=0.05)": "Reject H0 (Significant)" if zt.get("p_value", 1) < 0.05 else "Fail to Reject",
                })
            st.dataframe(pd.DataFrame(z_rows), hide_index=True)

        st.markdown("#### 2. One-Way ANOVA F-Tests across 4 Clusters")
        st.caption("F-test evaluating if clinical biomarker means differ significantly across all 4 cohorts.")
        anova = battery.get("anova_f_tests", {})
        if anova:
            f_rows = []
            for feat_key, f_res in anova.items():
                f_rows.append({
                    "Feature": f_res.get("feature_name", feat_key),
                    "F-Statistic": f"{f_res.get('f_statistic', 0):.3f}",
                    "df": f"({f_res.get('df_between')}, {f_res.get('df_within')})",
                    "P-Value": f"{f_res.get('p_value', 1):.4e}",
                    "Log Transformed": "Yes (ln)" if f_res.get("log_transformed") else "No (Raw)",
                    "Decision": "Reject H0" if f_res.get("p_value", 1) < 0.05 else "Fail to Reject",
                })
            st.dataframe(pd.DataFrame(f_rows), hide_index=True)

        st.markdown("#### 3. Log-Transformation (Normalizing Skewed Biomarkers)")
        st.caption("Glucose level is right-skewed. Taking natural log ln(glucose) reduces skewness and stabilizes variance for valid parametric tests.")
        g_raw = anova.get("glucose", {})
        g_log = anova.get("glucose_log_transformed", {})
        col_lg1, col_lg2 = st.columns(2)
        col_lg1.metric("Raw Glucose Skewness", f"{g_raw.get('mean_skew_before', 1.57):.2f}", "Heavily Skewed")
        col_lg2.metric("Log(Glucose) Skewness", f"{g_log.get('mean_skew_after', 0.88):.2f}", "-44% Normalised", delta_color="inverse")

    with stat_tab2:
        st.markdown("#### 🧪 Interactive Hypothesis Testing Lab")
        test_mode = st.radio("Select Test Type", ["Two-Sample Z-Test (Proportions)", "One-Sample Z-Test (Mean vs Baseline)"])
        if test_mode == "Two-Sample Z-Test (Proportions)":
            with st.form("interactive_z_prop_form"):
                st.caption("Compare stroke or disease rates between two custom cohorts.")
                c_c1, c_c2 = st.columns(2)
                p1_name = c_c1.text_input("Cohort 1 Label", "High-Risk Phenotype")
                p1_count = c_c1.number_input("Cohort 1 Events (Stroke=1)", min_value=0, value=40, step=1)
                p1_n = c_c1.number_input("Cohort 1 Total Size (n1)", min_value=1, value=200, step=1)

                p2_name = c_c2.text_input("Cohort 2 Label", "Low-Risk Phenotype")
                p2_count = c_c2.number_input("Cohort 2 Events (Stroke=1)", min_value=0, value=10, step=1)
                p2_n = c_c2.number_input("Cohort 2 Total Size (n2)", min_value=1, value=500, step=1)
                alpha_val = st.selectbox("Significance Level (α)", [0.05, 0.01, 0.001], index=0)
                submit_z = st.form_submit_button("Compute Z-Test & Evaluate P-Value")

            if submit_z:
                z_calc = ws.run_custom_z_test(
                    count1=int(p1_count), n1=int(p1_n),
                    count2=int(p2_count), n2=int(p2_n),
                    label1=p1_name, label2=p2_name,
                    alpha=float(alpha_val),
                )
                if z_calc:
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("Z-Statistic", f"{z_calc.get('z_statistic'):+.4f}")
                    c_res2.metric("P-Value", f"{z_calc.get('p_value'):.4e}")
                    c_res3.metric("Decision", "Reject H0" if z_calc.get("p_value", 1) < alpha_val else "Fail to Reject")
                    st.success(f"**Result:** {z_calc.get('decision')}")
                    st.info(f"**Interpretation:** {z_calc.get('interpretation')}")
                    ci = z_calc.get("ci_95", [0, 0])
                    st.markdown(f"**Difference in Proportions:** `{z_calc.get('difference'):.2%}` (95% CI: `[{ci[0]:.2%}, {ci[1]:.2%}]`)")
                    st.caption("✅ Test automatically appended to persistent statistical audit log.")

        elif test_mode == "One-Sample Z-Test (Mean vs Baseline)":
            with st.form("interactive_z_mean_form"):
                st.caption("Test if a cohort's average biomarker differs significantly from clinical baseline benchmark.")
                m_metric = st.text_input("Biomarker Name", "Average Glucose Level (mg/dL)")
                m_sample = st.number_input("Cohort Sample Mean", min_value=0.0, value=125.4, step=0.1)
                m_std = st.number_input("Cohort Standard Deviation", min_value=0.1, value=35.2, step=0.1)
                m_n = st.number_input("Cohort Sample Size (n)", min_value=2, value=150, step=1)
                m_mu0 = st.number_input("Clinical Reference Baseline (μ0)", min_value=0.0, value=100.0, step=0.1)
                m_alpha = st.selectbox("Significance Level (α)", [0.05, 0.01, 0.001], index=0, key="m_alpha")
                submit_m = st.form_submit_button("Compute One-Sample Z-Test")

            if submit_m:
                from stroke_prediction.statistical_tests import run_z_test_means
                zm_res = run_z_test_means(
                    sample_mean=float(m_sample),
                    sample_std=float(m_std),
                    n=int(m_n),
                    mu0=float(m_mu0),
                    metric_name=m_metric,
                    alpha=float(m_alpha),
                )
                if zm_res:
                    c_m1, c_m2, c_m3 = st.columns(3)
                    c_m1.metric("Z-Statistic", f"{zm_res.get('z_statistic'):+.4f}")
                    c_m2.metric("P-Value", f"{zm_res.get('p_value'):.4e}")
                    c_m3.metric("Decision", "Reject H0" if zm_res.get("p_value", 1) < m_alpha else "Fail to Reject")
                    st.success(f"**Result:** {zm_res.get('decision')}")
                    st.info(f"**Interpretation:** {zm_res.get('interpretation')}")
                    st.caption("✅ Test automatically appended to persistent statistical audit log.")

    with stat_tab3:
        st.markdown("#### 📜 Persistent Statistical Test Audit Log")
        st.caption("Auditable record of all automated and on-demand hypothesis tests (Z-Tests, F-Tests, ANOVA, P-values).")
        logs = ws.get_statistical_test_logs()
        if logs:
            log_table = []
            for item in logs[:25]:
                if "test_type" in item:
                    log_table.append({
                        "Timestamp": item.get("timestamp"),
                        "Test Type": item.get("test_type"),
                        "Metric": item.get("target_metric"),
                        "Statistic": f"{item.get('statistic_name')} = {item.get('statistic_value')}",
                        "P-Value": f"{item.get('p_value'):.4e}",
                        "Significant?": "✅ Yes" if item.get("is_significant") else "❌ No",
                        "Decision": item.get("decision", "").split("(")[0].strip(),
                    })
                elif "raw_entry" in item:
                    log_table.append({"Raw Log Entry": item.get("raw_entry")})
            if log_table:
                st.dataframe(pd.DataFrame(log_table), hide_index=True)
        else:
            st.info("No statistical tests logged yet.")


# Prediction Retrival Page Section
with st.sidebar.expander("Retrieve Past Predictions"):
    button = None
    st.title("Select a Search Mode")
    option = st.selectbox('Search Mode', ('< Select Option >',
                          'Per Patient', 'Window Period', 'Per file'))

    with st.form(key="Retrieve patients predictions by full name", clear_on_submit=True) as form_1:

        # Per Patient Full Name Search
        if option == 'Per Patient':
            st.title("Patient Full Name")
            search_patient_first_name = st.text_input(label='First Name')
            search_patient_last_name = st.text_input(label='Last Name')
            button = st.form_submit_button("Get Patient Records")

        elif option == 'Window Period':
            st.title("Select a Window Period")
            search_patient_from_date = st.date_input(
                "From Date", datetime.today() - relativedelta(years=1))
            search_patient_to_date = st.date_input("To Date", datetime.today())
            button = st.form_submit_button("Get Patients Records")

        elif option == 'Per file':
            st.title("Enter File Details")
            st.text(
                " No File extension is Required\nExample:\n'myfile.csv' => your input 'myfile'")
            search_file_name = st.text_input(label='File Name')
            search_created_on = st.date_input("Created On", datetime.today())
            button = st.form_submit_button("Get File Records")

if button:
    if validate_search_input_details():
        if option == 'Per Patient':
            data = search_patient_by_fullname()
            data = pd.DataFrame(data)
            if data.shape[0] > 0:
                data_frame_style_display(data)
            else:
                st.warning("No Records Found")

        elif option == 'Window Period':
            data = search_patient_by_window_period()
            data = pd.DataFrame(data)
            if data.shape[0] > 0:
                data_frame_style_display(data)
            else:
                st.warning("No Records Found")

        elif option == 'Per file':

            data = search_patients_file_by_date()
            data = pd.DataFrame(data)
            if data.shape[0] > 0:
                data_frame_style_display(data)
            else:
                st.warning("No Records Found")
