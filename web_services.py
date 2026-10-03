import pandas as pd
import json
from datetime import date

import requests

# Interact with FastAPI endpoint
import load_config as config
BACKEND_SERVER = config.get_backend_connection_server()

# Make Predictions Section


def get_prediction(features: dict) -> int:
    """_summary_
    Takes users Input from User Interface returns a Singular Prediction
    Returns:
        _type_: _description_
    """
    url = BACKEND_SERVER + "predict"
    response = requests.get(url, json=features)
    if response.status_code == 200:
        results = response.json()
        prediction = results["prediction"]
        return prediction
    else:
        print(response.status_code)
        return None


def get_prediction_document(filename: str, data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
     Takes File Input from User Interface returns a Prediction Dataframe
    Args:
        data (pd.DataFrame): _description_
    """
    data = data_frame_fix_column_with_Nan_float(data)
    list_of_json = data.to_dict(orient='records')
    data_json = dict()
    data_json["record"] = {"id": 0, "file_name": filename}
    data_json["patient"] = list_of_json
    url = BACKEND_SERVER + "predict_multiple"
    response = requests.get(url, json=data_json)
    if response.status_code == 200:
        result = json.loads(response.content)
        prediction_df = pd.read_json(result, orient='index')
        data["prediction"] = prediction_df["prediction"]
        if "cluster_cohort" in prediction_df.columns:
            data["cluster_cohort"] = prediction_df["cluster_cohort"]
        return data, True
    elif response.status_code == 422:
        return "File is Not withint the Correct Format !", False
    else:
        return pd.DataFrame([]), True


def get_cluster_profiles() -> list:
    """Retrieve all unsupervised patient cluster summaries."""
    url = BACKEND_SERVER + "cluster_profiles"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print("Cluster profiles fetch error:", e)
    return []



def data_frame_fix_column_with_Nan_float(data):
    """_summary_
    Fix Nan_float issues in dataframe , Pydantic model doesn't like
    Nan's in Float columns Json Encoder doesnt like to deal
    with Nan's in Float columns
    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    float_cols = data.select_dtypes(include=['float64', 'int64']).columns
    str_cols = data.select_dtypes(include=['object']).columns
    data.loc[:, float_cols] = data.loc[:, float_cols].fillna(0.0)
    data.loc[:, str_cols] = data.loc[:, str_cols].fillna('')
    return data

# Data Base Services


def search_patient_by_fullname(first_name: str,
                               last_name: str) -> pd.DataFrame:
    """_summary_
    Search patient records by full name
    Args:
        first_name (str): _description_
        last_name (str): _description_

    Returns:
        pd.DataFrame: _description_
    """
    url = BACKEND_SERVER + "search/patient/{firstname}&{lastname}"\
        .format(firstname=first_name if len(first_name) else "%%",
                lastname=last_name if len(last_name) else "%%",)
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return pd.DataFrame([])


def search_patient_by_window_period(from_date: date,
                                    to_date: date) -> pd.DataFrame:
    """_summary_
    Search patient records by Window Period
    Args:
        from_date (date): _description_
        to_date (date): _description_

    Returns:
        pd.DataFrame: _description_
    """
    url = BACKEND_SERVER + "search/patient/period/{fromdate}&{todate}"\
        .format(fromdate=from_date.strftime("%Y-%m-%d"),
                todate=to_date.strftime("%Y-%m-%d"))
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        return results
    else:
        return pd.DataFrame([])


def search_patients_file_by_date(file_name: str,
                                 created_on: date) -> pd.DataFrame:
    """_summary_
    Search File Patients records by Date Created On
    Args:
        file_name (str): _description_
        created_date (date): _description_

    Returns:
        pd.DataFrame: _description_
    """
    url = BACKEND_SERVER + "search/file/{filename}&{createdon}"\
        .format(filename=file_name, createdon=created_on.strftime("%Y-%m-%d"))
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json()
        if results is not None:
            return results
        else:
            return pd.DataFrame([])
    else:
        return None


def get_recommendations(features: dict) -> dict:
    """
    Call the /recommend API endpoint.
    Returns a dict with prediction result and full recommendations.
    Falls back to direct model execution if server is unreachable.
    """
    url = BACKEND_SERVER + "recommend"
    patient_data = features.get("patient", {})
    try:
        response = requests.post(url, json=patient_data, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Recommendation API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Backend unreachable, falling back to local recommendation engine: {e}")

    # Graceful local fallback
    try:
        from stroke_prediction.inference import make_prediction
        from stroke_prediction.recommendation_engine import generate_recommendations
        import pandas as pd
        df = pd.DataFrame([patient_data])
        df.drop(["firstname", "lastname"], axis=1, inplace=True, errors="ignore")
        pred = int(make_prediction(df)[0])
        recs = generate_recommendations(
            prediction=pred,
            age=patient_data.get("age", 50.0),
            bmi=patient_data.get("bmi", 25.0),
            avg_glucose_level=patient_data.get("avg_glucose_level", 100.0),
            hypertension=patient_data.get("hypertension", 0),
            heart_disease=patient_data.get("heart_disease", 0),
            smoking_status=patient_data.get("smoking_status", "never smoked"),
            gender=patient_data.get("gender", "Male"),
            ever_married=patient_data.get("ever_married", "Yes"),
            work_type=patient_data.get("work_type", "Private"),
            residence_type=patient_data.get("Residence_type", "Urban"),
        )
        return {
            "prediction": pred,
            "cluster_info": recs.get("cluster_profile", {}),
            "recommendations": recs,
        }
    except Exception as fallback_err:
        print(f"Local fallback error: {fallback_err}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Testing Web Services
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_statistical_battery() -> dict:
    """Fetch complete hypothesis testing battery across clinical clusters."""
    url = BACKEND_SERVER + "statistical_tests/battery"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Battery endpoint error: {e}, falling back to local computation")

    try:
        from stroke_prediction.statistical_tests import run_cluster_statistical_battery
        return run_cluster_statistical_battery()
    except Exception as err:
        print(f"Local battery error: {err}")
        return {}


def get_statistical_test_logs() -> list:
    """Fetch persistent and in-memory statistical hypothesis test audit logs."""
    url = BACKEND_SERVER + "statistical_tests/logs"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Logs endpoint error: {e}, falling back to local audit logs")

    try:
        from stroke_prediction.statistical_tests import get_statistical_audit_logs
        return get_statistical_audit_logs()
    except Exception as err:
        print(f"Local logs error: {err}")
        return []


def run_custom_z_test(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    label1: str = "Cohort 1",
    label2: str = "Cohort 2 / Baseline",
    alpha: float = 0.05,
) -> dict:
    """Execute custom Two-Sample Z-Test for proportions."""
    url = BACKEND_SERVER + "statistical_tests/z_test_proportion"
    payload = {
        "count1": count1,
        "n1": n1,
        "count2": count2,
        "n2": n2,
        "label1": label1,
        "label2": label2,
        "alpha": alpha,
    }
    try:
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    from stroke_prediction.statistical_tests import run_z_test_proportions
    return run_z_test_proportions(
        count1=count1, n1=n1, count2=count2, n2=n2,
        label1=label1, label2=label2, alpha=alpha
    )


def run_custom_f_test_anova(
    groups_dict: dict,
    feature_name: str = "Clinical Biomarker",
    log_transform: bool = False,
    alpha: float = 0.05,
) -> dict:
    """Execute One-Way ANOVA F-Test across patient groups."""
    from stroke_prediction.statistical_tests import run_f_test_anova
    return run_f_test_anova(
        groups_dict=groups_dict,
        feature_name=feature_name,
        log_transform=log_transform,
        alpha=alpha,
    )

