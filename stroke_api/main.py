import pandas as pd
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../stroke_prediction')
sys.path.insert(0, '../postgres')
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder._infrequent_enabled = False
# ML API
from stroke_prediction.inference import make_prediction
from stroke_prediction.recommendation_engine import generate_recommendations
from stroke_prediction.cluster_engine import (
    predict_patient_cluster,
    get_all_cluster_summaries,
    get_clustering_model,
    CLUSTER_PROFILES
)
from stroke_prediction.data_processing import pipeline
from stroke_prediction.statistical_tests import (
    run_z_test_proportions,
    run_z_test_means,
    run_f_test_anova,
    run_f_test_variance,
    run_cluster_statistical_battery,
    get_statistical_audit_logs,
)
# FastAPI
from typing import Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request, status
# Postgres init
import models
import dbApi as db
from datetime import datetime
from typing import List
from json import dumps

# FastAPI init
app = FastAPI()


# BaseModels from FastAPI
class Patient(BaseModel):
    id: int
    firstname: str
    lastname: str
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: Optional[float] = 0.0
    bmi: Optional[float] = 0.0
    smoking_status: str

    class Config:
        # Serialize our sql into json
        orm_mode = True


class Patient_in_db(Patient):
    record_id: int
    prediction: int


class Record(BaseModel):
    id: Optional[int] = 0
    file_name: Optional[str] = "-"
    doctor_first_name: Optional[str] = "N/A"
    doctor_last_name: Optional[str] = "N/A"
    createdon: Optional[datetime] = datetime.now()

    class Config:
        # Serialize our sql into json
        orm_mode = True

# Databse Calls


def save_patient_record(record: Record, patient: Patient, result) -> int:

    new_record = models.Record(
        file_name=record.file_name,
        doctor_first_name=record.doctor_first_name,
        doctor_last_name=record.doctor_last_name,
        createdon=datetime.now()
    )

    record_id_in_db = db.create_record(new_record)
    patient_in_db = Patient_in_db(
        **patient.dict(), record_id=record_id_in_db, prediction=result)
    new_patient = models.Patient(
        record_id=patient_in_db.record_id,
        firstname=patient_in_db.firstname,
        lastname=patient_in_db.lastname,
        gender=patient_in_db.gender,
        age=patient_in_db.age,
        hypertension=patient_in_db.hypertension,
        heart_disease=patient_in_db.heart_disease,
        ever_married=patient_in_db.ever_married,
        work_type=patient_in_db.work_type,
        Residence_type=patient_in_db.Residence_type,
        avg_glucose_level=patient_in_db.avg_glucose_level,
        bmi=patient_in_db.bmi,
        smoking_status=patient_in_db.smoking_status,
        prediction=patient_in_db.prediction
    )
    result = db.insert_patient(new_patient)
    return result


def save_list_patients_record(record: Record,
                              patients, prediction_results) -> int:

    new_record = models.Record(
        file_name=record.file_name,
        doctor_first_name=record.doctor_first_name,
        doctor_last_name=record.doctor_last_name,
        createdon=datetime.now()
    )

    record_id_in_db = db.create_record(new_record)

    list_of_patients = []
    for patient, result in zip(patients, prediction_results):
        new_patient = models.Patient(
            record_id=record_id_in_db,
            firstname=patient.firstname,
            lastname=patient.lastname,
            gender=patient.gender,
            age=patient.age,
            hypertension=patient.hypertension,
            heart_disease=patient.heart_disease,
            ever_married=patient.ever_married,
            work_type=patient.work_type,
            Residence_type=patient.Residence_type,
            avg_glucose_level=patient.avg_glucose_level,
            bmi=patient.bmi,
            smoking_status=patient.smoking_status,
            prediction=result
        )
        list_of_patients.append(new_patient)
    result = db.insert_patients(list_of_patients)
    return result


# ML Calls
def make_one_prediction(record: Record, patient: Patient) -> dict:
    """_summary_
    Make a prediction for a single patient

    Args:
        record (Record): _description_
        patient (Patient): _description_

    Returns:
        dict: _description_
    """
    pd_dict = patient.dict()
    prediction_df = pd.DataFrame.from_dict([pd_dict])
    prediction_df.drop(['firstname', 'lastname'], axis=1, inplace=True)
    prediction = int(make_prediction(prediction_df)[0])
    save_patient_record(record, patient, prediction)
    recommendations = generate_recommendations(
        prediction=prediction,
        age=patient.age,
        bmi=patient.bmi,
        avg_glucose_level=patient.avg_glucose_level,
        hypertension=patient.hypertension,
        heart_disease=patient.heart_disease,
        smoking_status=patient.smoking_status,
        gender=patient.gender,
        ever_married=patient.ever_married,
        work_type=patient.work_type,
        residence_type=patient.Residence_type,
    )
    cluster_info = recommendations.get("cluster_profile", {})
    return {
        "prediction": prediction,
        "cluster_info": cluster_info,
        "recommendations": recommendations,
    }


def make_mulitple_prediction(record: Record, patients: List[Patient]):
    """_summary_
    Make a prediction for multiple patients
    Args:
        record (Record): _description_
        patients (List[Patient]): _description_

    Returns:
        _type_: _description_
    """
    records = [dict(patients[patienindex])
               for patienindex in range(len(patients))]
    prediction_df = pd.DataFrame.from_records(records)
    prediction_df.drop(['firstname', 'lastname'], axis=1, inplace=True)
    prediction_df["prediction"] = make_prediction(prediction_df)
    # Unsupervised Patient Segmentation on the batch
    try:
        km = get_clustering_model()
        features_for_clustering = prediction_df.drop(["prediction"], axis=1, errors="ignore").copy()
        proc_features = pipeline(features_for_clustering)
        cluster_labels = km.predict(proc_features)
        prediction_df["cluster_id"] = cluster_labels
        prediction_df["cluster_cohort"] = [
            CLUSTER_PROFILES.get(c, {}).get("tag", f"Cluster {c}") for c in cluster_labels
        ]
    except Exception as e:
        print(f"Batch clustering warning: {e}")

    results = list(prediction_df["prediction"])
    save_list_patients_record(record, patients, results)
    return dumps(prediction_df.to_dict('index'))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/")
def read_root():
    return {"message": "We are ready to go !"}


@app.get("/predict")
async def predict(record: Record, patient: Patient):
    result = make_one_prediction(record, patient)
    return result


@app.get("/predict_multiple")
async def predict_file(record: Record, patient: List[Patient]):
    result = make_mulitple_prediction(record, patient)
    return result


@app.post("/recommend")
async def recommend(patient: Patient):
    """Predict stroke risk and return personalised recommendations for a patient."""
    pd_dict = patient.dict()
    prediction_df = pd.DataFrame.from_dict([pd_dict])
    prediction_df.drop(['firstname', 'lastname'], axis=1, inplace=True)
    prediction = int(make_prediction(prediction_df)[0])
    recommendations = generate_recommendations(
        prediction=prediction,
        age=patient.age,
        bmi=patient.bmi,
        avg_glucose_level=patient.avg_glucose_level,
        hypertension=patient.hypertension,
        heart_disease=patient.heart_disease,
        smoking_status=patient.smoking_status,
        gender=patient.gender,
        ever_married=patient.ever_married,
        work_type=patient.work_type,
        residence_type=patient.Residence_type,
    )
    cluster_info = recommendations.get("cluster_profile", {})
    return {
        "prediction": prediction,
        "cluster_info": cluster_info,
        "recommendations": recommendations,
    }


@app.post("/cluster")
async def cluster_patient(patient: Patient):
    """Segment a patient into their unsupervised clinical phenotype cluster."""
    pd_dict = patient.dict()
    patient_df = pd.DataFrame.from_dict([pd_dict])
    patient_df.drop(['firstname', 'lastname'], axis=1, inplace=True)
    cluster_profile = predict_patient_cluster(patient_df)
    return cluster_profile


@app.get("/cluster_profiles")
def get_clusters():
    """Retrieve summaries of all discovered unsupervised patient clinical phenotypes."""
    return get_all_cluster_summaries()


@app.get("/search/patient/{firstname}&{lastname}",
         response_model=List[Patient_in_db], status_code=200)
async def get_patient_by_name(firstname: str, lastname: str):
    patients = db.get_patient_by_full_name(firstname, lastname)
    return patients


@app.get("/search/patient/period/{fromdate}&{todate}",
         response_model=List[Patient_in_db], status_code=200)
async def get_patients_by_window_period(fromdate: str,
                                        todate: str) -> List[Patient]:
    from_to_dict = dict()
    from_to_dict["from_year"] = fromdate.split("-")[0]
    from_to_dict["from_month"] = fromdate.split("-")[1]
    from_to_dict["from_day"] = fromdate.split("-")[2]
    from_to_dict["to_year"] = todate.split("-")[0]
    from_to_dict["to_month"] = todate.split("-")[1]
    from_to_dict["to_day"] = todate.split("-")[2]
    patients = db.get_patients_by_window_period(from_to_dict)
    return patients


@app.get("/search/file/{filename}&{createdon}",
         response_model=List[Patient_in_db], status_code=200)
async def get_patient_by_file_name(filename: str, createdon: str):
    year = createdon.split("-")[0]
    month = createdon.split("-")[1]
    day = createdon.split("-")[2]
    patients = db.get_patients_file_by_date(filename, year, month, day)
    return patients


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL HYPOTHESIS TESTING & AUDIT LOG ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

class ZTestProportionPayload(BaseModel):
    count1: int
    n1: int
    count2: int
    n2: int
    label1: Optional[str] = "Cohort 1"
    label2: Optional[str] = "Cohort 2 / Baseline"
    alpha: Optional[float] = 0.05


class ZTestMeanPayload(BaseModel):
    sample_mean: float
    sample_std: float
    n: int
    mu0: float
    metric_name: Optional[str] = "Biomarker"
    alpha: Optional[float] = 0.05


class FTestVariancePayload(BaseModel):
    sample1: List[float]
    sample2: List[float]
    label1: Optional[str] = "Cohort 1"
    label2: Optional[str] = "Cohort 2"
    metric_name: Optional[str] = "Biomarker"
    alpha: Optional[float] = 0.05


@app.get("/statistical_tests/battery")
def get_cluster_statistical_battery():
    """Run and return the complete hypothesis testing battery on clinical clusters."""
    return run_cluster_statistical_battery()


@app.get("/statistical_tests/logs")
def get_stat_logs():
    """Retrieve all logged statistical test executions from audit trail."""
    return get_statistical_audit_logs()


@app.post("/statistical_tests/z_test_proportion")
def run_z_test_prop(payload: ZTestProportionPayload):
    """Execute a Two-Sample Z-Test for proportions and log to audit trail."""
    return run_z_test_proportions(
        count1=payload.count1,
        n1=payload.n1,
        count2=payload.count2,
        n2=payload.n2,
        label1=payload.label1,
        label2=payload.label2,
        alpha=payload.alpha,
    )


@app.post("/statistical_tests/z_test_mean")
def run_z_test_m(payload: ZTestMeanPayload):
    """Execute a One-Sample Z-Test for mean against benchmark and log to audit trail."""
    return run_z_test_means(
        sample_mean=payload.sample_mean,
        sample_std=payload.sample_std,
        n=payload.n,
        mu0=payload.mu0,
        metric_name=payload.metric_name,
        alpha=payload.alpha,
    )


@app.post("/statistical_tests/f_test_variance")
def run_f_test_var(payload: FTestVariancePayload):
    """Execute an F-Test for equality of variances and log to audit trail."""
    return run_f_test_variance(
        sample1=payload.sample1,
        sample2=payload.sample2,
        label1=payload.label1,
        label2=payload.label2,
        metric_name=payload.metric_name,
        alpha=payload.alpha,
    )

