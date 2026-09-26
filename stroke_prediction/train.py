from stroke_prediction.data_processing import (pipeline,
                                               build_model,
                                               evaluate_model,
                                               train_and_evaluate_all_models,
                                               get_supported_models,
                                               train_unsupervised_models,
                                               perform_hyperparameter_tuning)
from stroke_prediction.dl_models import train_dl_model, evaluate_dl_model


def make_model(df, model_name="logistic_regression"):
    xtrain, ytrain, xtest, ytest = pipeline(df)
    build_model(xtrain, ytrain, model_name=model_name)
    return evaluate_model(xtest, ytest, model_name=model_name)


def train_all(df):
    """Train and evaluate all supervised learning algorithms up to Naive Bayes, including XGBoost."""
    return train_and_evaluate_all_models(df)


def train_clustering(df):
    """Train all unsupervised models (clustering)."""
    return train_unsupervised_models(df)


def tune_and_evaluate(df, model_name="random_forest"):
    """Run hyperparameter tuning for a given model."""
    return perform_hyperparameter_tuning(df, model_name=model_name)


def train_and_evaluate_dl(df, model_type="cnn"):
    """Train and evaluate Deep Learning models (cnn or rnn)."""
    xtrain, ytrain, xtest, ytest = pipeline(df)
    model = train_dl_model(xtrain, ytrain, model_type=model_type)
    xtest_processed = pipeline(xtest.copy())
    return evaluate_dl_model(model, xtest_processed, ytest)
