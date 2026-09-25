from stroke_prediction.data_processing import (pipeline,
                                               build_model,
                                               evaluate_model,
                                               train_and_evaluate_all_models,
                                               get_supported_models)


def make_model(df, model_name="logistic_regression"):
    xtrain, ytrain, xtest, ytest = pipeline(df)
    build_model(xtrain, ytrain, model_name=model_name)
    return evaluate_model(xtest, ytest, model_name=model_name)


def train_all(df):
    """Train and evaluate all supervised learning algorithms up to Naive Bayes."""
    return train_and_evaluate_all_models(df)

