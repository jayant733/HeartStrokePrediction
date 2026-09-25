from stroke_prediction.data_processing import pipeline, get_model_path
import pickle


def make_prediction(X, model_name="classifier"):
    X = pipeline(X)
    filename = f"{model_name}.pickle" if not model_name.endswith(".pickle") else model_name
    file_path = get_model_path(filename)
    classifier = pickle.load(open(file_path, "rb"))
    return classifier.predict(X)
