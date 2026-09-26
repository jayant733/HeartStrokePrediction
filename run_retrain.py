import os
import pandas as pd
from stroke_prediction.train import train_all, train_and_evaluate_dl, train_clustering

print("Loading dataset...")
df_path = "data/Training_data/healthcare-dataset-stroke-data.csv"

if not os.path.exists(df_path):
    print(f"Error: Dataset not found at {df_path}")
else:
    df = pd.read_csv(df_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    print("\n--- Retraining Supervised Machine Learning Models ---")
    results = train_all(df)
    print(results)

    print("\n--- Retraining Deep Learning / Transformer Models ---")
    try:
        cnn_results = train_and_evaluate_dl(df, model_type="cnn")
        print(f"CNN Model  | Metrics: {cnn_results}")
    except Exception as e:
        print(f"CNN Model retraining error: {e}")

    try:
        rnn_results = train_and_evaluate_dl(df, model_type="rnn")
        print(f"RNN Model  | Metrics: {rnn_results}")
    except Exception as e:
        print(f"RNN Model retraining error: {e}")

    print("\n--- Retraining Unsupervised Models (Clustering) ---")
    try:
        clustering_results = train_clustering(df)
        print(f"Clustering Complete.")
    except Exception as e:
        print(f"Clustering error: {e}")

    print("\nAll models successfully retrained and updated in models/ directory!")
