import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path="../.env")

def get_Settings():
    DATABASE_URL = "sqlite:///stroke_prediction.db"
    return DATABASE_URL