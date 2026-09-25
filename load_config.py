import os

from pathlib import Path

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

def get_backend_connection_server():
    BACKEND_SERVER : str = os.getenv("BACKEND_SERVER", "http://localhost:8005/")
    return BACKEND_SERVER
