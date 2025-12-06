import os

RANDOM_STATE = 42

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
IMAGES_PATH = os.path.join(PROJECT_ROOT, "images")

# Ensure output directories exist
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
