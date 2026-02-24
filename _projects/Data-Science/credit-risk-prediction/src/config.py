import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Safely loads the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Define base directory for path joining
BASE_DIR = Path(__file__).resolve().parent.parent