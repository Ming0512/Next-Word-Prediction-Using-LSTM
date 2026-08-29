import os
import json
import pickle
from pathlib import Path

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from src.nextWordPredictor.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Read a YAML file and return its contents as a ConfigBox
    (dot-accessible dict)."""
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


def create_directories(path_to_directories: list, verbose: bool = True):
    """Create a list of directories if they don't already exist."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """Save a dict as a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


def load_json(path) -> ConfigBox:
    """Load a JSON file and return its contents as a ConfigBox."""
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)


def save_pickle(path: Path, obj) -> None:
    """Save any picklable Python object (used for the fitted Tokenizer)."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"pickle file saved at: {path}")


def load_pickle(path: Path):
    """Load a pickled Python object."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"pickle file loaded from: {path}")
    return obj


def get_size(path) -> str:
    """Return file size in KB as a display string."""
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"
