import json
import os

from loanApprovalPrediction.logger import logger


def create_directories(path_to_directories: list, verbose=True):
    for path in path_to_directories:
        if os.path.exists(path):
            if verbose:
                logger.info(f"directory already exists at: {path}")
            continue
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)
