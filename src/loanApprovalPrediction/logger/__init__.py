import os
import sys

from loguru import logger

from loanApprovalPrediction.constants import ROOT_DIR

LOG_DIR = ROOT_DIR / "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Remove default handler to configure our own
logger.remove()

# Add a default handler for console output
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)

# Add a file handler for logging to a file at root/logs/
logger.add(
    str(LOG_DIR / "file_{time:YYYY-MM-DD}.log"),
    level="DEBUG",
    rotation="1 day",
    compression="zip",
    enqueue=True,
    retention="7 days",
)
