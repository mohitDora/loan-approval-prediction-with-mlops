from loanApprovalPrediction.logger import logger
from loanApprovalPrediction.exception import MyException
import sys

logger.info("Logging has started")
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

try:
    a = 1/0
except Exception as e:
    raise MyException(e)