import sys
import os
from loanApprovalPrediction.logger import logger

def error_message_detail(error: Exception, exc_tb: type) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param exc_tb: The traceback object (from sys.exc_info()[2]).
    :return: A formatted error message string.
    """
    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message string with file name, line number, and the actual error
    line_number = exc_tb.tb_lineno
    # Use os.path.basename to get just the filename, not the full path
    error_message_str = f"Error occurred in python script name: [{os.path.basename(file_name)}] at line number [{line_number}]: {str(error)}"

    return error_message_str


class MyException(Exception):
    """
    Custom exception class for handling specific errors, designed to capture detailed
    traceback information when initialized within an 'except' block.
    """

    def __init__(self, error_message: str | Exception, error_detail: sys = sys):
        """
        Initializes MyException with a detailed error message.

        :param error_message: A string describing the error or the original exception object.
                              If an Exception object is passed, its details will be used.
        :param error_detail: The sys module to access traceback details. Defaults to sys.
        """
        # Get the current exception information. This will be (None, None, None)
        # if not called inside an 'except' block.
        exc_type, exc_obj, exc_tb = error_detail.exc_info()

        formatted_message = ""

        # Case 1: An actual exception object was passed as error_message
        if isinstance(error_message, Exception):
            # If there's an active traceback (meaning we are inside an except block
            # and this exception is being wrapped), use it.
            if exc_tb:
                formatted_message = error_message_detail(error_message, exc_tb)
            else:
                # If no active traceback, just use the string representation of the exception
                formatted_message = str(error_message)
        # Case 2: A string message was passed as error_message
        else:
            # If there's an active exception object and traceback, use them
            if exc_obj and exc_tb:
                formatted_message = error_message_detail(exc_obj, exc_tb)
            else:
                # Otherwise, just use the provided string message
                formatted_message = str(error_message)

        # Call the base class constructor with the formatted message
        super().__init__(formatted_message)

        # Store the formatted message
        self.error_message = formatted_message

        # Log the error when the custom exception is initialized
        logger.error(self.error_message)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message