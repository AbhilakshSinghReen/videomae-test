import logging
import os


src_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(src_dir)
logs_dir = os.path.join(repo_dir, "logs")

os.makedirs(logs_dir, exist_ok=True)


def setup_logger(name, level=logging.DEBUG):
    log_file_name = name + ".log"
    log_file_path = os.path.join(logs_dir, log_file_name)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if the logger is already configured
    if logger.hasHandlers():
        return logger

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a logging format
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_formatter = logging.Formatter(
        "%(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Example usage
if __name__ == "__main__":
    # Set up the logger
    logger = setup_logger("example")

    # Log some messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")