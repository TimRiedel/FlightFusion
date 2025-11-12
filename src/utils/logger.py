import logging
import os
import sys
from datetime import datetime

# --- Create log directory ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{datetime.utcnow():%Y-%m-%dT%H:%M:%S}.log")

# --- Global logger ---
logger = logging.getLogger("FlightFusion")
logger.setLevel(logging.DEBUG)  # process all messages

# --- File handler (detailed) ---
file_handler = logging.FileHandler(log_file)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)  # capture everything
logger.addHandler(file_handler)

# --- Console handler (simplified) ---
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter("%(message)s")  # raw output
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# --- Capture uncaught exceptions globally ---
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow Ctrl+C to terminate normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception\n\n", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception