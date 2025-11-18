import sys
import io
import contextlib
import logging
import re
from tqdm import tqdm
import time

class PrefixedIO(io.TextIOBase):
    def __init__(self, stream, prefix, file_handlers=None):
        self.stream = stream
        self.prefix = prefix
        self.file_handlers = file_handlers or []
        self._newline = True

    def _is_tqdm_output(self, text):
        """Detect if text is from tqdm progress bar output."""
        # tqdm output typically contains:
        # - Carriage returns (\r) for overwriting lines
        # - Progress bar patterns: |, #, percentage signs
        # - Time patterns: [HH:MM<HH:MM, ...]
        if '\r' in text:
            return True
        # Check for tqdm progress bar patterns
        tqdm_patterns = [
            r'\d+%\|',  # Percentage with bar: "50%|"
            r'\[\d+:\d+<\d+:\d+',  # Time pattern: "[00:10<00:05"
            r'\|[#\s]+\|',  # Progress bar: "|####    |"
        ]
        for pattern in tqdm_patterns:
            if re.search(pattern, text):
                return True
        return False

    def write(self, text):
        # Write to console with prefix (always, including tqdm)
        # Track what we write to console so we can write the same to file handlers
        console_output = []
        for chunk in text.splitlines(True):
            if self._newline and not chunk.startswith("\r"):
                self.stream.write(self.prefix)
                console_output.append(self.prefix + chunk)
            else:
                console_output.append(chunk)
            self.stream.write(chunk)
            self._newline = chunk.endswith("\n")
        self.stream.flush()
        
        # Also write to file handlers with prefix (same as console)
        # Skip tqdm output from being written to log files
        if self.file_handlers and text.strip() and not self._is_tqdm_output(text):
            for prefixed_chunk in console_output:
                if prefixed_chunk.strip():  # Only write non-empty lines
                    # Write the prefixed message (same as console) to file handlers
                    for handler in self.file_handlers:
                        handler.stream.write(prefixed_chunk)
                        handler.stream.flush()

    def flush(self):
        self.stream.flush()
        for handler in self.file_handlers:
            if hasattr(handler, 'stream'):
                handler.stream.flush()

    # Allow tqdm to detect terminal properties
    def isatty(self):
        return hasattr(self.stream, "isatty") and self.stream.isatty()

    def fileno(self):
        return self.stream.fileno() if hasattr(self.stream, "fileno") else 1

def _get_file_handlers():
    """Get all FileHandler instances from all loggers."""
    file_handlers = []
    # Check root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            file_handlers.append(handler)

    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler not in file_handlers:
                file_handlers.append(handler)
    return file_handlers

@contextlib.contextmanager
def console_output_prefix(prefix):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    
    # Get all file handlers to also write captured output to log files
    file_handlers = _get_file_handlers()
    
    wrapped_stdout = PrefixedIO(old_stdout, prefix, file_handlers)
    wrapped_stderr = PrefixedIO(old_stderr, prefix, file_handlers)
    sys.stdout, sys.stderr = wrapped_stdout, wrapped_stderr

    # Redirect existing logging handlers that use StreamHandler
    root_logger = logging.getLogger()
    old_handlers = []
    for h in root_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            old_handlers.append((h, h.stream))
            h.stream = wrapped_stderr

    # tqdm factory that uses wrapped_stderr
    def prefixed_tqdm(*args, **kwargs):
        return tqdm(*args, file=wrapped_stderr, **kwargs)

    try:
        yield prefixed_tqdm
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        for h, s in old_handlers:
            h.stream = s