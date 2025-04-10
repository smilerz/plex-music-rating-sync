import logging
import os
import sys
from typing import Optional, TextIO

from tqdm import tqdm

from manager.config_manager import LogLevel


class InfoFilter(logging.Filter):
    def filter(self, rec: logging.LogRecord) -> bool:
        return rec.levelno in (logging.DEBUG, logging.INFO)


class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that writes log messages via tqdm.write(),
    preventing them from overwriting or corrupting any active tqdm bars.
    """

    def __init__(self, stream: TextIO = sys.stdout, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.stream = stream

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


class LevelBasedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno in [logging.WARNING, logging.ERROR, logging.CRITICAL]:
            self._style._fmt = "[%(asctime)s] %(levelname)s " "[%(name)s.%(funcName)s:%(lineno)d] %(message)s"
        else:
            self._style._fmt = "[%(asctime)s] %(levelname)s: %(message)s"
        return super().format(record)


class LogManager:
    LOG_LEVEL_MAP = {
        LogLevel.CRITICAL: logging.CRITICAL,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.INFO: logging.INFO,
        LogLevel.DEBUG: logging.DEBUG,
    }
    LOG_DIR: str = "logs"
    LOG_FILE: str = "sync_ratings.log"
    MAX_BACKUPS: int = 5

    def __init__(self, log_dir: Optional[str] = None, log_file: Optional[str] = None, max_backups: Optional[int] = None) -> None:
        """Initialize logging manager with configurable paths and settings."""
        self.logger = logging.getLogger("PlexSync")
        self.log_dir = log_dir or self.LOG_DIR
        self.log_file = log_file or self.LOG_FILE
        self.max_backups = max_backups or self.MAX_BACKUPS

    def _rotate_logs(self) -> None:
        """Rotates log files within log_dir, handling up to max_backups."""
        base_log = os.path.join(self.log_dir, self.log_file)
        for i in range(self.max_backups, 0, -1):
            old_log = f"{base_log}.{i}"
            new_log = f"{base_log}.{i + 1}"
            if os.path.exists(old_log):
                if i == self.max_backups:
                    os.remove(old_log)
                else:
                    os.rename(old_log, new_log)

        if os.path.exists(base_log):
            os.rename(base_log, f"{base_log}.1")

    def setup_logging(self, log_level: str) -> logging.Logger:
        """Initializes custom logging with log rotation and multi-level console output."""
        os.makedirs(self.log_dir, exist_ok=True)
        self._rotate_logs()

        self.logger.setLevel(logging.DEBUG)

        # File Handler: Always at DEBUG, dynamic formatting
        file_formatter = LevelBasedFormatter(datefmt="%H:%M:%S")
        log_file_path = os.path.join(self.log_dir, self.log_file)
        fh = logging.FileHandler(filename=log_file_path, encoding="utf-8", mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)

        # Console: Warning+ => stderr with TqdmLoggingHandler
        warn_format = logging.Formatter("%(levelname)s [%(funcName)s:%(lineno)d]: %(message)s")
        ch_err = TqdmLoggingHandler(stream=sys.stderr, level=logging.WARNING)
        ch_err.setFormatter(warn_format)
        self.logger.addHandler(ch_err)

        # Console: Info => stdout with TqdmLoggingHandler
        info_format = logging.Formatter("%(levelname)s: %(message)s")
        ch_std = TqdmLoggingHandler(stream=sys.stdout)
        ch_std.setFormatter(info_format)
        ch_std.addFilter(InfoFilter())
        self.logger.addHandler(ch_std)

        level = -1
        if isinstance(log_level, str):
            try:
                normalized_level = log_level.lower()
                if normalized_level in LogLevel:
                    level = self.LOG_LEVEL_MAP[normalized_level]
            except (KeyError, ValueError):
                pass
        elif isinstance(log_level, int):
            if 0 <= log_level <= 50:
                level = log_level

        if level < 0:
            valid_levels = ", ".join(f"{level}" for level in LogLevel)
            print(f"Valid logging levels: {valid_levels}")
            raise RuntimeError(f"Invalid logging level selected: {log_level}")

        ch_err.setLevel(level)
        ch_std.setLevel(level)
        self.logger.info("Logging initialized with custom settings.")
        return self.logger
