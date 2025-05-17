import logging
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from manager.config_manager import LogLevel
from manager.log_manager import TRACE_LEVEL, InfoFilter, LevelBasedFormatter, LogManager, TqdmLoggingHandler


@pytest.fixture(autouse=True)
def clear_logger_handlers():
    logger = logging.getLogger("PlexSync")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


class TestLogManagerInitialization:
    def test_init_default_values_sets_expected_attributes(self):
        mgr = LogManager()
        assert mgr.log_dir == LogManager.LOG_DIR
        assert mgr.log_file == LogManager.LOG_FILE
        assert mgr.max_backups == LogManager.MAX_BACKUPS
        assert mgr.logger.name == "PlexSync"

    def test_init_custom_values_override_defaults(self):
        mgr = LogManager(log_dir="foo", log_file="bar.log", max_backups=42)
        assert mgr.log_dir == "foo"
        assert mgr.log_file == "bar.log"
        assert mgr.max_backups == 42


class TestLogManagerRotateLogs:
    @patch("os.path.exists")
    @patch("os.rename")
    @patch("os.remove")
    def test_rotate_logs_removes_oldest_and_renames_backups(self, mock_remove, mock_rename, mock_exists):
        # Simulate all backup files exist
        mock_exists.side_effect = lambda path: True
        mgr = LogManager(log_dir="logs", log_file="test.log", max_backups=3)
        mgr._rotate_logs()
        base = os.path.join("logs", "test.log")
        # Oldest removed
        mock_remove.assert_called_once_with(f"{base}.3")
        # Others renamed
        mock_rename.assert_any_call(f"{base}.2", f"{base}.3")
        mock_rename.assert_any_call(f"{base}.1", f"{base}.2")
        mock_rename.assert_any_call(base, f"{base}.1")

    @patch("os.path.exists")
    @patch("os.rename")
    @patch("os.remove")
    def test_rotate_logs_handles_missing_files_gracefully(self, mock_remove, mock_rename, mock_exists):
        # Simulate only base log exists
        def exists_side_effect(path):
            return path.endswith("test.log")

        mock_exists.side_effect = exists_side_effect
        mgr = LogManager(log_dir="logs", log_file="test.log", max_backups=2)
        mgr._rotate_logs()
        # Only base log renamed, no remove/rename for backups
        mock_rename.assert_called_once_with(os.path.join("logs", "test.log"), os.path.join("logs", "test.log.1"))
        mock_remove.assert_not_called()


class TestLogManagerSetupLogging:
    @patch("os.makedirs")
    @patch.object(LogManager, "_rotate_logs")
    @patch("logging.FileHandler")
    @patch("manager.log_manager.TqdmLoggingHandler")
    def test_setup_logging_creates_log_dir_and_handlers(self, mock_tqdm, mock_filehandler, mock_rotate, mock_makedirs):
        mgr = LogManager(log_dir="logs", log_file="foo.log")
        logger = mgr.logger
        # Remove all handlers before test
        logger.handlers.clear()
        # Simulate FileHandler and TqdmLoggingHandler
        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh
        mock_ch_err = MagicMock()
        mock_ch_std = MagicMock()
        mock_tqdm.side_effect = [mock_ch_err, mock_ch_std]
        result_logger = mgr.setup_logging("debug")
        mock_makedirs.assert_called_once_with("logs", exist_ok=True)
        mock_rotate.assert_called_once()
        # FileHandler and both console handlers added
        assert mock_fh in logger.handlers
        assert mock_ch_err in logger.handlers
        assert mock_ch_std in logger.handlers
        assert result_logger is logger

    @pytest.mark.parametrize("level", ["invalid", 999, None])
    def test_setup_logging_invalid_log_level_raises(self, level):
        mgr = LogManager()
        with pytest.raises(RuntimeError):
            mgr.setup_logging(level)

    @patch("os.makedirs")
    @patch.object(LogManager, "_rotate_logs")
    @patch("logging.FileHandler")
    @patch("manager.log_manager.TqdmLoggingHandler")
    def test_setup_logging_sets_logger_level_and_handlers(self, mock_tqdm, mock_filehandler, mock_rotate, mock_makedirs):
        mgr = LogManager()
        logger = mgr.logger
        logger.handlers.clear()
        mock_fh = MagicMock()
        mock_filehandler.return_value = mock_fh
        mock_ch_err = MagicMock()
        mock_ch_std = MagicMock()
        mock_tqdm.side_effect = [mock_ch_err, mock_ch_std]
        mgr.setup_logging("info")
        # Logger level set to INFO
        assert logger.level == logging.INFO
        # ch_err set to WARNING, ch_std set to TRACE_LEVEL
        mock_ch_err.setLevel.assert_called_with(logging.WARNING)
        mock_ch_std.setLevel.assert_called_with(TRACE_LEVEL)


class TestLogManagerUpdateLogLevel:
    @pytest.mark.parametrize(
        "input_level,expected",
        [
            ("debug", logging.DEBUG),
            (LogLevel.INFO, logging.INFO),
            ("trace", TRACE_LEVEL),
        ],
    )
    def test_update_log_level_accepts_string_and_enum(self, input_level, expected):
        mgr = LogManager()
        mgr.logger.setLevel(logging.WARNING)
        mgr.update_log_level(input_level)
        assert mgr.logger.level == expected

    def test_update_log_level_invalid_value_noop(self):
        mgr = LogManager()
        mgr.logger.setLevel(logging.ERROR)
        # Should not change level for invalid input
        mgr.update_log_level("notalevel")
        assert mgr.logger.level == logging.ERROR


class TestLevelBasedFormatter:
    def test_format_warning_and_error_levels_use_detailed_format(self):
        fmt = LevelBasedFormatter()
        record = logging.LogRecord("foo", logging.WARNING, "", 0, "msg", (), None)
        out = fmt.format(record)
        assert "[foo." in out and "] WARNING [" in out
        record = logging.LogRecord("foo", logging.ERROR, "", 0, "msg", (), None)
        out = fmt.format(record)
        assert "[foo." in out and "] ERROR [" in out

    def test_format_info_and_debug_levels_use_simple_format(self):
        fmt = LevelBasedFormatter()
        record = logging.LogRecord("foo", logging.INFO, "", 0, "msg", (), None)
        out = fmt.format(record)
        assert out.startswith("[") and ": msg" in out
        record = logging.LogRecord("foo", logging.DEBUG, "", 0, "msg", (), None)
        out = fmt.format(record)
        assert out.startswith("[") and ": msg" in out


class TestTqdmLoggingHandler:
    @patch("manager.log_manager.tqdm.write")
    def test_emit_writes_message_via_tqdm_write(self, mock_tqdm_write):
        handler = TqdmLoggingHandler(stream=sys.stdout)
        record = logging.LogRecord("foo", logging.INFO, "", 0, "msg", (), None)
        handler.format = MagicMock(return_value="formatted")
        handler.emit(record)
        mock_tqdm_write.assert_called_once_with("formatted", file=sys.stdout)

    @patch("manager.log_manager.tqdm.write", side_effect=Exception("fail"))
    def test_emit_handles_exceptions_gracefully(self, mock_tqdm_write):
        handler = TqdmLoggingHandler()
        record = logging.LogRecord("foo", logging.INFO, "", 0, "msg", (), None)
        handler.format = MagicMock(return_value="formatted")
        # Should not raise
        handler.emit(record)


class TestInfoFilter:
    @pytest.mark.parametrize(
        "level,expected",
        [
            (TRACE_LEVEL, True),
            (logging.DEBUG, True),
            (logging.INFO, True),
            (logging.WARNING, False),
            (logging.ERROR, False),
            (logging.CRITICAL, False),
        ],
    )
    def test_filter_accepts_trace_debug_info(self, level, expected):
        f = InfoFilter()
        rec = logging.LogRecord("foo", level, "", 0, "msg", (), None)
        assert f.filter(rec) is expected


class TestTraceLoggerExtension:
    def test_trace_method_logs_at_trace_level(self):
        logger = logging.getLogger("PlexSync")
        logger.handlers.clear()
        out = {}

        class DummyHandler(logging.Handler):
            def emit(self, record):
                out["level"] = record.levelno
                out["msg"] = record.getMessage()

        handler = DummyHandler()
        logger.addHandler(handler)
        logger.setLevel(TRACE_LEVEL)
        logger.trace("hello trace")
        assert out["level"] == TRACE_LEVEL
        assert out["msg"] == "hello trace"
