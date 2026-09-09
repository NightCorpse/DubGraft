"""Terminal progress and per-run diagnostic logging."""

import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from types import TracebackType
from typing import TextIO

_log_context: ContextVar[object | None] = ContextVar(
    "dubgraft_log_context", default=None
)
_log_lock = threading.Lock()
_active_log_outputs = 0
_saved_logger_state: tuple[int, bool] | None = None


class _RunLogFilter(logging.Filter):
    def __init__(self, run_key: object) -> None:
        super().__init__()
        self.run_key = run_key

    def filter(self, record: logging.LogRecord) -> bool:
        return _log_context.get() is self.run_key


class RunOutput:
    """Coordinate stable terminal output with an optional detailed log."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        quiet: bool = False,
        log_path: Path | None = None,
        stream: TextIO | None = None,
    ) -> None:
        self.verbose = verbose
        self.quiet = quiet
        self.stream = stream or sys.stderr
        self._interactive = self.stream.isatty()
        self._active_message: str | None = None
        self._active_width = 0
        self._logger = logging.getLogger("dubgraft")
        self._handler: logging.Handler | None = None
        self._run_key = object()
        self._context_token: Token[object | None] | None = None

        if log_path is not None:
            handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            handler.setLevel(logging.DEBUG)
            handler.addFilter(_RunLogFilter(self._run_key))
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                )
            )
            self._add_log_handler(handler)
            self._handler = handler

    def _add_log_handler(self, handler: logging.Handler) -> None:
        global _active_log_outputs, _saved_logger_state
        with _log_lock:
            if _active_log_outputs == 0:
                _saved_logger_state = (self._logger.level, self._logger.propagate)
                self._logger.setLevel(logging.DEBUG)
                self._logger.propagate = False
            self._logger.addHandler(handler)
            _active_log_outputs += 1

    def _replace_active_line(self, text: str, *, final: bool = False) -> None:
        padding = " " * max(0, self._active_width - len(text))
        self.stream.write(f"\r{text}{padding}")
        if final:
            self.stream.write("\n")
        self.stream.flush()
        self._active_width = len(text)

    def _start(self, message: str) -> None:
        self._logger.info("%s started", message)
        self._active_message = message
        text = f"{message}..."
        if not self.quiet:
            if self._interactive:
                self._replace_active_line(text)
            else:
                self.stream.write(text)
                self.stream.flush()

    def update(self, detail: str) -> None:
        if self.quiet or not self._interactive or self._active_message is None:
            return
        self._replace_active_line(f"{self._active_message}... {detail}")

    def _finish(self, outcome: str) -> None:
        if self._active_message is None:
            return
        message = self._active_message
        self._logger.info("%s %s", message, outcome)
        if not self.quiet:
            text = f"{message}... {outcome}"
            if self._interactive:
                self._replace_active_line(text, final=True)
            else:
                self.stream.write(f" {outcome}\n")
                self.stream.flush()
        self._active_message = None
        self._active_width = 0

    @contextmanager
    def stage(self, message: str) -> Iterator["RunOutput"]:
        self._start(message)
        try:
            yield self
        except BaseException:
            self._finish("failed")
            raise
        else:
            self._finish("done")

    def detail(self, message: str) -> None:
        self._logger.debug(message)
        if self.verbose and not self.quiet:
            print(f"  {message}", file=self.stream)

    def error(self, message: str) -> None:
        if self._handler is not None:
            self._logger.error(message)

    def warning(self, message: str) -> None:
        if self._handler is not None:
            self._logger.warning(message)

    def exception(self, message: str) -> None:
        if self._handler is not None:
            self._logger.exception(message)

    def close(self) -> None:
        global _active_log_outputs, _saved_logger_state
        if self._handler is None:
            return
        self._logger.info("DubGraft run finished")
        handler = self._handler
        self._handler = None
        with _log_lock:
            self._logger.removeHandler(handler)
            handler.close()
            _active_log_outputs -= 1
            if _active_log_outputs == 0 and _saved_logger_state is not None:
                level, propagate = _saved_logger_state
                self._logger.setLevel(level)
                self._logger.propagate = propagate
                _saved_logger_state = None

    def __enter__(self) -> "RunOutput":
        self._context_token = _log_context.set(self._run_key)
        if self._handler is not None:
            self._logger.info("DubGraft run started")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        finally:
            if self._context_token is not None:
                _log_context.reset(self._context_token)
                self._context_token = None
