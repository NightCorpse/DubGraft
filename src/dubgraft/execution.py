"""Centralized external process execution."""

import logging
import math
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, overload


logger = logging.getLogger(__name__)


class ExecutionError(RuntimeError):
    """Raised when an external command cannot complete successfully."""


class ExecutableNotFoundError(ExecutionError):
    """Raised when a required executable cannot be found in PATH."""

    def __init__(self, executable: str) -> None:
        self.executable = executable
        super().__init__(f"{executable} was not found in PATH")


class CommandLaunchError(ExecutionError):
    """Raised when an external command cannot be started or times out."""


class CommandTimeoutError(CommandLaunchError):
    """Raised when an external command exceeds its configured timeout."""

    def __init__(self, error: subprocess.TimeoutExpired) -> None:
        self.command = error.cmd
        self.timeout = error.timeout
        super().__init__(str(error))


class CommandExitError(ExecutionError):
    """Raised when an external command returns a non-zero exit code."""

    def __init__(
        self, command: Sequence[str], return_code: int, diagnostics: str | bytes
    ) -> None:
        self.command = tuple(command)
        self.return_code = return_code
        if isinstance(diagnostics, bytes):
            detail = diagnostics.decode("utf-8", errors="replace").strip()
        else:
            detail = diagnostics.strip()
        executable = Path(command[0]).name if command else "process"
        super().__init__(detail or f"{executable} exited with code {return_code}")


def resolve_executable(name: str) -> str:
    """Resolve an executable from PATH without caching environment state."""
    executable = shutil.which(name)
    if executable is None:
        raise ExecutableNotFoundError(name)
    return executable


@overload
def run_capture(
    command: Sequence[str], *, timeout: float | None = None, text: Literal[True] = True
) -> subprocess.CompletedProcess[str]: ...


@overload
def run_capture(
    command: Sequence[str], *, timeout: float | None = None, text: Literal[False]
) -> subprocess.CompletedProcess[bytes]: ...


@overload
def run_capture(
    command: Sequence[str], *, timeout: float | None = None, text: bool
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]: ...


def run_capture(
    command: Sequence[str], *, timeout: float | None = None, text: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    """Run a command and capture stdout and stderr in text or binary mode."""
    command_list = list(command)
    logger.debug("Running command: %s", shlex.join(command_list))
    try:
        if text:
            text_result = subprocess.run(
                command_list,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
            if text_result.returncode != 0:
                raise CommandExitError(
                    command_list, text_result.returncode, text_result.stderr
                )
            return text_result
        binary_result = subprocess.run(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
        if binary_result.returncode != 0:
            raise CommandExitError(
                command_list, binary_result.returncode, binary_result.stderr
            )
        return binary_result
    except subprocess.TimeoutExpired as error:
        raise CommandTimeoutError(error) from error
    except OSError as error:
        raise CommandLaunchError(str(error)) from error


def _progress_time(values: dict[str, str]) -> float | None:
    for key in ("out_time_us", "out_time_ms"):
        try:
            seconds = int(values[key]) / 1_000_000
        except (KeyError, ValueError):
            continue
        if math.isfinite(seconds):
            return seconds

    try:
        hours, minutes, seconds_text = values["out_time"].split(":", 2)
        seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds_text)
    except (KeyError, ValueError):
        return None
    return seconds if math.isfinite(seconds) else None


def iter_progress_times(lines: Iterable[str]) -> Iterator[float]:
    """Parse timestamps from FFmpeg's progress protocol."""
    values: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.strip().partition("=")
        if not separator:
            continue
        values[key] = value
        if key == "progress":
            seconds = _progress_time(values)
            if seconds is not None:
                yield seconds
            values.clear()


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            return
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
    except OSError:
        pass


def run_ffmpeg_progress(
    command: Sequence[str],
    *,
    expected_duration: float,
    progress: Callable[[float], None],
) -> None:
    """Run FFmpeg while reporting monotonic media-time progress."""
    command_list = list(command)
    progress_command = [
        command_list[0],
        "-nostats",
        "-progress",
        "pipe:1",
        *command_list[1:],
    ]
    logger.debug("Running command: %s", shlex.join(progress_command))
    callback_error: BaseException | None = None
    try:
        with tempfile.TemporaryFile(
            mode="w+", encoding="utf-8", errors="replace"
        ) as diagnostics:
            process = subprocess.Popen(
                progress_command,
                stdout=subprocess.PIPE,
                stderr=diagnostics,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            completed = 0.0
            try:
                for seconds in iter_progress_times(process.stdout):
                    current = min(max(seconds, completed, 0.0), expected_duration * 0.99)
                    if current > completed:
                        completed = current
                        try:
                            progress(completed)
                        except BaseException as error:
                            callback_error = error
                            raise
                return_code = process.wait()
            except BaseException:
                _stop_process(process)
                raise
            finally:
                process.stdout.close()

            diagnostics.seek(0)
            detail = diagnostics.read().strip()
    except OSError as error:
        if callback_error is not None:
            raise callback_error
        raise CommandLaunchError(str(error)) from error

    if return_code != 0:
        raise CommandExitError(progress_command, return_code, detail)
    progress(expected_duration)
