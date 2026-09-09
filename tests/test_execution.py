import io
import subprocess

import pytest

from dubgraft.execution import (
    CommandExitError,
    CommandLaunchError,
    CommandTimeoutError,
    ExecutableNotFoundError,
    resolve_executable,
    run_capture,
    run_ffmpeg_progress,
)


class _Process:
    def __init__(
        self,
        output: str,
        diagnostics: io.TextIOBase,
        *,
        return_code: int = 0,
        error: str = "",
    ) -> None:
        self.stdout = io.StringIO(output)
        self.return_code = return_code
        diagnostics.write(error)

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        return self.return_code

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


class _StubbornProcess(_Process):
    def __init__(self, output: str, diagnostics: io.TextIOBase) -> None:
        super().__init__(output, diagnostics)
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return 0 if self.killed else None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if timeout is not None and not self.killed:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return 0

    def kill(self) -> None:
        self.killed = True


class _UnkillableProcess(_StubbornProcess):
    def kill(self) -> None:
        raise OSError("access denied")


def test_resolve_executable_uses_current_path(monkeypatch: pytest.MonkeyPatch) -> None:
    paths = iter(["/first/ffmpeg", "/second/ffmpeg"])
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: next(paths))

    assert resolve_executable("ffmpeg") == "/first/ffmpeg"
    assert resolve_executable("ffmpeg") == "/second/ffmpeg"


def test_resolve_executable_reports_missing_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("dubgraft.execution.shutil.which", lambda name: None)

    with pytest.raises(ExecutableNotFoundError, match="ffprobe was not found in PATH"):
        resolve_executable("ffprobe")


def test_run_capture_configures_text_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "output", "")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)

    result = run_capture(["ffprobe", "-version"], timeout=10)

    assert result.stdout == "output"
    command, options = calls[0]
    assert command == ["ffprobe", "-version"]
    assert options == {
        "check": False,
        "timeout": 10,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }


def test_run_capture_configures_binary_execution_without_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(kwargs)
        return subprocess.CompletedProcess(command, 0, b"audio", b"")

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fake_run)

    result = run_capture(["ffmpeg", "pipe:1"], text=False)

    assert result.stdout == b"audio"
    assert calls[0] == {
        "capture_output": True,
        "check": False,
        "timeout": None,
    }


def test_run_capture_wraps_launch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = OSError("cannot start")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fail)

    with pytest.raises(CommandLaunchError) as error:
        run_capture(["ffprobe", "-version"])

    assert error.value.__cause__ is failure


def test_run_capture_preserves_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = subprocess.TimeoutExpired("ffprobe", 10)

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("dubgraft.execution.subprocess.run", fail)

    with pytest.raises(CommandTimeoutError) as error:
        run_capture(["ffprobe", "-version"], timeout=10)

    assert error.value.command == "ffprobe"
    assert error.value.timeout == 10
    assert error.value.__cause__ is failure


@pytest.mark.parametrize("diagnostics", ["invalid input", b"invalid input"])
def test_run_capture_reports_nonzero_exit(
    diagnostics: str | bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 2, type(diagnostics)(), diagnostics
        ),
    )

    with pytest.raises(CommandExitError, match="invalid input") as error:
        run_capture(["ffmpeg", "broken"], text=isinstance(diagnostics, str))

    assert error.value.return_code == 2


def test_run_capture_uses_executable_name_when_stderr_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 3, "", ""),
    )

    with pytest.raises(CommandExitError, match="ffprobe exited with code 3"):
        run_capture(["/usr/bin/ffprobe", "broken"])


def test_ffmpeg_progress_injects_protocol_and_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    updates: list[float] = []

    def fake_popen(command: list[str], **kwargs: object) -> _Process:
        commands.append(command)
        return _Process(
            "out_time_us=5000000\nprogress=continue\n"
            "out_time_us=12000000\nprogress=end\n",
            kwargs["stderr"],  # type: ignore[arg-type]
        )

    monkeypatch.setattr("dubgraft.execution.subprocess.Popen", fake_popen)

    run_ffmpeg_progress(
        ["ffmpeg", "-v", "error", "output.mkv"],
        expected_duration=10,
        progress=updates.append,
    )

    assert commands[0][1:4] == ["-nostats", "-progress", "pipe:1"]
    assert updates == [5, 9.9, 10]


def test_ffmpeg_progress_preserves_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.Popen",
        lambda command, **kwargs: _Process(
            "progress=end\n",
            kwargs["stderr"],
            return_code=1,
            error="mux failed",
        ),
    )

    with pytest.raises(CommandExitError, match="mux failed"):
        run_ffmpeg_progress(
            ["ffmpeg", "output.mkv"], expected_duration=10, progress=lambda value: None
        )


def test_ffmpeg_progress_kills_process_that_does_not_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[_StubbornProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> _StubbornProcess:
        process = _StubbornProcess(
            "out_time_us=1000000\nprogress=continue\n",
            kwargs["stderr"],  # type: ignore[arg-type]
        )
        processes.append(process)
        return process

    monkeypatch.setattr("dubgraft.execution.subprocess.Popen", fake_popen)

    with pytest.raises(KeyboardInterrupt):
        run_ffmpeg_progress(
            ["ffmpeg", "output.mkv"],
            expected_duration=10,
            progress=lambda value: (_ for _ in ()).throw(KeyboardInterrupt),
        )

    assert processes[0].terminated
    assert processes[0].killed
    assert processes[0].stdout.closed


def test_ffmpeg_progress_does_not_misclassify_callback_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "dubgraft.execution.subprocess.Popen",
        lambda command, **kwargs: _Process(
            "out_time_us=1000000\nprogress=continue\n", kwargs["stderr"]
        ),
    )
    failure = OSError("callback failed")

    with pytest.raises(OSError) as error:
        run_ffmpeg_progress(
            ["ffmpeg", "output.mkv"],
            expected_duration=10,
            progress=lambda value: (_ for _ in ()).throw(failure),
        )

    assert error.value is failure


def test_ffmpeg_progress_does_not_wait_after_failed_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[_UnkillableProcess] = []

    def fake_popen(command: list[str], **kwargs: object) -> _UnkillableProcess:
        process = _UnkillableProcess(
            "out_time_us=1000000\nprogress=continue\n",
            kwargs["stderr"],  # type: ignore[arg-type]
        )
        processes.append(process)
        return process

    monkeypatch.setattr("dubgraft.execution.subprocess.Popen", fake_popen)

    with pytest.raises(KeyboardInterrupt):
        run_ffmpeg_progress(
            ["ffmpeg", "output.mkv"],
            expected_duration=10,
            progress=lambda value: (_ for _ in ()).throw(KeyboardInterrupt),
        )

    assert processes[0].terminated
    assert not processes[0].killed
