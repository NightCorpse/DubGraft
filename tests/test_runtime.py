from io import StringIO
from pathlib import Path

from dubgraft.runtime import RunOutput


class _TTYBuffer(StringIO):
    def isatty(self) -> bool:
        return True


def test_progress_reuses_terminal_line() -> None:
    stream = _TTYBuffer()
    output = RunOutput(stream=stream)

    with output.stage("Analyzing alignment") as stage:
        stage.update("2/4 (50%)")

    rendered = stream.getvalue()
    assert "\rAnalyzing alignment..." in rendered
    assert "\rAnalyzing alignment... 2/4 (50%)" in rendered
    assert "\rAnalyzing alignment... done" in rendered
    assert rendered.endswith("\n")


def test_nested_logs_are_isolated(tmp_path: Path) -> None:
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"

    # Nested contexts expose accidental handler sharing without using threads.
    with RunOutput(log_path=first_log) as first:
        first.detail("first before")
        with RunOutput(log_path=second_log) as second:
            second.detail("second only")
        first.detail("first after")

    first_contents = first_log.read_text(encoding="utf-8")
    second_contents = second_log.read_text(encoding="utf-8")
    assert "first before" in first_contents
    assert "first after" in first_contents
    assert "second only" not in first_contents
    assert "second only" in second_contents
    assert "first before" not in second_contents
