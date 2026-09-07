import pytest

from dubgraft import __version__
from dubgraft.cli import main


def test_cli_without_arguments_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0

    output = capsys.readouterr().out
    assert "usage: dubgraft" in output
    assert (
        "Align and graft dubbed audio across media releases using distributed audio anchors."
        in " ".join(output.split())
    )


def test_cli_reports_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"dubgraft {__version__}"
