import json
from pathlib import Path

import pytest

from dubgraft.report import ReportError, write_json_report


def test_json_report_does_not_replace_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("existing", encoding="utf-8")

    with pytest.raises(ReportError, match="use --overwrite"):
        write_json_report(path, {"status": "completed"}, overwrite=False)

    assert path.read_text(encoding="utf-8") == "existing"


def test_json_report_overwrite_publishes_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("existing", encoding="utf-8")

    write_json_report(path, {"status": "completed"}, overwrite=True)

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed"}
