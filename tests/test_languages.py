import csv
from importlib.resources import files

import pytest

from dubgraft.languages import (
    LANGUAGE_CODES_URL,
    LanguageCodeError,
    normalize_language_code,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("pt", "por"), ("POR", "por"), ("fra", "fre"), ("deu", "ger")],
)
def test_language_codes_normalize(value: str, expected: str) -> None:
    assert normalize_language_code(value) == expected


def test_language_code_suggests_correction() -> None:
    with pytest.raises(LanguageCodeError) as error:
        normalize_language_code("egn")

    assert "did you mean 'eng'?" in str(error.value)
    assert LANGUAGE_CODES_URL in str(error.value)


@pytest.mark.parametrize("value", ["qaa", "qgz", "qtz"])
def test_language_local_codes(value: str) -> None:
    assert normalize_language_code(value) == value


def test_language_range_notation_is_rejected() -> None:
    with pytest.raises(LanguageCodeError):
        normalize_language_code("qaa-qtz")


def test_language_csv_is_complete() -> None:
    path = files("dubgraft").joinpath("data", "iso-639-2.csv")
    with path.open(encoding="utf-8", newline="") as language_file:
        rows = list(csv.DictReader(language_file))

    assert len(rows) == 487
    assert all(row["iso_639_2_b"] for row in rows)
