"""ISO 639 language-code validation backed by the bundled official list."""

import csv
from functools import cache
from importlib.resources import files
from itertools import product
from string import ascii_lowercase

LANGUAGE_CODES_URL = (
    "https://github.com/NightCorpse/DubGraft/blob/main/src/dubgraft/data/iso-639-2.csv"
)


class LanguageCodeError(ValueError):
    """Raised when a language code is not present in the bundled ISO list."""


@cache
def _language_codes() -> dict[str, str]:
    path = files("dubgraft").joinpath("data", "iso-639-2.csv")
    codes: dict[str, str] = {}
    with path.open(encoding="utf-8") as language_file:
        for row in csv.DictReader(language_file):
            canonical = row["iso_639_2_b"]
            if "-" in canonical:
                start, end = canonical.split("-", 1)
                for characters in product(ascii_lowercase, repeat=3):
                    code = "".join(characters)
                    if start <= code <= end:
                        codes[code] = code
                continue
            for column in ("iso_639_2_b", "iso_639_2_t", "iso_639_1"):
                if code := row[column]:
                    codes[code] = canonical
    return codes


def _suggest_language_code(code: str) -> str | None:
    transpositions = {
        code[:index] + code[index + 1] + code[index] + code[index + 2 :]
        for index in range(len(code) - 1)
    }
    matches = {
        _language_codes()[candidate]
        for candidate in transpositions
        if candidate in _language_codes()
    }
    if len(matches) == 1:
        return matches.pop()

    matches = {
        canonical
        for candidate, canonical in _language_codes().items()
        if len(candidate) == len(code)
        and sum(left != right for left, right in zip(candidate, code, strict=True)) == 1
    }
    return matches.pop() if len(matches) == 1 else None


def normalize_language_code(value: str) -> str:
    """Validate an ISO 639-1/2 code and return its ISO 639-2/B form."""
    code = value.casefold()
    canonical = _language_codes().get(code)
    if canonical is not None:
        return canonical

    match = _suggest_language_code(code)
    suggestion = f"; did you mean '{match}'?" if match else ""
    raise LanguageCodeError(
        f"invalid language code '{value}'{suggestion} See supported codes: "
        f"{LANGUAGE_CODES_URL}"
    )
