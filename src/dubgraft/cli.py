"""Command-line entry point for DubGraft."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from dubgraft import __version__
from dubgraft.config import ProcessingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dubgraft",
        description="Align and graft dubbed audio across media releases using distributed audio anchors.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("positional_source", nargs="?", metavar="SOURCE")
    parser.add_argument("positional_target", nargs="?", metavar="TARGET")
    parser.add_argument("positional_output", nargs="?", metavar="OUTPUT")
    parser.add_argument("-s", "--source", dest="explicit_source", metavar="PATH")
    parser.add_argument("-t", "--target", dest="explicit_target", metavar="PATH")
    parser.add_argument("-o", "--output", dest="explicit_output", metavar="PATH")
    parser.add_argument(
        "-y",
        "--overwrite",
        action="store_true",
        help="allow replacing an existing output file",
    )
    return parser


def _resolve_argument(
    parser: argparse.ArgumentParser,
    role: str,
    positional: str | None,
    explicit: str | None,
) -> Path:
    if positional is not None and explicit is not None:
        parser.error(f"{role} was provided more than once")
    value = explicit if explicit is not None else positional
    if value is None:
        parser.error(f"{role} is required")
    return Path(value)


def parse_processing_config(
    argv: Sequence[str], parser: argparse.ArgumentParser | None = None
) -> ProcessingConfig:
    parser = parser or build_parser()
    arguments = parser.parse_args(argv)
    return ProcessingConfig(
        source=_resolve_argument(
            parser, "SOURCE", arguments.positional_source, arguments.explicit_source
        ),
        target=_resolve_argument(
            parser, "TARGET", arguments.positional_target, arguments.explicit_target
        ),
        output=_resolve_argument(
            parser, "OUTPUT", arguments.positional_output, arguments.explicit_output
        ),
        overwrite=arguments.overwrite,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        parser.print_help()
        return 0
    parse_processing_config(arguments, parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
