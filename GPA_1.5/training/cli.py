from __future__ import annotations

import json
import sys

from .assets import build_default_path_report


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--print-default-paths" in args:
        print(json.dumps(build_default_path_report(), indent=2))
        return

    if "-h" in args or "--help" in args:
        from .runner import build_parser

        build_parser().print_help()
        return

    from .runner import train

    train(args)