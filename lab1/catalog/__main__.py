"""Запуск интерфейса или одного сценария: python -m catalog [ui|classify|interpret|lookup|recovery|dialogue]."""

import argparse
import sys
import traceback

from catalog.config import redact
from catalog.registry import BY_ID


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Каталог товаров по фразе на естественном языке")
    parser.add_argument(
        "command",
        nargs="?",
        default="ui",
        choices=("ui", *BY_ID),
    )
    args = parser.parse_args(argv)
    if args.command == "ui":
        from catalog.ui import serve

        serve()
        return
    try:
        BY_ID[args.command]["run"]()
    except Exception:
        sys.stderr.write(redact(traceback.format_exc()))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
