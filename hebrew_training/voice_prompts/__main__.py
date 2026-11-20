"""`python -m hebrew_training.voice_prompts extract|dedupe ...` (see README.md)."""

from __future__ import annotations

import argparse

from hebrew_training.voice_prompts import dedupe, extract


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m hebrew_training.voice_prompts")
    commands = parser.add_subparsers(dest="command", required=True)
    extract.add_arguments(commands.add_parser("extract", help="cut prompts at word-gap seams"))
    dedupe.add_arguments(commands.add_parser("dedupe", help="remove same-voice duplicates"))
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract.check_arguments(parser, args)
        extract.report(extract.run(args), args)
    else:
        dedupe.run(args)


if __name__ == "__main__":
    main()
