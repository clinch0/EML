"""Compute the completion percentage from the checklist in docs/REQUIREMENTS.md."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ITEM_RE = re.compile(r"^\s*-\s*\[( |x|X)\]\s*(R\d+)\b")
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "docs" / "REQUIREMENTS.md"


def count_items(text: str) -> tuple[int, int, list[str]]:
    """Count requirement items.

    Args:
        text: Contents of the requirements file.

    Returns:
        Tuple ``(done, total, open_ids)``.
    """
    done = 0
    total = 0
    open_ids: list[str] = []
    for line in text.splitlines():
        match = ITEM_RE.match(line)
        if match is None:
            continue
        total += 1
        if match.group(1).lower() == "x":
            done += 1
        else:
            open_ids.append(match.group(2))
    return done, total, open_ids


def main(argv: list[str] | None = None) -> int:
    """Print the completion percentage.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args(argv)
    done, total, open_ids = count_items(args.path.read_text(encoding="utf-8"))
    if total == 0:
        print("No requirement items found.")
        return 1
    print(f"Completed: {done}/{total} = {100.0 * done / total:.1f}%")
    if open_ids:
        print("Open: " + ", ".join(open_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
