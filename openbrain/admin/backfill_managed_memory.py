from __future__ import annotations

import argparse

from openbrain import build_openbrain


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill managed memories from existing raw captures.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing managed memories before rebuilding.",
    )
    args = parser.parse_args()

    app = build_openbrain()
    print(app.rebuild_managed_memories(reset=args.reset))


if __name__ == "__main__":
    main()
