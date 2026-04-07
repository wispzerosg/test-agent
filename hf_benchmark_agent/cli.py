from __future__ import annotations

import argparse
import json
import sys

from .agent import run_agent_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Return top-5 AI models and scores for the most relevant Arena leaderboard."
        )
    )
    parser.add_argument("request", help="Natural language request, e.g. 'best coding model'.")
    parser.add_argument(
        "--arena-base-url",
        default=None,
        help="Arena base URL (default: ARENA_BASE_URL env var or https://arena.ai).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_agent_sync(args.request, arena_base_url=args.arena_base_url)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
