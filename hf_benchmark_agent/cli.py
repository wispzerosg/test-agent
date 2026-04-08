from __future__ import annotations

import argparse
import json
import sys

from .agent import BenchmarkAgentResult, ModelScore, run_agent_sync
from .telegram_bot import TelegramOutputRelay, build_telegram_summary_text


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


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _build_cost_table(top_models: list[ModelScore]) -> str:
    lines = ["Top-5 cost summary:"]
    lines.append("rank | model                        | in$/1M  | out$/1M | $/image | $/second")
    for model in top_models:
        label = f"#{model.rank}" if model.rank is not None else "-"
        name = model.model_id
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(
            f"{label:>4} | {name:<28} | "
            f"{_format_cost(model.input_cost_per_million):>7} | "
            f"{_format_cost(model.output_cost_per_million):>7} | "
            f"{_format_cost(model.price_per_image):>7} | "
            f"{_format_cost(model.price_per_second):>8}"
        )
    return "\n".join(lines)


def _print_result(result: BenchmarkAgentResult) -> None:
    print(json.dumps(result.to_dict(), indent=2))
    print()
    print(_build_cost_table(result.top_models))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_agent_sync(args.request, arena_base_url=args.arena_base_url)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    _print_result(result)
    relay = TelegramOutputRelay()
    try:
        relay.send_text_copy(build_telegram_summary_text(result))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "telegram_warning": (
                        f"Output not copied to Telegram: {exc}. "
                        "Send any message to the bot first so chat_id can be discovered."
                    )
                },
                indent=2,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
