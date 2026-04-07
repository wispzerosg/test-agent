from __future__ import annotations

import argparse
import json
import sys

from .agent import BenchmarkAgentResult, ModelScore, run_agent_sync


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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable terminal ASCII plot output.",
    )
    return parser


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _build_score_plot(top_models: list[ModelScore], width: int = 40) -> str:
    scored = [m for m in top_models if m.score is not None]
    if not scored:
        return "Top-5 score plot: no numeric scores available."

    min_score = min(m.score for m in scored if m.score is not None)
    max_score = max(m.score for m in scored if m.score is not None)
    score_span = max(max_score - min_score, 1e-9)

    lines = ["Top-5 score plot (relative scale):"]
    for model in top_models:
        label = f"#{model.rank}" if model.rank is not None else "-"
        name = model.model_id
        if len(name) > 28:
            name = name[:25] + "..."

        if model.score is None:
            bar = ""
        else:
            normalized = (model.score - min_score) / score_span
            bar_len = max(1, int(round(normalized * width)))
            bar = "█" * bar_len
        lines.append(f"{label:>4} | {name:<28} | {_format_score(model.score):>8} | {bar}")
    return "\n".join(lines)


def _print_result(result: BenchmarkAgentResult, show_plot: bool) -> None:
    print(json.dumps(result.to_dict(), indent=2))
    if show_plot:
        print()
        print(_build_score_plot(result.top_models))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_agent_sync(args.request, arena_base_url=args.arena_base_url)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    _print_result(result, show_plot=not args.no_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
