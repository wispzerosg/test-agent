from __future__ import annotations

import argparse
import json
import sys

from .agent import run_agent_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Return top-5 AI models and scores for the most relevant Hugging Face benchmark."
        )
    )
    parser.add_argument("request", help="Natural language request, e.g. 'best coding model'.")
    parser.add_argument(
        "--mcp-url",
        default=None,
        help="MCP server URL (default: HF_MCP_URL env var or https://huggingface.co/mcp).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face access token (default: HF_TOKEN env var).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_agent_sync(args.request, mcp_url=args.mcp_url, hf_token=args.hf_token)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
