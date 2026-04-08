from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from typing import Any

import requests

from .agent import BenchmarkAgent, BenchmarkAgentResult

TELEGRAM_BOT_TOKEN = "7618745447:AAHa1p4c8udJe2tnSV_lYFDgnMDba9rQpPg"
BENCHMARK_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:/)?benchmark(?:@\w+)?\s+(.+)",
    flags=re.IGNORECASE,
)


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_telegram_summary_text(result: BenchmarkAgentResult) -> str:
    lines: list[str] = []
    lines.append(f"Request: {result.request}")
    lines.append(f"Selected benchmark: {result.selected_benchmark.dataset_id}")
    lines.append(f"Rating link: {result.selected_benchmark.url}")
    lines.append("")
    lines.append("Top-5 models:")
    for model in result.top_models:
        rank = model.rank if model.rank is not None else "-"
        score = "n/a" if model.score is None else f"{model.score:.4f}"
        in_cost = _format_cost(model.input_cost_per_million)
        out_cost = _format_cost(model.output_cost_per_million)
        lines.append(
            f"- #{rank} {model.model_id}: score={score}, "
            f"in$/1M={in_cost}, out$/1M={out_cost}"
        )
    return "\n".join(lines)


def split_message(text: str, max_len: int = 3500) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def _response_json_or_error(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise requests.HTTPError("Unexpected Telegram response shape", response=response)
    if not payload.get("ok"):
        description = payload.get("description") or "Unknown Telegram API error"
        raise requests.HTTPError(description, response=response)
    return payload


def _parse_telegram_result(payload: dict[str, Any], endpoint: str | None = None) -> list[dict[str, Any]]:
    if not payload.get("ok"):
        description = payload.get("description") or "Unknown Telegram API error"
        prefix = f"{endpoint}: " if endpoint else ""
        raise RuntimeError(f"{prefix}{description}")
    result = payload.get("result")
    return result if isinstance(result, list) else []


def _extract_benchmark_requests(
    updates: list[dict[str, Any]],
    *,
    now_ts: int | None = None,
    window_hours: int = 24,
) -> list[dict[str, Any]]:
    now = int(time.time()) if now_ts is None else now_ts
    cutoff = now - (window_hours * 3600)
    out: list[dict[str, Any]] = []

    for update in updates:
        message = update.get("message")
        if not isinstance(message, dict):
            continue
        msg_ts = message.get("date")
        if not isinstance(msg_ts, int) or msg_ts < cutoff:
            continue
        chat = message.get("chat")
        if not isinstance(chat, dict):
            continue
        chat_id = chat.get("id")
        if not isinstance(chat_id, int):
            continue
        text = message.get("text")
        if not isinstance(text, str):
            continue

        request_text: str | None = None
        for line in text.splitlines():
            match = BENCHMARK_COMMAND_RE.search(line.strip())
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    request_text = candidate
                    break

        if request_text is None:
            match = BENCHMARK_COMMAND_RE.search(text.strip())
            if match:
                candidate = match.group(1).strip()
                if candidate:
                    request_text = candidate

        if request_text:
            out.append(
                {
                    "chat_id": chat_id,
                    "request": request_text,
                    "date": msg_ts,
                    "update_id": update.get("update_id"),
                }
            )

    return out


class TelegramOutputRelay:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.token = TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.timeout_seconds = timeout_seconds

    def _telegram_request(
        self,
        http_method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = requests.request(
            method=http_method,
            url=f"{self.base_url}/{endpoint}",
            params=params,
            json=json_payload,
            timeout=self.timeout_seconds,
        )
        return _response_json_or_error(response)

    def _send_message(self, chat_id: int | str, text: str) -> None:
        self._telegram_request(
            "POST",
            "sendMessage",
            json_payload={"chat_id": chat_id, "text": text},
        )

    def get_latest_chat_id(self) -> int:
        updates = self._get_updates(timeout=1)
        for update in reversed(updates):
            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat")
            if not isinstance(chat, dict):
                continue
            chat_id = chat.get("id")
            if isinstance(chat_id, int):
                return chat_id
        raise RuntimeError("No chat_id found. Send any message to the bot first.")

    def _get_updates(
        self, *, timeout: int = 1, limit: int = 100, offset: int | None = None
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout, "limit": limit}
        if offset is not None:
            params["offset"] = offset
        payload = self._telegram_request("GET", "getUpdates", params=params)
        return _parse_telegram_result(payload, endpoint="getUpdates")

    def send_text_copy(self, text: str, chat_id: int | None = None) -> int:
        target_chat_id = chat_id if chat_id is not None else self.get_latest_chat_id()
        sent = 0
        for chunk in split_message(text):
            self._send_message(target_chat_id, chunk)
            sent += 1
        return sent

    def read_bot(
        self, *, arena_base_url: str | None = None, hours: int = 24, limit: int = 100
    ) -> list[dict[str, Any]]:
        updates = self._get_updates(timeout=1, limit=limit)
        requests_to_process = _extract_benchmark_requests(
            updates, window_hours=max(1, hours)
        )
        if not requests_to_process:
            return []

        agent = BenchmarkAgent(arena_base_url=arena_base_url)
        processed: list[dict[str, Any]] = []
        for item in requests_to_process:
            chat_id = item["chat_id"]
            request_text = item["request"]
            try:
                result = asyncio.run(agent.run(request_text))
                text = build_telegram_summary_text(result)
            except Exception as exc:
                text = (
                    f"Request: {request_text}\n"
                    f"benchmark processing failed: {exc}"
                )
            self.send_text_copy(text, chat_id=chat_id)
            processed.append(item)
        return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send terminal output copy or process benchmark commands from Telegram."
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to send. If omitted, use --stdin.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read text content from stdin and send it.",
    )
    parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Optional explicit chat id. Defaults to latest chat from getUpdates.",
    )
    parser.add_argument(
        "--read-bot",
        action="store_true",
        help="Read last 24h (configurable) bot history, parse 'benchmark ...' and process requests.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run --read-bot on a recurring schedule (requires --read-bot).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scheduled scans (default: 300). Minimum 10.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="History window in hours for --read-bot (default: 24).",
    )
    parser.add_argument(
        "--arena-base-url",
        default=None,
        help="Arena base URL for read_bot request processing (default: https://arena.ai).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum Telegram updates to inspect in --read-bot mode (default: 100).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.schedule and not args.read_bot:
        print(
            json.dumps({"error": "--schedule requires --read-bot"}, indent=2),
            file=sys.stderr,
        )
        return 1

    if args.read_bot and args.schedule:
        import logging
        from .scheduler import BenchmarkScanScheduler

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        scheduler = BenchmarkScanScheduler(
            interval_seconds=max(10, args.interval),
            hours=args.hours,
            limit=args.limit,
            arena_base_url=args.arena_base_url,
        )
        try:
            scheduler.run_forever()
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        return 0

    relay = TelegramOutputRelay()

    if args.read_bot:
        try:
            processed = relay.read_bot(
                arena_base_url=args.arena_base_url,
                hours=args.hours,
                limit=args.limit,
            )
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps({"processed": len(processed), "items": processed}, indent=2))
        return 0

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = " ".join(args.text)

    text = text.strip()
    if not text:
        print(json.dumps({"error": "No text provided. Pass text or use --stdin."}, indent=2), file=sys.stderr)
        return 1

    try:
        relay.send_text_copy(text, chat_id=args.chat_id)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
