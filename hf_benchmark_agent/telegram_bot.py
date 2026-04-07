from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from typing import Any

import requests

from .agent import BenchmarkAgent, BenchmarkAgentResult
from .cli import _build_cost_table


def _build_result_message(result: BenchmarkAgentResult) -> str:
    payload = json.dumps(result.to_dict(), indent=2)
    cost_table = _build_cost_table(result.top_models)
    return f"{payload}\n\n{cost_table}"


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


class TelegramBotRunner:
    def __init__(
        self,
        token: str,
        arena_base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.agent = BenchmarkAgent(arena_base_url=arena_base_url)
        self.timeout_seconds = timeout_seconds

    def run_forever(self, poll_timeout: int = 30) -> None:
        offset: int | None = None
        while True:
            try:
                updates = self._get_updates(offset=offset, timeout=poll_timeout)
                for update in updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1
                    self._handle_update(update)
            except requests.RequestException:
                time.sleep(2)

    def _get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        response = requests.get(
            f"{self.base_url}/getUpdates",
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            return []
        result = payload.get("result")
        return result if isinstance(result, list) else []

    def _send_message(self, chat_id: int | str, text: str) -> None:
        requests.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=self.timeout_seconds,
        )

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        text = message.get("text")
        chat = message.get("chat")
        if not isinstance(text, str) or not isinstance(chat, dict):
            return
        chat_id = chat.get("id")
        if chat_id is None:
            return

        text = text.strip()
        if not text:
            return
        if text in {"/start", "/help"}:
            self._send_message(
                chat_id,
                "Send any text request (e.g. 'best coding model') and I will return top-5 models with normalized scores and costs.",
            )
            return

        self._send_message(chat_id, "Working on your request...")
        try:
            result = asyncio.run(self.agent.run(text))
            output = _build_result_message(result)
        except Exception as exc:
            output = json.dumps({"error": str(exc)}, indent=2)

        for chunk in split_message(output):
            self._send_message(chat_id, chunk)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Telegram bot that returns Arena benchmark results."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Telegram bot token (default: TELEGRAM_BOT_TOKEN env var).",
    )
    parser.add_argument(
        "--arena-base-url",
        default=None,
        help="Arena base URL (default: ARENA_BASE_URL env var or https://arena.ai).",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=30,
        help="Telegram getUpdates long-poll timeout in seconds (default: 30).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print(json.dumps({"error": "Telegram token missing. Use --token or TELEGRAM_BOT_TOKEN."}, indent=2))
        return 1

    runner = TelegramBotRunner(
        token=token,
        arena_base_url=args.arena_base_url,
        timeout_seconds=max(args.poll_timeout + 10, 30),
    )
    runner.run_forever(poll_timeout=args.poll_timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
