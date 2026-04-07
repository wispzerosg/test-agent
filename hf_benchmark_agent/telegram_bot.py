from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

import requests

from .agent import BenchmarkAgent, BenchmarkAgentResult
from .cli import _build_cost_table

TELEGRAM_BOT_TOKEN = "406067963:AAHs9OUwjbxSelsrZBhi_WYBp9ulS3ez1Xc"


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


class TelegramBotRunner:
    def __init__(
        self,
        arena_base_url: str | None = None,
        timeout_seconds: float = 30.0,
        debug: bool = False,
    ) -> None:
        self.token = TELEGRAM_BOT_TOKEN
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.agent = BenchmarkAgent(arena_base_url=arena_base_url)
        self.timeout_seconds = timeout_seconds
        self.debug = debug

    def run_forever(self, poll_timeout: int = 30, once: bool = False) -> None:
        offset: int | None = None
        while True:
            try:
                updates = self._get_updates(offset=offset, timeout=poll_timeout)
                if self.debug:
                    print(f"[telegram] fetched {len(updates)} updates", file=sys.stderr)
                for update in updates:
                    update_id = update.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1
                    self._handle_update(update)
                if once:
                    return
            except requests.RequestException as exc:
                if self.debug:
                    print(f"[telegram] request error: {exc}", file=sys.stderr)
                time.sleep(2)
            except Exception as exc:
                if self.debug:
                    print(f"[telegram] runtime error: {exc}", file=sys.stderr)
                time.sleep(2)

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

    def _get_updates(self, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        payload = self._telegram_request("GET", "getUpdates", params=params)
        return _parse_telegram_result(payload)

    def _send_message(self, chat_id: int | str, text: str) -> None:
        self._telegram_request(
            "POST",
            "sendMessage",
            json_payload={"chat_id": chat_id, "text": text},
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
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one polling cycle and exit (useful for diagnostics).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print Telegram polling/debug logs to stderr.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    runner = TelegramBotRunner(
        arena_base_url=args.arena_base_url,
        timeout_seconds=max(args.poll_timeout + 10, 30),
        debug=args.debug,
    )
    runner.run_forever(poll_timeout=args.poll_timeout, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
