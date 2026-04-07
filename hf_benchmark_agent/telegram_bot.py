from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import requests

from .agent import BenchmarkAgentResult

TELEGRAM_BOT_TOKEN = "406067963:AAHs9OUwjbxSelsrZBhi_WYBp9ulS3ez1Xc"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _build_cost_table(top_models: list[Any]) -> str:
    lines = ["Top-5 cost summary:"]
    lines.append("rank | model                        | in$/1M  | out$/1M | $/image | $/second")
    for model in top_models:
        if hasattr(model, "rank"):
            rank_value = model.rank
            model_id = model.model_id
            input_cost = model.input_cost_per_million
            output_cost = model.output_cost_per_million
            per_image = model.price_per_image
            per_second = model.price_per_second
        else:
            rank_value = model.get("rank")
            model_id = str(model.get("model_id", "unknown"))
            input_cost = model.get("input_cost_per_million")
            output_cost = model.get("output_cost_per_million")
            per_image = model.get("price_per_image")
            per_second = model.get("price_per_second")

        label = f"#{rank_value}" if rank_value is not None else "-"
        name = model_id
        if len(name) > 28:
            name = name[:25] + "..."
        lines.append(
            f"{label:>4} | {name:<28} | "
            f"{_format_cost(input_cost):>7} | "
            f"{_format_cost(output_cost):>7} | "
            f"{_format_cost(per_image):>7} | "
            f"{_format_cost(per_second):>8}"
        )
    return "\n".join(lines)


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
        payload = self._telegram_request(
            "GET",
            "getUpdates",
            params={"timeout": 1},
        )
        updates = _parse_telegram_result(payload, endpoint="getUpdates")
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

    def send_text_copy(self, text: str, chat_id: int | None = None) -> int:
        target_chat_id = chat_id if chat_id is not None else self.get_latest_chat_id()
        sent = 0
        for chunk in split_message(text):
            self._send_message(target_chat_id, chunk)
            sent += 1
        return sent

    def send_output_copy(self, result: BenchmarkAgentResult, chat_id: int | None = None) -> int:
        return self.send_text_copy(_build_result_message(result), chat_id=chat_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send terminal text output copy to Telegram latest chat."
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = " ".join(args.text)

    text = text.strip()
    if not text:
        print(json.dumps({"error": "No text provided. Pass text or use --stdin."}, indent=2), file=sys.stderr)
        return 1

    relay = TelegramOutputRelay()
    try:
        relay.send_text_copy(text, chat_id=args.chat_id)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    return 0
