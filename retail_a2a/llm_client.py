from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(slots=True)
class LocalLLMClient:
    """Simple client for locally deployed OpenAI-compatible chat APIs."""

    base_url: str
    model: str
    timeout_seconds: int = 20
    api_key: str | None = None

    @classmethod
    def from_env(cls) -> "LocalLLMClient":
        return cls(
            base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("LOCAL_LLM_MODEL", "llama3.1"),
            timeout_seconds=int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "20")),
            api_key=os.getenv("LOCAL_LLM_API_KEY"),
        )

    def chat(self, *, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as exc:
            return (
                "Local LLM unavailable. Returning deterministic fallback analysis. "
                f"Reason: {exc}"
            )

        try:
            parsed: dict[str, Any] = json.loads(response_body)
            return str(parsed["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            return (
                "Local LLM returned an unexpected response format. "
                "Fallback recommendation: expedite replenishment, monitor demand spikes, "
                "and align safety stock with lead-time variability."
            )
