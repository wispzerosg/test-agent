from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_SUBPROCESS_CMD_TEMPLATE = [
    sys.executable, "-m", "hf_benchmark_agent.telegram_bot",
    "--read-bot",
]


@dataclass
class ScanResult:
    timestamp: float
    returncode: int
    processed: int
    items: list[dict[str, Any]]
    error: str | None


def _build_scan_command(
    *,
    hours: int = 24,
    limit: int = 100,
    arena_base_url: str | None = None,
) -> list[str]:
    cmd = list(_SUBPROCESS_CMD_TEMPLATE)
    cmd.extend(["--hours", str(hours)])
    cmd.extend(["--limit", str(limit)])
    if arena_base_url:
        cmd.extend(["--arena-base-url", arena_base_url])
    return cmd


def run_scan_subprocess(
    *,
    hours: int = 24,
    limit: int = 100,
    arena_base_url: str | None = None,
    timeout_seconds: float = 120.0,
) -> ScanResult:
    cmd = _build_scan_command(hours=hours, limit=limit, arena_base_url=arena_base_url)
    ts = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return ScanResult(
            timestamp=ts, returncode=-1, processed=0, items=[],
            error="Scan subprocess timed out",
        )
    except OSError as exc:
        return ScanResult(
            timestamp=ts, returncode=-1, processed=0, items=[],
            error=f"Failed to start scan subprocess: {exc}",
        )

    if proc.returncode != 0:
        stderr_text = proc.stderr.strip()
        error_msg = stderr_text or f"Scan subprocess exited with code {proc.returncode}"
        return ScanResult(
            timestamp=ts, returncode=proc.returncode, processed=0, items=[],
            error=error_msg,
        )

    stdout_text = proc.stdout.strip()
    try:
        payload = json.loads(stdout_text)
    except (json.JSONDecodeError, ValueError):
        return ScanResult(
            timestamp=ts, returncode=proc.returncode, processed=0, items=[],
            error=f"Invalid JSON from scan subprocess: {stdout_text[:200]}",
        )

    processed = payload.get("processed", 0) if isinstance(payload, dict) else 0
    items = payload.get("items", []) if isinstance(payload, dict) else []
    return ScanResult(
        timestamp=ts, returncode=proc.returncode,
        processed=processed, items=items, error=None,
    )


class BenchmarkScanScheduler:
    """Periodically spawns a subprocess to scan Telegram history for benchmark requests."""

    def __init__(
        self,
        interval_seconds: int = 300,
        hours: int = 24,
        limit: int = 100,
        arena_base_url: str | None = None,
        subprocess_timeout: float = 120.0,
    ) -> None:
        if interval_seconds < 10:
            raise ValueError("interval_seconds must be >= 10")
        self.interval_seconds = interval_seconds
        self.hours = hours
        self.limit = limit
        self.arena_base_url = arena_base_url
        self.subprocess_timeout = subprocess_timeout
        self._stop = False
        self._scan_count = 0
        self._history: list[ScanResult] = []

    @property
    def scan_count(self) -> int:
        return self._scan_count

    @property
    def history(self) -> list[ScanResult]:
        return list(self._history)

    def _run_one_scan(self) -> ScanResult:
        result = run_scan_subprocess(
            hours=self.hours,
            limit=self.limit,
            arena_base_url=self.arena_base_url,
            timeout_seconds=self.subprocess_timeout,
        )
        self._scan_count += 1
        self._history.append(result)
        return result

    def stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        """Block and scan on the configured interval until stop() or SIGINT/SIGTERM."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_shutdown(signum: int, frame: Any) -> None:
            logger.info("Received signal %s, shutting down scheduler.", signum)
            self.stop()

        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)

        logger.info(
            "Scheduler started: scanning every %ds (hours=%d, limit=%d).",
            self.interval_seconds, self.hours, self.limit,
        )

        try:
            while not self._stop:
                result = self._run_one_scan()
                if result.error:
                    logger.warning("Scan #%d failed: %s", self._scan_count, result.error)
                else:
                    logger.info(
                        "Scan #%d completed: processed %d benchmark request(s).",
                        self._scan_count, result.processed,
                    )

                deadline = time.monotonic() + self.interval_seconds
                while not self._stop and time.monotonic() < deadline:
                    time.sleep(min(1.0, deadline - time.monotonic()))
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        logger.info("Scheduler stopped after %d scan(s).", self._scan_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Periodically scan Telegram history for benchmark requests via subprocess.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between scans (default: 300). Minimum 10.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="History window in hours (default: 24).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max Telegram updates per scan (default: 100).",
    )
    parser.add_argument(
        "--arena-base-url",
        default=None,
        help="Arena base URL (default: https://arena.ai).",
    )
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    scheduler = BenchmarkScanScheduler(
        interval_seconds=max(10, args.interval),
        hours=args.hours,
        limit=args.limit,
        arena_base_url=args.arena_base_url,
    )
    try:
        scheduler.run_forever()
    except Exception as exc:
        logger.error("Scheduler failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
