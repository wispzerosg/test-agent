import unittest

from hf_benchmark_agent.telegram_bot import (
    TELEGRAM_BOT_TOKEN,
    _extract_benchmark_requests,
    split_message,
)


class TestTelegramBot(unittest.TestCase):
    def test_hardcoded_token_constant(self):
        self.assertTrue(TELEGRAM_BOT_TOKEN.startswith("7618745447:"))

    def test_split_message_short(self):
        parts = split_message("hello", max_len=50)
        self.assertEqual(parts, ["hello"])

    def test_split_message_long(self):
        text = "x" * 9000
        parts = split_message(text, max_len=4000)
        self.assertEqual(len(parts), 3)
        self.assertEqual(sum(len(p) for p in parts), 9000)
        self.assertTrue(all(len(p) <= 4000 for p in parts))

    def test_extract_benchmark_requests_within_24h(self):
        now = 1_700_000_000
        updates = [
            {
                "update_id": 1,
                "message": {
                    "date": now - 100,
                    "text": "benchmark best coding model",
                    "chat": {"id": 123},
                },
            },
            {
                "update_id": 2,
                "message": {
                    "date": now - 200,
                    "text": "/benchmark best vision model",
                    "chat": {"id": 124},
                },
            },
            {
                "update_id": 3,
                "message": {
                    "date": now - (25 * 3600),
                    "text": "benchmark too old",
                    "chat": {"id": 125},
                },
            },
            {
                "update_id": 4,
                "message": {
                    "date": now - 50,
                    "text": "hello world",
                    "chat": {"id": 126},
                },
            },
        ]
        extracted = _extract_benchmark_requests(updates, now_ts=now, window_hours=24)
        self.assertEqual(len(extracted), 2)
        self.assertEqual(extracted[0]["request"], "best coding model")
        self.assertEqual(extracted[1]["request"], "best vision model")

    def test_extract_benchmark_requests_skips_answered_ids(self):
        now = 1_700_000_000
        updates = [
            {
                "update_id": 10,
                "message": {
                    "date": now - 100,
                    "text": "benchmark best coding model",
                    "chat": {"id": 123},
                },
            },
            {
                "update_id": 11,
                "message": {
                    "date": now - 200,
                    "text": "/benchmark best vision model",
                    "chat": {"id": 124},
                },
            },
            {
                "update_id": 12,
                "message": {
                    "date": now - 300,
                    "text": "benchmark reasoning model",
                    "chat": {"id": 125},
                },
            },
        ]
        extracted = _extract_benchmark_requests(
            updates, now_ts=now, window_hours=24, answered_update_ids={10, 12},
        )
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["request"], "best vision model")
        self.assertEqual(extracted[0]["update_id"], 11)

    def test_extract_benchmark_requests_empty_answered_ids(self):
        now = 1_700_000_000
        updates = [
            {
                "update_id": 10,
                "message": {
                    "date": now - 100,
                    "text": "benchmark test query",
                    "chat": {"id": 123},
                },
            },
        ]
        extracted = _extract_benchmark_requests(
            updates, now_ts=now, window_hours=24, answered_update_ids=set(),
        )
        self.assertEqual(len(extracted), 1)

if __name__ == "__main__":
    unittest.main()
