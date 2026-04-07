import unittest

from hf_benchmark_agent.telegram_bot import TELEGRAM_BOT_TOKEN, split_message


class TestTelegramBot(unittest.TestCase):
    def test_hardcoded_token_constant(self):
        self.assertTrue(TELEGRAM_BOT_TOKEN.startswith("406067963:"))

    def test_split_message_short(self):
        parts = split_message("hello", max_len=50)
        self.assertEqual(parts, ["hello"])

    def test_split_message_long(self):
        text = "x" * 9000
        parts = split_message(text, max_len=4000)
        self.assertEqual(len(parts), 3)
        self.assertEqual(sum(len(p) for p in parts), 9000)
        self.assertTrue(all(len(p) <= 4000 for p in parts))

if __name__ == "__main__":
    unittest.main()
