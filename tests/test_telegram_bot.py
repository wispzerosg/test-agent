import unittest

from hf_benchmark_agent.telegram_bot import _build_result_message, split_message
from hf_benchmark_agent.agent import BenchmarkAgentResult, ModelScore, SelectedBenchmark


class TestTelegramBot(unittest.TestCase):
    def test_split_message_short(self):
        parts = split_message("hello", max_len=50)
        self.assertEqual(parts, ["hello"])

    def test_split_message_long(self):
        text = "x" * 9000
        parts = split_message(text, max_len=4000)
        self.assertEqual(len(parts), 3)
        self.assertEqual(sum(len(p) for p in parts), 9000)
        self.assertTrue(all(len(p) <= 4000 for p in parts))

    def test_build_result_message_contains_cost_table(self):
        result = BenchmarkAgentResult(
            request="best coding model",
            selected_benchmark=SelectedBenchmark(
                dataset_id="arena/code:webdev",
                url="https://arena.ai/leaderboard/code",
                relevance_score=1.2,
            ),
            top_models=[
                ModelScore(
                    rank=1,
                    model_id="m1",
                    score=1.1,
                    verified=None,
                    input_cost_per_million=5.0,
                    output_cost_per_million=25.0,
                    price_per_image=None,
                    price_per_second=None,
                )
            ],
        )
        text = _build_result_message(result)
        self.assertIn('"dataset_id": "arena/code:webdev"', text)
        self.assertIn("Top-5 cost summary", text)


if __name__ == "__main__":
    unittest.main()
