import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from hf_benchmark_agent.agent import BenchmarkAgent, BenchmarkAgentResult, ModelScore, SelectedBenchmark
from hf_benchmark_agent.cli import _build_cost_table, main


def _make_leaderboard_html(arena_slug, leaderboard_slug, category, entries_json):
    return (
        '<script>self.__next_f.push([1,"x:'
        + json.dumps({
            "arenaSlug": arena_slug,
            "leaderboardSlug": leaderboard_slug,
            "params": {"category": category},
            "entries": entries_json,
        }).replace('"', '\\"')
        + '"])</script>'
    )


class FakeBenchmarkAgent(BenchmarkAgent):
    async def _safe_get_text(self, url: str) -> str:
        if url.endswith("/leaderboard/code"):
            return _make_leaderboard_html("code", "webdev", "webdev", [
                {"rank": 1, "modelDisplayName": "model/a", "rating": 66.1,
                 "inputPricePerMillion": 0.5, "outputPricePerMillion": 1.0},
                {"rank": 2, "modelDisplayName": "model/b", "rating": 64.0,
                 "inputPricePerMillion": 0.6, "outputPricePerMillion": 1.2},
                {"rank": 3, "modelDisplayName": "model/c", "rating": 63.5},
                {"rank": 4, "modelDisplayName": "model/d", "rating": 62.9},
                {"rank": 5, "modelDisplayName": "model/e", "rating": 62.0},
                {"rank": 6, "modelDisplayName": "model/f", "rating": 61.0},
            ])
        if url.endswith("/leaderboard/text"):
            return _make_leaderboard_html("text", "overall", "overall", [
                {"rank": 1, "modelDisplayName": "text-model", "rating": 80.0},
            ])
        if url.endswith("/leaderboard/image-edit"):
            return _make_leaderboard_html("image-edit", "overall", "overall", [
                {"rank": 1, "modelDisplayName": "img-edit-model", "rating": 55.0},
            ])
        if url.endswith("/leaderboard/vision"):
            return _make_leaderboard_html("vision", "overall", "overall", [
                {"rank": 1, "modelDisplayName": "vision-model", "rating": 70.0},
            ])
        return ""


class FallbackLeaderboardAgent(BenchmarkAgent):
    async def _safe_get_text(self, url: str) -> str:
        if url.endswith("/leaderboard/code"):
            return """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"code\\",\\"leaderboardSlug\\":\\"webdev\\",\\"params\\":{\\"category\\":\\"webdev\\"},\\"entries\\":[]}"])</script>
"""
        if url.endswith("/leaderboard/text"):
            return """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"text\\",\\"leaderboardSlug\\":\\"overall\\",\\"params\\":{\\"category\\":\\"overall\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"org/r1\\",\\"rating\\":42.0},{\\"rank\\":2,\\"modelDisplayName\\":\\"org/r2\\",\\"rating\\":39.5}]}"])</script>
"""
        return ""


class TestBenchmarkAgent(unittest.IsolatedAsyncioTestCase):
    async def test_run_selects_relevant_benchmark_and_returns_top_five(self):
        agent = FakeBenchmarkAgent(arena_base_url="https://arena.ai")
        result = await agent.run("best coding model for software engineering")

        self.assertEqual(result.selected_benchmark.dataset_id, "arena/code:webdev")
        self.assertEqual(len(result.top_models), 5)
        self.assertEqual(result.top_models[0].model_id, "model/a")
        self.assertAlmostEqual(result.top_models[0].score or 0.0, 66.1 / ((66.1 + 64.0 + 63.5 + 62.9 + 62.0 + 61.0) / 6))
        self.assertEqual(result.top_models[0].input_cost_per_million, 0.5)
        self.assertEqual(result.top_models[0].output_cost_per_million, 1.0)

    def test_extract_top_models_supports_arena_fields(self):
        agent = BenchmarkAgent()
        entries = [
            {
                "rank": "1",
                "modelDisplayName": "org/m1",
                "rating": "91.2",
                "inputPricePerMillion": "2.5",
                "outputPricePerMillion": "10.0",
                "pricePerImage": "0.04",
                "pricePerSecond": "0.20",
            },
            {"rank": "2", "modelDisplayName": "org/m2", "rating": "90.0"},
        ]
        models = agent._extract_top_models_from_entries(entries, limit=5)
        self.assertEqual([m.model_id for m in models], ["org/m1", "org/m2"])
        self.assertAlmostEqual(models[0].score or 0.0, 91.2 / 90.6)
        self.assertEqual(models[0].input_cost_per_million, 2.5)
        self.assertEqual(models[0].output_cost_per_million, 10.0)
        self.assertEqual(models[0].price_per_image, 0.04)
        self.assertEqual(models[0].price_per_second, 0.2)

    def test_extract_arena_leaderboards_from_html(self):
        agent = BenchmarkAgent()
        html = """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"code\\",\\"leaderboardSlug\\":\\"webdev\\",\\"params\\":{\\"category\\":\\"webdev\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"a\\",\\"rating\\":1.0}]}"])</script>
"""
        payloads = agent._extract_arena_leaderboards_from_html(html)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0].arena_slug, "code")

    def test_extract_arena_leaderboards_drops_unused_fields(self):
        agent = BenchmarkAgent()
        html = """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"code\\",\\"leaderboardSlug\\":\\"webdev\\",\\"params\\":{\\"category\\":\\"webdev\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"a\\",\\"rating\\":1.0,\\"unused\\":\\"x\\"}],\\"unusedRoot\\":\\"root\\"}"])</script>
"""
        payloads = agent._extract_arena_leaderboards_from_html(html)
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload.arena_slug, "code")
        self.assertEqual(payload.leaderboard_slug, "webdev")
        self.assertEqual(payload.category, "webdev")
        self.assertEqual(len(payload.entries), 1)

    def test_leaderboard_page_url(self):
        agent = BenchmarkAgent()
        url = agent._leaderboard_page_url("code")
        self.assertEqual(url, "https://arena.ai/leaderboard/code")

    async def test_run_tries_next_candidate_when_first_has_no_entries(self):
        agent = FallbackLeaderboardAgent()
        result = await agent.run("math reasoning benchmark")
        self.assertEqual(result.selected_benchmark.dataset_id, "arena/text:overall")
        self.assertEqual(result.top_models[0].model_id, "org/r1")
        self.assertAlmostEqual(result.top_models[0].score or 0.0, 42.0 / ((42.0 + 39.5) / 2))

    def test_scores_use_top_100_mean(self):
        agent = BenchmarkAgent()
        entries = []
        for rank in range(1, 121):
            entries.append({"rank": rank, "modelDisplayName": f"m{rank}", "rating": float(rank)})
        models = agent._extract_top_models_from_entries(entries, limit=5)
        # Mean is computed over rank 1..100 => 50.5
        self.assertAlmostEqual(models[0].score or 0.0, 1.0 / 50.5)

    def test_cost_table_renders(self):
        table = _build_cost_table(
            [
                ModelScore(
                    rank=1,
                    model_id="a",
                    score=1.1,
                    verified=None,
                    input_cost_per_million=2.0,
                    output_cost_per_million=8.0,
                    price_per_image=None,
                    price_per_second=None,
                ),
                ModelScore(
                    rank=2,
                    model_id="b",
                    score=1.0,
                    verified=None,
                    input_cost_per_million=None,
                    output_cost_per_million=None,
                    price_per_image=0.03,
                    price_per_second=0.15,
                ),
            ]
        )
        self.assertIn("Top-5 cost summary", table)
        self.assertIn("a", table)
        self.assertIn("b", table)
        self.assertIn("in$/1M", table)

    def test_page_prioritization_quality(self):
        """Verify the top prioritized page is correct for a range of requests."""
        agent = BenchmarkAgent()
        cases = [
            ("russian language model", "text"),
            ("best coding model", "code"),
            ("image generation", "text-to-image"),
            ("video editing tool", "video-edit"),
            ("best vision model", "vision"),
            ("document parsing", "document"),
            ("search engine model", "search"),
            ("math reasoning", "text"),
            ("translate english to french", "text"),
            ("best model for writing", "text"),
            ("text to image", "text-to-image"),
            ("edit my photo", "image-edit"),
            ("best llm", "text"),
            ("multilingual chatbot", "text"),
            ("best model for OCR", "vision"),
            ("generate a video", "text-to-video"),
            ("python code assistant", "code"),
            ("pdf extraction", "document"),
            ("web retrieval", "search"),
        ]
        failures = []
        for request, expected_page in cases:
            tokens = agent._tokenize(request)
            pages = agent._prioritized_pages(tokens)
            if pages[0] != expected_page:
                failures.append(
                    f"{request!r}: got {pages[0]!r}, expected {expected_page!r} "
                    f"(top-3: {pages[:3]})"
                )
        self.assertEqual(
            failures, [],
            f"{len(failures)} matching failure(s):\n" + "\n".join(failures),
        )

    def test_hint_alignment_rewards_matching_category(self):
        agent = BenchmarkAgent()
        request_tokens = agent._tokenize("russian language model")
        text_context = agent._tokenize("text text overall overall")
        image_context = agent._tokenize("image-edit image-edit overall overall")
        text_score = agent._hint_alignment_score(request_tokens, text_context)
        image_score = agent._hint_alignment_score(request_tokens, image_context)
        self.assertGreater(text_score, image_score)

    async def test_russian_language_model_matches_text_not_image_edit(self):
        agent = FakeBenchmarkAgent(arena_base_url="https://arena.ai")
        result = await agent.run("russian language model")
        self.assertIn("text", result.selected_benchmark.dataset_id)
        self.assertNotIn("image", result.selected_benchmark.dataset_id)

    async def test_coding_request_matches_code(self):
        agent = FakeBenchmarkAgent(arena_base_url="https://arena.ai")
        result = await agent.run("best coding model for software engineering")
        self.assertIn("code", result.selected_benchmark.dataset_id)

    def test_hint_alignment_penalizes_wrong_category(self):
        agent = BenchmarkAgent()
        request_tokens = agent._tokenize("edit my photo")
        correct_context = agent._tokenize("image-edit image-edit overall")
        wrong_context = agent._tokenize("text-to-image text-to-image overall")
        correct_score = agent._hint_alignment_score(request_tokens, correct_context)
        wrong_score = agent._hint_alignment_score(request_tokens, wrong_context)
        self.assertGreater(correct_score, wrong_score)

    def test_cli_prints_json_and_costs(self):
        fake_result = BenchmarkAgentResult(
            request="best coding model",
            selected_benchmark=SelectedBenchmark(
                dataset_id="arena/code:webdev",
                url="https://arena.ai/leaderboard/code",
                relevance_score=1.2,
            ),
            top_models=[
                ModelScore(
                    rank=1,
                    model_id="model/a",
                    score=1.2,
                    verified=None,
                    input_cost_per_million=2.0,
                    output_cost_per_million=8.0,
                    price_per_image=None,
                    price_per_second=None,
                ),
                ModelScore(
                    rank=2,
                    model_id="model/b",
                    score=1.1,
                    verified=None,
                    input_cost_per_million=1.5,
                    output_cost_per_million=6.0,
                    price_per_image=0.02,
                    price_per_second=0.1,
                ),
            ],
        )

        stdout = io.StringIO()
        with patch("hf_benchmark_agent.cli.run_agent_sync", return_value=fake_result):
            with patch("hf_benchmark_agent.cli.TelegramOutputRelay") as relay_cls:
                relay_cls.return_value.send_text_copy.return_value = 1
                with patch("sys.argv", ["hf-benchmark-agent", "best coding model"]):
                    with redirect_stdout(stdout):
                        exit_code = main()
                relay_cls.return_value.send_text_copy.assert_called_once()
                sent_text = relay_cls.return_value.send_text_copy.call_args.args[0]
                self.assertIn("Rating link:", sent_text)
                self.assertNotIn('"request"', sent_text)

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"dataset_id": "arena/code:webdev"', output)
        self.assertIn("Top-5 cost summary", output)


if __name__ == "__main__":
    unittest.main()
