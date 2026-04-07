import unittest

from hf_benchmark_agent.agent import BenchmarkAgent


class FakeBenchmarkAgent(BenchmarkAgent):
    async def _safe_get_text(self, url: str) -> str:
        if url.endswith("/leaderboard/code"):
            return """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"code\\",\\"leaderboardSlug\\":\\"webdev\\",\\"params\\":{\\"category\\":\\"webdev\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"model/a\\",\\"rating\\":66.1},{\\"rank\\":2,\\"modelDisplayName\\":\\"model/b\\",\\"rating\\":64.0},{\\"rank\\":3,\\"modelDisplayName\\":\\"model/c\\",\\"rating\\":63.5},{\\"rank\\":4,\\"modelDisplayName\\":\\"model/d\\",\\"rating\\":62.9},{\\"rank\\":5,\\"modelDisplayName\\":\\"model/e\\",\\"rating\\":62.0},{\\"rank\\":6,\\"modelDisplayName\\":\\"model/f\\",\\"rating\\":61.0}]}"])</script>
"""
        if url.endswith("/leaderboard/text"):
            return """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"text\\",\\"leaderboardSlug\\":\\"overall\\",\\"params\\":{\\"category\\":\\"overall\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"text-model\\",\\"rating\\":80.0}]}"])</script>
"""
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
        self.assertEqual(result.top_models[0].score, 66.1)

    def test_extract_top_models_supports_arena_fields(self):
        agent = BenchmarkAgent()
        entries = [
            {"rank": "1", "modelDisplayName": "org/m1", "rating": "91.2"},
            {"rank": "2", "modelDisplayName": "org/m2", "rating": "90.0"},
        ]
        models = agent._extract_top_models_from_entries(entries, limit=5)
        self.assertEqual([m.model_id for m in models], ["org/m1", "org/m2"])
        self.assertAlmostEqual(models[0].score or 0.0, 91.2)

    def test_extract_arena_leaderboards_from_html(self):
        agent = BenchmarkAgent()
        html = """
<script>self.__next_f.push([1,"x:{\\"arenaSlug\\":\\"code\\",\\"leaderboardSlug\\":\\"webdev\\",\\"params\\":{\\"category\\":\\"webdev\\"},\\"entries\\":[{\\"rank\\":1,\\"modelDisplayName\\":\\"a\\",\\"rating\\":1.0}]}"])</script>
"""
        payloads = agent._extract_arena_leaderboards_from_html(html)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["arenaSlug"], "code")

    def test_expanded_search_terms_include_hint_tokens(self):
        agent = BenchmarkAgent()
        terms = agent._expanded_search_terms("best coding model")
        self.assertIn("swe", terms)
        self.assertIn("programming", terms)

    def test_leaderboard_page_url(self):
        agent = BenchmarkAgent()
        url = agent._leaderboard_page_url("code")
        self.assertEqual(url, "https://arena.ai/leaderboard/code")

    async def test_run_tries_next_candidate_when_first_has_no_entries(self):
        agent = FallbackLeaderboardAgent()
        result = await agent.run("math reasoning benchmark")
        self.assertEqual(result.selected_benchmark.dataset_id, "arena/text:overall")
        self.assertEqual(result.top_models[0].model_id, "org/r1")


if __name__ == "__main__":
    unittest.main()
