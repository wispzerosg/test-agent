import unittest

from hf_benchmark_agent.agent import BenchmarkAgent


class _FakeText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolResult:
    def __init__(self, content) -> None:
        self.content = content


class FakeBenchmarkAgent(BenchmarkAgent):
    async def _search_benchmarks_via_mcp(self, request: str) -> list[str]:
        return ["SWE-bench/SWE-bench_Verified", "cais/hle"]

    async def _get_json(self, url: str, params=None):
        if url.endswith("/api/datasets/SWE-bench%2FSWE-bench_Verified"):
            return {
                "id": "SWE-bench/SWE-bench_Verified",
                "description": "Software engineering coding benchmark for LLMs.",
            }
        if url.endswith("/api/datasets/cais%2Fhle"):
            return {"id": "cais/hle", "description": "Reasoning benchmark."}
        if url.endswith("/api/datasets/SWE-bench%2FSWE-bench_Verified/leaderboard"):
            return [
                {"rank": 1, "model_id": "model/a", "value": 66.1, "verified": True},
                {"rank": 2, "model_id": "model/b", "value": 64.0, "verified": True},
                {"rank": 3, "model_id": "model/c", "value": 63.5, "verified": False},
                {"rank": 4, "model_id": "model/d", "value": 62.9, "verified": True},
                {"rank": 5, "model_id": "model/e", "value": 62.0, "verified": None},
                {"rank": 6, "model_id": "model/f", "value": 61.0, "verified": True},
            ]
        return {}


class FallbackLeaderboardAgent(BenchmarkAgent):
    async def _search_benchmarks_via_mcp(self, request: str) -> list[str]:
        return ["empty/benchmark", "cais/hle"]

    async def _get_json(self, url: str, params=None):
        if url.endswith("/api/datasets/empty%2Fbenchmark"):
            return {"id": "empty/benchmark", "description": "Empty leaderboard benchmark."}
        if url.endswith("/api/datasets/cais%2Fhle"):
            return {"id": "cais/hle", "description": "Reasoning benchmark with scores."}
        if url.endswith("/api/datasets/empty%2Fbenchmark/leaderboard"):
            return []
        if url.endswith("/api/datasets/cais%2Fhle/leaderboard"):
            return [
                {"rank": 1, "model_id": "org/r1", "value": 42.0, "verified": True},
                {"rank": 2, "model_id": "org/r2", "value": 39.5, "verified": False},
            ]
        return {}


class TestBenchmarkAgent(unittest.IsolatedAsyncioTestCase):
    async def test_run_selects_relevant_benchmark_and_returns_top_five(self):
        agent = FakeBenchmarkAgent(mcp_url="https://huggingface.co/mcp", hf_token="dummy")
        result = await agent.run("best coding model for software engineering")

        self.assertEqual(result.selected_benchmark.dataset_id, "SWE-bench/SWE-bench_Verified")
        self.assertEqual(len(result.top_models), 5)
        self.assertEqual(result.top_models[0].model_id, "model/a")
        self.assertEqual(result.top_models[0].score, 66.1)

    def test_extract_top_models_supports_alternate_keys(self):
        agent = BenchmarkAgent()
        payload = {
            "entries": [
                {"rank": "1", "modelId": "org/m1", "score": "91.2", "verified": True},
                {"rank": "2", "modelId": "org/m2", "score": "90.0", "verified": False},
            ]
        }
        models = agent._extract_top_models(payload, limit=5)
        self.assertEqual([m.model_id for m in models], ["org/m1", "org/m2"])
        self.assertAlmostEqual(models[0].score or 0.0, 91.2)

    def test_extract_dataset_ids_from_text_content(self):
        agent = BenchmarkAgent()
        fake_result = _FakeToolResult(
            content=[
                _FakeText("Try https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified"),
                _FakeText("and cais/hle"),
            ]
        )
        dataset_ids = agent._extract_dataset_ids_from_tool_result(fake_result)
        self.assertIn("SWE-bench/SWE-bench_Verified", dataset_ids)

    async def test_run_tries_next_candidate_when_first_has_no_leaderboard(self):
        agent = FallbackLeaderboardAgent()
        result = await agent.run("math reasoning benchmark")
        self.assertEqual(result.selected_benchmark.dataset_id, "cais/hle")
        self.assertEqual(result.top_models[0].model_id, "org/r1")


if __name__ == "__main__":
    unittest.main()
