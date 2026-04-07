from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any
from urllib.parse import quote

import httpx


DATASET_ID_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
DATASET_URL_RE = re.compile(
    r"https?://huggingface\.co/datasets/([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)"
)

REQUEST_HINTS: dict[str, set[str]] = {
    "coding": {"code", "coding", "programming", "software", "swe", "bug", "debug"},
    "math": {"math", "mathematics", "reasoning", "aime", "gsm", "algebra", "proof"},
    "chat": {"chat", "assistant", "instruction", "instruct", "dialog", "conversation"},
    "multilingual": {"multilingual", "translation", "language", "cross-lingual"},
    "vision": {"vision", "image", "vqa", "ocr", "caption", "multimodal"},
    "safety": {"safety", "toxic", "harm", "jailbreak", "alignment", "robustness"},
}


@dataclass(frozen=True)
class ModelScore:
    rank: int | None
    model_id: str
    score: float | None
    verified: bool | None


@dataclass(frozen=True)
class SelectedBenchmark:
    dataset_id: str
    url: str
    relevance_score: float


@dataclass(frozen=True)
class BenchmarkAgentResult:
    request: str
    selected_benchmark: SelectedBenchmark
    top_models: list[ModelScore]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BenchmarkAgent:
    """Find benchmark most relevant to a user request and return top models."""

    def __init__(
        self,
        mcp_url: str | None = None,
        hf_token: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.mcp_url = (mcp_url or os.getenv("HF_MCP_URL") or "https://huggingface.co/mcp").strip()
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.timeout_seconds = timeout_seconds

    async def run(self, request: str) -> BenchmarkAgentResult:
        request = request.strip()
        if not request:
            raise ValueError("Request text must not be empty.")

        candidate_dataset_ids = await self._search_benchmarks_via_mcp(request)
        # Always augment MCP candidates with direct Hub API discovery.
        api_candidates = await self._fallback_benchmark_ids(request)
        candidate_dataset_ids.extend(ds for ds in api_candidates if ds not in candidate_dataset_ids)
        if not candidate_dataset_ids:
            raise RuntimeError("No benchmark datasets found.")

        ranked = await self._rank_candidates(request, candidate_dataset_ids)
        if not ranked:
            raise RuntimeError("Could not rank any benchmark candidates.")

        selected_dataset_id: str | None = None
        relevance_score = 0.0
        top_models: list[ModelScore] = []
        for candidate_dataset_id, candidate_score in ranked:
            leaderboard_url = self._dataset_leaderboard_api_url(candidate_dataset_id)
            leaderboard_payload = await self._safe_get_json(leaderboard_url)
            candidate_top_models = self._extract_top_models(leaderboard_payload, limit=5)
            if candidate_top_models:
                selected_dataset_id = candidate_dataset_id
                relevance_score = candidate_score
                top_models = candidate_top_models
                break

        if selected_dataset_id is None:
            raise RuntimeError("No candidate benchmark returned leaderboard scores.")

        return BenchmarkAgentResult(
            request=request,
            selected_benchmark=SelectedBenchmark(
                dataset_id=selected_dataset_id,
                url=f"https://huggingface.co/datasets/{selected_dataset_id}",
                relevance_score=relevance_score,
            ),
            top_models=top_models,
        )

    async def _search_benchmarks_via_mcp(self, request: str) -> list[str]:
        search_args = {
            "query": request,
            "repo_types": ["dataset"],
            "filters": ["benchmark:official"],
            "limit": 40,
        }
        fallback_args = {
            "query": "",
            "repo_types": ["dataset"],
            "filters": ["benchmark:official"],
            "sort": "trendingScore",
            "limit": 40,
        }

        for tool_name in ("hub_repo_search", "Huggingface-skills-hub_repo_search"):
            try:
                primary = await self._call_mcp_tool(tool_name, search_args)
                dataset_ids = self._extract_dataset_ids_from_tool_result(primary)
                if len(dataset_ids) < 5:
                    broad = await self._call_mcp_tool(tool_name, fallback_args)
                    dataset_ids.extend(
                        ds for ds in self._extract_dataset_ids_from_tool_result(broad) if ds not in dataset_ids
                    )
                if dataset_ids:
                    return dataset_ids
            except Exception:
                continue

        return []

    async def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamable_http_client as streamablehttp_client

        headers: dict[str, str] | None = None
        if self.hf_token:
            headers = {"Authorization": f"Bearer {self.hf_token}"}

        async with streamablehttp_client(url=self.mcp_url, headers=headers) as transport:
            if len(transport) == 3:
                read_stream, write_stream, _get_session_id = transport
            else:
                read_stream, write_stream = transport

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    def _extract_dataset_ids_from_tool_result(self, tool_result: Any) -> list[str]:
        values: list[str] = []

        structured = getattr(tool_result, "structuredContent", None)
        if structured is None:
            structured = getattr(tool_result, "structured_content", None)

        if structured is not None:
            self._collect_dataset_ids(structured, values)

        content = getattr(tool_result, "content", None)
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    self._collect_dataset_ids(text, values)
                else:
                    self._collect_dataset_ids(item, values)

        self._collect_dataset_ids(tool_result, values)

        unique: list[str] = []
        for dataset_id in values:
            if dataset_id not in unique:
                unique.append(dataset_id)
        return unique

    def _collect_dataset_ids(self, value: Any, out: list[str]) -> None:
        if value is None:
            return

        if isinstance(value, str):
            for match in DATASET_URL_RE.finditer(value):
                out.append(match.group(1))

            if DATASET_ID_RE.match(value):
                out.append(value)

            # Parse JSON strings when a tool returns compact JSON in text blocks.
            if value.lstrip().startswith("{") or value.lstrip().startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    self._collect_dataset_ids(parsed, out)
            return

        if isinstance(value, list):
            for item in value:
                self._collect_dataset_ids(item, out)
            return

        if isinstance(value, dict):
            for key, item in value.items():
                lower_key = key.lower()
                if lower_key in {"id", "dataset_id", "datasetid", "repo_id", "repoid"}:
                    if isinstance(item, str) and DATASET_ID_RE.match(item):
                        out.append(item)
                self._collect_dataset_ids(item, out)
            return

        for attr in ("id", "dataset_id", "repo_id"):
            if hasattr(value, attr):
                item = getattr(value, attr)
                if isinstance(item, str) and DATASET_ID_RE.match(item):
                    out.append(item)

        # Some MCP SDK objects expose dict-like payload through model_dump.
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except Exception:
                dumped = None
            if dumped is not None:
                self._collect_dataset_ids(dumped, out)

    async def _fallback_benchmark_ids(self, request: str) -> list[str]:
        base_url = "https://huggingface.co/api/datasets"

        ids: list[str] = []
        for term in self._expanded_search_terms(request):
            searched = await self._safe_get_json(
                base_url, params={"filter": "benchmark:official", "search": term, "limit": 60}
            )
            for dataset_id in self._extract_dataset_ids_from_api_list(searched):
                if dataset_id not in ids:
                    ids.append(dataset_id)
        if ids:
            return ids

        broad = await self._get_json(base_url, params={"filter": "benchmark:official", "limit": 60})
        return self._extract_dataset_ids_from_api_list(broad)

    def _expanded_search_terms(self, request: str) -> list[str]:
        request_tokens = self._tokenize(request)
        terms: list[str] = [request]
        for hint_tokens in REQUEST_HINTS.values():
            if request_tokens & hint_tokens:
                for token in sorted(hint_tokens):
                    if token not in terms:
                        terms.append(token)
        return terms

    def _extract_dataset_ids_from_api_list(self, payload: Any) -> list[str]:
        if not isinstance(payload, list):
            return []
        ids: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                dataset_id = item.get("id")
                if isinstance(dataset_id, str) and DATASET_ID_RE.match(dataset_id):
                    ids.append(dataset_id)
        return ids

    async def _rank_candidates(
        self, request: str, dataset_ids: list[str]
    ) -> list[tuple[str, float]]:
        request_tokens = self._tokenize(request)
        ranked: list[tuple[str, float]] = []
        unique_dataset_ids = []
        for dataset_id in dataset_ids:
            if dataset_id not in unique_dataset_ids:
                unique_dataset_ids.append(dataset_id)

        # Limit remote metadata fetches while keeping a decent candidate set.
        for dataset_id in unique_dataset_ids[:25]:
            metadata = await self._safe_get_json(self._dataset_api_url(dataset_id))
            benchmark_text = self._build_benchmark_text(dataset_id, metadata)
            benchmark_tokens = self._tokenize(benchmark_text)

            overlap = len(request_tokens & benchmark_tokens)
            hint_score = self._hint_alignment_score(request_tokens, benchmark_tokens)
            ratio = SequenceMatcher(None, request.lower(), benchmark_text.lower()).ratio()
            relevance = overlap * 2.0 + hint_score + ratio
            ranked.append((dataset_id, relevance))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _build_benchmark_text(self, dataset_id: str, metadata: Any) -> str:
        parts = [dataset_id]
        if isinstance(metadata, dict):
            for key in ("id", "description", "pretty_name", "cardData"):
                value = metadata.get(key)
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, dict):
                    for nested_key in ("pretty_name", "description", "task_categories", "tags"):
                        nested_val = value.get(nested_key)
                        if isinstance(nested_val, str):
                            parts.append(nested_val)
                        elif isinstance(nested_val, list):
                            parts.extend(str(item) for item in nested_val)
                elif isinstance(value, list):
                    parts.extend(str(item) for item in value)
        return " ".join(parts)

    def _hint_alignment_score(
        self, request_tokens: set[str], benchmark_tokens: set[str]
    ) -> float:
        score = 0.0
        for hint_tokens in REQUEST_HINTS.values():
            request_hit = request_tokens & hint_tokens
            benchmark_hit = benchmark_tokens & hint_tokens
            if request_hit and benchmark_hit:
                score += 1.25 + (0.15 * len(request_hit & benchmark_hit))
            elif request_hit and not benchmark_hit:
                score -= 0.75
        return score

    def _extract_top_models(self, payload: Any, limit: int) -> list[ModelScore]:
        entries: list[Any]
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            entries = (
                payload.get("leaderboard")
                or payload.get("entries")
                or payload.get("results")
                or payload.get("rows")
                or []
            )
            if not isinstance(entries, list):
                entries = []
        else:
            entries = []

        models: list[ModelScore] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = (
                entry.get("model_id")
                or entry.get("modelId")
                or entry.get("model")
                or entry.get("model_name")
            )
            if not isinstance(model_id, str):
                continue

            rank = self._to_int(entry.get("rank"))
            score = self._to_float(entry.get("value"))
            if score is None:
                score = self._to_float(entry.get("score"))
            verified = entry.get("verified") if isinstance(entry.get("verified"), bool) else None
            models.append(ModelScore(rank=rank, model_id=model_id, score=score, verified=verified))

        models.sort(
            key=lambda item: (
                item.rank is None,
                item.rank if item.rank is not None else 10**9,
                -item.score if item.score is not None else 0.0,
            )
        )
        return models[:limit]

    def _to_int(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _to_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    async def _safe_get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        try:
            return await self._get_json(url, params=params)
        except Exception:
            return {}

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def _dataset_api_url(self, dataset_id: str) -> str:
        # Hugging Face dataset API expects namespace/repo with an unescaped slash.
        normalized = quote(dataset_id, safe="/._-")
        return f"https://huggingface.co/api/datasets/{normalized}"

    def _dataset_leaderboard_api_url(self, dataset_id: str) -> str:
        normalized = quote(dataset_id, safe="/._-")
        return f"https://huggingface.co/api/datasets/{normalized}/leaderboard"


def run_agent_sync(
    request: str, mcp_url: str | None = None, hf_token: str | None = None
) -> BenchmarkAgentResult:
    return asyncio.run(BenchmarkAgent(mcp_url=mcp_url, hf_token=hf_token).run(request))
