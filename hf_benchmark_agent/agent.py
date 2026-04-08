from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx


NEXT_FLIGHT_CHUNK_RE = re.compile(
    r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>',
    flags=re.S,
)

ARENA_PAGE_SLUGS = [
    "overview",
    "text",
    "code",
    "vision",
    "document",
    "search",
    "text-to-image",
    "image-edit",
    "text-to-video",
    "image-to-video",
    "video-edit",
]

ARENA_HINTS: dict[str, set[str]] = {
    "code": {"code", "coding", "programming", "software", "swe", "webdev", "react", "html"},
    "text": {"chat", "assistant", "reasoning", "math", "multilingual", "translation"},
    "vision": {"vision", "image", "vqa", "ocr", "caption", "multimodal"},
    "document": {"document", "pdf", "file", "paper", "doc"},
    "search": {"search", "browse", "web", "retrieval"},
    "text-to-image": {"text-to-image", "t2i", "image generation"},
    "image-edit": {"image edit", "inpainting", "edit image"},
    "text-to-video": {"text-to-video", "video generation", "t2v"},
    "image-to-video": {"image-to-video", "i2v"},
    "video-edit": {"video edit", "video-edit"},
}


@dataclass(frozen=True)
class ModelScore:
    rank: int | None
    model_id: str
    score: float | None
    verified: bool | None
    input_cost_per_million: float | None
    output_cost_per_million: float | None
    price_per_image: float | None
    price_per_second: float | None


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


@dataclass(frozen=True)
class ArenaLeaderboardCandidate:
    page_slug: str
    arena_slug: str
    leaderboard_slug: str
    top_models: list[ModelScore]
    relevance_score: float


class BenchmarkAgent:
    """Find the most relevant Arena leaderboard and return top models."""

    def __init__(
        self,
        arena_base_url: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.arena_base_url = (arena_base_url or os.getenv("ARENA_BASE_URL") or "https://arena.ai").strip().rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def run(self, request: str) -> BenchmarkAgentResult:
        request = request.strip()
        if not request:
            raise ValueError("Request text must not be empty.")

        ranked = await self._rank_candidates(request)
        if not ranked:
            raise RuntimeError("No Arena leaderboard candidates found for this request.")

        selected: ArenaLeaderboardCandidate | None = None
        top_models: list[ModelScore] = []
        for candidate in ranked:
            if candidate.top_models:
                selected = candidate
                top_models = candidate.top_models
                break
        if selected is None:
            raise RuntimeError("No Arena leaderboard candidate returned model scores.")

        return BenchmarkAgentResult(
            request=request,
            selected_benchmark=SelectedBenchmark(
                dataset_id=f"arena/{selected.arena_slug}:{selected.leaderboard_slug}",
                url=self._leaderboard_page_url(selected.page_slug),
                relevance_score=selected.relevance_score,
            ),
            top_models=top_models,
        )

    async def _rank_candidates(self, request: str) -> list[ArenaLeaderboardCandidate]:
        request_tokens = self._tokenize(request)
        ranked: list[ArenaLeaderboardCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        for page_slug in self._prioritized_pages(request_tokens):
            html = await self._safe_get_text(self._leaderboard_page_url(page_slug))
            if not html:
                continue

            for payload in self._extract_arena_leaderboards_from_html(html):
                if not payload.arena_slug or not payload.leaderboard_slug:
                    continue

                dedupe_key = (page_slug, payload.arena_slug, payload.leaderboard_slug)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                context = " ".join(
                    part
                    for part in [page_slug, payload.arena_slug, payload.leaderboard_slug, payload.category]
                    if part
                )
                context_tokens = self._tokenize(context)
                overlap = len(request_tokens & context_tokens)
                hint_score = self._hint_alignment_score(request_tokens, context_tokens)
                ratio = SequenceMatcher(None, request.lower(), context.lower()).ratio()
                relevance = overlap * 2.0 + hint_score + ratio
                ranked.append(
                    ArenaLeaderboardCandidate(
                        page_slug=page_slug,
                        arena_slug=payload.arena_slug,
                        leaderboard_slug=payload.leaderboard_slug,
                        top_models=self._extract_top_models_from_entries(payload.entries, limit=5),
                        relevance_score=relevance,
                    )
                )

        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked

    def _hint_alignment_score(
        self, request_tokens: set[str], context_tokens: set[str]
    ) -> float:
        score = 0.0
        for page_slug, hint_tokens in ARENA_HINTS.items():
            request_hit = request_tokens & hint_tokens
            context_hit = context_tokens & (hint_tokens | self._tokenize(page_slug))
            if request_hit and context_hit:
                score += 1.4 + (0.2 * len(request_hit & context_hit))
            elif request_hit and not context_hit:
                score -= 0.75
        return score

    def _extract_top_models_from_entries(self, entries: list[Any], limit: int) -> list[ModelScore]:
        models: list[ModelScore] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_id = (
                entry.get("modelDisplayName")
                or entry.get("model_name")
                or entry.get("model_id")
                or entry.get("modelId")
                or entry.get("model")
                or entry.get("name")
            )
            if not isinstance(model_id, str):
                continue

            rank = self._to_int(entry.get("rank"))
            score = self._to_float(entry.get("rating"))
            if score is None:
                score = self._to_float(entry.get("score"))
            if score is None:
                score = self._to_float(entry.get("value"))
            verified = entry.get("verified") if isinstance(entry.get("verified"), bool) else None
            input_cost_per_million = self._to_float(entry.get("inputPricePerMillion"))
            output_cost_per_million = self._to_float(entry.get("outputPricePerMillion"))
            price_per_image = self._to_float(entry.get("pricePerImage"))
            price_per_second = self._to_float(entry.get("pricePerSecond"))
            models.append(
                ModelScore(
                    rank=rank,
                    model_id=model_id,
                    score=score,
                    verified=verified,
                    input_cost_per_million=input_cost_per_million,
                    output_cost_per_million=output_cost_per_million,
                    price_per_image=price_per_image,
                    price_per_second=price_per_second,
                )
            )

        models.sort(
            key=lambda item: (
                item.rank is None,
                item.rank if item.rank is not None else 10**9,
                -item.score if item.score is not None else 0.0,
            )
        )
        top_100_scores = [m.score for m in models[:100] if m.score is not None]
        if not top_100_scores:
            return models[:limit]

        top_100_mean = sum(top_100_scores) / len(top_100_scores)
        if top_100_mean == 0:
            return models[:limit]

        normalized: list[ModelScore] = []
        for model in models[:limit]:
            normalized_score = model.score / top_100_mean if model.score is not None else None
            normalized.append(
                ModelScore(
                    rank=model.rank,
                    model_id=model.model_id,
                    score=normalized_score,
                    verified=model.verified,
                    input_cost_per_million=model.input_cost_per_million,
                    output_cost_per_million=model.output_cost_per_million,
                    price_per_image=model.price_per_image,
                    price_per_second=model.price_per_second,
                )
            )
        return normalized

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

    def _prioritized_pages(self, request_tokens: set[str]) -> list[str]:
        scored: list[tuple[str, float]] = []
        for slug in ARENA_PAGE_SLUGS:
            slug_tokens = self._tokenize(slug)
            hint_tokens = ARENA_HINTS.get(slug, set())
            overlap = len(request_tokens & (slug_tokens | hint_tokens))
            score = overlap + (0.25 * SequenceMatcher(None, " ".join(sorted(request_tokens)), slug).ratio())
            scored.append((slug, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        ordered = [slug for slug, _ in scored]
        # Keep traversal bounded while still allowing fallback exploration.
        return ordered[:4] + [slug for slug in ordered[4:] if slug not in ordered[:4]]

    async def _safe_get_text(self, url: str) -> str:
        try:
            return await self._get_text(url)
        except Exception:
            return ""

    async def _get_text(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    def _leaderboard_page_url(self, page_slug: str) -> str:
        return f"{self.arena_base_url}/leaderboard/{page_slug}"

    def _extract_arena_leaderboards_from_html(self, html: str) -> list[ArenaLeaderboardPayload]:
        parts: list[str] = []
        for raw in NEXT_FLIGHT_CHUNK_RE.findall(html):
            try:
                parts.append(raw.encode("utf-8").decode("unicode_escape"))
            except UnicodeDecodeError:
                parts.append(raw)

        blob = "\n".join(parts)
        out: list[ArenaLeaderboardPayload] = []
        seen: set[tuple[str, str, str]] = set()
        index = 0
        while True:
            start = blob.find('{"arenaSlug"', index)
            if start == -1:
                break
            object_text, end = self._extract_json_object(blob, start)
            index = max(start + 1, end)
            if not object_text:
                continue

            try:
                parsed = json.loads(object_text)
            except json.JSONDecodeError:
                continue

            if not isinstance(parsed, dict):
                continue
            entries = parsed.get("entries")
            if not isinstance(entries, list):
                continue
            arena_slug = parsed.get("arenaSlug")
            leaderboard_slug = parsed.get("leaderboardSlug")
            params = parsed.get("params") if isinstance(parsed.get("params"), dict) else {}
            category = params.get("category")
            if not isinstance(arena_slug, str) or not isinstance(leaderboard_slug, str):
                continue
            key = (arena_slug, leaderboard_slug, str(category or ""))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                ArenaLeaderboardPayload(
                    arena_slug=arena_slug,
                    leaderboard_slug=leaderboard_slug,
                    category=str(category or ""),
                    entries=entries,
                )
            )

        return out

    def _extract_json_object(self, text: str, start: int) -> tuple[str, int]:
        if start < 0 or start >= len(text) or text[start] != "{":
            return "", start

        in_string = False
        escaped = False
        depth = 0
        for idx in range(start, len(text)):
            char = text[idx]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1], idx + 1
        return "", start + 1


def run_agent_sync(request: str, arena_base_url: str | None = None) -> BenchmarkAgentResult:
    return asyncio.run(BenchmarkAgent(arena_base_url=arena_base_url).run(request))


@dataclass(frozen=True)
class ArenaLeaderboardPayload:
    arena_slug: str
    leaderboard_slug: str
    category: str
    entries: list[dict[str, Any]]
