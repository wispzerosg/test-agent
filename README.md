# Arena Benchmark Agent

Lightweight agent that:
1. receives a plain-text request (for example: "best coding model"),
2. maps the request to the most relevant leaderboard on https://arena.ai/,
3. extracts leaderboard entries from Arena pages,
4. returns the top-5 models with scores.

## Quick start

```bash
python3 -m pip install -e .
python3 -m hf_benchmark_agent.cli "best coding model"
```

If your shell has `~/.local/bin` on `PATH`, the console script also works:

```bash
hf-benchmark-agent "best coding model"
```

Optional environment variables:

- `ARENA_BASE_URL` (default: `https://arena.ai`)

Example with custom base URL:

```bash
ARENA_BASE_URL=https://arena.ai python3 -m hf_benchmark_agent.cli "reasoning model"
```

## Output format

The CLI prints JSON with:

- `request`: original text query
- `selected_benchmark`: Arena leaderboard chosen as most relevant
- `top_models`: top 5 model entries (`rank`, `model_id`, `score`, `verified`)

## How relevance is computed

The agent scores candidate Arena leaderboard candidates using:

- token overlap between request and arena/category/leaderboard identifiers,
- overlap with Arena-specific keyword hints,
- weak sequence similarity against leaderboard context.

Highest relevance score wins.
