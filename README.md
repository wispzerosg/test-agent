# HF Benchmark MCP Agent

Lightweight agent that:
1. receives a plain-text request (for example: "best coding model"),
2. uses the Hugging Face MCP server to discover official benchmark datasets,
3. picks the most relevant benchmark,
4. returns the top-5 models with leaderboard scores.

## Quick start

```bash
python3 -m hf_benchmark_agent.cli "best coding model"
```

Optional environment variables:

- `HF_MCP_URL` (default: `https://huggingface.co/mcp`)
- `HF_TOKEN` (recommended for authenticated access)

Example with token:

```bash
HF_TOKEN=hf_xxx python3 -m hf_benchmark_agent.cli "reasoning benchmark"
```

## Output format

The CLI prints JSON with:

- `request`: original text query
- `selected_benchmark`: benchmark dataset chosen as most relevant
- `top_models`: top 5 model entries (`rank`, `model_id`, `score`, `verified`)

## How relevance is computed

The agent scores candidate benchmark datasets using:

- token overlap between request and benchmark metadata,
- overlap with benchmark-specific keyword hints,
- weak sequence similarity against dataset ID/title.

Highest relevance score wins.
