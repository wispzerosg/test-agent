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
  - `score` is normalized as: `model_score / mean(top_100_scores)`
  - includes cost fields when available:
    - `input_cost_per_million`
    - `output_cost_per_million`
    - `price_per_image`
    - `price_per_second`

After JSON output, the CLI prints a terminal cost summary table for the top-5 models.

## Telegram bot integration

Run the bot (long polling) and send text prompts to it:

```bash
python3 -m pip install -e .
hf-benchmark-telegram-bot --token "<YOUR_TELEGRAM_BOT_TOKEN>"
```

Or with environment variable:

```bash
TELEGRAM_BOT_TOKEN="<YOUR_TELEGRAM_BOT_TOKEN>" hf-benchmark-telegram-bot
```

Optional bot settings:

- `--arena-base-url` (default: `https://arena.ai`)
- `--poll-timeout` (default: `30`)
- `--debug` (prints polling and send/receive diagnostics)

For each text message, the bot replies with:

1. selected Arena leaderboard
2. top-5 normalized model scores
3. top-5 cost summary table

Quick connectivity check:

```bash
python3 -m hf_benchmark_agent.telegram_bot \
  --token "<YOUR_TELEGRAM_BOT_TOKEN>" \
  --once --debug
```

## How relevance is computed

The agent scores candidate Arena leaderboard candidates using:

- token overlap between request and arena/category/leaderboard identifiers,
- overlap with Arena-specific keyword hints,
- weak sequence similarity against leaderboard context.

Highest relevance score wins.
