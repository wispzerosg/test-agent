# Retail A2A Out-of-Stock Simulation Agent

This project provides a lightweight **agent-to-agent (A2A)** reference implementation for testing
cross-company communication between a retail company and an external partner.

It is designed to:

- simulate a retail out-of-stock incident,
- exchange structured A2A messages with an external partner agent,
- and optionally use a **locally deployed LLM** (OpenAI-compatible API) for recovery recommendations.

## Architecture

Two HTTP agents communicate over a simple JSON A2A protocol:

1. **Retailer Agent**
   - Receives stock-out simulation requests (`/simulate/out-of-stock`)
   - Sends `stock_inquiry` A2A messages to partner
   - Receives `replenishment_offer`
   - Produces fill-rate projection + LLM-powered action guidance

2. **Partner Agent**
   - Receives retailer `stock_inquiry` messages (`/a2a/message`)
   - Returns `replenishment_offer` with allocatable quantity and ETA

## Project Structure

```
retail_a2a/
  __init__.py
  agents.py        # retailer/partner business logic and A2A messaging
  llm_client.py    # local OpenAI-compatible LLM connector
  models.py        # A2A message and stock-out data models
  server.py        # deployable HTTP server and CLI
run_local_demo.py  # end-to-end local simulation (starts both agents)
```

## Run Locally

### 1) Start partner agent

```bash
python3 -m retail_a2a.server \
  --role partner \
  --company-name "External Supplier Co." \
  --port 8002
```

### 2) Start retailer agent

```bash
python3 -m retail_a2a.server \
  --role retailer \
  --company-name "RetailCo" \
  --port 8001 \
  --partner-url "http://localhost:8002"
```

### 3) Send a stock-out scenario to retailer

```bash
curl -sS -X POST "http://localhost:8001/simulate/out-of-stock" \
  -H "Content-Type: application/json" \
  -d '{
    "sku": "SKU-1001",
    "store_id": "STORE-44",
    "product_name": "Bottled Water 1L",
    "current_stock": 20,
    "daily_demand": 55,
    "lead_time_days": 3
  }' | python3 -m json.tool
```

### 4) One-command local demo

```bash
python3 run_local_demo.py
```

## Connect to a Local LLM

The retailer agent expects an OpenAI-compatible endpoint:

- default base URL: `http://localhost:11434/v1`
- endpoint used: `/chat/completions`

Set environment variables before starting the retailer:

```bash
export LOCAL_LLM_BASE_URL="http://localhost:11434/v1"
export LOCAL_LLM_MODEL="llama3.1"
export LOCAL_LLM_TIMEOUT_SECONDS="20"
# export LOCAL_LLM_API_KEY="..."  # optional
```

You can also override via CLI:

```bash
python3 -m retail_a2a.server \
  --role retailer \
  --company-name "RetailCo" \
  --port 8001 \
  --partner-url "http://localhost:8002" \
  --local-llm-base-url "http://localhost:11434/v1" \
  --local-llm-model "llama3.1"
```

If no local model is reachable, the system returns deterministic fallback guidance so A2A integration tests can still run.
