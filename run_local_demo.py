from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

from retail_a2a.agents import PartnerAgent, RetailerAgent
from retail_a2a.llm_client import LocalLLMClient
from retail_a2a.server import create_server


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body)


def main() -> None:
    partner_agent = PartnerAgent(company_name="External Supplier Co.", agent_id="partner-agent")
    retailer_agent = RetailerAgent(
        company_name="RetailCo",
        agent_id="retailer-agent",
        llm_client=LocalLLMClient.from_env(),
        partner_endpoints={"partner-agent": "http://127.0.0.1:9102"},
    )

    partner_server = create_server(partner_agent, "127.0.0.1", 9102)
    retailer_server = create_server(retailer_agent, "127.0.0.1", 9101)

    threads = [
        threading.Thread(target=partner_server.serve_forever, daemon=True),
        threading.Thread(target=retailer_server.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()

    time.sleep(0.3)

    scenario = {
        "sku": "SKU-1001",
        "store_id": "STORE-44",
        "product_name": "Bottled Water 1L",
        "current_stock": 20,
        "daily_demand": 55,
        "lead_time_days": 3,
    }

    result = _post_json("http://127.0.0.1:9101/simulate/out-of-stock", scenario)
    print(json.dumps(result, indent=2))

    retailer_server.shutdown()
    partner_server.shutdown()
    retailer_server.server_close()
    partner_server.server_close()


if __name__ == "__main__":
    main()
