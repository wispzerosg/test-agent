from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agents import BaseA2AAgent, PartnerAgent, RetailerAgent
from .llm_client import LocalLLMClient
from .models import A2AMessage, StockOutEvent


def create_server(agent: BaseA2AAgent, host: str, port: int) -> ThreadingHTTPServer:
    class AgentHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "agent_id": agent.agent_id,
                        "company_name": agent.company_name,
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                data = self._read_json()
                if self.path == "/a2a/message":
                    inbound = A2AMessage.from_dict(data)
                    outbound = agent.handle_a2a_message(inbound)
                    self._send_json(HTTPStatus.OK, outbound.to_dict())
                    return

                if self.path == "/simulate/out-of-stock":
                    if not isinstance(agent, RetailerAgent):
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Simulation endpoint is only available on retailer agents."},
                        )
                        return
                    event = StockOutEvent.from_dict(data)
                    result = agent.simulate_out_of_stock(event)
                    self._send_json(HTTPStatus.OK, result)
                    return

                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except Exception as exc:  # noqa: BLE001
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    return ThreadingHTTPServer((host, port), AgentHandler)


def run_agent_server(agent: BaseA2AAgent, host: str, port: int) -> None:
    server = create_server(agent, host, port)
    print(f"[{agent.agent_id}] Listening on http://{host}:{port}")
    server.serve_forever()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an A2A retail out-of-stock simulation agent server."
    )
    parser.add_argument("--role", choices=("retailer", "partner"), required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--company-name", required=True)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument(
        "--partner-url",
        default="http://localhost:8002",
        help="Partner base URL (used by retailer role).",
    )
    parser.add_argument("--local-llm-base-url", default=None)
    parser.add_argument("--local-llm-model", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.role == "partner":
        agent = PartnerAgent(
            company_name=args.company_name,
            agent_id=args.agent_id or "partner-agent",
        )
    else:
        llm_client = LocalLLMClient.from_env()
        if args.local_llm_base_url:
            llm_client.base_url = args.local_llm_base_url
        if args.local_llm_model:
            llm_client.model = args.local_llm_model
        agent = RetailerAgent(
            company_name=args.company_name,
            agent_id=args.agent_id or "retailer-agent",
            llm_client=llm_client,
            partner_endpoints={"partner-agent": args.partner_url},
        )

    run_agent_server(agent, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
