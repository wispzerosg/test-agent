from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from typing import Any
from urllib.request import Request, urlopen

from .llm_client import LocalLLMClient
from .models import A2AMessage, StockOutEvent


class BaseA2AAgent(ABC):
    def __init__(
        self,
        *,
        company_name: str,
        agent_id: str,
        partner_endpoints: dict[str, str] | None = None,
    ) -> None:
        self.company_name = company_name
        self.agent_id = agent_id
        self.partner_endpoints = partner_endpoints or {}

    def send_message(
        self,
        *,
        receiver_id: str,
        message_type: str,
        payload: dict[str, Any],
        conversation_id: str | None = None,
        in_reply_to: str | None = None,
    ) -> A2AMessage:
        if receiver_id not in self.partner_endpoints:
            raise ValueError(f"Unknown receiver '{receiver_id}'. Configure partner endpoint first.")
        outbound = A2AMessage.create(
            sender=self.agent_id,
            receiver=receiver_id,
            message_type=message_type,
            payload=payload,
            conversation_id=conversation_id,
            in_reply_to=in_reply_to,
            metadata={"sender_company": self.company_name},
        )

        request = Request(
            url=f"{self.partner_endpoints[receiver_id].rstrip('/')}/a2a/message",
            data=json.dumps(outbound.to_dict()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=20) as response:
            response_body = response.read().decode("utf-8")
        parsed_response = json.loads(response_body)
        return A2AMessage.from_dict(parsed_response)

    @abstractmethod
    def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
        raise NotImplementedError


class PartnerAgent(BaseA2AAgent):
    def __init__(
        self,
        *,
        company_name: str,
        agent_id: str = "partner-agent",
        partner_endpoints: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            company_name=company_name,
            agent_id=agent_id,
            partner_endpoints=partner_endpoints,
        )

    def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
        if message.message_type != "stock_inquiry":
            return A2AMessage.create(
                sender=self.agent_id,
                receiver=message.sender,
                message_type="unsupported_message",
                payload={"detail": f"Unsupported message type '{message.message_type}'"},
                conversation_id=message.conversation_id,
                in_reply_to=message.message_id,
            )

        requested_qty = max(0, int(message.payload.get("requested_qty", 0)))
        partner_available_stock = max(0, int(message.payload.get("partner_available_stock", 220)))
        safety_buffer = max(10, int(partner_available_stock * 0.15))
        allocatable_qty = max(0, min(requested_qty, partner_available_stock - safety_buffer))

        eta_days = 1 if allocatable_qty > 0 else max(2, int(message.payload.get("lead_time_days", 3)))
        offer_payload = {
            "sku": message.payload.get("sku"),
            "requested_qty": requested_qty,
            "allocatable_qty": allocatable_qty,
            "eta_days": eta_days,
            "priority_level": "expedite" if allocatable_qty >= math.ceil(requested_qty * 0.7) else "standard",
            "substitute_sku": message.payload.get("substitute_sku", "SUB-ALT-001"),
            "partner_company": self.company_name,
        }
        return A2AMessage.create(
            sender=self.agent_id,
            receiver=message.sender,
            message_type="replenishment_offer",
            payload=offer_payload,
            conversation_id=message.conversation_id,
            in_reply_to=message.message_id,
        )


class RetailerAgent(BaseA2AAgent):
    def __init__(
        self,
        *,
        company_name: str,
        llm_client: LocalLLMClient,
        partner_receiver_id: str = "partner-agent",
        agent_id: str = "retailer-agent",
        partner_endpoints: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            company_name=company_name,
            agent_id=agent_id,
            partner_endpoints=partner_endpoints,
        )
        self.llm_client = llm_client
        self.partner_receiver_id = partner_receiver_id

    def handle_a2a_message(self, message: A2AMessage) -> A2AMessage:
        return A2AMessage.create(
            sender=self.agent_id,
            receiver=message.sender,
            message_type="ack",
            payload={"detail": f"Retailer received '{message.message_type}'"},
            conversation_id=message.conversation_id,
            in_reply_to=message.message_id,
        )

    def simulate_out_of_stock(self, event: StockOutEvent) -> dict[str, Any]:
        days_of_cover = round(event.current_stock / max(event.daily_demand, 1), 2)
        severity = "critical" if days_of_cover < 1 else ("high" if days_of_cover < 2 else "medium")
        requested_qty = max(0, event.daily_demand * event.lead_time_days - event.current_stock)
        requested_qty += math.ceil(event.daily_demand * 0.3)

        response_message = self.send_message(
            receiver_id=self.partner_receiver_id,
            message_type="stock_inquiry",
            payload={
                "sku": event.sku,
                "product_name": event.product_name,
                "store_id": event.store_id,
                "requested_qty": requested_qty,
                "lead_time_days": event.lead_time_days,
                "daily_demand": event.daily_demand,
            },
        )

        partner_offer = response_message.payload
        fill_rate = (
            round(partner_offer.get("allocatable_qty", 0) / requested_qty, 2) if requested_qty else 1.0
        )
        recommendation_prompt = self._build_llm_prompt(event, severity, requested_qty, partner_offer, fill_rate)
        llm_analysis = self.llm_client.chat(
            system_prompt=(
                "You are a retail supply-chain AI assistant. Provide concise and practical "
                "steps for managing stock-out risk with partner collaboration."
            ),
            user_prompt=recommendation_prompt,
        )

        return {
            "retailer_company": self.company_name,
            "event": event.to_dict(),
            "assessment": {
                "days_of_cover": days_of_cover,
                "severity": severity,
                "recommended_request_qty": requested_qty,
            },
            "partner_offer": partner_offer,
            "service_recovery_projection": {
                "projected_fill_rate": fill_rate,
                "needs_store_level_rationing": fill_rate < 0.85,
            },
            "llm_analysis": llm_analysis,
        }

    @staticmethod
    def _build_llm_prompt(
        event: StockOutEvent,
        severity: str,
        requested_qty: int,
        partner_offer: dict[str, Any],
        fill_rate: float,
    ) -> str:
        return (
            "Retail stock-out event and partner response:\n"
            f"- SKU: {event.sku} ({event.product_name})\n"
            f"- Store: {event.store_id}\n"
            f"- Current stock: {event.current_stock}\n"
            f"- Daily demand: {event.daily_demand}\n"
            f"- Lead time: {event.lead_time_days} days\n"
            f"- Severity: {severity}\n"
            f"- Requested qty to partner: {requested_qty}\n"
            f"- Partner allocatable qty: {partner_offer.get('allocatable_qty')}\n"
            f"- Partner ETA days: {partner_offer.get('eta_days')}\n"
            f"- Projected fill rate: {fill_rate}\n\n"
            "Return:\n"
            "1) Three immediate actions for retailer operations.\n"
            "2) Two recommendations for collaborative planning with the partner.\n"
            "3) One KPI to track over next 7 days."
        )
