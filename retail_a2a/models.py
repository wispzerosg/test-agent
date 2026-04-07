from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass(slots=True)
class A2AMessage:
    message_id: str
    conversation_id: str
    sender: str
    receiver: str
    message_type: str
    payload: dict[str, Any]
    timestamp: str
    in_reply_to: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        sender: str,
        receiver: str,
        message_type: str,
        payload: dict[str, Any],
        conversation_id: str | None = None,
        in_reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "A2AMessage":
        return cls(
            message_id=str(uuid4()),
            conversation_id=conversation_id or str(uuid4()),
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            payload=payload,
            timestamp=_utc_now_iso(),
            in_reply_to=in_reply_to,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "A2AMessage":
        required_fields = (
            "message_id",
            "conversation_id",
            "sender",
            "receiver",
            "message_type",
            "payload",
            "timestamp",
        )
        missing = [field_name for field_name in required_fields if field_name not in data]
        if missing:
            raise ValueError(f"Invalid A2A message. Missing fields: {', '.join(missing)}")
        return cls(
            message_id=str(data["message_id"]),
            conversation_id=str(data["conversation_id"]),
            sender=str(data["sender"]),
            receiver=str(data["receiver"]),
            message_type=str(data["message_type"]),
            payload=dict(data.get("payload", {})),
            timestamp=str(data["timestamp"]),
            in_reply_to=str(data["in_reply_to"]) if data.get("in_reply_to") else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class StockOutEvent:
    sku: str
    store_id: str
    product_name: str
    current_stock: int
    daily_demand: int
    lead_time_days: int
    target_service_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StockOutEvent":
        required_fields = (
            "sku",
            "store_id",
            "product_name",
            "current_stock",
            "daily_demand",
            "lead_time_days",
        )
        missing = [field_name for field_name in required_fields if field_name not in data]
        if missing:
            raise ValueError(f"Invalid stock out event. Missing fields: {', '.join(missing)}")
        return cls(
            sku=str(data["sku"]),
            store_id=str(data["store_id"]),
            product_name=str(data["product_name"]),
            current_stock=max(0, int(data["current_stock"])),
            daily_demand=max(1, int(data["daily_demand"])),
            lead_time_days=max(1, int(data["lead_time_days"])),
            target_service_level=float(data.get("target_service_level", 0.95)),
        )
