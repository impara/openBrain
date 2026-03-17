from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel


class ContactSummary(BaseModel):
    id: int
    full_name: str
    company: Optional[str] = None
    tags: List[str] = []
    last_interaction_at: Optional[datetime] = None


class ContactDetail(BaseModel):
    id: int
    full_name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    tags: List[str] = []
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class Interaction(BaseModel):
    id: int
    contact_id: int
    summary: str
    channel: str
    direction: str
    source: Optional[str] = None
    occurred_at: datetime
    created_at: datetime
    raw_capture_id: Optional[int] = None


def row_to_contact_summary(row: dict[str, Any]) -> ContactSummary:
    return ContactSummary(
        id=row["id"],
        full_name=row["full_name"],
        company=row.get("company"),
        tags=row.get("tags") or [],
        last_interaction_at=row.get("last_interaction_at"),
    )


def row_to_contact_detail(row: dict[str, Any]) -> ContactDetail:
    return ContactDetail(
        id=row["id"],
        full_name=row["full_name"],
        first_name=row.get("first_name"),
        last_name=row.get("last_name"),
        email=row.get("email"),
        phone=row.get("phone"),
        company=row.get("company"),
        role=row.get("role"),
        location=row.get("location"),
        tags=row.get("tags") or [],
        notes=row.get("notes"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def row_to_interaction(row: dict[str, Any]) -> Interaction:
    return Interaction(
        id=row["id"],
        contact_id=row["contact_id"],
        summary=row["summary"],
        channel=row["channel"],
        direction=row["direction"],
        source=row.get("source"),
        occurred_at=row["occurred_at"],
        created_at=row["created_at"],
        raw_capture_id=row.get("raw_capture_id"),
    )

