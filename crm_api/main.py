from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query

from .db import ensure_crm_schema, fetchall, fetchone
from .models import (
    ContactDetail,
    ContactSummary,
    Interaction,
    row_to_contact_detail,
    row_to_contact_summary,
    row_to_interaction,
)


app = FastAPI(title="OpenBrain Personal CRM API")


@app.on_event("startup")
def _startup():
    # Ensure schema exists even if OpenBrain MCP wasn't rebuilt/restarted yet.
    ensure_crm_schema()


def get_query_limit(limit: int = Query(50, ge=1, le=200)) -> int:
    return limit


@app.get("/contacts", response_model=List[ContactSummary])
def list_contacts(q: Optional[str] = Query(None), limit: int = Depends(get_query_limit)):
    """
    List contacts with optional text search over name/company and
    include the timestamp of the most recent interaction.
    """
    params: list = []
    where_clauses: list[str] = []
    if q:
        like = f"%{q}%"
        where_clauses.append(
            "(c.full_name ILIKE %s OR c.company ILIKE %s OR c.email ILIKE %s)"
        )
        params.extend([like, like, like])
    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT
            c.id,
            c.full_name,
            c.company,
            c.tags,
            MAX(i.occurred_at) AS last_interaction_at
        FROM crm.contacts c
        LEFT JOIN crm.interactions i ON i.contact_id = c.id
        {where_sql}
        GROUP BY c.id, c.full_name, c.company, c.tags
        ORDER BY COALESCE(MAX(i.occurred_at), c.created_at) DESC
        LIMIT %s;
    """
    params.append(limit)
    rows = fetchall(query, tuple(params))
    return [row_to_contact_summary(row) for row in rows]


@app.get("/contacts/{contact_id}", response_model=ContactDetail)
def get_contact(contact_id: int):
    row = fetchone(
        """
        SELECT
            id,
            full_name,
            first_name,
            last_name,
            email,
            phone,
            company,
            role,
            location,
            tags,
            notes,
            created_at,
            updated_at
        FROM crm.contacts
        WHERE id = %s;
        """,
        (contact_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Contact not found")
    return row_to_contact_detail(row)


@app.get("/contacts/{contact_id}/interactions", response_model=List[Interaction])
def list_interactions(contact_id: int, limit: int = Depends(get_query_limit)):
    rows = fetchall(
        """
        SELECT
            id,
            contact_id,
            raw_capture_id,
            channel,
            direction,
            summary,
            source,
            occurred_at,
            created_at
        FROM crm.interactions
        WHERE contact_id = %s
        ORDER BY occurred_at DESC
        LIMIT %s;
        """,
        (contact_id, limit),
    )
    return [row_to_interaction(row) for row in rows]


@app.get("/health")
def health():
    return {"status": "ok"}

