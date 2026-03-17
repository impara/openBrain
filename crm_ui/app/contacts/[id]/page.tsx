// @ts-nocheck
import React from "react";
import Link from "next/link";

type ContactDetail = {
  id: number;
  full_name: string;
  first_name?: string | null;
  last_name?: string | null;
  email?: string | null;
  phone?: string | null;
  company?: string | null;
  role?: string | null;
  location?: string | null;
  tags: string[];
  notes?: string | null;
  created_at: string;
  updated_at: string;
};

type Interaction = {
  id: number;
  contact_id: number;
  summary: string;
  channel: string;
  direction: string;
  source?: string | null;
  occurred_at: string;
  created_at: string;
  raw_capture_id?: number | null;
};

const API_URL = process.env.API_URL ?? "http://localhost:8002";

async function fetchContact(id: string): Promise<ContactDetail> {
  const res = await fetch(`${API_URL}/contacts/${id}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error("Failed to load contact");
  }
  return res.json();
}

async function fetchInteractions(id: string): Promise<Interaction[]> {
  const res = await fetch(`${API_URL}/contacts/${id}/interactions`, { cache: "no-store" });
  if (!res.ok) {
    return [];
  }
  return res.json();
}

export default async function ContactDetailPage({ params }: { params: { id: string } }) {
  const [contact, interactions] = await Promise.all([
    fetchContact(params.id),
    fetchInteractions(params.id),
  ]);

  return (
    <main>
      <div style={{ marginBottom: "1rem" }}>
        <Link href="/" style={{ color: "#9ca3af", fontSize: "0.875rem" }}>
          ← Back to contacts
        </Link>
      </div>
      <section
        style={{
          marginBottom: "1.5rem",
          padding: "1rem 1.25rem",
          borderRadius: "0.5rem",
          border: "1px solid #1f2937",
          backgroundColor: "#020617",
        }}
      >
        <h2 style={{ fontSize: "1.25rem", fontWeight: 600 }}>{contact.full_name}</h2>
        <div style={{ marginTop: "0.5rem", color: "#9ca3af", fontSize: "0.9rem" }}>
          {contact.company && <div>{contact.company}</div>}
          {contact.role && <div>{contact.role}</div>}
          {contact.location && <div>{contact.location}</div>}
          {contact.email && <div>Email: {contact.email}</div>}
          {contact.phone && <div>Phone: {contact.phone}</div>}
        </div>
        {contact.tags?.length > 0 && (
          <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.25rem", flexWrap: "wrap" }}>
            {contact.tags.map((tag) => (
              <span
                key={tag}
                style={{
                  borderRadius: "999px",
                  padding: "0.1rem 0.5rem",
                  fontSize: "0.75rem",
                  backgroundColor: "#111827",
                  color: "#e5e7eb",
                  border: "1px solid #1f2937",
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        {contact.notes && (
          <p style={{ marginTop: "0.75rem", fontSize: "0.9rem", lineHeight: 1.5 }}>{contact.notes}</p>
        )}
      </section>

      <section>
        <h3 style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.75rem" }}>Interactions</h3>
        {interactions.length === 0 ? (
          <p style={{ color: "#9ca3af", fontSize: "0.9rem" }}>No interactions logged yet.</p>
        ) : (
          <ul
            style={{
              listStyle: "none",
              padding: 0,
              margin: 0,
              display: "flex",
              flexDirection: "column",
              gap: "0.75rem",
            }}
          >
            {interactions.map((ix) => (
              <li
                key={ix.id}
                style={{
                  padding: "0.75rem 1rem",
                  borderRadius: "0.5rem",
                  border: "1px solid #1f2937",
                  backgroundColor: "#020617",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    gap: "0.75rem",
                    marginBottom: "0.25rem",
                    fontSize: "0.8rem",
                    color: "#9ca3af",
                  }}
                >
                  <div>
                    <span style={{ textTransform: "capitalize" }}>{ix.channel}</span>{" "}
                    <span>• {ix.direction}</span>
                    {ix.source && <span> • {ix.source}</span>}
                  </div>
                  <div>
                    {new Date(ix.occurred_at).toLocaleString(undefined, {
                      year: "numeric",
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                </div>
                <p style={{ margin: 0, fontSize: "0.9rem", lineHeight: 1.5 }}>{ix.summary}</p>
                {ix.raw_capture_id != null && (
                  <div style={{ marginTop: "0.25rem", fontSize: "0.75rem", color: "#6b7280" }}>
                    Linked raw capture ID: {ix.raw_capture_id}
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

