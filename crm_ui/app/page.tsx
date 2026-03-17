// @ts-nocheck
import React from "react";
import Link from "next/link";

type ContactSummary = {
  id: number;
  full_name: string;
  company?: string | null;
  tags: string[];
  last_interaction_at?: string | null;
};

const API_URL = process.env.API_URL ?? "http://localhost:8002";

async function fetchContacts(): Promise<ContactSummary[]> {
  const res = await fetch(`${API_URL}/contacts`, {
    // Avoid caching in dev; for production you can adjust.
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error("Failed to load contacts");
  }
  return res.json();
}

export default async function ContactsPage() {
  let contacts: ContactSummary[] = [];
  let error = false;
  try {
    contacts = await fetchContacts();
  } catch {
    error = true;
  }

  return (
    <main>
      <h2 style={{ fontSize: "1.25rem", fontWeight: 600, marginBottom: "1rem" }}>Contacts</h2>
      {error && (
        <p style={{ color: "#f87171", marginBottom: "0.75rem", fontSize: "0.9rem" }}>
          Could not reach the CRM API. Is the <code>crm-api</code> service running?
        </p>
      )}
      {contacts.length === 0 ? (
        <p style={{ color: "#9ca3af" }}>
          No contacts yet. Once your agent starts logging CRM interactions, they&apos;ll show up here.
        </p>
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
          {contacts.map((contact) => (
            <li
              key={contact.id}
              style={{
                padding: "0.75rem 1rem",
                borderRadius: "0.5rem",
                border: "1px solid #1f2937",
                backgroundColor: "#020617",
              }}
            >
              <Link
                href={`/contacts/${contact.id}`}
                style={{ textDecoration: "none", color: "inherit", display: "block" }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: "0.5rem" }}>
                  <div>
                    <div style={{ fontWeight: 600 }}>{contact.full_name}</div>
                    {contact.company && (
                      <div style={{ color: "#9ca3af", fontSize: "0.875rem" }}>{contact.company}</div>
                    )}
                    {contact.tags?.length > 0 && (
                      <div style={{ marginTop: "0.25rem", display: "flex", gap: "0.25rem", flexWrap: "wrap" }}>
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
                  </div>
                  <div style={{ textAlign: "right", fontSize: "0.75rem", color: "#9ca3af" }}>
                    {contact.last_interaction_at ? (
                      <>
                        <div>Last interaction</div>
                        <div>
                          {new Date(contact.last_interaction_at).toLocaleDateString(undefined, {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })}
                        </div>
                      </>
                    ) : (
                      <div>No interactions yet</div>
                    )}
                  </div>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}

