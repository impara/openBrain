"use client";

import type { ReactNode } from "react";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          backgroundColor: "#020617",
          color: "#e5e7eb",
        }}
      >
        <div
          style={{
            maxWidth: 960,
            margin: "0 auto",
            padding: "1.5rem",
          }}
        >
          <header
            style={{
              marginBottom: "1.5rem",
              borderBottom: "1px solid #1f2937",
              paddingBottom: "0.75rem",
            }}
          >
            <h1 style={{ fontSize: "1.5rem", fontWeight: 600 }}>OpenBrain • Personal CRM</h1>
            <p style={{ marginTop: "0.25rem", color: "#9ca3af" }}>
              Contacts and interactions backed by your OpenBrain memory.
            </p>
          </header>
          {children}
        </div>
      </body>
    </html>
  );
}

