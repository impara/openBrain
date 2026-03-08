from __future__ import annotations

from contextlib import contextmanager

from psycopg2 import pool

from openbrain.settings import OpenBrainSettings


class Database:
    def __init__(self, settings: OpenBrainSettings):
        self.settings = settings
        self.pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=max(6, settings.ingest_workers + 6),
            dbname=settings.database.dbname,
            user=settings.database.user,
            password=settings.database.password,
            host=settings.database.host,
            port=settings.database.port,
        )

    @contextmanager
    def cursor(self, *, commit: bool = False, age: bool = False):
        conn = self.pool.getconn()
        try:
            with conn.cursor() as cur:
                if age:
                    cur.execute("LOAD 'age';")
                    cur.execute("SET search_path TO ag_catalog, public;")
                yield conn, cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                if not conn.closed:
                    conn.rollback()
            except Exception:
                pass
            self.pool.putconn(conn)

    def close(self) -> None:
        self.pool.closeall()
