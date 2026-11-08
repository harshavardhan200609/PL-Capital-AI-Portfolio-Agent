"""Shared read-only SQLite connection and authorizer helpers."""

import sqlite3
from pathlib import Path


_READ_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _READ_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)

_DENIED_FUNCTIONS = {"load_extension", "writefile"}


def _read_only_authorizer(action, arg1, arg2, database, trigger):
    """Permit query reads and deny all database changes and unsafe functions."""
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (arg2 or arg1 or "").lower()
        if function_name in _DENIED_FUNCTIONS:
            return sqlite3.SQLITE_DENY
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def install_read_only_protection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Protect a SQLite connection against writes, schema changes, and PRAGMAs."""
    # Temporarily clear a prior authorizer so the connection-level safeguard
    # can be applied idempotently; no user SQL runs during this short window.
    conn.set_authorizer(None)
    try:
        conn.execute("PRAGMA query_only = ON")
    finally:
        conn.set_authorizer(_read_only_authorizer)
    return conn


def connect_read_only(db_path: str) -> sqlite3.Connection:
    """Open an existing SQLite database with URI read-only mode and an authorizer."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found at '{path}'.")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        install_read_only_protection(conn)
    except Exception:
        conn.close()
        raise
    return conn
