"""
setup_database.py
-----------------
Creates portfolio_database.db, applies database_schema.sql, and imports
all nine CSV files from the data/ directory into their corresponding tables.

Usage:
    python setup_database.py
"""

import sqlite3
import logging
import os
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "portfolio_database.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "database_schema.sql")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Maps table name -> CSV filename (in dependency / FK-safe insertion order)
TABLE_CSV_MAP = [
    ("sectors",              "sectors.csv"),
    ("securities",           "securities.csv"),
    ("benchmarks",           "benchmarks.csv"),
    ("portfolios",           "portfolios.csv"),
    ("holdings",             "holdings.csv"),
    ("transactions",         "transactions.csv"),
    ("historical_prices",    "historical_prices.csv"),
    ("portfolio_performance","portfolio_performance.csv"),
    ("risk_metrics",         "risk_metrics.csv"),
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key enforcement enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def apply_schema(conn: sqlite3.Connection, schema_path: str) -> None:
    """Execute the schema SQL file against the open connection."""
    log.info("Applying schema from %s", schema_path)
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema_sql = fh.read()
    conn.executescript(schema_sql)
    conn.commit()
    log.info("Schema applied successfully.")


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Read a CSV into a DataFrame.
    Strips whitespace from column names.
    Replaces empty strings with None so SQLite stores NULL, not ''.
    """
    df = pd.read_csv(csv_path, dtype=str)           # read everything as str first
    df.columns = df.columns.str.strip()
    df = df.where(df.notna(), other=None)           # NaN -> None (NULL in SQLite)
    # Replace empty strings with None as well
    df = df.map(lambda x: None if isinstance(x, str) and x.strip() == "" else x)
    return df


def import_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    """
    Insert rows into *table* using pandas to_sql with 'append' (schema already
    created).  Returns the number of rows inserted.
    """
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table};")
    count = cursor.fetchone()[0]
    return count


def verify_foreign_keys(conn: sqlite3.Connection) -> list[str]:
    """
    Run PRAGMA foreign_key_check on every table and collect violations.
    Returns a list of violation strings (empty list = all FK constraints OK).
    """
    cursor = conn.execute("PRAGMA foreign_key_check;")
    rows = cursor.fetchall()
    violations = []
    for row in rows:
        table, rowid, parent, fkid = row
        violations.append(
            f"  FK violation: table={table!r} rowid={rowid} "
            f"references parent={parent!r} (fk_index={fkid})"
        )
    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Remove stale DB so we always start clean --------------------
    if os.path.exists(DB_PATH):
        log.info("Removing existing database at %s", DB_PATH)
        os.remove(DB_PATH)

    conn = get_connection(DB_PATH)
    log.info("Opened database at %s", DB_PATH)

    try:
        # 1. Apply schema
        apply_schema(conn, SCHEMA_PATH)

        # 2. Import each CSV
        log.info("-" * 55)
        log.info("%-28s  %s", "Table", "Rows imported")
        log.info("-" * 55)

        total_rows = 0
        for table, csv_file in TABLE_CSV_MAP:
            csv_path = os.path.join(DATA_DIR, csv_file)

            if not os.path.exists(csv_path):
                log.error("CSV not found: %s — skipping table %r", csv_path, table)
                continue

            df = load_csv(csv_path)
            count = import_table(conn, table, df)
            log.info("%-28s  %d", table, count)
            total_rows += count

        log.info("-" * 55)
        log.info("%-28s  %d", "TOTAL", total_rows)

        # 3. Validate foreign-key integrity
        log.info("")
        log.info("Running foreign-key integrity check …")
        violations = verify_foreign_keys(conn)
        if violations:
            log.error("Foreign-key violations found:")
            for v in violations:
                log.error(v)
        else:
            log.info("Foreign-key check passed — no violations.")

        # 4. Quick sanity: confirm all 9 tables exist and are non-empty
        log.info("")
        log.info("Final table record counts:")
        log.info("-" * 55)
        for table, _ in TABLE_CSV_MAP:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table};")
            n = cursor.fetchone()[0]
            status = "OK" if n > 0 else "EMPTY — WARNING"
            log.info("  %-26s  %5d  [%s]", table, n, status)

    except Exception as exc:
        log.exception("Setup failed: %s", exc)
        raise
    finally:
        conn.close()
        log.info("")
        log.info("Database connection closed.")


if __name__ == "__main__":
    main()
