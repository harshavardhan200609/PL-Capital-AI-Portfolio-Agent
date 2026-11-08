"""
sql_tool.py
-----------
SQL Query Tool: converts a natural-language question into a safe, read-only
SQLite query using the configured LLM provider, executes it, and returns human-readable
results.

Public API
----------
    run_sql_query(question: str, conn: sqlite3.Connection) -> str
"""

import re
import sqlite3
import logging
import time
from decimal import Decimal, InvalidOperation

# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
from llm_client import llm_call as _llm_call
from db_utils import install_read_only_protection

# Database schema hint sent to the configured provider for query generation
# ---------------------------------------------------------------------------

_SCHEMA_CONTEXT = """
You are a SQLite expert working on a portfolio management database.

DATABASE SCHEMA:
----------------
sectors(sector_id PK, sector_name, sector_description, industry_group)

securities(security_id PK, symbol, company_name,
           asset_type TEXT CHECK IN ('Stock','Bond'),
           sector_id FK->sectors, market_cap, current_price, currency,
           exchange, country, listing_date, maturity_date, coupon_rate)

benchmarks(benchmark_id PK, benchmark_name, benchmark_symbol,
           benchmark_type, description, inception_date)

portfolios(portfolio_id PK, portfolio_name, creation_date,
           target_risk_level, total_aum, strategy_type,
           benchmark_index, status)

holdings(holding_id PK, portfolio_id FK->portfolios,
         security_id FK->securities, quantity, purchase_price,
         purchase_date, current_weight, cost_basis,
         UNIQUE(portfolio_id, security_id))

transactions(transaction_id PK, portfolio_id FK->portfolios,
             security_id FK->securities,
             transaction_type CHECK IN ('BUY','SELL'),
             quantity, price, transaction_date, fees,
             settlement_date, notes)

historical_prices(price_id PK, security_id FK->securities, price_date,
                  open_price, high_price, low_price, close_price,
                  volume, adjusted_close,
                  UNIQUE(security_id, price_date))

portfolio_performance(performance_id PK, portfolio_id FK->portfolios,
                      performance_date, nav, total_return_1m,
                      total_return_3m, total_return_6m, total_return_1y,
                      volatility, sharpe_ratio, max_drawdown,
                      UNIQUE(portfolio_id, performance_date))

risk_metrics(risk_id PK, portfolio_id FK->portfolios, calculation_date,
             var_95, var_99, cvar_95, beta, correlation_sp500,
             tracking_error, information_ratio, sortino_ratio,
             UNIQUE(portfolio_id, calculation_date))

RULES:
- Generate only a single SELECT statement. No INSERT, UPDATE, DELETE, DROP, CREATE, or any DDL/DML.
- Use standard SQLite syntax (no ILIKE, use LIKE instead).
- Column names are exactly as listed above.
- String comparisons are case-sensitive in SQLite. Use the exact case shown in the data:
    portfolios.status values are: 'Active', 'Passive'  (capital first letter)
    securities.asset_type values are: 'Stock', 'Bond'  (capital first letter)
    sectors.sector_name values are e.g. 'Technology', 'Financials'  (title case)
- Always include human-readable descriptive columns (e.g. portfolio_name, symbol, company_name, sector_name) rather than only IDs or numeric values.
- For "top N holdings" requests, include these columns in this exact order:
    h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis.
  Sort by the criterion requested by the user and apply the requested LIMIT.
- IMPORTANT â€” two distinct concepts, never substitute one for the other:
    AUM (Assets Under Management): always use portfolios.total_aum.
      Use this whenever the question mentions AUM, total AUM, Assets Under Management,
      or fund size/value in a business context.
      Example: SELECT SUM(total_aum) FROM portfolios WHERE ...
    Calculated holdings market value: always use SUM(holdings.quantity * securities.current_price).
      Use this whenever the question asks for the market value of holdings,
      computed portfolio value, or position value derived from quantities and prices.
      Example: SELECT SUM(h.quantity * s.current_price) FROM holdings h JOIN securities s ...
- When asked for Technology sector value and percentage of total portfolio value, follow this pattern:
    CTE 1: SUM(h.quantity * s.current_price) WHERE sector = 'Technology'   -> tech_value
    CTE 2: SUM(h.quantity * s.current_price) across ALL holdings            -> total_value
    Output: portfolio_name, tech_value, total_value, ROUND(tech_value/total_value*100, 2) as tech_percentage
- When asked for sector diversification metrics (portfolios with >N sectors), follow this pattern:
    CTE 1: COUNT(DISTINCT sec.sector_id) WHERE asset_type='Stock' HAVING count>5 -> sector_count, portfolio_name
    CTE 2: COUNT(h.holding_id) from ALL holdings (including bonds)                -> total_holdings
    Output: portfolio_name, sector_count, total_holdings, ROUND(CAST(sector_count AS REAL)/total_holdings, 3) as diversification_ratio
- Return ONLY the raw SQL query â€” no markdown fences, no explanation, no trailing semicolons.
"""

# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|ATTACH|DETACH|VACUUM|REINDEX)",
    re.IGNORECASE,
)


def _remove_leading_comments(sql: str) -> str:
    """Remove SQL comments before validating the first statement keyword."""
    # Some provider responses begin with a UTF-8 BOM. It is formatting, not
    # part of the SQL statement, and must not hide the SELECT/WITH keyword.
    remaining = sql.lstrip("\ufeff").lstrip()
    while remaining.startswith("--") or remaining.startswith("/*"):
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            if newline < 0:
                return ""
            remaining = remaining[newline + 1:].lstrip()
        else:
            end = remaining.find("*/", 2)
            if end < 0:
                return ""
            remaining = remaining[end + 2:].lstrip()
    return remaining


def _is_safe(sql: str) -> tuple[bool, str]:
    """Return (True, '') if the SQL is read-only, else (False, reason)."""
    cleaned = _remove_leading_comments(sql.strip()).rstrip(";").strip()
    if _FORBIDDEN.match(cleaned):
        return False, "Disallowed database modification statement."
    if not re.match(r"^\s*(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        return False, "Only SELECT / WITH â€¦ SELECT queries are permitted."
    # sqlite3.Connection.execute rejects stacked statements. The read-only
    # authorizer below independently blocks changes, including write CTEs.
    return True, ""


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def _format_results(cursor: sqlite3.Cursor, max_rows: int = 50) -> str:
    """Turn a cursor result set into a readable string table."""
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchmany(max_rows + 1)

    if not rows:
        return "Query returned no results."

    truncated = len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]

    widths = [len(c) for c in columns]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val) if val is not None else "NULL"))

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    header = "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns)) + " |"

    lines = [sep, header, sep]
    for row in rows:
        line = "| " + " | ".join(
            str(v if v is not None else "NULL").ljust(widths[i])
            for i, v in enumerate(row)
        ) + " |"
        lines.append(line)
    lines.append(sep)

    if truncated:
        lines.append(f"  (showing first {max_rows} rows; more results exist)")

    return "\n".join(lines)


def _total_aum_summary(question: str, result_table: str, sql: str = "") -> str | None:
    """Build a deterministic total-AUM answer from the returned AUM column."""
    asks_for_total = re.search(
        r"\b(?:total|combined)\b.{0,100}\b(?:aum|assets\s+under\s+management)\b|"
        r"\b(?:aum|assets\s+under\s+management)\b.{0,100}\b(?:total|combined)\b",
        question,
        re.IGNORECASE,
    )
    if not asks_for_total or "(showing first " in result_table.lower():
        return None

    table_lines = [line for line in result_table.splitlines() if line.startswith("| ")]
    if len(table_lines) < 2:
        return None
    table = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in table_lines
    ]
    headers = table[0]
    aum_index = next(
        (index for index, header in enumerate(headers)
         if re.search(r"\b(?:total_aum|aum)\b", header, re.IGNORECASE)),
        None,
    )
    if aum_index is None:
        return None

    values = []
    for row in table[1:]:
        if len(row) != len(headers):
            continue
        try:
            values.append(float(row[aum_index].replace(",", "")))
        except ValueError:
            return None
    if not values:
        return None

    total = sum(values)
    formatted_total = f"{int(total):,}" if total.is_integer() else f"{total:,.2f}".rstrip("0").rstrip(".")
    if re.search(r"\bSUM\s*\(", sql, re.IGNORECASE):
        return f"The total AUM is {formatted_total}."
    portfolio_count = len(values)
    return f"The combined AUM is {formatted_total} across {portfolio_count} returned portfolio(s)."


def _count_summary(question: str, result_table: str, sql: str) -> str | None:
    """Summarize a scalar COUNT result using its actual returned value."""
    if not (
        re.search(r"\b(?:how many|number of|count)\b", question, re.IGNORECASE)
        or re.search(r"\bCOUNT\s*\(", sql, re.IGNORECASE)
    ):
        return None
    lines = [line for line in result_table.splitlines() if line.startswith("| ")]
    if len(lines) != 2:
        return None
    cells = [cell.strip() for cell in lines[1].strip().strip("|").split("|")]
    if len(cells) != 1:
        return None
    try:
        count = int(float(cells[0]))
    except ValueError:
        return None
    if re.search(r"\bportfolios?\b", question, re.IGNORECASE):
        return f"We have {count} portfolios in total."
    entity = next(
        (name for pattern, name in (
            (r"\bsecurities\b", "securities"),
            (r"\bholdings?\b", "holdings"),
            (r"\btransactions?\b", "transactions"),
            (r"\bsectors?\b", "sectors"),
        ) if re.search(pattern, question, re.IGNORECASE)),
        None,
    )
    return f"There are {count} {entity} in total." if entity else f"The count is {count}."


def _sum_summary(result_table: str, sql: str) -> str | None:
    """Summarize a scalar SUM result from its returned value and expression."""
    if not re.search(r"\bSUM\s*\(", sql, re.IGNORECASE):
        return None
    lines = [line for line in result_table.splitlines() if line.startswith("| ")]
    if len(lines) != 2:
        return None
    cells = [cell.strip() for cell in lines[1].strip().strip("|").split("|")]
    if len(cells) != 1:
        return None
    try:
        total = float(cells[0].replace(",", ""))
    except ValueError:
        return None
    formatted = f"{int(total):,}" if total.is_integer() else f"{total:,.2f}".rstrip("0").rstrip(".")
    expression = re.search(r"\bSUM\s*\(\s*([\w.]+)", sql, re.IGNORECASE)
    metric = expression.group(1).split(".")[-1].replace("_", " ") if expression else ""
    return f"The total {metric} is {formatted}." if metric else f"The total is {formatted}."


def _parse_result_table(result_table: str) -> tuple[list[str], list[list[str]]] | None:
    """Read the headers and data rows from the SQL tool's displayed table."""
    lines = [line for line in result_table.splitlines() if line.startswith("| ")]
    if len(lines) < 2:
        return None
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    headers = cells[0]
    rows = [row for row in cells[1:] if len(row) == len(headers)]
    return (headers, rows) if rows else None


def _decimal_value(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", "").strip())
    except InvalidOperation:
        return None


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"{int(value):,}"
    return format(value, ",f").rstrip("0").rstrip(".")


def _list_summary(question: str, result_table: str) -> str | None:
    """Explain returned rows and numeric comparisons without adding unsupported facts."""
    parsed = _parse_result_table(result_table)
    if not parsed:
        return None
    headers, rows = parsed

    def find_column(*names: str) -> int | None:
        return next((i for i, header in enumerate(headers) if header.casefold() in names), None)

    name_index = next(
        (i for i, header in enumerate(headers)
         if header.casefold() in {"portfolio_name", "sector_name", "company_name", "symbol", "security_name"}),
        0,
    )
    company_index = find_column("company_name")
    symbol_index = find_column("symbol")
    name_header = headers[name_index].casefold()
    entity = {
        "portfolio_name": "portfolios",
        "sector_name": "sectors",
        "company_name": "securities",
        "symbol": "securities",
        "security_name": "securities",
        "holding_id": "holdings",
    }.get(name_header, "results")

    numeric_columns = []
    for index, header in enumerate(headers):
        normalized = header.casefold()
        if normalized == "id" or normalized.endswith("_id"):
            continue
        values = [_decimal_value(row[index]) for row in rows]
        if values and all(value is not None for value in values):
            numeric_columns.append((index, header, values))

    question_lower = question.casefold()
    comparison = bool(re.search(
        r"\b(compare|comparison|highest|lowest|top|bottom|most|least|rank|largest|smallest|difference)\b",
        question_lower,
    )) or (len(rows) > 1 and bool(re.search(r"\b(each|per|by)\b", question_lower)) and bool(numeric_columns))

    if comparison and numeric_columns:
        # Prefer columns directly named by the question; otherwise summarize
        # up to three meaningful numeric columns from the returned table.
        hinted = []
        for item in numeric_columns:
            header = item[1].casefold().replace("_", " ")
            header_terms = set(header.split())
            if (
                any(term in question_lower for term in (header, header.replace(" percentage", "")))
                or ("value" in header_terms and "value" in question_lower)
                or ("percentage" in header_terms and "percentage" in question_lower)
            ):
                hinted.append(item)
        selected = hinted or numeric_columns

        def describe_entity(row: list[str]) -> str:
            if company_index is not None and symbol_index is not None:
                return f"{row[company_index]} ({row[symbol_index]})"
            return row[name_index]

        observations = []
        for index, header, values in selected[:3]:
            low_position = min(range(len(values)), key=values.__getitem__)
            high_position = max(range(len(values)), key=values.__getitem__)
            low_value, high_value = values[low_position], values[high_position]
            if low_position == high_position:
                observations.append(
                    f"All returned rows have {header.replace('_', ' ')} of {_format_decimal(high_value)}"
                )
                continue
            low_name, high_name = describe_entity(rows[low_position]), describe_entity(rows[high_position])
            spread = high_value - low_value
            unit = " percentage points" if "percent" in header.casefold() or "exposure" in header.casefold() else ""
            observations.append(
                f"{high_name} has the highest {header.replace('_', ' ')} "
                f"({_format_decimal(high_value)}); {low_name} has the lowest "
                f"({_format_decimal(low_value)}), a difference of {_format_decimal(spread)}{unit}"
            )
        if observations:
            summary = ". ".join(observations).rstrip(".") + "."
            if "(showing first " in result_table.lower():
                summary += " These comparisons use only the first 50 displayed rows."
            return f"{summary}\n\n{result_table}"

    def describe_row(row: list[str]) -> str:
        name = row[name_index]
        if company_index is not None and symbol_index is not None:
            name = f"{row[symbol_index]} ({row[company_index]})"
        metrics = []
        for index, header, _ in numeric_columns[:3]:
            metrics.append(f"{header.replace('_', ' ')}: {row[index]}")
        return f"{name}" + (f" ({'; '.join(metrics)})" if metrics else "")

    if len(rows) == 1:
        summary = f"The query returned {describe_row(rows[0])}."
    else:
        labels = [describe_row(row) for row in rows[:5]]
        shown = ", ".join(labels)
        remainder = len(rows) - len(labels)
        if remainder > 0:
            shown += f", and {remainder} more"
        singular = {"portfolios": "portfolio", "sectors": "sector", "securities": "security", "holdings": "holding"}.get(entity, "result")
        noun = entity if len(rows) != 1 else singular
        summary = f"The query returned {len(rows)} {noun}: {shown}."
        if "(showing first " in result_table.lower():
            summary = f"The query returned at least {len(rows)} {entity}: {shown}."
    return f"{summary}\n\n{result_table}"


_KNOWN_PORTFOLIO_NAMES = (
    "Growth Equity Fund", "Conservative Income Fund", "Tech Innovation Fund",
    "Balanced Portfolio", "ESG Sustainable Fund", "Small Cap Value Fund",
    "International Equity Fund", "Fixed Income Plus", "Dividend Aristocrats Fund",
    "Emerging Markets Fund", "Total Stock Market Index Fund",
    "Total Bond Market Index Fund", "Total International Index Fund",
)
_KNOWN_SECTOR_NAMES = (
    "Technology", "Financials", "Healthcare", "Energy", "Consumer Staples",
    "Consumer Discretionary", "Automotive", "Industrials", "Materials", "Utilities",
    "Communication Services",
)


def _deterministic_sql(question: str) -> str | None:
    """Return safe, schema-based SQL for a few unambiguous common requests."""
    current_question = re.search(r"\bCurrent question:\s*(.+)$", question, re.IGNORECASE | re.DOTALL)
    source_question = current_question.group(1) if current_question else question
    q = re.sub(r"\s+", " ", source_question).strip().lower()

    # Leave questions that combine SQL findings with exposure for the hybrid
    # router, even if one half matches a deterministic SQL template.
    asks_exposure = re.search(r"\b(?:sector\s+)?(?:exposures?|allocations?|breakdowns?)\b", q)
    asks_sql_findings = re.search(r"\b(?:holdings?|transactions?|performance|risk|market value|count|how many)\b", q)
    if asks_exposure and asks_sql_findings:
        return None

    sector_name = next(
        (name for name in _KNOWN_SECTOR_NAMES if re.search(rf"\b{re.escape(name)}\b", q, re.IGNORECASE)),
        None,
    )
    asks_sector_share = (
        sector_name
        and re.search(r"\bfor each portfolio\b", q)
        and re.search(r"\bpercentage\b", q)
        and re.search(r"\btotal\s+(?:portfolio\s+)?value\b", q)
        and re.search(r"\bholdings?\b", q)
    )
    if asks_sector_share:
        escaped_sector = sector_name.replace("'", "''")
        sector_alias = re.sub(r"\W+", "_", sector_name.lower()).strip("_")
        return (
            "WITH portfolio_sector_value AS ("
            "SELECT p.portfolio_id, p.portfolio_name, "
            "SUM(h.quantity * s.current_price) AS sector_value "
            "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
            "JOIN securities s ON h.security_id = s.security_id "
            "JOIN sectors sec ON s.sector_id = sec.sector_id "
            f"WHERE sec.sector_name = '{escaped_sector}' "
            "GROUP BY p.portfolio_id, p.portfolio_name), "
            "portfolio_total_value AS ("
            "SELECT p.portfolio_id, SUM(h.quantity * s.current_price) AS total_value "
            "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
            "JOIN securities s ON h.security_id = s.security_id "
            "GROUP BY p.portfolio_id) "
            f"SELECT psv.portfolio_name, psv.sector_value AS {sector_alias}_value, "
            "ptv.total_value, "
            f"ROUND(psv.sector_value * 100.0 / NULLIF(ptv.total_value, 0), 2) AS {sector_alias}_percentage "
            "FROM portfolio_sector_value psv JOIN portfolio_total_value ptv "
            "ON psv.portfolio_id = ptv.portfolio_id "
            f"ORDER BY {sector_alias}_percentage DESC"
        )

    sector_threshold = re.search(
        r"\b(?:more\s+than|greater\s+than|over)\s+(\d+)\s+(?:different\s+)?sectors?\b",
        q,
    )
    if sector_threshold and re.search(r"\bdiversification\b|\bsector\s+count\b", q):
        threshold = int(sector_threshold.group(1))
        return (
            "WITH portfolio_sector_count AS ("
            "SELECT p.portfolio_id, p.portfolio_name, "
            "COUNT(DISTINCT sec.sector_id) AS sector_count "
            "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
            "JOIN securities s ON h.security_id = s.security_id "
            "JOIN sectors sec ON s.sector_id = sec.sector_id "
            "WHERE s.asset_type = 'Stock' "
            "GROUP BY p.portfolio_id, p.portfolio_name "
            f"HAVING COUNT(DISTINCT sec.sector_id) > {threshold}), "
            "portfolio_holdings_count AS ("
            "SELECT p.portfolio_id, COUNT(h.holding_id) AS total_holdings "
            "FROM portfolios p JOIN holdings h ON p.portfolio_id = h.portfolio_id "
            "GROUP BY p.portfolio_id) "
            "SELECT psc.portfolio_name, psc.sector_count, phc.total_holdings, "
            "ROUND(CAST(psc.sector_count AS REAL) / NULLIF(phc.total_holdings, 0), 3) AS diversification_ratio "
            "FROM portfolio_sector_count psc JOIN portfolio_holdings_count phc "
            "ON psc.portfolio_id = phc.portfolio_id ORDER BY diversification_ratio DESC"
        )

    if re.search(r"\baverage\b", q) and re.search(r"\bcurrent\s+price\b", q) and re.search(r"\beach\s+sector\b", q):
        return (
            "SELECT sec.sector_name, AVG(s.current_price) AS average_current_price "
            "FROM securities s JOIN sectors sec ON s.sector_id = sec.sector_id "
            "GROUP BY sec.sector_id, sec.sector_name ORDER BY sec.sector_name"
        )

    if sector_name and re.search(r"\bsecurities\b", q) and re.search(r"\bsector\b", q):
        escaped_sector = sector_name.replace("'", "''")
        return (
            "SELECT s.symbol, s.company_name FROM securities s "
            "JOIN sectors sec ON s.sector_id = sec.sector_id "
            f"WHERE sec.sector_name = '{escaped_sector}' ORDER BY s.symbol"
        )

    if re.search(r"\bactive\s+portfolios?\b", q) and re.search(r"\b(?:names?|list|show|what|all)\b", q):
        return "SELECT portfolio_name FROM portfolios WHERE status = 'Active' ORDER BY portfolio_name"

    if re.search(r"\b(?:how many|number of|count(?: of)?)\b.*\bportfolios?\b|\bportfolio count\b", q):
        return "SELECT COUNT(*) AS portfolio_count FROM portfolios"

    if (
        re.search(r"\b(?:total|combined)\b.{0,80}\b(?:aum|assets\s+under\s+management)\b", q)
        and re.search(r"\b(?:high|higher)\b.{0,35}\b(?:target\s+)?risk\b|\bhigh[- ]risk\b", q)
    ):
        return (
            "SELECT SUM(total_aum) AS total_aum FROM portfolios "
            "WHERE target_risk_level = 'High'"
        )

    portfolio_name = next(
        (name for name in _KNOWN_PORTFOLIO_NAMES if name.lower() in q),
        None,
    )
    if not portfolio_name:
        return None

    has_sector_exposure = bool(re.search(r"\b(?:sector\s+)?(?:exposures?|allocations?|breakdowns?)\b", q))
    if has_sector_exposure:
        return None

    if re.search(r"\btop\s+(\d+)\s+holdings?\b", q) and re.search(r"\bcost\s+basis\b", q):
        limit = int(re.search(r"\btop\s+(\d+)\s+holdings?\b", q).group(1))
        escaped_name = portfolio_name.replace("'", "''")
        return (
            "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis "
            "FROM holdings h JOIN securities s ON h.security_id = s.security_id "
            "JOIN portfolios p ON h.portfolio_id = p.portfolio_id "
            f"WHERE p.portfolio_name = '{escaped_name}' "
            f"ORDER BY h.cost_basis DESC LIMIT {limit}"
        )

    # Handle clear list/follow-up requests only. More specific holding filters
    # (for example, by date or market value) remain with the SQL generator.
    is_holdings_list = (
        re.search(r"\b(?:what about|show|list|display|give me)\b.*\bholdings?\b", q)
        or re.search(r"\bholdings?\b.*\b(?:for|in|of)\b", q)
    )
    has_extra_filter = re.search(r"\b(?:where|with|above|below|greater|less|after|before)\b", q)
    if is_holdings_list and not has_extra_filter:
        escaped_name = portfolio_name.replace("'", "''")
        return (
            "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis "
            "FROM holdings h JOIN securities s ON h.security_id = s.security_id "
            "JOIN portfolios p ON h.portfolio_id = p.portfolio_id "
            f"WHERE p.portfolio_name = '{escaped_name}' "
            "ORDER BY h.holding_id"
        )
    return None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def generate_sql(question: str) -> str:
    """Ask the configured LLM to convert a natural-language question into a SQL query."""
    if not question:
        raise ValueError("Question must not be empty.")
    deterministic_sql = _deterministic_sql(question)
    if deterministic_sql is not None:
        return deterministic_sql
    prompt = (
        f"{_SCHEMA_CONTEXT}\n\n"
        f"USER QUESTION: {question}\n\n"
        "SQL QUERY:"
    )
    # Trim before handling markdown fences: providers sometimes indent a
    # fenced SQL response, which otherwise leaves ```sql at the front and
    # causes the read-only validator to reject the actual SELECT statement.
    sql = _llm_call(prompt).strip().lstrip("\ufeff").strip()

    # Strip accidental markdown fences
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^```\s*",    "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```$",       "", sql).strip()
    sql = sql.rstrip(";").strip()
    return sql


def _summarize_results(question: str, result_table: str) -> str:
    """Explain executed SQL results in plain language without adding facts."""
    prompt = (
        "Answer the user's portfolio analytics question in concise, natural language. "
        "Use only the database result shown below. Preserve all relevant names and "
        "numbers exactly; do not infer facts that are not present. If the result "
        "contains multiple rows, mention the key findings. Return only the answer.\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"DATABASE RESULT:\n{result_table}"
    )
    return _llm_call(prompt).strip()


def _sql_result(answer: str, sql: str | None, status: str, return_details: bool):
    """Optionally return execution metadata captured from this SQL run."""
    if not return_details:
        return answer
    return {
        "answer": answer,
        "tool_details": [{
            "tool": "sql_tool",
            "sql": sql,
            "execution_status": status,
        }],
    }


def run_sql_query(
    question: str,
    conn: sqlite3.Connection,
    return_details: bool = False,
) -> str | dict:
    """
    Full pipeline: question -> generated SQL -> safety check -> execute -> summarize.

    Returns a human-readable string (result table or error message).
    """
    started = time.perf_counter()

    try:
        sql = generate_sql(question)
    except Exception as exc:
        log.error("[sql_tool] SQL generation failed (%s).", type(exc).__name__)
        return _sql_result(
            "I couldn't generate a database query right now. Please try rephrasing your question.",
            None,
            "generation_failed",
            return_details,
        )

    safe, reason = _is_safe(sql)
    if not safe:
        log.warning("[sql_tool] Unsafe SQL rejected: %s", reason)
        return _sql_result(
            "I couldn't safely run the generated database query. Please try rephrasing your question.",
            sql,
            "rejected",
            return_details,
        )

    try:
        install_read_only_protection(conn)
        cursor = conn.execute(sql)
        result = _format_results(cursor)
        if result == "Query returned no results.":
            log.info("[sql_tool] Query returned no rows in %.3fs.", time.perf_counter() - started)
            return _sql_result(result, sql, "executed_no_rows", return_details)
        count_summary = _count_summary(question, result, sql)
        if count_summary:
            log.info("[sql_tool] Portfolio count read from query results in %.3fs.", time.perf_counter() - started)
            return _sql_result(f"{count_summary}\n\n{result}", sql, "executed", return_details)
        aum_summary = _total_aum_summary(question, result, sql)
        if aum_summary:
            # The numeric total is derived from the actual query result, so a
            # second LLM call cannot omit rows or introduce unsupported claims.
            log.info("[sql_tool] Total AUM calculated from query results in %.3fs.", time.perf_counter() - started)
            return _sql_result(f"{aum_summary}\n\n{result}", sql, "executed", return_details)
        sum_summary = _sum_summary(result, sql)
        if sum_summary:
            log.info("[sql_tool] SUM value read from query results in %.3fs.", time.perf_counter() - started)
            return _sql_result(f"{sum_summary}\n\n{result}", sql, "executed", return_details)
        column_count = len(cursor.description or [])
        list_summary = _list_summary(question, result)
        if list_summary:
            log.info("[sql_tool] Result table returned without LLM summary in %.3fs.", time.perf_counter() - started)
            return _sql_result(list_summary, sql, "executed", return_details)
        try:
            summary = _summarize_results(question, result)
            if summary:
                data_rows = max(0, sum(line.startswith("| ") for line in result.splitlines()) - 1)
                column_count = len(cursor.description or [])
                log.info("[sql_tool] Query and summary completed in %.3fs.", time.perf_counter() - started)
                if data_rows > 1 or column_count > 1:
                    answer = f"{summary}\n\n{result}"
                else:
                    answer = summary
                return _sql_result(answer, sql, "executed", return_details)
            log.warning("[sql_tool] LLM returned an empty summary; returning result table.")
        except Exception as exc:
            log.warning("[sql_tool] Result summarization failed (%s).", type(exc).__name__)
        return _sql_result(result, sql, "executed", return_details)
    except sqlite3.Error as exc:
        log.error("[sql_tool] SQL execution failed (%s) after %.3fs.", type(exc).__name__, time.perf_counter() - started)
        return _sql_result(
            "The database couldn't complete that query. Please try a simpler or more specific question.",
            sql,
            "failed",
            return_details,
        )

