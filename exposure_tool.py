"""
exposure_tool.py
----------------
Sector Exposure Calculator: given a portfolio name, returns the percentage
exposure to each sector using equity holdings only (bonds are excluded).

Exposure is computed by summing each holding's current_weight for equities,
then re-normalising the sum to 100% so the output is always relative to the
equity-only slice of the portfolio.

Public API
----------
    calculate_exposure(portfolio_name: str, conn: sqlite3.Connection) -> str
"""

import sqlite3
import logging
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

_EXPOSURE_SQL = """
SELECT
    sec.sector_name,
    SUM(h.current_weight) AS total_weight
FROM holdings h
JOIN securities s  ON h.security_id  = s.security_id
JOIN sectors   sec ON s.sector_id     = sec.sector_id
JOIN portfolios p  ON h.portfolio_id  = p.portfolio_id
WHERE
    p.portfolio_name = ?
    AND s.asset_type  = 'Stock'
GROUP BY
    sec.sector_name
ORDER BY
    total_weight DESC;
"""

_PORTFOLIO_EXISTS_SQL = """
SELECT portfolio_id, portfolio_name
FROM portfolios
WHERE portfolio_name = ?;
"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def calculate_exposure(
    portfolio_name: str,
    conn: sqlite3.Connection,
    return_details: bool = False,
) -> str | dict:
    """
    Calculate equity-only sector exposures for *portfolio_name*.

    Returns a human-readable formatted string with sector percentages,
    or an informative error message if the portfolio is not found / has no
    equity holdings.
    """
    started = time.perf_counter()
    details = {
        "tool": "exposure_calculator",
        "portfolio_name": portfolio_name,
        "included_asset_type": "Stock",
        "excluded_asset_type": "Bond",
        "weight_source": "holdings.current_weight",
    }

    def finish(answer: str, status: str, **extra):
        trace = dict(details)
        trace.update(extra)
        trace["execution_status"] = status
        trace["duration_seconds"] = round(time.perf_counter() - started, 4)
        if return_details:
            return {"answer": answer, "tool_details": [trace]}
        return answer

    # 1. Confirm the portfolio exists
    try:
        cursor = conn.execute(_PORTFOLIO_EXISTS_SQL, (portfolio_name,))
        row = cursor.fetchone()
    except sqlite3.Error as exc:
        log.error("[exposure_tool] Portfolio lookup failed (%s).", type(exc).__name__)
        return finish("Error: The portfolio database is unavailable. Please try again later.", "failed")

    if row is None:
        log.info("[exposure_tool] Portfolio lookup returned no match.")
        return finish(
            f"Portfolio '{portfolio_name}' was not found in the database.\n"
            "Please check the portfolio name and try again.",
            "portfolio_not_found",
        )

    # 2. Fetch sector weights for equity holdings
    try:
        cursor = conn.execute(_EXPOSURE_SQL, (portfolio_name,))
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        log.error("[exposure_tool] Exposure query failed (%s).", type(exc).__name__)
        return finish("Error: Could not retrieve holdings from the portfolio database.", "failed")

    if not rows:
        log.info("[exposure_tool] No equity holdings found.")
        return finish(
            f"Portfolio '{portfolio_name}' has no equity holdings.\n"
            "Sector exposure cannot be calculated (bonds are excluded).",
            "no_equity_holdings",
        )

    # 3. Normalise to 100 % across equity-only weights
    try:
        sector_weights = [(sector, float(weight or 0)) for sector, weight in rows]
    except (TypeError, ValueError):
        log.error("[exposure_tool] Invalid holding weight data.")
        return finish(
            "Error: Holdings contain invalid weight data; exposure could not be calculated.",
            "invalid_weight_data",
        )
    total_equity_weight = sum(weight for _, weight in sector_weights)

    if total_equity_weight <= 0:
        return finish(
            f"Portfolio '{portfolio_name}': equity holding weights sum to zero. "
            "Cannot compute exposure percentages.",
            "zero_equity_weight",
            sector_weights=[{"sector": sector, "weight": weight} for sector, weight in sector_weights],
            total_equity_weight=total_equity_weight,
        )

    exposures = [
        (sector, weight / total_equity_weight * 100.0)
        for sector, weight in sector_weights
    ]

    # 4. Format output
    lines = [
        f"Sector Exposure — {portfolio_name}",
        f"(Equity holdings only; bonds excluded)",
        "",
        f"{'Sector':<28}  {'Exposure':>8}",
        "-" * 40,
    ]
    for sector, pct in exposures:
        lines.append(f"  {sector:<26}  {pct:>7.2f}%")

    lines.append("-" * 40)
    lines.append(f"  {'Total (equity)':<26}  {'100.00':>7}%")

    log.info("[exposure_tool] Calculated %d sector(s) in %.3fs.", len(exposures), time.perf_counter() - started)
    highest_sector, highest_pct = max(exposures, key=lambda item: item[1])
    if len(exposures) == 1:
        explanation = (
            f"{highest_sector} represents {highest_pct:.2f}% of "
            f"{portfolio_name}'s equity exposure."
        )
    else:
        lowest_sector, lowest_pct = min(exposures, key=lambda item: item[1])
        spread = highest_pct - lowest_pct
        explanation = (
            f"{highest_sector} has the largest exposure at {highest_pct:.2f}%; "
            f"{lowest_sector} has the smallest at {lowest_pct:.2f}%, "
            f"a difference of {spread:.2f} percentage points."
        )
    answer = f"{explanation}\n\n" + "\n".join(lines)
    return finish(
        answer,
        "calculated",
        sector_weights=[{"sector": sector, "weight": weight} for sector, weight in sector_weights],
        total_equity_weight=total_equity_weight,
        exposures=[{"sector": sector, "percentage": round(pct, 4)} for sector, pct in exposures],
    )
