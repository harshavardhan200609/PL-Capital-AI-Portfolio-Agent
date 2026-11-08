"""
evaluator.py
------------
Agent Evaluator: loads ground_truth_dataset.json, runs each question through
the PortfolioAgent, compares the result against the expected answer, and
reports pass/fail with an overall accuracy score.

Usage:
    python evaluator.py
    python evaluator.py --verbose
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from typing import Any

from dotenv import load_dotenv

from agent import PortfolioAgent
from db_utils import connect_read_only

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_PATH = os.path.join(BASE_DIR, "ground_truth_dataset.json")
DB_PATH = os.path.join(BASE_DIR, "portfolio_database.db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth SQL execution
# ---------------------------------------------------------------------------

def _run_gt_sql(sql: str, conn: sqlite3.Connection) -> list[tuple]:
    """Execute the ground-truth SQL and return all rows."""
    cursor = conn.execute(sql)
    return cursor.fetchall()


def _run_gt_query(sql: str, conn: sqlite3.Connection):
    """Return expected column names and rows from ground-truth SQL."""
    cursor = conn.execute(sql)
    columns = [description[0] for description in (cursor.description or [])]
    return columns, cursor.fetchall()


def _result_to_str(rows: list[tuple]) -> str:
    """Flatten a result set to a normalised string for comparison."""
    return " | ".join(
        " ".join(str(v).strip() for v in row)
        for row in rows
    ).lower()


def _ascii_data_rows(answer: str) -> list[list[str]]:
    """Extract data rows from the SQL Tool's displayed ASCII result table."""
    lines = [line for line in answer.splitlines() if line.startswith("| ")]
    if len(lines) < 3:
        return []
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    header = cells[0]
    return [row for row in cells[1:] if len(row) == len(header)]


def _cell_matches(expected, actual) -> bool:
    """Compare one expected result cell with one displayed table cell."""
    try:
        expected_number = float(str(expected).replace(",", ""))
        actual_number = float(str(actual).replace(",", ""))
        return abs(expected_number - actual_number) <= max(0.01, 0.02 * abs(expected_number))
    except (TypeError, ValueError):
        return str(expected).strip().lower() == str(actual).strip().lower()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a string."""
    return [
        float(value.replace(",", ""))
        for value in re.findall(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text)
    ]


def _numbers_close(a: list[float], b: list[float], tol: float = 0.05) -> bool:
    """True if two numeric lists have the same length and all values within tol."""
    if len(a) != len(b):
        return False
    return all(
        abs(x - y) <= tol * max(1.0, abs(y))
        for x, y in zip(sorted(a), sorted(b))
    )


def _numeric_columns_present(gt_rows: list[tuple], columns: list[str], agent_answer: str,
                              tol: float = 0.02) -> tuple[bool, str]:
    """
    Check that every *business-significant* numeric value in the GT result set
    has a sufficiently close match somewhere in the agent answer.

    Skips:
      - None / NULL values
      - Small whole-number integers (<= 9999 with no decimal part) that are
        likely surrogate keys (holding_id, security_id, portfolio_id, etc.)
        A correct agent answer omits these in favour of human-readable names.

    Returns (ok: bool, reason: str).
    Uses relative tolerance *tol* (2 % by default).
    """
    agent_floats = _extract_numbers(agent_answer)

    missing = []
    for row in gt_rows:
        for index, val in enumerate(row):
            if val is None:
                continue
            column_name = columns[index].lower() if index < len(columns) else ""
            if column_name == "id" or column_name.endswith("_id"):
                continue
            try:
                gt_float = float(val)
            except (TypeError, ValueError):
                continue  # skip non-numeric cells

            matched = any(
                abs(gt_float - a) <= max(0.01, tol * abs(gt_float))
                for a in agent_floats
            )
            if not matched:
                missing.append(gt_float)

    if missing:
        sample = ", ".join(str(v) for v in missing[:5])
        suffix = f" (+{len(missing)-5} more)" if len(missing) > 5 else ""
        return False, f"Numeric values missing from answer: {sample}{suffix}"
    return True, ""


def _evaluate_text2sql(
    question_id: int,
    question: str,
    ground_truth: dict,
    agent_answer: str,
    gt_conn: sqlite3.Connection,
) -> dict:
    """
    For text2sql questions compare the agent's answer against the result set
    produced by the ground-truth SQL.

    Strategy:
      1. Run GT SQL -> get GT rows.
      2. Extract numbers from both GT result and agent answer.
      3. If single_value: check numeric match.
      4. If list/table: check that key terms from GT rows appear in agent answer.
    """
    gt_sql = ground_truth.get("sql_query", "")
    result_type = ground_truth.get("expected_result_type", "")

    try:
        columns, gt_rows = _run_gt_query(gt_sql, gt_conn)
    except sqlite3.Error as exc:
        return {
            "status": "ERROR",
            "reason": f"GT SQL execution failed: {exc}",
        }

    gt_str = _result_to_str(gt_rows)
    agent_lower = agent_answer.lower()

    # --- single value ---
    if result_type == "single_value":
        gt_nums = _extract_numbers(gt_str)
        agent_nums = _extract_numbers(agent_answer)
        if gt_nums and agent_nums:
            if _numbers_close(gt_nums[:1], agent_nums[:1]):
                return {"status": "PASS", "reason": "Numeric value matches GT."}
            else:
                return {
                    "status": "FAIL",
                    "reason": f"Expected ~{gt_nums[0]}, got {agent_nums[0]}.",
                }
        # Fallback: both empty or non-numeric
        if gt_str in agent_lower or agent_lower in gt_str:
            return {"status": "PASS", "reason": "Text match."}
        return {"status": "FAIL", "reason": f"No match. GT={gt_str!r}"}

    # --- list or table ---
    if not gt_rows:
        if "no result" in agent_lower or "not found" in agent_lower:
            return {"status": "PASS", "reason": "Ground truth and agent both indicate no results."}
        return {"status": "FAIL", "reason": "Ground truth returned no rows, but the answer did not indicate that."}

    displayed_rows = _ascii_data_rows(agent_answer)
    missing_rows = []
    for row_number, row in enumerate(gt_rows, start=1):
        expected_values = [
            value for index, value in enumerate(row)
            if value is not None
            and not (columns[index].lower() == "id" or columns[index].lower().endswith("_id"))
        ]
        if not expected_values:
            continue
        if displayed_rows:
            row_found = any(
                all(any(_cell_matches(value, cell) for cell in shown_row) for value in expected_values)
                for shown_row in displayed_rows
            )
        else:
            row_found = all(
                (str(value).strip().lower() in agent_lower if not isinstance(value, (int, float))
                 else any(abs(float(value) - number) <= max(0.01, 0.02 * abs(float(value)))
                          for number in _extract_numbers(agent_answer)))
                for value in expected_values
            )
        if not row_found:
            missing_rows.append(row_number)
    if missing_rows:
        return {
            "status": "FAIL",
            "reason": f"Missing expected result row(s): {', '.join(map(str, missing_rows))} of {len(gt_rows)}.",
        }

    # Require every expected non-ID value; this catches missing rows and fields.
    missing_by_column = {}
    for col_index, column in enumerate(columns):
        if column.lower() == "id" or column.lower().endswith("_id"):
            continue
        for row_index, row in enumerate(gt_rows):
            value = row[col_index]
            if value is None:
                continue
            if isinstance(value, (int, float)):
                present = any(
                    abs(float(value) - actual) <= max(0.01, 0.02 * abs(float(value)))
                    for actual in _extract_numbers(agent_answer)
                )
            else:
                present = str(value).strip().lower() in agent_lower
            if not present:
                missing_by_column.setdefault(column, []).append(row_index + 1)

    if missing_by_column:
        details = "; ".join(
            f"{column}: {len(rows)} missing row value(s)" 
            for column, rows in missing_by_column.items()
        )
        return {"status": "FAIL", "reason": f"Missing expected row/column values — {details}."}

    return {
        "status": "PASS",
        "reason": f"All {len(gt_rows)} expected row(s) and non-ID column values are represented.",
    }


def _evaluate_exposure(
    question_id: int,
    question: str,
    ground_truth: dict,
    agent_answer: str,
    gt_conn: sqlite3.Connection,
) -> dict:
    """
    For exposure_calculator questions check that:
      - The answer contains sector names.
      - Percentages are present and sum to ~100.
      - The named portfolio appears in the output.
    """
    portfolio_name = (
        ground_truth.get("parameters", {}).get("portfolio_name", "")
    )
    agent_lower = agent_answer.lower()

    if "error" in agent_lower and "not found" in agent_lower:
        return {
            "status": "FAIL",
            "reason": "Agent reported portfolio not found.",
        }

    # Portfolio name must appear
    if portfolio_name.lower() not in agent_lower:
        return {
            "status": "FAIL",
            "reason": f"Portfolio name '{portfolio_name}' missing from answer.",
        }

    # Compare every displayed sector percentage with the expected normalized weight.
    verify_sql = """
        SELECT sec.sector_name, SUM(h.current_weight) AS w
        FROM holdings h
        JOIN securities s  ON h.security_id = s.security_id
        JOIN sectors sec   ON s.sector_id    = sec.sector_id
        JOIN portfolios p  ON h.portfolio_id = p.portfolio_id
        WHERE p.portfolio_name = ?
          AND s.asset_type = 'Stock'
        GROUP BY sec.sector_name
    """
    try:
        rows = gt_conn.execute(verify_sql, (portfolio_name,)).fetchall()
        if not rows:
            return {"status": "FAIL", "reason": "Ground truth contains no equity sector weights."}
        total_weight = sum(float(row[1] or 0) for row in rows)
        if total_weight <= 0:
            return {"status": "FAIL", "reason": "Ground-truth equity weights sum to zero."}
        expected = {row[0]: float(row[1] or 0) / total_weight * 100 for row in rows}

        actual = {}
        for line in agent_answer.splitlines():
            match = re.match(r"^\s*(.+?)\s{2,}(-?\d+(?:\.\d+)?)\s*%\s*$", line)
            if match and "total" not in match.group(1).lower():
                actual[match.group(1).strip()] = float(match.group(2))

        missing = [sector for sector in expected if sector not in actual]
        unexpected = [sector for sector in actual if sector not in expected]
        mismatched = [
            f"{sector} expected {expected[sector]:.2f}%, got {actual[sector]:.2f}%"
            for sector in expected.keys() & actual.keys()
            if abs(expected[sector] - actual[sector]) > 0.03
        ]
        if missing or unexpected or mismatched:
            reasons = []
            if missing:
                reasons.append("missing sectors: " + ", ".join(missing))
            if unexpected:
                reasons.append("unexpected sectors: " + ", ".join(unexpected))
            if mismatched:
                reasons.append("percentage mismatch: " + "; ".join(mismatched))
            return {"status": "FAIL", "reason": ". ".join(reasons) + "."}
        actual_total = sum(actual.values())
        if abs(actual_total - 100.0) > 0.03 * len(expected):
            return {"status": "FAIL", "reason": f"Sector percentages sum to {actual_total:.2f}%, expected 100%."}
        return {
            "status": "PASS",
            "reason": f"All {len(expected)} sector percentages match normalized ground truth.",
        }
    except sqlite3.Error as exc:
        return {"status": "ERROR", "reason": f"Could not verify expected exposures ({type(exc).__name__})."}


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def run_evaluation(verbose: bool = False) -> None:
    if not os.path.exists(GT_PATH):
        print(f"Error: Ground truth file not found at '{GT_PATH}'")
        sys.exit(1)

    with open(GT_PATH, "r", encoding="utf-8") as fh:
        ground_truth_data = json.load(fh)

    questions = ground_truth_data.get("questions", [])
    if not questions:
        print("Error: No questions found in ground truth dataset.")
        sys.exit(1)

    try:
        gt_conn = connect_read_only(DB_PATH)
    except (OSError, sqlite3.Error) as exc:
        print(f"Error: Could not open the database read-only ({type(exc).__name__}).")
        sys.exit(1)

    results = []

    print("\n" + "=" * 70)
    print("  Portfolio Analytics Agent — Evaluation Report")
    print("=" * 70)

    with PortfolioAgent(DB_PATH) as agent:
        for item in questions:
            qid = item["id"]
            qtype = item["type"]
            difficulty = item.get("difficulty", "?")
            question = item["question"]
            gt = item["ground_truth"]

            print(f"\n[Q{qid:02d}] ({qtype} / {difficulty})")
            print(f"  Question : {question}")

            # Run the agent
            try:
                agent_answer = agent.answer(question)
            except Exception as exc:
                log.error("Agent failed for question %s (%s).", qid, type(exc).__name__)
                agent_answer = "AGENT ERROR: The request could not be completed."

            if verbose:
                print(f"  Answer   :\n{agent_answer}\n")

            # Evaluate
            if qtype == "text2sql":
                eval_result = _evaluate_text2sql(
                    qid, question, gt, agent_answer, gt_conn
                )
            elif qtype == "exposure_calculator":
                eval_result = _evaluate_exposure(
                    qid, question, gt, agent_answer, gt_conn
                )
            else:
                eval_result = {"status": "SKIP", "reason": f"Unknown type: {qtype}"}

            status = eval_result["status"]
            reason = eval_result["reason"]
            symbol = {"PASS": "✓", "FAIL": "✗", "ERROR": "!", "SKIP": "~"}.get(status, "?")
            print(f"  Result   : [{symbol}] {status}  —  {reason}")

            results.append({
                "id": qid,
                "type": qtype,
                "difficulty": difficulty,
                "status": status,
            })

    gt_conn.close()

    # Summary
    passed  = sum(1 for r in results if r["status"] == "PASS")
    failed  = sum(1 for r in results if r["status"] == "FAIL")
    errors  = sum(1 for r in results if r["status"] == "ERROR")
    total   = len(results)
    accuracy = passed / total * 100 if total else 0

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total questions : {total}")
    print(f"  PASS            : {passed}")
    print(f"  FAIL            : {failed}")
    print(f"  ERROR           : {errors}")
    print(f"  Accuracy        : {accuracy:.1f}%")
    print("=" * 70)

    # Per-difficulty breakdown
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            p = sum(1 for r in subset if r["status"] == "PASS")
            print(f"  {diff.capitalize():<10} : {p}/{len(subset)} passed")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio Analytics Agent Evaluator")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print each agent answer in full.",
    )
    args = parser.parse_args()

    run_evaluation(verbose=args.verbose)


if __name__ == "__main__":
    main()
