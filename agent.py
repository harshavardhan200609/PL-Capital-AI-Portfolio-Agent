"""
agent.py
--------
Portfolio Analytics Agent — command-line interface.

The agent accepts a natural-language question, uses the configured provider
to decide which tool to invoke, calls that tool, and prints the result.

Usage (interactive):
    python agent.py

Usage (single question):
    python agent.py --question "How many portfolios do we have?"
    python agent.py --question "What are the sector exposures for Tech Innovation Fund?"
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time

from dotenv import load_dotenv

from sql_tool import run_sql_query
from exposure_tool import calculate_exposure
from db_utils import connect_read_only

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "portfolio_database.db")

from llm_client import llm_call as _llm_call

# ---------------------------------------------------------------------------
# Tool-routing prompt
# ---------------------------------------------------------------------------

_ROUTING_PROMPT = """
You are a routing agent for a portfolio analytics system. Choose one tool unless
the question genuinely requires both SQL data and a sector exposure calculation.

1. sql_tool
   Use for: questions about portfolio data that require querying the database —
   counts, lists, aggregations, rankings, comparisons, joins across tables, etc.
   Examples: "How many portfolios do we have?", "Show top 5 holdings by cost basis."

2. exposure_calculator
   Use for: questions that ask about sector exposure, sector allocation, or
   sector breakdown for a specific portfolio. Bonds are excluded; equities only.
   Examples: "What are the sector exposures for Tech Innovation Fund?",
             "Calculate sector exposure breakdown for International Equity."

3. hybrid
   Use only when the question requires both a database result and sector
   exposure for the same portfolio. Return both findings as separate parts.
   Example: "Show the holdings market value and sector exposure for Growth Equity Fund."

Known portfolio names (use EXACT spelling):
- Growth Equity Fund
- Conservative Income Fund
- Tech Innovation Fund
- Balanced Portfolio
- ESG Sustainable Fund
- Small Cap Value Fund
- International Equity Fund
- Fixed Income Plus
- Dividend Aristocrats Fund
- Emerging Markets Fund
- Total Stock Market Index Fund
- Total Bond Market Index Fund
- Total International Index Fund

Respond with a JSON object containing exactly these two keys:
  "tool"            : "sql_tool", "exposure_calculator", "hybrid", or "unsupported"
  "portfolio_name"  : the EXACT portfolio name from the list above if tool is
                      "exposure_calculator" or "hybrid", otherwise null.
                      Match the user's mention to the closest exact name.

If the question is ambiguous between the two, default to "sql_tool".
If the question is completely unrelated to portfolio analytics, set
  "tool": "unsupported"

Return ONLY the raw JSON — no markdown fences, no explanation.

USER QUESTION: {question}
"""


# ---------------------------------------------------------------------------
# Known portfolio names (fallback for name extraction)
# ---------------------------------------------------------------------------

_KNOWN_PORTFOLIOS = [
    "Growth Equity Fund",
    "Conservative Income Fund",
    "Tech Innovation Fund",
    "Balanced Portfolio",
    "ESG Sustainable Fund",
    "Small Cap Value Fund",
    "International Equity Fund",
    "Fixed Income Plus",
    "Dividend Aristocrats Fund",
    "Emerging Markets Fund",
    "Total Stock Market Index Fund",
    "Total Bond Market Index Fund",
    "Total International Index Fund",
]


# Local responses for standalone conversational messages. Keep these checks
# ahead of the LLM router so simple conversation never makes an API call.
_CONVERSATIONAL_RESPONSES = (
    (re.compile(r"^(?:h+i+|h+e+l+o+|h+e+y+)(?:\s+there)?[!.]*$", re.IGNORECASE),
     "Hello! I can help answer questions about your portfolio data."),
    (re.compile(
        r"^(?:what\s+can\s+(?:you|u)\s+do|what\s+do\s+(?:you|u)\s+do|"
        r"how\s+can\s+(?:you|u)\s+help(?:\s+me)?|"
        r"what\s+can\s+(?:you|u)\s+help\s+me\s+with|"
        r"what\s+are\s+you\s+able\s+to\s+do|"
        r"what\s+are\s+(?:your|ur)\s+capabilities)[?.!]*$",
        re.IGNORECASE,
    ),
     "I can answer questions about portfolios, holdings, securities, sectors, "
     "AUM, performance, transactions, risk metrics, and diversification. "
     "I can also calculate a portfolio's sector exposure using equity holdings; "
     "bonds are excluded from that calculation."),
    (re.compile(
        r"^(?:how\s+do\s+you\s+work|how\s+does\s+this\s+work|"
        r"how\s+will\s+you\s+answer\s+my\s+questions|"
        r"how\s+do\s+you\s+answer\s+questions|how\s+will\s+this\s+work)[?.!]*$",
        re.IGNORECASE,
    ),
     "I route portfolio data questions to a database query tool, and sector "
     "exposure questions to a calculator that uses equity holdings only. "
     "Greetings, capability questions, thanks, and goodbyes get local responses."),
    (re.compile(
        r"^(?:thanks?(?:\s+(?:so\s+much|a\s+lot|for\s+your\s+help))?|"
        r"thank\s+you(?:\s+(?:so\s+much|for\s+your\s+help))?|"
        r"got\s+it|okay|ok|understood|great|awesome)[!.]*$",
        re.IGNORECASE,
    ),
     "You're welcome! Let me know if you have another portfolio question."),
    (re.compile(r"^(?:bye|goodbye|see\s+you|see\s+you\s+later|farewell)[!.]*$", re.IGNORECASE),
     "Goodbye!"),
)


def _conversational_response(message: str) -> str | None:
    """Return a predefined response for a standalone conversational message."""
    normalized = re.sub(r"\s+", " ", message).strip().lower()
    normalized = re.sub(r"[.!?…]+", " | ", normalized)
    normalized = re.sub(r"\b(?:u|ur)\b", lambda m: "you" if m[0] == "u" else "your", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,|\t")

    # A greeting may precede a standalone conversation request. Leave the
    # remainder intact for matching so an analytics question still routes on.
    without_greeting = re.sub(
        r"^(?:h+i+|h+e+l+o+|h+e+y+)(?:\s+there)?(?:\s*[,|]\s*|\s+)+",
        "",
        normalized,
        count=1,
        flags=re.IGNORECASE,
    ).strip(" ,|\t")
    if without_greeting:
        normalized = without_greeting

    # Conversational questions are often phrased as a short series, such as
    # "tell me what can u do and what are ur capabilities". Expand common
    # polite/filler prefixes, then accept only if every clause is local chat.
    clauses = [
        part.strip(" ,|\t")
        for part in re.split(r"\s+(?:and|also)\s+|[,|]", normalized)
        if part.strip(" ,|\t")
    ]
    if not clauses:
        clauses = [normalized]
    matched = []
    for clause in clauses:
        clause = re.sub(
            r"^(?:(?:please\s+)?(?:tell\s+me\s+|can\s+you\s+tell\s+me\s+|i\s+want\s+to\s+know\s+))*",
            "",
            clause,
        ).strip()
        response = next(
            (response for pattern, response in _CONVERSATIONAL_RESPONSES
             if pattern.fullmatch(clause)),
            None,
        )
        if response is None:
            matched = []
            break
        matched.append(response)
    if matched:
        capability_response = _CONVERSATIONAL_RESPONSES[1][1]
        return capability_response if capability_response in matched else matched[-1]

    # Greetings on their own remain local; mixed greeting plus analytics does
    # not match and continues through the existing portfolio router.
    if re.fullmatch(r"(?:h+i+|h+e+l+o+|h+e+y+)(?:\s+there)?", normalized, re.IGNORECASE):
        return _CONVERSATIONAL_RESPONSES[0][1]
    return None


def _extract_portfolio_name(question: str) -> str:
    """Return a portfolio name found verbatim (or by partial match) in *question*."""
    q = question.lower()
    # Exact match first
    for name in _KNOWN_PORTFOLIOS:
        if name.lower() in q:
            return name
    # Partial / fuzzy: find the portfolio whose words best overlap the question
    best_name, best_score = "", 0
    for name in _KNOWN_PORTFOLIOS:
        words = set(name.lower().split())
        # Exclude generic words
        words -= {"fund", "portfolio", "index"}
        if not words:
            continue
        score = sum(1 for w in words if w in q)
        if score > best_score:
            best_score, best_name = score, name
    return best_name if best_score >= 1 else ""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _route(question: str) -> dict:
    """Ask the configured provider which tool to use and return its JSON decision."""
    prompt = _ROUTING_PROMPT.format(question=question)
    raw = _llm_call(prompt)

    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*",     "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"```$",        "", raw).strip()

    try:
        decision = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("Router returned invalid JSON; using the SQL tool fallback.")
        if re.search(r'"tool"\s*:\s*"hybrid"', raw, re.IGNORECASE):
            decision = {"tool": "hybrid", "portfolio_name": None}
        elif "exposure_calculator" in raw:
            decision = {"tool": "exposure_calculator", "portfolio_name": None}
        else:
            decision = {"tool": "sql_tool", "portfolio_name": None}

    if not isinstance(decision, dict):
        return {"tool": "sql_tool", "portfolio_name": None}
    return decision


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PortfolioAgent:
    """Thin orchestrator that routes questions to the correct tool."""

    def __init__(self, db_path: str = DB_PATH):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Database not found at '{db_path}'.\n"
                "Run 'python setup_database.py' first."
            )
        self.conn = connect_read_only(db_path)
        log.info("Agent opened the portfolio database read-only.")

    def close(self):
        self.conn.close()

    def answer(self, question: str) -> str:
        """Route *question* to the right tool and return the answer string."""
        return self.answer_with_details(question)["answer"]

    def answer_with_details(self, question: str) -> dict:
        """Answer a question and return details only for tools actually invoked."""
        def payload(result, fallback_tool: str | None = None) -> tuple[str, list[dict]]:
            if isinstance(result, dict) and "answer" in result:
                return result["answer"], result.get("tool_details", [])
            details = []
            if fallback_tool:
                details = [{"tool": fallback_tool, "execution_status": "unknown"}]
            return str(result), details

        if not isinstance(question, str):
            return {"answer": "Please enter your question as text.", "tool_details": []}
        question = question.strip()
        if not question:
            return {"answer": "Please enter a question.", "tool_details": []}
        conversational = _conversational_response(question)
        if conversational is not None:
            return {"answer": conversational, "tool_details": []}

        started = time.perf_counter()

        try:
            decision = _route(question)
        except Exception as exc:
            log.error("Routing failed (%s).", type(exc).__name__)
            return {
                "answer": "I couldn't route that request right now. Please try again shortly.",
                "tool_details": [],
            }

        tool = decision.get("tool", "sql_tool")
        portfolio_name = decision.get("portfolio_name") or ""
        if not isinstance(portfolio_name, str):
            portfolio_name = ""
        if tool not in {"sql_tool", "exposure_calculator", "hybrid", "unsupported"}:
            log.warning("Router selected an unknown tool; using SQL fallback.")
            tool = "sql_tool"
        log.info("Selected tool: %s", tool)

        if tool == "exposure_calculator":
            if not portfolio_name:
                portfolio_name = _extract_portfolio_name(question)
            if not portfolio_name:
                return {"answer": (
                    "I couldn't identify which portfolio you're asking about.\n"
                    "Please include the portfolio name in your question, e.g.\n"
                    "'What are the sector exposures for Tech Innovation Fund?'"
                ), "tool_details": []}
            answer, tool_details = payload(
                calculate_exposure(portfolio_name, self.conn, return_details=True),
                "exposure_calculator",
            )
            log.info("Exposure calculation completed in %.3fs.", time.perf_counter() - started)
            return {"answer": answer, "tool_details": tool_details}

        elif tool == "sql_tool":
            answer, tool_details = payload(
                run_sql_query(question, self.conn, return_details=True),
                "sql_tool",
            )
            log.info("SQL request completed in %.3fs.", time.perf_counter() - started)
            return {"answer": answer, "tool_details": tool_details}

        elif tool == "hybrid":
            if not portfolio_name:
                portfolio_name = _extract_portfolio_name(question)
            if not portfolio_name:
                return {
                    "answer": "Please include the portfolio name for the sector exposure part of that question.",
                    "tool_details": [],
                }
            sql_answer, sql_details = payload(
                run_sql_query(question, self.conn, return_details=True),
                "sql_tool",
            )
            exposure_answer, exposure_details = payload(
                calculate_exposure(portfolio_name, self.conn, return_details=True),
                "exposure_calculator",
            )
            log.info("Hybrid request completed in %.3fs.", time.perf_counter() - started)
            return {
                "answer": (
                    f"Portfolio data findings:\n{sql_answer}\n\n"
                    f"Sector exposure for {portfolio_name}:\n{exposure_answer}"
                ),
                "tool_details": sql_details + exposure_details,
            }

        elif tool == "unsupported":
            log.info("Unsupported request handled locally in %.3fs.", time.perf_counter() - started)
            return {"answer": (
                "I can help with portfolio holdings, securities, sectors, AUM, "
                "performance, transactions, risk, diversification, and equity-only "
                "sector exposure."
            ), "tool_details": []}
        else:
            log.info("Unsupported request handled locally in %.3fs.", time.perf_counter() - started)
            return {"answer": (
                "I can help with portfolio holdings, securities, sectors, AUM, "
                "performance, transactions, risk, diversification, and equity-only "
                "sector exposure."
            ), "tool_details": []}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_interactive(agent: PortfolioAgent) -> None:
    print("\nPortfolio Analytics Agent")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not question:
            continue
        answer = agent.answer(question)
        print(f"\nAgent:\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Portfolio Analytics Agent",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Ask a single question and exit.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        with PortfolioAgent() as agent:
            if args.question:
                answer = agent.answer(args.question)
                print(answer)
            else:
                _run_interactive(agent)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except sqlite3.Error:
        log.exception("Could not open the portfolio database.")
        print("Error: Could not open the portfolio database.")
        sys.exit(1)


if __name__ == "__main__":
    main()
