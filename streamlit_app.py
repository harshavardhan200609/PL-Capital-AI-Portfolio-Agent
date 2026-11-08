"""Chat-style Streamlit interface for the Portfolio Analytics Agent."""

import hashlib
import html
import re
from collections import OrderedDict

import streamlit as st

from agent import (
    DB_PATH,
    _KNOWN_PORTFOLIOS,
    PortfolioAgent,
    _conversational_response,
)


WELCOME_MESSAGE = (
    "Hello! I’m your portfolio analytics assistant. Ask me about holdings, "
    "securities, AUM, performance, risk, diversification, or equity-only sector exposure."
)
CACHE_LIMIT = 32
MAX_CONTEXT_LENGTH = 4000


_CUSTOM_CSS = """
<style>
:root { color-scheme: dark; }
:root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
  background: #101113 !important; color: #E9EAEC !important;
}
[data-testid="stSidebar"] { display: none; }
/* Hide Streamlit's menu/toolbar so its theme selector is not exposed. */
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { height: 2.5rem; background: #101113 !important; }
main .block-container { max-width: 1040px; padding: 3.25rem 1.5rem 8.5rem; }
.brand { margin: .3rem 0 2.3rem; }
.brand-kicker { color: #9A9CA2; font-size: .72rem; letter-spacing: .2em; text-transform: uppercase; }
.brand-title { color: #F5F5F6; font-size: clamp(2rem, 4vw, 2.65rem); font-weight: 680; letter-spacing: -.035em; line-height: 1.15; margin: .65rem 0 .55rem; }
.brand-subtitle { color: #A5A7AD; font-size: 1rem; line-height: 1.6; max-width: 42rem; }
.user-row { display: flex; justify-content: flex-end; margin: 1.25rem 0 1.8rem; width: 100%; }
.user-bubble { max-width: min(78%, 680px); padding: .78rem 1.08rem; border: 1px solid #303136; border-radius: 1.25rem; background: #25262A; color: #F2F2F3; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
.assistant-row { display: flex; justify-content: flex-start; width: 100%; margin: 1.3rem 0 2.1rem; }
.assistant-content { width: min(100%, 820px); color: #E9EAEC; line-height: 1.7; }
.assistant-label { color: #8F9198; font-size: .68rem; letter-spacing: .14em; margin: 0 0 .55rem; }
.assistant-content [data-testid="stMarkdownContainer"] p { margin-bottom: .75rem; }
.typing { display: inline-flex; align-items: center; gap: .35rem; padding: .55rem .8rem; border: 1px solid #2D2E33; border-radius: 1rem; background: #191A1D; }
.typing span { width: .38rem; height: .38rem; border-radius: 50%; background: #A3A5AB; animation: typing 1.1s infinite ease-in-out; }
.typing span:nth-child(2) { animation-delay: .15s; }
.typing span:nth-child(3) { animation-delay: .3s; }
@keyframes typing { 0%, 60%, 100% { opacity: .35; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-3px); } }
/* Streamlit's chat input is fixed by the component; this creates a padded dock. */
[data-testid="stChatInput"] { position: fixed; z-index: 100; bottom: 0; left: 0; width: 100%; box-sizing: border-box; border: 0 !important; background: linear-gradient(180deg, rgba(16,17,19,0), #101113 24%); padding: 1.3rem max(1.5rem, calc((100vw - 1040px) / 2)) 1rem !important; }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] form { border: 0 !important; outline: 0 !important; background: transparent !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea { min-height: 3.1rem; color: #F1F1F2 !important; background: #202125 !important; border: 1px solid #3A3B40 !important; border-radius: 1.2rem !important; box-shadow: none !important; }
[data-testid="stChatInput"] textarea:focus, [data-testid="stChatInput"] textarea:focus-visible { border-color: #686A70 !important; outline: none !important; box-shadow: 0 0 0 2px rgba(155,157,164,.13) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #92949B !important; }
[data-testid="stChatInput"] button { color: #E4E5E7 !important; background: #37383D !important; border: 1px solid #48494F !important; border-radius: .8rem !important; }
[data-testid="stChatInput"] button:hover { background: #45464C !important; }
[data-testid="stDataFrame"] { overflow: hidden; border: 1px solid #34353B; border-radius: .75rem; background: #17181B; }
[data-testid="stDataFrame"] iframe { color-scheme: dark; }
[data-testid="stExpander"] { border: 1px solid #34353A; border-radius: .8rem; background: #17181B; }
[data-testid="stExpander"] summary { color: #E6E7E9; }
[data-testid="stCode"] { border: 1px solid #303137; border-radius: .65rem; }
[data-testid="stMetric"] { padding: .7rem .85rem; border: 1px solid #34353A; border-radius: .7rem; background: #191A1D; }
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] { border: 1px solid #34353A; border-radius: .75rem; background: #17181B; padding: .4rem; }
.stMarkdown, .stCaption { color: #D8D9DC; }
@media (max-width: 640px) {
  main .block-container { padding: 2.5rem 1rem 8rem; }
  .brand { margin-bottom: 1.7rem; }
  .user-bubble { max-width: 90%; }
  [data-testid="stChatInput"] { padding: 1rem 1rem .8rem !important; }
}
</style>
"""


def parse_ascii_table(answer: str):
    """Parse the SQL Tool's ASCII result table into headers and data rows."""
    lines = [line for line in answer.splitlines() if line.startswith("| ")]
    if len(lines) < 2:
        return None
    cells = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    headers = cells[0]
    rows = [row for row in cells[1:] if len(row) == len(headers)]
    return (headers, rows) if rows else None


def parse_exposures(answer: str):
    """Extract sector percentages from Exposure Calculator output."""
    values = []
    for line in answer.splitlines():
        match = re.match(r"^\s*(.+?)\s{2,}(\d+(?:\.\d+)?)\s*%\s*$", line)
        if match and "total" not in match.group(1).lower():
            values.append((match.group(1).strip(), float(match.group(2))))
    return values


def format_result_cell(header: str, value: str) -> str:
    """Add readable grouping and precision without guessing a currency symbol."""
    text = str(value).strip()
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return text
    try:
        number = float(text)
    except ValueError:
        return text
    name = header.casefold()
    if any(token in name for token in ("id", "count", "quantity", "volume", "holdings")):
        return f"{int(number):,}" if number.is_integer() else text
    precision = 2 if any(token in name for token in ("aum", "value", "price", "cost", "amount", "nav", "fee")) else 4
    return f"{number:,.{precision}f}"


def resolve_portfolio_followup(question: str, messages: list[dict]) -> str:
    """Resolve explicit possessive follow-ups only from one recent named portfolio."""
    if not re.search(r"\b(?:its|that portfolio)\b", question, re.IGNORECASE):
        return question
    for message in reversed(messages):
        if message.get("role") != "user" or message.get("local"):
            continue
        prior_text = message.get("content", "")
        mentioned = [name for name in _KNOWN_PORTFOLIOS if name.casefold() in prior_text.casefold()]
        if not mentioned:
            continue
        # A single prior question may compare portfolios; in that case the
        # referent is not clear enough to expand automatically.
        if len(mentioned) != 1:
            return question
        name = mentioned[0]
        expanded = re.sub(r"\bits\b", f"{name}'s", question, flags=re.IGNORECASE)
        expanded = re.sub(r"\bthat portfolio\b", name, expanded, flags=re.IGNORECASE)
        return expanded
    return question


def build_contextual_question(question: str, messages: list[dict]) -> str:
    """Attach a small recent conversation window while staying under input limits."""
    previous = [
        message for message in messages
        if not message.get("welcome") and not message.get("local")
    ]
    # Repeated identical questions should reuse their cached response in the
    # same context instead of adding the just-completed exchange to the prompt.
    if (
        len(previous) >= 2
        and previous[-2].get("role") == "user"
        and previous[-2].get("content", "").strip().casefold() == question.strip().casefold()
        and previous[-1].get("role") == "assistant"
    ):
        previous = previous[:-2]
    previous = previous[-4:]

    if not previous:
        return question

    label = "Recent conversation (use only to resolve follow-up references):\n"
    resolved_question = resolve_portfolio_followup(question, previous)
    ending = f"\n\nCurrent question: {resolved_question}"
    # Keep the added conversation context bounded while preserving the full
    # current question. Long questions simply travel without prior context.
    available = MAX_CONTEXT_LENGTH - len(label) - len(ending) - len(question)
    if available <= 0:
        return question

    entries = []
    for message in previous:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = re.sub(r"\s+", " ", message.get("content", "")).strip()
        entries.append(f"{role}: {content}")
    context = "\n".join(entries)
    if len(context) > available:
        context = context[-available:]
    return f"{label}{context}{ending}"


def cache_key(prompt: str) -> str:
    """Return a compact key without retaining prompt text as the cache key."""
    normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def cache_response(cache: OrderedDict, key: str, answer: str) -> None:
    """Store a result in a bounded per-session least-recently-used cache."""
    cache[key] = answer
    cache.move_to_end(key)
    while len(cache) > CACHE_LIMIT:
        cache.popitem(last=False)


def is_cacheable(answer: str) -> bool:
    """Allow retries for failures instead of caching transient errors."""
    lowered = answer.strip().lower()
    return not lowered.startswith((
        "error:",
        "i couldn't",
        "the database couldn't",
        "the database could not",
        "the request could not",
        "portfolio database not found",
    ))


def get_cached_answer(cache: OrderedDict, key: str, make_answer):
    """Return a cached answer payload, invoking and caching a miss once."""
    if key in cache:
        cache.move_to_end(key)
        cached = cache[key]
        if isinstance(cached, str):
            cached = {"answer": cached, "tool_details": []}
        return cached, True
    payload = make_answer()
    if isinstance(payload, str):
        payload = {"answer": payload, "tool_details": []}
    if is_cacheable(payload.get("answer", "")):
        cache_response(cache, key, payload)
    return payload, False


def render_answer(answer: str) -> None:
    """Render natural-language text, SQL tables, and exposure charts."""
    table = parse_ascii_table(answer)
    display_text = answer
    if table:
        first_border = answer.find("+-")
        display_text = answer[:first_border].strip()
    if display_text:
        st.markdown(display_text)

    if table:
        headers, rows = table
        formatted_rows = [
            {
                header: format_result_cell(header, row[index])
                for index, header in enumerate(headers)
            }
            for row in rows
        ]
        st.dataframe(
            formatted_rows,
            use_container_width=True,
            hide_index=True,
        )

    exposures = parse_exposures(answer)
    if exposures:
        st.caption("Equity holdings only · bonds excluded")
        st.bar_chart(
            {
                "Sector": [sector for sector, _ in exposures],
                "Exposure (%)": [percentage for _, percentage in exposures],
            },
            x="Sector",
            y="Exposure (%)",
        )


def render_tool_details(tool_details: list[dict]) -> None:
    """Show captured backend traces only for tools that actually ran."""
    for detail in tool_details:
        tool = detail.get("tool")
        status = detail.get("execution_status", "unknown").replace("_", " ").title()
        if tool == "sql_tool":
            with st.expander("View SQL Query", expanded=False):
                st.caption(f"Execution status: {status}")
                sql = detail.get("sql")
                if sql:
                    st.code(sql, language="sql")
                else:
                    st.info("No SQL query was generated, so nothing was executed.")
        elif tool == "exposure_calculator":
            with st.expander("View Exposure Calculation", expanded=False):
                st.caption(f"Calculation status: {status}")
                if detail.get("portfolio_name"):
                    st.write(f"**Portfolio:** {detail['portfolio_name']}")
                st.write(
                    f"**Included:** {detail.get('included_asset_type', 'Stock')} holdings  ·  "
                    f"**Excluded:** {detail.get('excluded_asset_type', 'Bond')} holdings"
                )
                st.write(f"**Weight source:** `{detail.get('weight_source', 'holdings.current_weight')}`")
                if "total_equity_weight" in detail:
                    st.metric("Total equity weight before normalization", f"{detail['total_equity_weight']:.4f}")
                sector_weights = detail.get("sector_weights") or []
                if sector_weights:
                    st.caption("Sector weights used")
                    st.dataframe(sector_weights, use_container_width=True, hide_index=True)
                exposures = detail.get("exposures") or []
                if exposures:
                    st.caption("Normalized exposure percentages")
                    st.dataframe(exposures, use_container_width=True, hide_index=True)


def render_message(message: dict) -> None:
    """Place user messages on the right and assistant messages on the left."""
    if message.get("role") == "user":
        safe_content = html.escape(message.get("content", ""))
        st.markdown(
            f'<div class="user-row"><div class="user-bubble">{safe_content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    column, spacer = st.columns([8.5, 1.5])
    with column:
        st.markdown('<div class="assistant-label">PORTFOLIO ASSISTANT</div>', unsafe_allow_html=True)
        with st.container():
            render_answer(message.get("content", ""))
            render_tool_details(message.get("tool_details", []))


def render_typing_indicator() -> None:
    """Display a small animated typing indicator without Streamlit's spinner."""
    st.markdown(
        '<div class="assistant-row"><div class="assistant-content">'
        '<div class="assistant-label">PORTFOLIO ASSISTANT</div>'
        '<div class="typing" aria-label="Assistant is typing">'
        '<span></span><span></span><span></span></div></div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="PL Capital | Portfolio Assistant",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="brand">'
        '<div class="brand-kicker">PL CAPITAL · PORTFOLIO INTELLIGENCE</div>'
        '<div class="brand-title">Portfolio Assistant</div>'
        '<div class="brand-subtitle">Ask a question. Get a clear answer from your portfolio data.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "welcome": True}
        ]
    if "response_cache" not in st.session_state:
        st.session_state.response_cache = OrderedDict()

    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Message your portfolio assistant…")
    if not question:
        return

    question = question.strip()
    if not question:
        return

    previous_messages = list(st.session_state.messages)
    contextual_question = build_contextual_question(question, previous_messages)
    is_local_conversation = _conversational_response(question) is not None
    key = cache_key(contextual_question)
    st.session_state.messages.append({
        "role": "user", "content": question, "local": is_local_conversation,
    })
    # Render the submitted question in this run before any potentially slow
    # routing or provider request, so it appears immediately after Enter.
    render_message(st.session_state.messages[-1])

    render_typing_indicator()
    try:
        def make_answer():
            effective_question = question if is_local_conversation else contextual_question
            with PortfolioAgent(DB_PATH) as portfolio_agent:
                return portfolio_agent.answer_with_details(effective_question)

        answer_payload, _cache_hit = get_cached_answer(
            st.session_state.response_cache, key, make_answer
        )
    except FileNotFoundError:
        answer_payload = {
            "answer": "Portfolio database not found. Contact the project owner to restore it.",
            "tool_details": [],
        }
    except Exception as exc:
        answer_payload = {
            "answer": f"The request could not be completed ({type(exc).__name__}). Please try again.",
            "tool_details": [],
        }

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_payload.get("answer", ""),
        "tool_details": answer_payload.get("tool_details", []),
        "local": is_local_conversation,
    })
    st.rerun()


if __name__ == "__main__":
    main()
