# PL Capital Portfolio Analytics Agent

## Overview

This project answers natural-language questions about the supplied portfolio dataset. It combines an LLM-routed agent, a read-only SQLite Text-to-SQL tool, and an equity-only sector exposure calculator. The command-line interface and optional Streamlit chat interface use the same `PortfolioAgent` backend.

The solution is designed to demonstrate tool routing, safe database access, portfolio analytics, grounded result summaries, and evaluation against the supplied ground truth.

## Architecture and workflow

1. The CLI or Streamlit interface receives a question. The Streamlit interface keeps conversation messages in session state and maintains a bounded in-memory response cache. It has limited follow-up resolution for explicit references such as “its” or “that portfolio”: it substitutes a portfolio name when a single recognized portfolio is present in recent user context. If the context is missing or names more than one portfolio, it leaves the reference unchanged. This does not provide general conversation understanding for arbitrary pronouns.
2. Greetings, thanks, capability questions, and related simple conversation receive local predefined answers without routing to an LLM.
3. Other questions are sent to the configured provider. The agent asks it to select `sql_tool`, `exposure_calculator`, `hybrid`, or `unsupported`.
4. The SQL Tool generates or selects a query, validates it, runs it against a read-only SQLite connection, and returns a result table with a concise result-grounded summary when available.
5. The Exposure Calculator queries equity holdings, excludes bonds, normalizes sector weights over the equity portion, and reports the sector percentages.
6. Hybrid routing runs both tools and combines their results. Unsupported requests receive a local explanation of the assistant's portfolio analytics scope.

The common SQL summaries and exposure explanations are derived from actual returned rows and values. For other result shapes the SQL Tool may use the configured provider to summarize the returned table. The result table remains part of the SQL answer.

## Technology stack

- Python (3.10 or newer recommended)
- SQLite and Python's built-in `sqlite3`
- Gemini via `google-genai`
- NVIDIA NIM via the OpenAI-compatible `openai` client
- `python-dotenv` for environment loading
- Pandas for database setup/CSV ingestion
- Optional Streamlit UI

## Project structure

```text
data/                    Supplied CSV input files
agent.py                 Conversational behavior, LLM routing, CLI
sql_tool.py              SQL generation, safety validation, execution, summaries
exposure_tool.py         Equity-only sector exposure calculation
db_utils.py              Read-only SQLite connections and authorizer
llm_client.py            Gemini/NVIDIA provider abstraction
streamlit_app.py         Optional chat UI, session memory, UI response cache
setup_database.py        Rebuilds the SQLite database from schema and CSVs
database_schema.sql      Database tables, relationships, and indexes
portfolio_database.db    Included project database, ready to use after cloning
ground_truth_dataset.json Company-provided evaluation questions and ground truth
evaluator.py             Evaluates agent answers against the ground truth
test_offline.py          Offline regression and safety tests
requirements.txt         Core dependencies
requirements-streamlit.txt Core dependencies plus Streamlit
.env.example             Environment-variable template
```

## SQLite database

The schema defines `portfolios`, `holdings`, `securities`, `sectors`, `benchmarks`, `transactions`, `historical_prices`, `portfolio_performance`, and `risk_metrics`. Portfolio holdings join to securities; securities may join to sectors. The schema includes foreign keys, uniqueness constraints, and indexes for common joins and lookups.

The repository includes `portfolio_database.db`, so the CLI, Streamlit app, and evaluator can use the database immediately after cloning. The application opens it in SQLite URI read-only mode, enables `PRAGMA query_only`, and installs an authorizer that permits reads while denying writes, schema changes, and other actions.

Rebuilding is optional. `setup_database.py` deletes the existing database and recreates it from the supplied CSVs and schema. Run it only when you intentionally want to rebuild:

```powershell
py setup_database.py
```

## SQL Query Tool

`sql_tool.py` handles portfolio questions that require database queries, such as counts, lists, aggregations, rankings, comparisons, and joins. `generate_sql()` first checks a set of deterministic templates for common, unambiguous patterns. These include portfolio counts, active portfolio names, high-risk AUM, top holdings by cost basis, sector securities and average prices, per-portfolio sector value/percentage, diversification thresholds, and clear named-portfolio holdings lists. Questions that do not match a template use the configured LLM provider to generate SQL. The templates cover only those patterns; they are not a general SQL replacement. The SQL validator accepts a single `SELECT` or `WITH` query; execution protections independently enforce read-only access. Database and provider errors are returned as concise messages.

The tool formats SQL result tables and creates concise summaries from returned values for common aggregates, lists, and comparisons. It can include the actual executed SQL and execution status in tool details for the UI. It does not execute SQL displayed only as text.

## Sector Exposure Calculator

`exposure_tool.py` calculates exposure from `holdings.current_weight`, grouped by sector. It includes securities whose `asset_type` is `Stock`, excludes bonds, and normalizes the included weights to sum to 100%. It uses parameterized SQL for portfolio lookups. The output includes a sector breakdown and a short statement about the largest and smallest exposures when those values are available.

## LLM routing and providers

`agent.py` routes analytics questions through the configured LLM provider. The provider chooses the SQL Tool, Exposure Calculator, both for a hybrid question, or the unsupported response. Simple predefined conversational replies do not call the provider. The agent also provides interactive and single-question CLI modes.

`llm_client.py` supports:

- Google Gemini through `google-genai`
- NVIDIA NIM through the OpenAI-compatible `openai` client

Provider SDK clients are initialized lazily on the first call for that provider. `LLM_PROVIDER` is read when `llm_client.py` is imported, so restart the CLI or Streamlit process after changing provider settings. The selected provider's API key is required for analytics requests. API keys are not stored in source code.

## Environment configuration

Copy `.env.example` to `.env` and set the desired provider and its credentials. Do not commit `.env`.

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash-lite

# For NVIDIA instead, set LLM_PROVIDER=nvidia and configure:
NVIDIA_API_KEY=...
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
```

The example template defaults to Gemini. Only the selected provider's key is needed at runtime.

## Installation and execution

Run these commands in PowerShell from the project directory. This Windows environment has Python 3.13.7 available through the `py` launcher; `python` is not recognized. `py -m pip` uses the same Python 3.13 installation used by the project commands. No project virtual environment is currently present.

Confirm the interpreter and pip:

```powershell
py --version
py -m pip --version
```

Install core dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run the interactive CLI:

```powershell
py agent.py
```

Run one question and exit:

```powershell
py agent.py --question "How many portfolios do we have?"
py agent.py --question "What are the sector exposures for Tech Innovation Fund?"
```

Install the optional Streamlit dependency and launch the chat UI:

```powershell
py -m pip install -r requirements-streamlit.txt
py -m streamlit run streamlit_app.py
```

The Streamlit command was verified to start the app with the installed dependencies. In this environment port 8501 was already occupied, so Streamlit started on port 8502. The UI displays chat history, formatted SQL results, tool details when a tool ran, and an exposure chart when exposure results are present. Its response cache is in-memory, bounded, and limited to the running application session; it is not persistent storage.

## Tests and evaluation

Run offline tests, which use in-memory or temporary databases and mocked provider clients:

```powershell
py test_offline.py
```

Run the evaluator against the original ten questions and the existing database:

```powershell
py evaluator.py --verbose
```

The evaluator makes provider calls for normal analytics questions. The command was verified in this environment and passed **10/10 questions (100%)** using the configured provider. This result applies to the supplied ground-truth questions and the database/provider configuration used in that run; it does not guarantee accuracy for every possible question or a different dataset/provider response.

## Limitations

- Answers depend on the contents and coverage of the supplied SQLite database. The project does not fetch live market prices or refresh source data.
- Natural-language routing and non-template SQL generation depend on the selected provider, credentials, network availability, and model response.
- Sector exposure is based on stored holding weights and the available sector assignments. It is an equity-only allocation calculation, not a complete portfolio risk model.
- Evaluation covers the ten supplied ground-truth questions and does not establish performance for all natural-language questions.
- The Streamlit response cache and conversation state are in-memory and are not shared or persisted across sessions.
