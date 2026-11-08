"""Offline regression tests for the safety and routing enhancements."""

import os
import sqlite3
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

# Keep the suite runnable without installing optional runtime dependencies.
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda: False
    sys.modules["dotenv"] = dotenv_stub

import agent
import evaluator
import llm_client
from streamlit_app import build_contextual_question, resolve_portfolio_followup
from db_utils import connect_read_only, install_read_only_protection
from exposure_tool import calculate_exposure
from sql_tool import _is_safe, generate_sql, run_sql_query


class ReadOnlyDatabaseTests(unittest.TestCase):
    def test_read_only_uri_and_authorizer_reject_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sample.db")
            writable = sqlite3.connect(path)
            writable.execute("CREATE TABLE sample (value INTEGER)")
            writable.execute("INSERT INTO sample VALUES (7)")
            writable.commit()
            writable.close()

            conn = connect_read_only(path)
            try:
                self.assertEqual(conn.execute("SELECT value FROM sample").fetchone(), (7,))
                attacks = (
                    "INSERT INTO sample VALUES (8)",
                    "UPDATE sample SET value = 8",
                    "DELETE FROM sample",
                    "DROP TABLE sample",
                    "ALTER TABLE sample ADD COLUMN extra TEXT",
                    "CREATE TABLE other (value INTEGER)",
                    "WITH x AS (SELECT 1) DELETE FROM sample",
                    "PRAGMA user_version = 9",
                )
                for statement in attacks:
                    with self.subTest(statement=statement):
                        with self.assertRaises(sqlite3.DatabaseError):
                            conn.execute(statement)
            finally:
                conn.close()

    def test_select_and_with_validation(self):
        self.assertEqual(_is_safe("SELECT 1"), (True, ""))
        self.assertEqual(_is_safe("WITH x AS (SELECT 1) SELECT * FROM x"), (True, ""))
        self.assertEqual(_is_safe("\ufeffSELECT SUM(total_aum) FROM portfolios WHERE target_risk_level = 'High'"), (True, ""))
        self.assertFalse(_is_safe("-- comment\nDROP TABLE sample")[0])
        self.assertFalse(_is_safe("PRAGMA user_version")[0])

    def test_sql_tool_installs_authorizer_for_external_connections(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sample (value INTEGER)")
        conn.execute("INSERT INTO sample VALUES (1)")
        install_read_only_protection(conn)
        self.assertEqual(conn.execute("SELECT value FROM sample").fetchone(), (1,))
        with self.assertRaises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM sample")
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1; SELECT 2")
        conn.close()


class SQLToolFailureTests(unittest.TestCase):
    def test_diversification_threshold_template_counts_stock_sectors_and_all_holdings(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);
            CREATE TABLE holdings (holding_id INTEGER, portfolio_id INTEGER, security_id INTEGER);
            CREATE TABLE securities (security_id INTEGER, asset_type TEXT, sector_id INTEGER);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            INSERT INTO portfolios VALUES (1, 'Alpha Fund'), (2, 'Beta Fund');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials'), (3, 'Healthcare'),
                                      (4, 'Energy'), (5, 'Automotive'), (6, 'Industrials');
            INSERT INTO securities VALUES
                (101, 'Stock', 1), (102, 'Stock', 2), (103, 'Stock', 3),
                (104, 'Stock', 4), (105, 'Stock', 5), (106, 'Stock', 6),
                (107, 'Bond', NULL), (108, 'Bond', NULL), (109, 'Bond', NULL),
                (110, 'Bond', NULL), (111, 'Bond', NULL), (112, 'Bond', NULL),
                (201, 'Stock', 1), (202, 'Stock', 2), (203, 'Stock', 3),
                (204, 'Stock', 4), (205, 'Stock', 5);
            INSERT INTO holdings VALUES
                (1,1,101),(2,1,102),(3,1,103),(4,1,104),(5,1,105),(6,1,106),
                (7,1,107),(8,1,108),(9,1,109),(10,1,110),(11,1,111),(12,1,112),
                (13,2,201),(14,2,202),(15,2,203),(16,2,204),(17,2,205);
        """)
        question = (
            "Find portfolios that have holdings in more than 5 different sectors "
            "and show their diversification metrics"
        )
        with patch("sql_tool._llm_call", side_effect=AssertionError("SQL/summary API called")):
            result = run_sql_query(question, conn, return_details=True)
        self.assertEqual(result["tool_details"][0]["execution_status"], "executed")
        self.assertIn("Alpha Fund", result["answer"])
        self.assertNotIn("Beta Fund", result["answer"])
        self.assertIn("| 6 ", result["answer"])
        self.assertIn("| 12 ", result["answer"])
        self.assertIn("0.5", result["answer"])
        conn.close()

    def test_common_sql_lists_and_sector_average_need_no_llm(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT, status TEXT);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            CREATE TABLE securities (security_id INTEGER, symbol TEXT, company_name TEXT,
                                     sector_id INTEGER, current_price REAL);
            INSERT INTO portfolios VALUES (1, 'Alpha Fund', 'Active'), (2, 'Beta Fund', 'Passive');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials');
            INSERT INTO securities VALUES (11, 'TECH', 'Tech Company', 1, 100),
                                          (12, 'BANK', 'Bank Company', 2, 200);
        """)
        cases = [
            ("What are the names of all active portfolios?", "Alpha Fund", "Beta Fund"),
            ("Which securities are in the Technology sector?", "Tech Company", "Bank Company"),
            ("What is the average current price of securities in each sector?", "100.0", ""),
        ]
        with patch("sql_tool._llm_call", side_effect=AssertionError("SQL/summary API called")):
            for question, expected, unexpected in cases:
                with self.subTest(question=question):
                    result = run_sql_query(question, conn, return_details=True)
                    self.assertEqual(result["tool_details"][0]["execution_status"], "executed")
                    self.assertIn(expected, result["answer"])
                    if "each sector" in question:
                        self.assertIn("has the highest average current price", result["answer"])
                        self.assertIn("has the lowest", result["answer"])
                    else:
                        self.assertIn("The query returned", result["answer"])
                    if "average current price" in question:
                        self.assertIn("200.0", result["answer"])
                    if unexpected:
                        self.assertNotIn(unexpected, result["answer"])
        conn.close()

    def test_comparison_summary_uses_actual_high_low_and_difference(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sector_prices (sector_name TEXT, average_price REAL)")
        conn.executemany(
            "INSERT INTO sector_prices VALUES (?, ?)",
            [("Technology", 125.5), ("Financials", 80.0), ("Energy", 100.0)],
        )
        with patch("sql_tool.generate_sql", return_value="SELECT sector_name, average_price FROM sector_prices"), \
             patch("sql_tool._summarize_results", side_effect=AssertionError("summary API called")):
            result = run_sql_query("Compare average prices by sector.", conn, return_details=True)
        self.assertIn("Technology has the highest average price (125.5)", result["answer"])
        self.assertIn("Financials has the lowest (80)", result["answer"])
        self.assertIn("difference of 45.5", result["answer"])
        self.assertIn("| average_price |", result["answer"])
        conn.close()

    def test_exposure_summary_reports_largest_and_smallest_actual_sectors(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);
            CREATE TABLE holdings (portfolio_id INTEGER, security_id INTEGER, current_weight REAL);
            CREATE TABLE securities (security_id INTEGER, sector_id INTEGER, asset_type TEXT);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            INSERT INTO portfolios VALUES (1, 'Demo Fund');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials');
            INSERT INTO securities VALUES (11, 1, 'Stock'), (12, 2, 'Stock');
            INSERT INTO holdings VALUES (1, 11, 75), (1, 12, 25);
        """)
        result = calculate_exposure("Demo Fund", conn, return_details=True)
        answer = result["answer"]
        self.assertIn("Technology has the largest exposure at 75.00%", answer)
        self.assertIn("Financials has the smallest at 25.00%", answer)
        gt = {"parameters": {"portfolio_name": "Demo Fund"}}
        self.assertEqual(evaluator._evaluate_exposure(1, "", gt, answer, conn)["status"], "PASS")
        conn.close()

    def test_scalar_sum_summary_uses_returned_value_without_llm(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sample (quantity INTEGER)")
        conn.executemany("INSERT INTO sample VALUES (?)", [(10,), (5,)])
        with patch("sql_tool.generate_sql", return_value="SELECT SUM(quantity) FROM sample"), \
             patch("sql_tool._summarize_results", side_effect=AssertionError("summary API called")):
            result = run_sql_query("What is the total quantity?", conn, return_details=True)
        self.assertIn("The total quantity is 15.", result["answer"])
        self.assertIn("| SUM(quantity) |", result["answer"])
        conn.close()

    def test_sector_value_and_portfolio_percentage_template_is_correct(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);
            CREATE TABLE holdings (holding_id INTEGER, portfolio_id INTEGER, security_id INTEGER,
                                   quantity REAL, current_weight REAL);
            CREATE TABLE securities (security_id INTEGER, current_price REAL, sector_id INTEGER);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            INSERT INTO portfolios VALUES (1, 'Alpha'), (2, 'Beta');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials');
            INSERT INTO securities VALUES (11, 100, 1), (12, 100, 2), (13, 50, 1);
            INSERT INTO holdings VALUES (1, 1, 11, 2, 0.5),
                                        (2, 1, 12, 2, 0.5),
                                        (3, 2, 13, 1, 1.0);
        """)
        question = (
            "For each portfolio, show the total value of Technology sector holdings "
            "and what percentage it represents of the total portfolio value"
        )
        with patch("sql_tool._llm_call", side_effect=AssertionError("SQL/summary API called")):
            result = run_sql_query(question, conn, return_details=True)
        self.assertEqual(result["tool_details"][0]["execution_status"], "executed")
        self.assertIn("technology_value", result["answer"])
        self.assertIn("technology_percentage", result["answer"])
        self.assertIn("Beta", result["answer"])
        self.assertIn("Alpha", result["answer"])
        self.assertIn("100.0", result["answer"])
        self.assertIn("50.0", result["answer"])
        self.assertIn("Alpha has the highest technology value (200)", result["answer"])
        self.assertIn("Beta has the lowest (50)", result["answer"])
        self.assertIn("Beta has the highest technology percentage (100)", result["answer"])
        self.assertIn("Alpha has the lowest (50)", result["answer"])
        conn.close()

    def test_q04_sums_returned_aum_rows_without_summary_api_call(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE portfolios (portfolio_name TEXT, total_aum REAL, target_risk_level TEXT)"
        )
        conn.executemany(
            "INSERT INTO portfolios VALUES (?, ?, ?)",
            [
                ("Portfolio A", 10_000_000, "High"),
                ("Portfolio B", 25_000_000, "High"),
                ("Portfolio C", 30_000_000, "High"),
                ("Portfolio D", 20_000_000, "High"),
                ("Portfolio E", 50_000_000, "Low"),
            ],
        )
        question = "What is the total Assets Under Management (AUM) for portfolios with high target risk level?"
        sql = (
            "SELECT portfolio_name, total_aum FROM portfolios "
            "WHERE target_risk_level = 'High'"
        )
        with patch("sql_tool.generate_sql", return_value=sql), \
             patch("sql_tool._summarize_results") as summarize:
            result = run_sql_query(question, conn, return_details=True)

        self.assertIn("The combined AUM is 85,000,000", result["answer"])
        self.assertEqual(evaluator._extract_numbers(result["answer"])[0], 85_000_000)
        self.assertEqual(result["tool_details"][0]["execution_status"], "executed")
        summarize.assert_not_called()
        conn.close()

    def test_q05_top_holdings_prompt_and_result_columns_follow_ground_truth(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(
            "CREATE TABLE holdings (holding_id INTEGER, portfolio_id INTEGER, security_id INTEGER, quantity REAL, cost_basis REAL);"
            "CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);"
            "CREATE TABLE securities (security_id INTEGER, symbol TEXT, company_name TEXT);"
            "INSERT INTO holdings VALUES (1, 3, 7, 10, 1250.50);"
            "INSERT INTO holdings VALUES (2, 3, 8, 5, 800.00);"
            "INSERT INTO portfolios VALUES (3, 'Growth Equity Fund');"
            "INSERT INTO securities VALUES (7, 'ACME', 'Acme Company');"
            "INSERT INTO securities VALUES (8, 'BETA', 'Beta Company');"
        )
        sql = (
            "SELECT h.holding_id, s.symbol, s.company_name, h.quantity, h.cost_basis "
            "FROM holdings h JOIN securities s ON h.security_id = s.security_id "
            "JOIN portfolios p ON h.portfolio_id = p.portfolio_id "
            "WHERE p.portfolio_name = 'Growth Equity Fund' "
            "ORDER BY h.cost_basis DESC LIMIT 5"
        )
        question = "Show me the top 5 holdings by cost basis in the Growth Equity Fund"
        with patch("sql_tool._llm_call", side_effect=AssertionError("SQL generation API called")):
            self.assertEqual(generate_sql(question), sql)
        with patch("sql_tool._summarize_results", side_effect=AssertionError("summary API called")):
            result = run_sql_query(
                question,
                conn,
                return_details=True,
            )

        self.assertEqual(
            result["tool_details"][0]["sql"],
            sql,
        )
        lines = [line for line in result["answer"].splitlines() if line.startswith("| ")]
        headers = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
        self.assertEqual(headers, ["holding_id", "symbol", "company_name", "quantity", "cost_basis"])
        self.assertIn("Acme Company (ACME) has the highest cost basis", result["answer"])
        self.assertIn("Beta Company (BETA) has the lowest", result["answer"])
        conn.close()

    def test_q04_indented_markdown_select_with_where_is_accepted_read_only(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE portfolios (total_aum REAL, target_risk_level TEXT)")
        conn.executemany(
            "INSERT INTO portfolios VALUES (?, ?)",
            [(50_000_000, "High"), (35_000_000, "High"), (99_000_000, "Low")],
        )
        generated = (
            "  \n  ```sql\n"
            "SELECT SUM(total_aum) FROM portfolios WHERE target_risk_level = 'High'\n"
            "  ```  \n"
        )
        with patch("sql_tool._llm_call", return_value=generated):
            parsed_sql = generate_sql("Show AUM by risk level")
        with patch("sql_tool.generate_sql", return_value=parsed_sql), \
             patch("sql_tool._summarize_results", side_effect=AssertionError("summary API called")):
            result = run_sql_query(
                "What is the total Assets Under Management (AUM) for portfolios with high target risk level?",
                conn,
                return_details=True,
            )

        self.assertEqual(result["tool_details"][0]["execution_status"], "executed")
        self.assertEqual(
            result["tool_details"][0]["sql"],
            "SELECT SUM(total_aum) FROM portfolios WHERE target_risk_level = 'High'",
        )
        self.assertIn("85000000", result["answer"])
        self.assertIn("The total AUM is 85,000,000.", result["answer"])
        self.assertNotIn("1 returned portfolio", result["answer"])
        with self.assertRaises(sqlite3.DatabaseError):
            conn.execute("UPDATE portfolios SET total_aum = 0")
        conn.close()

    def test_api_failure_is_friendly_and_does_not_echo_exception(self):
        conn = sqlite3.connect(":memory:")
        with patch("sql_tool._llm_call", side_effect=RuntimeError("offline-dummy-secret")):
            answer = run_sql_query("How many rows?", conn)
        self.assertIn("couldn't generate", answer.lower())
        self.assertNotIn("offline-dummy-secret", answer)
        conn.close()

    def test_stacked_write_query_cannot_modify_database(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE sample (value INTEGER)")
        conn.execute("INSERT INTO sample VALUES (3)")
        with patch("sql_tool._llm_call", return_value="SELECT 1; DELETE FROM sample"):
            answer = run_sql_query("Show one", conn)
        self.assertIn("database couldn't complete", answer.lower())
        self.assertEqual(conn.execute("SELECT value FROM sample").fetchall(), [(3,)])
        conn.close()


class AgentInputAndRoutingTests(unittest.TestCase):
    def test_conversation_is_local_and_does_not_route(self):
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        with patch.object(agent, "_route", side_effect=AssertionError("LLM route called")):
            answer = instance.answer("hii, what can u do..? what are ur capabilities")
        self.assertIn("I can answer questions", answer)

    def test_empty_and_oversized_input(self):
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        with patch.object(agent, "_route", side_effect=AssertionError("route called")):
            self.assertIn("enter a question", instance.answer("   ").lower())
        long_question = "x" * 2500
        with patch.object(agent, "_route", return_value={"tool": "unsupported"}) as route:
            self.assertIn("I can help with portfolio holdings", instance.answer(long_question))
            route.assert_called_once_with(long_question)

    def test_mixed_greeting_and_analytics_reaches_sql_tool(self):
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        instance.conn = object()
        with patch.object(agent, "_route", return_value={"tool": "sql_tool"}), \
             patch.object(agent, "run_sql_query", return_value="SQL answer") as sql_call:
            self.assertEqual(instance.answer("Hi, how many portfolios?"), "SQL answer")
        sql_call.assert_called_once()

    def test_single_tool_requests_do_not_call_the_other_tool(self):
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        instance.conn = object()
        with patch.object(agent, "_route", return_value={"tool": "sql_tool"}), \
             patch.object(agent, "run_sql_query", return_value="SQL answer") as sql_call, \
             patch.object(agent, "calculate_exposure") as exposure_call:
            self.assertEqual(instance.answer("How many portfolios?"), "SQL answer")
        sql_call.assert_called_once()
        exposure_call.assert_not_called()

        with patch.object(agent, "_route", return_value={
            "tool": "exposure_calculator", "portfolio_name": "Growth Equity Fund"
        }), patch.object(agent, "run_sql_query") as sql_call, \
             patch.object(agent, "calculate_exposure", return_value="Exposure answer") as exposure_call:
            self.assertEqual(instance.answer("Sector exposure for Growth Equity Fund"), "Exposure answer")
        exposure_call.assert_called_once()
        sql_call.assert_not_called()

    def test_portfolio_count_uses_actual_value_after_llm_routing(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE portfolios (portfolio_name TEXT)")
        conn.executemany(
            "INSERT INTO portfolios VALUES (?)",
            [(f"Portfolio {number}",) for number in range(1, 14)],
        )
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        instance.conn = conn
        with patch.object(agent, "_route", return_value={"tool": "sql_tool"}) as route, \
             patch("sql_tool._llm_call", side_effect=AssertionError("SQL/summary API called")):
            answer = instance.answer("How many portfolios do we have?")
        self.assertIn("We have 13 portfolios in total.", answer)
        self.assertIn("| portfolio_count |", answer)
        route.assert_called_once_with("How many portfolios do we have?")
        conn.close()

    def test_tech_exposure_followup_references_only_tech_portfolio_holdings(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            CREATE TABLE securities (security_id INTEGER, symbol TEXT, company_name TEXT,
                                    asset_type TEXT, sector_id INTEGER);
            CREATE TABLE holdings (holding_id INTEGER, portfolio_id INTEGER, security_id INTEGER,
                                   quantity REAL, cost_basis REAL, current_weight REAL);
            INSERT INTO portfolios VALUES (1, 'Tech Innovation Fund'), (2, 'Balanced Portfolio');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials');
            INSERT INTO securities VALUES
                (11, 'TECH', 'Tech Company', 'Stock', 1),
                (12, 'BANK', 'Other Company', 'Stock', 2);
            INSERT INTO holdings VALUES (101, 1, 11, 10, 1000, 0.6),
                                        (102, 2, 12, 20, 2000, 1.0);
        """)
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        instance.conn = conn
        first_question = "What are the sector exposures for Tech Innovation Fund?"
        with patch.object(agent, "_route", return_value={
            "tool": "exposure_calculator", "portfolio_name": "Tech Innovation Fund"
        }) as first_route, \
             patch("sql_tool._llm_call", side_effect=AssertionError("LLM API called")):
            exposure = instance.answer_with_details(first_question)
        first_route.assert_called_once_with(first_question)
        self.assertIn("Technology", exposure["answer"])
        self.assertEqual(exposure["tool_details"][0]["tool"], "exposure_calculator")

        messages = [
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": exposure["answer"]},
        ]
        contextual = build_contextual_question("What about its holdings?", messages)
        self.assertIn("What about Tech Innovation Fund's holdings?", contextual)
        with patch.object(agent, "_route", return_value={"tool": "sql_tool", "portfolio_name": None}) as route, \
             patch("sql_tool._llm_call", side_effect=AssertionError("SQL/summary API called")):
            holdings = instance.answer_with_details(contextual)
        self.assertIn("Tech Company", holdings["answer"])
        self.assertNotIn("Other Company", holdings["answer"])
        self.assertIn("Tech Innovation Fund", holdings["tool_details"][0]["sql"])
        route.assert_called_once()
        self.assertIn("What about Tech Innovation Fund's holdings?", route.call_args.args[0])
        conn.close()

    def test_ambiguous_or_missing_context_does_not_expand_its(self):
        question = "What about its holdings?"
        self.assertEqual(resolve_portfolio_followup(question, []), question)
        ambiguous = [{
            "role": "user",
            "content": "Compare Tech Innovation Fund and Growth Equity Fund",
        }]
        self.assertEqual(resolve_portfolio_followup(question, ambiguous), question)

    def test_hybrid_runs_both_tools(self):
        instance = agent.PortfolioAgent.__new__(agent.PortfolioAgent)
        instance.conn = object()
        decision = {"tool": "hybrid", "portfolio_name": "Growth Equity Fund"}
        with patch.object(agent, "_route", return_value=decision) as route, \
             patch.object(agent, "run_sql_query", return_value="Portfolio count: 13") as sql_call, \
             patch.object(agent, "calculate_exposure", return_value="Technology 100.00%") as exposure_call:
            answer = instance.answer("Show portfolio count and sector exposure for Growth Equity Fund")
        sql_call.assert_called_once()
        exposure_call.assert_called_once()
        self.assertIn("Portfolio data findings", answer)
        self.assertIn("Sector exposure", answer)
        route.assert_called_once()


class EvaluatorTests(unittest.TestCase):
    def test_table_evaluation_detects_missing_row_values(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE expected (name TEXT, amount REAL)")
        conn.executemany("INSERT INTO expected VALUES (?, ?)", [("Alpha", 10.0), ("Beta", 20.0)])
        ground_truth = {"sql_query": "SELECT name, amount FROM expected", "expected_result_type": "table"}
        complete = "| name | amount |\n| Alpha | 10.0 |\n| Beta | 20.0 |"
        missing = "| name | amount |\n| Alpha | 10.0 |"
        self.assertEqual(evaluator._evaluate_text2sql(1, "", ground_truth, complete, conn)["status"], "PASS")
        result = evaluator._evaluate_text2sql(1, "", ground_truth, missing, conn)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("missing expected result row", result["reason"].lower())
        conn.close()

    def test_exposure_evaluation_checks_each_percentage(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript("""
            CREATE TABLE portfolios (portfolio_id INTEGER, portfolio_name TEXT);
            CREATE TABLE holdings (portfolio_id INTEGER, security_id INTEGER, current_weight REAL);
            CREATE TABLE securities (security_id INTEGER, sector_id INTEGER, asset_type TEXT);
            CREATE TABLE sectors (sector_id INTEGER, sector_name TEXT);
            INSERT INTO portfolios VALUES (1, 'Demo Fund');
            INSERT INTO sectors VALUES (1, 'Technology'), (2, 'Financials');
            INSERT INTO securities VALUES (11, 1, 'Stock'), (12, 2, 'Stock');
            INSERT INTO holdings VALUES (1, 11, 60), (1, 12, 40);
        """)
        gt = {"parameters": {"portfolio_name": "Demo Fund"}}
        correct = "Demo Fund\n  Technology                  60.00%\n  Financials                  40.00%\n  Total (equity)              100.00%"
        incorrect = "Demo Fund\n  Technology                  50.00%\n  Financials                  50.00%\n  Total (equity)              100.00%"
        self.assertEqual(evaluator._evaluate_exposure(1, "", gt, correct, conn)["status"], "PASS")
        self.assertEqual(evaluator._evaluate_exposure(1, "", gt, incorrect, conn)["status"], "FAIL")
        conn.close()


class ProviderTests(unittest.TestCase):
    def test_both_provider_branches_use_mock_clients_without_network(self):
        class GeminiResponse:
            text = "SELECT 1"

        class GeminiModels:
            def generate_content(self, **kwargs):
                return GeminiResponse()

        class GeminiClient:
            models = GeminiModels()

        class NvidiaMessage:
            content = "SELECT 1"

        class NvidiaChoice:
            message = NvidiaMessage()

        class NvidiaResponse:
            choices = [NvidiaChoice()]

        class NvidiaCompletions:
            def create(self, **kwargs):
                return NvidiaResponse()

        class NvidiaChat:
            completions = NvidiaCompletions()

        class NvidiaClient:
            chat = NvidiaChat()

        google_stub = types.ModuleType("google")
        google_stub.__path__ = []
        google_genai_stub = types.ModuleType("google.genai")
        google_genai_stub.__path__ = []
        errors_stub = types.ModuleType("google.genai.errors")
        errors_stub.ClientError = type("ClientError", (Exception,), {})
        google_genai_stub.errors = errors_stub
        openai_stub = types.ModuleType("openai")
        openai_stub.RateLimitError = type("RateLimitError", (Exception,), {})
        fake_modules = {
            "google": google_stub,
            "google.genai": google_genai_stub,
            "google.genai.errors": errors_stub,
            "openai": openai_stub,
        }
        with patch.dict(sys.modules, fake_modules), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "offline-dummy-key"}), \
             patch.object(llm_client, "LLM_PROVIDER", "gemini"), \
             patch.object(llm_client, "_get_gemini_client", return_value=GeminiClient()):
            self.assertEqual(llm_client.llm_call("offline test"), "SELECT 1")
        with patch.dict(sys.modules, fake_modules), \
             patch.dict(os.environ, {"NVIDIA_API_KEY": "offline-dummy-key"}), \
             patch.object(llm_client, "LLM_PROVIDER", "nvidia"), \
             patch.object(llm_client, "_get_nvidia_client", return_value=NvidiaClient()):
            self.assertEqual(llm_client.llm_call("offline test"), "SELECT 1")

    def test_interactive_loop_accepts_local_conversation_and_exit(self):
        class FakeAgent:
            def __init__(self):
                self.questions = []

            def answer(self, question):
                self.questions.append(question)
                return "Hello!"

        fake_agent = FakeAgent()
        output = StringIO()
        with patch("builtins.input", side_effect=["hi", "quit"]), redirect_stdout(output):
            agent._run_interactive(fake_agent)
        self.assertEqual(fake_agent.questions, ["hi"])
        self.assertIn("Hello!", output.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
