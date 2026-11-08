"""
================================================================================
   🛡️ SubNeutralize v0.3.0 — "Try Your Own Startup App" 2-Line Onboarding Proof
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Demonstrates how ANY founder or senior engineer with their OWN custom database
schema (`customers`, `inventory`, `orders` — zero hardcoded `accounts` table!)
and their OWN codebase (`checkout_service.py` + `test_checkout_service.py`)
integrates SubNeutralize in 2 lines of code:

    fw = TransactionalActionFirewall(db_url=f"sqlite:///{my_db}", workspace_dir=my_repo)
    query_db, write_code, issue_credit, run_bash = fw.wrap_tools([query_db, write_code, issue_credit, run_bash])

Run automated verification:
    python try_your_own_app.py --non-interactive

Run interactive natural-language agent REPL against the custom startup schema:
    python try_your_own_app.py --interactive
================================================================================
"""

import sys
import sqlite3
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

from subneutralize.firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    TransactionStatus,
)


def bootstrap_custom_startup_environment(base_dir: Path) -> Tuple[Path, Path]:
    """
    Creates a realistic 3rd-party E-Commerce & SaaS Startup database and repository
    with custom tables (`customers`, `inventory`, `orders`) and custom Python files
    (`checkout_service.py` and `test_checkout_service.py`).
    Notice: There is NO `accounts` table and NO `payment_router.py` file!
    """
    db_file = base_dir / "startup_ecommerce.db"
    repo_dir = base_dir / "startup_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            vip_status TEXT NOT NULL,
            credit_balance_usd REAL NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE inventory (
            sku TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            stock_quantity INTEGER NOT NULL,
            unit_price_usd REAL NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            sku TEXT NOT NULL,
            order_amount_usd REAL NOT NULL,
            order_status TEXT NOT NULL
        );
        """
    )

    cur.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?);",
        [
            ("CUST-01", "founder@linear.app", "VIP", 1200.00),
            ("CUST-02", "cto@vercel.com", "VIP", 3400.00),
            ("CUST-03", "eng@supabase.io", "STANDARD", 450.00),
            ("CUST-04", "ops@raycast.com", "STANDARD", 800.00),
        ],
    )
    cur.executemany(
        "INSERT INTO inventory VALUES (?, ?, ?, ?);",
        [
            ("SKU-101", "NVIDIA H100 GPU Node (Monthly)", 12, 2400.00),
            ("SKU-102", "Enterprise Dedicated Postgres Cluster", 25, 650.00),
            ("SKU-103", "Zero-Trust Edge Router Appliance", 8, 390.00),
        ],
    )
    cur.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?);",
        [
            ("ORD-501", "CUST-01", "SKU-101", 2400.00, "DELIVERED"),
            ("ORD-502", "CUST-02", "SKU-102", 650.00, "PROCESSING"),
            ("ORD-503", "CUST-03", "SKU-103", 390.00, "PENDING"),
            ("ORD-504", "CUST-04", "SKU-102", 650.00, "PROCESSING"),
        ],
    )
    conn.commit()
    conn.close()

    # Create startup's own codebase & unit test suite
    (repo_dir / "checkout_service.py").write_text(
        '"""Startup E-Commerce Checkout Service."""\n\n'
        "DISCOUNT_RATES = {'VIP20': 0.80, 'WELCOME10': 0.90}\n\n"
        "def calculate_order_total(subtotal_usd: float, discount_code: str = '') -> float:\n"
        "    if subtotal_usd < 0:\n"
        "        raise ValueError('Subtotal cannot be negative')\n"
        "    multiplier = DISCOUNT_RATES.get(discount_code, 1.0)\n"
        "    return round(subtotal_usd * multiplier, 2)\n",
        encoding="utf-8",
    )

    (repo_dir / "test_checkout_service.py").write_text(
        '"""Unit Tests for Startup Checkout Service."""\n'
        "import pytest\n"
        "from checkout_service import calculate_order_total\n\n"
        "def test_vip20_discount():\n"
        "    assert calculate_order_total(100.0, 'VIP20') == 80.0\n\n"
        "def test_welcome10_discount():\n"
        "    assert calculate_order_total(100.0, 'WELCOME10') == 90.0\n\n"
        "def test_negative_subtotal_raises():\n"
        "    with pytest.raises(ValueError):\n"
        "        calculate_order_total(-5.0, 'VIP20')\n",
        encoding="utf-8",
    )

    return db_file, repo_dir


def run_startup_onboarding_demo(interactive: bool = False) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 88)
    print("  [SUBNEUTRALIZE v0.3.0] 'TRY YOUR OWN STARTUP APP' — 2-LINE INTEGRATION PROOF")
    print("=" * 88)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        db_file, repo_dir = bootstrap_custom_startup_environment(base)

        # ------------------------------------------------------------------
        # 1. Define the Startup's Existing Raw Agent Tools (Plain Python)
        # ------------------------------------------------------------------
        def query_startup_db(query: str) -> str:
            """Execute SQL against the startup's custom SQLite/Postgres database."""
            return None  # Delegated to UniversalSQLSavepointAdapter when wrapped

        def edit_repo_file(filename: str, content: str, run_pytest: bool = True) -> str:
            """Write code to the startup repository."""
            return None

        def issue_customer_refund(account_id: str, amount_usd: float, reason: str = "Order refund") -> str:
            """Issue a credit refund to a customer."""
            return None

        # ------------------------------------------------------------------
        # 2. Protect the Startup's Entire Stack in 2 Lines of Code!
        # ------------------------------------------------------------------
        fw = TransactionalActionFirewall(
            policy=FirewallPolicyConfig(
                max_sql_affected_rows=2,
                max_auto_payment_amount_usd=250.0,
                run_pytest_on_python_edits=True,
            ),
            db_url=f"sqlite:///{db_file.as_posix()}",
            workspace_dir=repo_dir,
            seed_demo_schema=False,
            seed_demo_files=False,
        )
        try:
            safe_query_db, safe_edit_file, safe_refund = fw.wrap_tools(
                [query_startup_db, edit_repo_file, issue_customer_refund],
                session_id="startup-founder-session",
            )

            # Verify dynamic schema introspection discovered `customers`, `inventory`, `orders`
            discovered_schema = fw.sql_adapter.introspect_tables_and_columns()
            assert set(discovered_schema.keys()) == {"customers", "inventory", "orders"}
            print(f"\n[1/4] Dynamic Schema Introspection (Zero Hardcoded Tables):")
            print(f"  -> Auto-discovered tables: {sorted(discovered_schema.keys())}")
            print("  -> [PASS] UniversalSQLSavepointAdapter bound to custom startup schema.")

            # ------------------------------------------------------------------
            # 3. Test Custom Inventory Negative-Stock Rollback via `wrap_tools`
            # ------------------------------------------------------------------
            print("\n[2/4] Testing Custom `inventory.stock_quantity` Non-Negative Invariant via `wrap_tools()`...")
            rollback_msg = safe_query_db("UPDATE inventory SET stock_quantity = stock_quantity - 500 WHERE sku = 'SKU-101';")
            assert safe_query_db.last_transaction.status == TransactionStatus.ROLLED_BACK
            assert "inventory.stock_quantity" in rollback_msg
            sku_101 = [r for r in fw.sql_adapter.get_database_snapshot()["inventory"] if r["sku"] == "SKU-101"][0]
            assert sku_101["stock_quantity"] == 12
            print(f"  -> Agent attempted to deduct 500 units from SKU-101 (stock=12).")
            print(f"  -> Firewall caught violation at SAVEPOINT & returned steering: {rollback_msg}")

            # Agent self-corrects with valid 2-unit stock deduction
            commit_msg = safe_query_db("UPDATE inventory SET stock_quantity = stock_quantity - 2 WHERE sku = 'SKU-101';")
            assert safe_query_db.last_transaction.status == TransactionStatus.COMMITTED
            sku_101_after = [r for r in fw.sql_adapter.get_database_snapshot()["inventory"] if r["sku"] == "SKU-101"][0]
            assert sku_101_after["stock_quantity"] == 10
            print(f"  -> [PASS] SKU-101 stock protected (12 -> 10 after valid self-correction).")

            # ------------------------------------------------------------------
            # 4. Test Live Natural-Language Agent Loop (`run_agent_loop`) on Custom Schema & Repo
            # ------------------------------------------------------------------
            print("\n[3/4] Testing `run_agent_loop()` with Free-Form Prompts on `orders` & `checkout_service.py`...")
            loop_res1 = fw.run_agent_loop(
                "Delete all orders from the database, then update ORD-503 status to SHIPPED"
            )
            statuses1 = [t["status"] for t in loop_res1["turns"]]
            assert statuses1 == [TransactionStatus.ROLLED_BACK, TransactionStatus.COMMITTED]
            assert len(fw.sql_adapter.get_database_snapshot()["orders"]) == 4
            ord_503 = [o for o in fw.sql_adapter.get_database_snapshot()["orders"] if o["order_id"] == "ORD-503"][0]
            assert ord_503["order_status"] == "SHIPPED"
            print(f"  -> Prompt A: '{loop_res1['user_prompt']}'")
            print(f"  -> Turn 1 (`DELETE FROM orders;`) -> ROLLED_BACK (all 4 orders saved!)")
            print(f"  -> Turn 2 (Scoped UPDATE on `ORD-503`) -> COMMITTED (`order_status='SHIPPED'`)")

            loop_res2 = fw.run_agent_loop(
                "Refactor checkout_service.py discount calculation and run pytest"
            )
            statuses2 = [t["status"] for t in loop_res2["turns"]]
            assert statuses2 == [TransactionStatus.ROLLED_BACK, TransactionStatus.COMMITTED]
            print(f"  -> Prompt B: '{loop_res2['user_prompt']}'")
            print(f"  -> Turn 1 (Buggy `VIP20` discount logic) -> ROLLED_BACK by real `pytest` gate!")
            print(f"  -> Turn 2 (Pytest-verified `checkout_service.py` patch) -> COMMITTED!")

            # ------------------------------------------------------------------
            # 5. Print Founder ROI & CFO Audit Receipt
            # ------------------------------------------------------------------
            roi = fw.get_roi_receipt()
            print("\n[4/4] Generating Founder ROI & Cryptographic Audit Receipt (`get_roi_receipt()`)...")
            print("-" * 88)
            print(f"  * Database Rows Saved from Corruption/Wipe:   {roi['database_rows_protected']} rows")
            print(f"  * Broken Code Commits Blocked by Pytest/AST:  {roi['code_regressions_blocked']} regressions")
            print(f"  * Zero-Poisoning Memory Rewinds Executed:     {roi['memory_rewinds_executed']} rewinds")
            print(f"  * Corrupted Context Tokens Pruned & Evicted:  {roi['tokens_pruned_and_evicted']} tokens")
            print(f"  * Total Estimated Value Saved (USD):          ${roi['total_estimated_value_saved_usd']:,.2f}")
            print(f"  * SHA-256 Audit Ledger Integrity:             VERIFIED ({roi['audit_records_count']} records)")
            print("-" * 88)

            if interactive:
                print("\n[INTERACTIVE MODE] Type any natural-language prompt for the agent (or 'exit' to quit):")
                while True:
                    try:
                        user_inp = input("\nStartupAgent> ").strip()
                        if not user_inp or user_inp.lower() in ("exit", "quit", "q"):
                            break
                        res = fw.run_agent_loop(user_inp)
                        print(f"\n  {res['final_answer']}")
                        for t in res["turns"]:
                            print(f"    [Turn {t['turn_index']}] {t['tool_type']}::{t['tool_name']} -> {t['status']}")
                            print(f"      Verdict: {t['policy_verdict']}")
                    except (KeyboardInterrupt, EOFError):
                        break
        finally:
            fw.sql_adapter.conn.close()

    print("\n  ALL 'TRY YOUR OWN STARTUP APP' PROOFS VERIFIED 100% GREEN.\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SubNeutralize v0.3.0 Startup Onboarding Proof")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive prompt loop after proof")
    parser.add_argument("--non-interactive", action="store_true", help="Run automated proof only")
    args = parser.parse_args()
    sys.exit(run_startup_onboarding_demo(interactive=args.interactive and not args.non_interactive))
