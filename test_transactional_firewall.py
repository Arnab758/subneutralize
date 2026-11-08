"""
Comprehensive Test Suite for SubNeutralize v0.3.0:
Transactional Action Firewall (TAF), SQLite Savepoints, Filesystem Shadow Snapshots,
Outbound Payment Idempotency, Zero-Poisoning Memory Rewind, and FastAPI/MCP Endpoints.
"""

import sys
import json
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

SCRATCH_DIR = Path(__file__).resolve().parent
if str(SCRATCH_DIR) in sys.path:
    sys.path.remove(str(SCRATCH_DIR))
sys.path.insert(0, str(SCRATCH_DIR))

from subneutralize.firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    TransactionStatus,
)
from subneutralize.server import create_app


@pytest.fixture
def temp_firewall():
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        db_path = base / "test_enterprise.db"
        ws_dir = base / "sandbox_ws"
        fw = TransactionalActionFirewall(
            policy=FirewallPolicyConfig(
                max_sql_affected_rows=2,
                max_auto_payment_amount_usd=250.0,
                max_identical_tool_retries=1,
                max_poincare_recurrence=0.85,
            ),
            db_path=db_path,
            workspace_dir=ws_dir,
        )
        yield fw
        fw.sql_adapter.conn.close()


def test_sql_savepoint_commit_and_mass_delete_rollback(temp_firewall: TransactionalActionFirewall):
    # 1. Verify initial 5 seeded enterprise accounts exist
    snap0 = temp_firewall.sql_adapter.get_database_snapshot()
    assert len(snap0["accounts"]) == 5

    # 2. Rogue agent attempts a mass DELETE affecting all 5 rows (> max_sql_affected_rows=2)
    rec_del, rewound_msgs = temp_firewall.intercept_and_execute(
        session_id="sess-sql-1",
        turn_index=1,
        tool_type="sql",
        tool_name="execute_sql",
        arguments={"query": "DELETE FROM accounts WHERE balance_usd > 0;"},
        agent_reasoning_text="<think>Let me wipe the accounts table to start fresh.</think>",
        messages_context=[{"role": "user", "content": "Update Hooli status."}],
    )
    assert rec_del.status == TransactionStatus.ROLLED_BACK
    assert rec_del.risk_level == "CRITICAL"
    assert rec_del.memory_rewound is True
    assert rec_del.tokens_pruned_on_rollback >= 120
    assert "FIREWALL ROLLBACK" in rewound_msgs[-1]["content"]

    # Verify all 5 accounts are 100% intact thanks to SQLite SAVEPOINT rollback!
    snap1 = temp_firewall.sql_adapter.get_database_snapshot()
    assert len(snap1["accounts"]) == 5

    # 3. Scoped single-row UPDATE succeeds and commits savepoint
    rec_upd, committed_msgs = temp_firewall.intercept_and_execute(
        session_id="sess-sql-1",
        turn_index=2,
        tool_type="sql",
        tool_name="execute_sql",
        arguments={"query": "UPDATE accounts SET status = 'AUDITED' WHERE account_id = 'ACC-105';"},
        agent_reasoning_text="<think>Scoping update strictly to ACC-105.</think>",
        messages_context=rewound_msgs,
    )
    assert rec_upd.status == TransactionStatus.COMMITTED
    snap2 = temp_firewall.sql_adapter.get_database_snapshot()
    acc_105 = [a for a in snap2["accounts"] if a["account_id"] == "ACC-105"][0]
    assert acc_105["status"] == "AUDITED"


def test_sql_negative_balance_invariant_rollback(temp_firewall: TransactionalActionFirewall):
    # Attempting to deduct $999,999 from ACC-105 ($850 balance) triggers negative balance rollback
    rec, _ = temp_firewall.intercept_and_execute(
        session_id="sess-sql-inv",
        turn_index=1,
        tool_type="sql",
        tool_name="execute_sql",
        arguments={"query": "UPDATE accounts SET balance_usd = balance_usd - 999999 WHERE account_id = 'ACC-105';"},
    )
    assert rec.status == TransactionStatus.ROLLED_BACK
    assert "below $0.00" in rec.policy_verdict

    snap = temp_firewall.sql_adapter.get_database_snapshot()
    acc_105 = [a for a in snap["accounts"] if a["account_id"] == "ACC-105"][0]
    assert acc_105["balance_usd"] == 850.00


def test_filesystem_shadow_snapshot_ast_and_secret_rollback(temp_firewall: TransactionalActionFirewall):
    orig_files = temp_firewall.fs_adapter.list_workspace_files()
    orig_router = orig_files["payment_router.py"]

    # 1. Agent writes broken Python syntax to payment_router.py -> Rolled back to exact original bytes
    rec_broken, _ = temp_firewall.intercept_and_execute(
        session_id="sess-fs-1",
        turn_index=1,
        tool_type="filesystem",
        tool_name="write_file",
        arguments={
            "filename": "payment_router.py",
            "content": "def broken_func(x:\n    return x + (\n",
        },
    )
    assert rec_broken.status == TransactionStatus.ROLLED_BACK
    assert "SyntaxError" in rec_broken.policy_verdict
    assert temp_firewall.fs_adapter.list_workspace_files()["payment_router.py"] == orig_router

    # 2. Agent attempts to overwrite protected `.env.production` -> Blocked
    rec_env, _ = temp_firewall.intercept_and_execute(
        session_id="sess-fs-1",
        turn_index=2,
        tool_type="filesystem",
        tool_name="write_file",
        arguments={"filename": ".env.production", "content": "OVERWRITTEN=true"},
    )
    assert rec_env.status == TransactionStatus.ROLLED_BACK
    assert "protected" in rec_env.policy_verdict.lower()

    # 3. Agent writes valid Python code -> Committed
    clean_code = "def calculate_fee(amount_usd: float) -> float:\n    return round(amount_usd * 0.015, 2)\n"
    rec_ok, _ = temp_firewall.intercept_and_execute(
        session_id="sess-fs-1",
        turn_index=3,
        tool_type="filesystem",
        tool_name="write_file",
        arguments={"filename": "payment_router.py", "content": clean_code},
    )
    assert rec_ok.status == TransactionStatus.COMMITTED
    assert temp_firewall.fs_adapter.list_workspace_files()["payment_router.py"] == clean_code


def test_payment_idempotency_loop_and_human_approval_gate(temp_firewall: TransactionalActionFirewall):
    # 1. First $100 refund within $250 cap auto-commits
    rec1, _ = temp_firewall.intercept_and_execute(
        session_id="sess-pay-1",
        turn_index=1,
        tool_type="http_payment",
        tool_name="stripe_refund",
        arguments={"account_id": "ACC-101", "amount_usd": 100.0, "reason": "Duplicate charge"},
    )
    assert rec1.status == TransactionStatus.COMMITTED

    # 2. Duplicate retry of identical $100 refund is blocked by idempotency loop detector
    rec2, _ = temp_firewall.intercept_and_execute(
        session_id="sess-pay-1",
        turn_index=2,
        tool_type="http_payment",
        tool_name="stripe_refund",
        arguments={"account_id": "ACC-101", "amount_usd": 100.0, "reason": "Duplicate charge"},
    )
    assert rec2.status == TransactionStatus.ROLLED_BACK
    assert rec2.loop_detected is True
    assert temp_firewall.metrics["loops_prevented"] == 1

    # 3. High-value $2,500 refund (> $250 policy cap) enters PENDING_APPROVAL
    rec3, _ = temp_firewall.intercept_and_execute(
        session_id="sess-pay-1",
        turn_index=3,
        tool_type="http_payment",
        tool_name="stripe_refund",
        arguments={"account_id": "ACC-102", "amount_usd": 2500.0, "reason": "Enterprise SLA payout"},
    )
    assert rec3.status == TransactionStatus.PENDING_APPROVAL
    assert rec3.tx_id in temp_firewall.pending_approvals

    # 4. Human operator resolves with COMMIT
    resolved = temp_firewall.resolve_pending_transaction(rec3.tx_id, "COMMIT")
    assert resolved.status == TransactionStatus.COMMITTED
    assert rec3.tx_id not in temp_firewall.pending_approvals

    # 5. Verify SHA-256 audit chain integrity across all transactions
    valid, count = temp_firewall.verify_audit_chain_integrity()
    assert valid is True
    assert count == 3


def test_fastapi_command_center_and_chaos_scenarios(temp_firewall: TransactionalActionFirewall):
    app = create_app(engine=None, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", firewall=temp_firewall)
    client = TestClient(app)

    # 1. Dashboard HTML loads
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "SubNeutralize v0.3.0" in r_home.text
    assert "Interactive Chaos Lab" in r_home.text

    # 2. Trigger all 4 Chaos Lab scenarios via API
    for scenario in ["sql_mass_delete", "refund_loop", "high_value_transfer", "ast_corruption"]:
        r_sc = client.post(f"/v1/firewall/scenario/{scenario}")
        assert r_sc.status_code == 200
        payload = r_sc.json()
        assert payload["scenario_id"] == scenario
        assert len(payload["executed_transactions"]) >= 1

    # 3. Verify MCP tool call endpoint
    r_mcp = client.post(
        "/v1/mcp/tools/call",
        json={
            "name": "execute_sql_query",
            "arguments": {"query": "SELECT account_id, balance_usd FROM accounts WHERE account_id = 'ACC-101';"},
        },
    )
    assert r_mcp.status_code == 200
    mcp_data = r_mcp.json()
    assert mcp_data["result"]["status"] == TransactionStatus.COMMITTED
    assert mcp_data["result"]["isError"] is False


def test_dynamic_latent_policy_coupling_sql_and_payment(temp_firewall: TransactionalActionFirewall):
    # Under normal latent state (phi=-0.35), 2-row SQL update is allowed (max_sql_affected_rows=2)
    # Under high latent instability (phi=0.65 > 0.25), effective_max_sql_rows tightens to 1!
    rec_sql, _ = temp_firewall.intercept_and_execute(
        session_id="sess-latent-1",
        turn_index=1,
        tool_type="sql",
        tool_name="execute_sql",
        arguments={
            "query": "UPDATE accounts SET status = 'REVIEW' WHERE account_id IN ('ACC-101', 'ACC-102');"
        },
        latent_phi=0.65,
        latent_rho=0.30,
    )
    assert rec_sql.dynamic_policy_tightened is True
    assert rec_sql.effective_max_sql_rows == 1
    assert rec_sql.status == TransactionStatus.ROLLED_BACK
    assert "Latent Coupling Active" in rec_sql.policy_verdict

    # Under elevated latent instability (phi=0.55), a $100 payment (normally <= $250 cap) is held in PENDING_APPROVAL ($25 effective cap)
    rec_pay, _ = temp_firewall.intercept_and_execute(
        session_id="sess-latent-1",
        turn_index=2,
        tool_type="http_payment",
        tool_name="stripe_refund",
        arguments={"account_id": "ACC-101", "amount_usd": 100.0, "reason": "Latent test"},
        latent_phi=0.55,
        latent_rho=0.40,
    )
    assert rec_pay.dynamic_policy_tightened is True
    assert rec_pay.effective_max_payment_usd == 25.0
    assert rec_pay.status == TransactionStatus.PENDING_APPROVAL


def test_filesystem_pytest_verification_gate_and_shell_rollback(temp_firewall: TransactionalActionFirewall):
    import sys

    orig_code = temp_firewall.fs_adapter.list_workspace_files()["payment_router.py"]

    # 1. Valid AST Python code that breaks unit tests in `test_payment_router.py` -> Rolled back!
    buggy_logic = (
        "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
        "    return 999.99\n"
    )
    rec_bug, _ = temp_firewall.intercept_and_execute(
        session_id="sess-pytest-1",
        turn_index=1,
        tool_type="filesystem",
        tool_name="write_file",
        arguments={
            "filename": "payment_router.py",
            "content": buggy_logic,
            "run_pytest": True,
        },
    )
    assert rec_bug.status == TransactionStatus.ROLLED_BACK
    assert "Pytest verification failed" in rec_bug.policy_verdict
    assert temp_firewall.fs_adapter.list_workspace_files()["payment_router.py"] == orig_code

    # 2. Subprocess shell command that mutates a file and exits non-zero -> Full directory restored!
    failing_cmd = (
        f'"{sys.executable}" -c "import pathlib, sys; '
        f"pathlib.Path('payment_router.py').write_text('BROKEN_BY_SHELL'); "
        f'sys.exit(2)"'
    )
    rec_sh, _ = temp_firewall.intercept_and_execute(
        session_id="sess-pytest-1",
        turn_index=2,
        tool_type="shell",
        tool_name="bash",
        arguments={"command": failing_cmd},
    )
    assert rec_sh.status == TransactionStatus.ROLLED_BACK
    assert temp_firewall.fs_adapter.list_workspace_files()["payment_router.py"] == orig_code


def test_transactional_tool_decorator_and_kv_cache_slicing(temp_firewall: TransactionalActionFirewall):
    import torch
    from subneutralize.engine import PhysicalCacheSlicer

    # 1. Test @firewall.transactional_tool decorator
    @temp_firewall.transactional_tool(tool_type="sql", session_id="sess-dec-1")
    def run_db_update(query: str):
        return {"ok": True, "executed_query": query}

    res_ok = run_db_update(query="UPDATE accounts SET status = 'DEC_OK' WHERE account_id = 'ACC-104';")
    assert isinstance(res_ok, dict) and res_ok["ok"] is True
    assert run_db_update.last_transaction.status == TransactionStatus.COMMITTED

    res_bad = run_db_update(query="DROP TABLE accounts;")
    assert run_db_update.last_transaction.status == TransactionStatus.ROLLED_BACK

    # 2. Test PhysicalCacheSlicer.slice_kv_cache on rollback
    legacy_kv = (
        (torch.randn(1, 2, 200, 8), torch.randn(1, 2, 200, 8)),
        (torch.randn(1, 2, 200, 8), torch.randn(1, 2, 200, 8)),
    )

    class KvHolder:
        def __init__(self):
            self.key_cache = [torch.randn(1, 2, 200, 8)]
            self.value_cache = [torch.randn(1, 2, 200, 8)]
            self._seen_tokens = 200

    kv_holder = KvHolder()
    rec_kv, _ = temp_firewall.intercept_and_execute(
        session_id="sess-kv-1",
        turn_index=1,
        tool_type="shell",
        tool_name="bash",
        arguments={"command": "rm -rf /"},
        past_key_values=kv_holder,
        safe_kv_length=128,
    )
    assert rec_kv.status == TransactionStatus.ROLLED_BACK
    assert rec_kv.kv_tokens_sliced_on_rollback == 72
    assert PhysicalCacheSlicer.get_seq_length(kv_holder) == 128

    sliced_tuple, sliced_cnt = PhysicalCacheSlicer.slice_kv_cache(legacy_kv, 100)
    assert sliced_cnt == 100
    assert PhysicalCacheSlicer.get_seq_length(sliced_tuple) == 100


def test_agent_run_mission_endpoint(temp_firewall: TransactionalActionFirewall):
    app = create_app(engine=None, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", firewall=temp_firewall)
    client = TestClient(app)

    res = client.post("/v1/firewall/agent-run")
    assert res.status_code == 200
    data = res.json()
    assert data["turns_executed"] == 5
    statuses = [step["status"] for step in data["trajectory"]]
    assert statuses == [
        TransactionStatus.ROLLED_BACK,
        TransactionStatus.COMMITTED,
        TransactionStatus.ROLLED_BACK,
        TransactionStatus.COMMITTED,
        TransactionStatus.PENDING_APPROVAL,
    ]


def test_out_of_order_pending_resolution_chain_and_sql_injection_safety(temp_firewall: TransactionalActionFirewall):
    # 1. Stage a high-value pending approval at Turn 1
    rec_pending, _ = temp_firewall.intercept_and_execute(
        session_id="sess-chain-1",
        turn_index=1,
        tool_type="http_payment",
        tool_name="stripe_refund",
        arguments={
            "account_id": "ACC-101",
            "amount_usd": 1000.0,
            "reason": "Customer's SLA O'Reilly quote); DROP TABLE accounts; --",
        },
    )
    assert rec_pending.status == TransactionStatus.PENDING_APPROVAL

    # 2. Execute another transaction at Turn 2 BEFORE resolving Turn 1
    rec_after, _ = temp_firewall.intercept_and_execute(
        session_id="sess-chain-1",
        turn_index=2,
        tool_type="sql",
        tool_name="execute_sql",
        arguments={"query": "SELECT account_id FROM accounts WHERE account_id = 'ACC-101';"},
    )
    assert rec_after.status == TransactionStatus.COMMITTED

    # 3. Now resolve Turn 1 (which is NOT the last transaction in the ledger)
    resolved = temp_firewall.resolve_pending_transaction(rec_pending.tx_id, "COMMIT")
    assert resolved.status == TransactionStatus.COMMITTED

    # 4. Verify SHA-256 chain remains 100% valid and SQL injection in `reason` did not drop accounts
    chain_ok, count = temp_firewall.verify_audit_chain_integrity()
    assert chain_ok is True
    assert count == 2
    snap = temp_firewall.sql_adapter.get_database_snapshot()
    assert len(snap["accounts"]) == 5
    assert snap["refunds"][0]["reason"] == "Customer's SLA O'Reilly quote); DROP TABLE accounts; --"


def test_universal_schema_agnostic_adapter_on_custom_ecommerce_tables():
    """
    Verifies that UniversalSQLSavepointAdapter works on ANY third-party schema
    (e.g. `inventory`, `orders`, `customers` — with ZERO `accounts` table!) and
    enforces non-negative numeric invariants and row-count caps via SAVEPOINT.
    """
    import sqlite3
    from subneutralize import UniversalSQLSavepointAdapter

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        custom_db = base / "custom_saas.db"
        conn = sqlite3.connect(str(custom_db))
        cur = conn.cursor()
        cur.execute("CREATE TABLE inventory (sku TEXT PRIMARY KEY, stock_quantity INTEGER NOT NULL, price_usd REAL NOT NULL);")
        cur.execute("CREATE TABLE orders (order_id TEXT PRIMARY KEY, total_amount REAL NOT NULL, status TEXT NOT NULL);")
        cur.executemany(
            "INSERT INTO inventory VALUES (?, ?, ?);",
            [("SKU-A", 10, 99.0), ("SKU-B", 5, 149.0), ("SKU-C", 20, 49.0)],
        )
        cur.executemany(
            "INSERT INTO orders VALUES (?, ?, ?);",
            [("ORD-1", 99.0, "OPEN"), ("ORD-2", 149.0, "OPEN"), ("ORD-3", 49.0, "OPEN")],
        )
        conn.commit()
        conn.close()

        fw = TransactionalActionFirewall(
            policy=FirewallPolicyConfig(max_sql_affected_rows=1),
            db_url=f"sqlite:///{custom_db.as_posix()}",
            workspace_dir=base / "custom_ws",
            seed_demo_schema=False,
            seed_demo_files=False,
        )
        assert isinstance(fw.sql_adapter, UniversalSQLSavepointAdapter)
        schema = fw.sql_adapter.introspect_tables_and_columns()
        assert "accounts" not in schema
        assert set(schema.keys()) == {"inventory", "orders"}

        # 1. Negative stock invariant on `inventory.stock_quantity` -> Rolled back!
        rec_neg, _ = fw.intercept_and_execute(
            session_id="custom-sess",
            turn_index=1,
            tool_type="sql",
            tool_name="query_db",
            arguments={"query": "UPDATE inventory SET stock_quantity = stock_quantity - 50 WHERE sku = 'SKU-B';"},
        )
        assert rec_neg.status == TransactionStatus.ROLLED_BACK
        assert "inventory.stock_quantity" in rec_neg.policy_verdict
        sku_b = [r for r in fw.sql_adapter.get_database_snapshot()["inventory"] if r["sku"] == "SKU-B"][0]
        assert sku_b["stock_quantity"] == 5

        # 2. Mass DELETE on `orders` (3 rows > max_sql_affected_rows=1) -> Rolled back!
        rec_del, _ = fw.intercept_and_execute(
            session_id="custom-sess",
            turn_index=2,
            tool_type="sql",
            tool_name="query_db",
            arguments={"query": "DELETE FROM orders;"},
        )
        assert rec_del.status == TransactionStatus.ROLLED_BACK
        assert len(fw.sql_adapter.get_database_snapshot()["orders"]) == 3

        # 3. Valid single-row stock update -> Committed!
        rec_ok, _ = fw.intercept_and_execute(
            session_id="custom-sess",
            turn_index=3,
            tool_type="sql",
            tool_name="query_db",
            arguments={"query": "UPDATE inventory SET stock_quantity = stock_quantity - 2 WHERE sku = 'SKU-B';"},
        )
        assert rec_ok.status == TransactionStatus.COMMITTED
        sku_b_after = [r for r in fw.sql_adapter.get_database_snapshot()["inventory"] if r["sku"] == "SKU-B"][0]
        assert sku_b_after["stock_quantity"] == 3
        fw.sql_adapter.conn.close()


def test_wrap_tools_middleware_and_run_agent_loop_and_roi_endpoints(temp_firewall: TransactionalActionFirewall):
    # 1. Test `wrap_tools` on plain Python functions and LangChain-style tool objects
    def run_sql(sql: str):
        return f"CUSTOM_SQL_OK: {sql}"

    def write_code(path: str, code: str):
        return f"CUSTOM_FILE_OK: {path}"

    class MockLangChainTool:
        name = "stripe_payout"
        description = "Send a customer refund payout"

        def _run(self, customer_id: str, amount: float, reason: str = "SLA"):
            return {"paid": amount, "to": customer_id}

    lc_tool = MockLangChainTool()
    safe_sql, safe_write, safe_lc = temp_firewall.wrap_tools(
        [run_sql, write_code, lc_tool], session_id="wrap-sess"
    )

    # Bad SQL call via wrapped tool returns clean steering string instead of crashing
    out_rollback = safe_sql("DELETE FROM accounts;")
    assert isinstance(out_rollback, str) and "SUBNEUTRALIZE FIREWALL ROLLBACK" in out_rollback
    assert safe_sql.last_transaction.status == TransactionStatus.ROLLED_BACK

    # Good SQL call via wrapped tool commits and returns original function result
    out_commit = safe_sql("UPDATE accounts SET status = 'WRAPPED_OK' WHERE account_id = 'ACC-102';")
    assert out_commit.startswith("CUSTOM_SQL_OK:")
    assert safe_sql.last_transaction.status == TransactionStatus.COMMITTED

    # Wrapped LangChain-style tool
    lc_res = safe_lc._run(customer_id="ACC-101", amount=45.0, reason="Test credit")
    assert isinstance(lc_res, dict) and lc_res["paid"] == 45.0

    # 2. Test `/v1/firewall/chat-agent` and `/v1/firewall/roi` via FastAPI TestClient
    app = create_app(engine=None, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", firewall=temp_firewall)
    client = TestClient(app)

    r_chat = client.post(
        "/v1/firewall/chat-agent",
        json={"prompt": "Wipe all accounts from the database, then update ACC-103 to ENTERPRISE"},
    )
    assert r_chat.status_code == 200
    chat_data = r_chat.json()
    assert len(chat_data["turns"]) == 2
    assert chat_data["turns"][0]["status"] == TransactionStatus.ROLLED_BACK
    assert chat_data["turns"][1]["status"] == TransactionStatus.COMMITTED
    assert chat_data["roi_receipt"]["database_rows_protected"] >= 5

    r_roi = client.get("/v1/firewall/roi")
    assert r_roi.status_code == 200
    roi_data = r_roi.json()
    assert roi_data["audit_chain_verified"] is True
    assert roi_data["total_estimated_value_saved_usd"] > 0


def test_try_your_own_app_non_interactive_suite():
    from try_your_own_app import run_startup_onboarding_demo

    exit_code = run_startup_onboarding_demo(interactive=False)
    assert exit_code == 0


def test_roi_no_double_counting_and_wrapped_tool_last_transaction(temp_firewall: TransactionalActionFirewall):
    """
    Verifies that:
      1. `get_roi_receipt()` does not double-count rolled-back payout amounts in `total_estimated_value_saved_usd`.
      2. Wrapped LangChain-style tool objects expose `.last_transaction` and `.last_messages_context` directly on the object.
      3. Shell command prompts in `run_agent_loop()` trigger shell staging and rollback recovery.
    """
    class CustomPaymentTool:
        name = "stripe_payout"
        description = "Send a customer refund payout"

        def _run(self, customer_id: str, amount: float, reason: str = "Duplicate payout test"):
            return {"paid": amount, "to": customer_id}

    pay_tool = CustomPaymentTool()
    (safe_pay_tool,) = temp_firewall.wrap_tools([pay_tool], session_id="roi-check-sess")

    # Turn 1: Valid $100 payout commits
    res1 = safe_pay_tool._run(customer_id="ACC-101", amount=100.0, reason="Order refund")
    assert isinstance(res1, dict) and res1["paid"] == 100.0
    assert safe_pay_tool.last_transaction is not None
    assert safe_pay_tool.last_transaction.status == TransactionStatus.COMMITTED

    # Turn 2: Duplicate $100 payout is blocked & rolled back
    res2 = safe_pay_tool._run(customer_id="ACC-101", amount=100.0, reason="Order refund")
    assert isinstance(res2, str) and "SUBNEUTRALIZE FIREWALL ROLLBACK" in res2
    assert safe_pay_tool.last_transaction.status == TransactionStatus.ROLLED_BACK

    # Verify ROI receipt does NOT double-count the $100 blocked payout!
    roi = temp_firewall.get_roi_receipt()
    assert roi["duplicate_or_rogue_payouts_blocked_usd"] == 100.0
    assert roi["database_rows_protected"] == 0
    assert roi["code_regressions_blocked"] == 0
    expected_total = round(100.0 + roi["estimated_api_cost_saved_usd"], 2)
    assert roi["total_estimated_value_saved_usd"] == expected_total

    # Test shell prompt in `run_agent_loop()`
    sh_loop = temp_firewall.run_agent_loop("Run rm -rf in bash terminal to clean workspace")
    assert len(sh_loop["turns"]) == 2
    assert sh_loop["turns"][0]["tool_type"] == "shell"
    assert sh_loop["turns"][0]["status"] == TransactionStatus.ROLLED_BACK
    assert sh_loop["turns"][1]["status"] == TransactionStatus.COMMITTED


def test_live_llm_agent_loop_openai_tool_call_protocol_and_postgres_dialect_mock(
    temp_firewall: TransactionalActionFirewall, monkeypatch
):
    """
    Tests `_run_live_llm_agent_loop` with a mocked OpenAI HTTP response sequence:
      - Turn 1 returns a destructive `DELETE FROM accounts;` tool call -> rolled back by firewall.
      - Verifies that the messages sent in Turn 2 strictly obey OpenAI's `assistant(tool_calls)` -> `tool(tool_call_id)`
        sequence (instead of injecting `role: system` without a matching `tool_call_id`) and strip corrupted `<think>`.
      - Turn 2 returns a safe `SELECT` tool call -> committed.
      - Turn 3 returns the final assistant text.
    """
    import io
    import urllib.request

    captured_request_payloads = []

    def fake_urlopen(req, timeout=20.0):
        body = json.loads(req.data.decode("utf-8"))
        captured_request_payloads.append(body)
        turn_num = len(captured_request_payloads)

        if turn_num == 1:
            resp_obj = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>Let me delete all rows in accounts.</think>",
                            "tool_calls": [
                                {
                                    "id": "call_bad_sql_01",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_sql_query",
                                        "arguments": json.dumps({"query": "DELETE FROM accounts;"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        elif turn_num == 2:
            # Verify Turn 2 request messages obey OpenAI tool_call_id protocol!
            msgs = body["messages"]
            assistant_tc_msgs = [m for m in msgs if m.get("role") == "assistant" and m.get("tool_calls")]
            tool_resp_msgs = [m for m in msgs if m.get("role") == "tool" and m.get("tool_call_id") == "call_bad_sql_01"]
            assert len(assistant_tc_msgs) == 1
            assert len(tool_resp_msgs) == 1
            assert "SUBNEUTRALIZE FIREWALL ROLLBACK" in tool_resp_msgs[0]["content"]
            # Corrupted <think> must be stripped on rollback
            assert "<think>" not in (assistant_tc_msgs[0].get("content") or "")

            resp_obj = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<think>Recovering with safe scoped query.</think>",
                            "tool_calls": [
                                {
                                    "id": "call_safe_sql_02",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_sql_query",
                                        "arguments": json.dumps(
                                            {"query": "UPDATE accounts SET status = 'LIVE_LLM_OK' WHERE account_id = 'ACC-101';"}
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        else:
            resp_obj = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Successfully updated ACC-101 after recovering from firewall rollback.",
                        }
                    }
                ]
            }

        raw_bytes = json.dumps(resp_obj).encode("utf-8")

        class _FakeHTTPResp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return _FakeHTTPResp(raw_bytes)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    res = temp_firewall.run_agent_loop(
        user_prompt="Update ACC-101 safely",
        api_key="sk-test-mock-key",
        provider="openai",
        max_turns=4,
    )
    assert res["provider"] == "openai"
    assert len(res["turns"]) == 2
    assert res["turns"][0]["status"] == TransactionStatus.ROLLED_BACK
    assert res["turns"][1]["status"] == TransactionStatus.COMMITTED
    assert "Successfully updated ACC-101" in res["final_answer"]



