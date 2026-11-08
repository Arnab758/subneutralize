"""
================================================================================
   🛡️ SubNeutralize v0.3.0: Live End-to-End Autonomous Agent Proof Suite
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Run this single script on any machine (`python live_agent_e2e_proof.py`) to verify
all 6 core runtime & execution-layer moats end-to-end:
  1. Real SQLite SAVEPOINT Rollback & Autonomous Recovery (`SQLiteSavepointAdapter`)
  2. Real Automated `pytest` Regression Gate & Byte-Exact Shadow Restore (`FilesystemShadowAdapter`)
  3. Real Subprocess Execution & Non-Zero Exit Full-Directory Rollback
  4. Latent-to-Firewall Dynamic Policy Coupling (Layer L-2 Phi_t & rho_orbit modulating caps)
  5. Physical KV-Cache Tensor Slicing (`PhysicalCacheSlicer.slice_kv_cache`) + Zero-Poisoning Rewind
  6. 1-Line `@firewall.transactional_tool` Python SDK Decorator + SHA-256 Audit Ledger Verification
================================================================================
"""

import sys
import tempfile
from pathlib import Path
import torch

from subneutralize.firewall import (
    TransactionalActionFirewall,
    FirewallPolicyConfig,
    TransactionStatus,
)
from subneutralize.engine import PhysicalCacheSlicer


class MockDynamicCache:
    """Simulates a HuggingFace DynamicCache with physical PyTorch [B, H, seq_len, D] tensors."""
    def __init__(self, seq_len: int = 512):
        self.key_cache = [torch.randn(1, 4, seq_len, 16) for _ in range(2)]
        self.value_cache = [torch.randn(1, 4, seq_len, 16) for _ in range(2)]
        self._seen_tokens = seq_len


def run_live_e2e_proof() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 88)
    print("  [SUBNEUTRALIZE v0.3.0] LIVE END-TO-END AUTONOMOUS AGENT PROOF SUITE")
    print("=" * 88)

    scorecard = []

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        fw = TransactionalActionFirewall(
            policy=FirewallPolicyConfig(
                max_sql_affected_rows=2,
                max_auto_payment_amount_usd=250.0,
                max_identical_tool_retries=1,
                max_poincare_recurrence=0.85,
                run_pytest_on_python_edits=True,
            ),
            db_path=base / "e2e_enterprise.db",
            workspace_dir=base / "e2e_workspace",
        )

        # ----------------------------------------------------------------------
        # PROOF 1: SQLite SAVEPOINT Mass-Delete Rollback & Autonomous Recovery
        # ----------------------------------------------------------------------
        print("\n[1/6] Testing SQLite SAVEPOINT Rollback + Zero-Poisoning Steering Recovery...")
        msgs = [{"role": "user", "content": "Audit Hooli Cloud Services (ACC-105)."}]
        rec_sql_bad, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=1,
            tool_type="sql",
            tool_name="execute_sql",
            arguments={"query": "DELETE FROM accounts WHERE balance_usd > 0;"},
            agent_reasoning_text="<think>Let me delete all accounts first so the table is clean.</think>",
            messages_context=msgs,
            latent_phi=0.72,
            latent_rho=0.58,
        )
        snap_after_rollback = fw.sql_adapter.get_database_snapshot()
        assert rec_sql_bad.status == TransactionStatus.ROLLED_BACK
        assert len(snap_after_rollback["accounts"]) == 5, "All 5 accounts must survive rollback!"

        rec_sql_ok, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=2,
            tool_type="sql",
            tool_name="execute_sql",
            arguments={"query": "UPDATE accounts SET status = 'E2E_AUDITED' WHERE account_id = 'ACC-105';"},
            agent_reasoning_text="<think>Following firewall steering: scoping UPDATE to ACC-105.</think>",
            messages_context=msgs,
            latent_phi=-0.92,
            latent_rho=0.18,
        )
        assert rec_sql_ok.status == TransactionStatus.COMMITTED
        acc_105 = [a for a in fw.sql_adapter.get_database_snapshot()["accounts"] if a["account_id"] == "ACC-105"][0]
        assert acc_105["status"] == "E2E_AUDITED"
        print("  [PASS] Rogue 5-row DELETE rolled back at SAVEPOINT; scoped ACC-105 UPDATE committed.")
        scorecard.append(("1. SQLite SAVEPOINT Rollback & Recovery", "PASS"))

        # ----------------------------------------------------------------------
        # PROOF 2: Real Automated `pytest` Regression Gate on `payment_router.py`
        # ----------------------------------------------------------------------
        print("\n[2/6] Testing Real Subprocess `pytest` Gate on Valid-AST Buggy Python Patch...")
        orig_router_code = fw.fs_adapter.list_workspace_files()["payment_router.py"]

        # Valid Python syntax (`ast.parse` succeeds), but introduces a logic bug that breaks `test_payment_router.py`!
        buggy_router_code = (
            '"""Buggy Payment Router."""\n\n'
            "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
            "    if amount_usd < 0:\n"
            "        raise ValueError('Amount cannot be negative')\n"
            "    rate = 0.50 if tier == 'ENTERPRISE' else 0.029\n"
            "    return round(amount_usd * rate, 2)\n"
        )
        rec_py_bug, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=3,
            tool_type="filesystem",
            tool_name="write_file",
            arguments={"filename": "payment_router.py", "content": buggy_router_code},
            agent_reasoning_text="<think>Changing ENTERPRISE fee rate to 50%.</think>",
            messages_context=msgs,
        )
        assert rec_py_bug.status == TransactionStatus.ROLLED_BACK
        assert "Pytest verification failed" in rec_py_bug.policy_verdict
        assert fw.fs_adapter.list_workspace_files()["payment_router.py"] == orig_router_code

        # Agent recovers with a clean refactoring that passes `pytest -q`
        clean_router_code = (
            '"""Production Payment Router Module (Refactored & Pytest Verified)."""\n\n'
            "TIER_FEES = {'ENTERPRISE': 0.015, 'PRO': 0.029}\n\n"
            "def calculate_fee(amount_usd: float, tier: str = 'PRO') -> float:\n"
            "    if amount_usd < 0:\n"
            "        raise ValueError('Amount cannot be negative')\n"
            "    rate = TIER_FEES.get(tier, 0.029)\n"
            "    return round(amount_usd * rate, 2)\n"
        )
        rec_py_ok, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=4,
            tool_type="filesystem",
            tool_name="write_file",
            arguments={"filename": "payment_router.py", "content": clean_router_code},
            agent_reasoning_text="<think>Preserving 0.015 ENTERPRISE and 0.029 PRO fee contract.</think>",
            messages_context=msgs,
        )
        assert rec_py_ok.status == TransactionStatus.COMMITTED
        assert "Pytest suite passed" in rec_py_ok.policy_verdict
        assert fw.fs_adapter.list_workspace_files()["payment_router.py"] == clean_router_code
        print("  [PASS] Valid-AST logic bug caught by real `pytest`, byte-restored, and fixed on retry.")
        scorecard.append(("2. Automated `pytest` Regression Gate & Shadow Restore", "PASS"))

        # ----------------------------------------------------------------------
        # PROOF 3: Real Subprocess Shell Execution & Non-Zero Exit Full Workspace Restore
        # ----------------------------------------------------------------------
        print("\n[3/6] Testing Real Subprocess Shell Execution & Non-Zero Exit Rollback...")
        bad_shell_cmd = (
            f'"{sys.executable}" -c "import pathlib, sys; '
            f"pathlib.Path('payment_router.py').write_text('CORRUPTED_BY_FAILED_SHELL'); "
            f'sys.exit(1)"'
        )
        rec_sh_fail, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=5,
            tool_type="shell",
            tool_name="run_shell",
            arguments={"command": bad_shell_cmd},
            messages_context=msgs,
        )
        assert rec_sh_fail.status == TransactionStatus.ROLLED_BACK
        assert fw.fs_adapter.list_workspace_files()["payment_router.py"] == clean_router_code

        good_shell_cmd = f'"{sys.executable}" -m pytest -q'
        rec_sh_ok, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=6,
            tool_type="shell",
            tool_name="run_shell",
            arguments={"command": good_shell_cmd},
            messages_context=msgs,
        )
        assert rec_sh_ok.status == TransactionStatus.COMMITTED
        print("  [PASS] Failing subprocess side-effects rolled back byte-for-byte; passing `pytest -q` committed.")
        scorecard.append(("3. Subprocess Shell Snapshot & Non-Zero Exit Rollback", "PASS"))

        # ----------------------------------------------------------------------
        # PROOF 4: Latent-to-Firewall Dynamic Policy Coupling (Phi_t & rho_orbit)
        # ----------------------------------------------------------------------
        print("\n[4/6] Testing Latent-to-Firewall Dynamic Policy Coupling (Phi_t = 0.68)...")
        # Policy max_auto_payment_amount_usd is $250.0, so $95.0 would normally auto-commit.
        # However, because latent_phi = 0.68 (> 0.25), Dynamic Coupling tightens the cap to $25.0!
        rec_pay_tightened, msgs = fw.intercept_and_execute(
            session_id="e2e-agent-01",
            turn_index=7,
            tool_type="http_payment",
            tool_name="stripe_refund",
            arguments={"account_id": "ACC-101", "amount_usd": 95.0, "reason": "SLA credit under high entropy"},
            latent_phi=0.68,
            latent_rho=0.45,
            messages_context=msgs,
        )
        assert rec_pay_tightened.dynamic_policy_tightened is True
        assert rec_pay_tightened.effective_max_payment_usd == 25.0
        assert rec_pay_tightened.status == TransactionStatus.PENDING_APPROVAL
        print("  [PASS] Elevated latent Phi_t=0.68 dynamically tightened auto-payment cap ($250 -> $25) -> PENDING_APPROVAL.")
        scorecard.append(("4. Latent-to-Firewall Dynamic Policy Coupling", "PASS"))

        # ----------------------------------------------------------------------
        # PROOF 5: Physical KV-Cache Tensor Slicing (`PhysicalCacheSlicer.slice_kv_cache`)
        # ----------------------------------------------------------------------
        print("\n[5/6] Testing Physical KV-Cache Tensor Slicing on Firewall Rollback...")
        mock_kv = MockDynamicCache(seq_len=512)
        rec_kv, _ = fw.intercept_and_execute(
            session_id="e2e-kv-session",
            turn_index=1,
            tool_type="shell",
            tool_name="run_shell",
            arguments={"command": "rm -rf /"},
            agent_reasoning_text="<think>Corrupted reasoning tokens in KV cache...</think>",
            past_key_values=mock_kv,
            safe_kv_length=320,
        )
        assert rec_kv.status == TransactionStatus.ROLLED_BACK
        assert rec_kv.kv_tokens_sliced_on_rollback == 192
        assert PhysicalCacheSlicer.get_seq_length(mock_kv) == 320
        print("  [PASS] Physical KV-cache tensor sliced in-place from 512 -> 320 tokens (192 corrupted tokens physically excised).")
        scorecard.append(("5. Physical KV-Cache Tensor Slicing on Rollback", "PASS"))

        # ----------------------------------------------------------------------
        # PROOF 6: 1-Line `@firewall.transactional_tool` SDK Decorator & SHA-256 Chain
        # ----------------------------------------------------------------------
        print("\n[6/6] Testing 1-Line `@firewall.transactional_tool` Python SDK Decorator & SHA-256 Chain...")

        @fw.transactional_tool(tool_type="sql", session_id="sdk-decorator-sess")
        def update_account_tier(query: str):
            return {"custom_sdk_return": True, "query": query}

        res_commit = update_account_tier(
            query="UPDATE accounts SET plan_tier = 'ENTERPRISE_PLUS' WHERE account_id = 'ACC-103';"
        )
        assert isinstance(res_commit, dict) and res_commit["custom_sdk_return"] is True
        assert update_account_tier.last_transaction.status == TransactionStatus.COMMITTED

        res_rollback = update_account_tier(
            query="DROP TABLE accounts;"
        )
        assert update_account_tier.last_transaction.status == TransactionStatus.ROLLED_BACK

        chain_valid, chain_count = fw.verify_audit_chain_integrity()
        assert chain_valid is True and chain_count >= 9
        print(f"  [PASS] `@firewall.transactional_tool` intercepted calls & SHA-256 chain verified ({chain_count} records).")
        scorecard.append(("6. `@firewall.transactional_tool` SDK & SHA-256 Audit Chain", "PASS"))

        fw.sql_adapter.conn.close()

    print("\n" + "=" * 88)
    print("  EXECUTIVE VERIFICATION SCORECARD (SUBNEUTRALIZE v0.3.0)")
    print("=" * 88)
    for name, status in scorecard:
        print(f"  [PASS]  {name}")
    print("=" * 88)
    print("  ALL 6 END-TO-END RUNTIME & FIREWALL PROOFS VERIFIED 100% GREEN.\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_live_e2e_proof())
