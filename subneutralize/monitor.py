"""
================================================================================
   📊 SubNeutralize v0.2.0: Real-Time Terminal Telemetry Monitor (CLI Top)
   Author: Arnab Dutta | Apache-2.0 License
================================================================================
Connects to a live SubNeutralize Inference Gateway (/v1/telemetry) and renders
a real-time, terminal dashboard showing:
  - Active Model & Hardware Status
  - Cumulative Request Volume & In-Flight Manifold Interventions
  - Stateful KV-Cache Eviction Counter & % Memory Saved
  - Visual KV Memory Compression Gauge
================================================================================
"""

import sys
import time
import os
import json
from typing import Optional, Dict, Any

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False


def fetch_telemetry(host: str = "127.0.0.1", port: int = 8000, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    """Fetches real-time telemetry from the SubNeutralize server."""
    url = f"http://{host}:{port}/v1/telemetry"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SubNeutralize-Monitor/0.2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data
    except Exception:
        return None
    return None


def render_bar(percentage: float, width: int = 30) -> str:
    """Renders a Unicode progress bar for memory savings."""
    clamped = max(0.0, min(100.0, percentage))
    filled_len = int(width * (clamped / 100.0))
    bar = "█" * filled_len + "░" * (width - filled_len)
    return f"[{bar}] {clamped:5.1f}%"


def format_number(n: int) -> str:
    """Formats large token counts into human-readable strings."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def generate_dashboard_text(telemetry: Optional[Dict[str, Any]], host: str, port: int) -> str:
    """Constructs the ASCII / Unicode terminal telemetry view."""
    lines = []
    lines.append("=" * 78)
    lines.append("        SUBNEUTRALIZE v0.2.0 : REAL-TIME INFERENCE GATEWAY MONITOR        ")
    lines.append("=" * 78)

    if telemetry is None:
        lines.append(f" [STATUS] Offline / Disconnected")
        lines.append(f" [TARGET] http://{host}:{port}/v1/telemetry")
        lines.append("-" * 78)
        lines.append(" Waiting for SubNeutralize gateway to start...")
        lines.append(" Tip: Run in another terminal: subneutralize serve --port 8000")
        lines.append("=" * 78)
        return "\n".join(lines)

    model = telemetry.get("model", "unknown")
    engine = telemetry.get("engine_version", "SubNeutralize v0.2.0")
    reqs = telemetry.get("total_requests", 0)
    prompt_toks = telemetry.get("total_prompt_tokens", 0)
    compl_toks = telemetry.get("total_completion_tokens", 0)
    evicted = telemetry.get("total_prefill_tokens_evicted", 0)
    compacted_reqs = telemetry.get("compacted_multi_turn_requests", 0)
    interventions = telemetry.get("manifold_interventions", 0)

    # Compute savings percentage
    total_prefill_raw = prompt_toks + evicted
    savings_pct = (evicted / max(1, total_prefill_raw)) * 100.0 if total_prefill_raw > 0 else 0.0

    lines.append(f" [STATUS] ONLINE (Connected to http://{host}:{port})")
    lines.append(f" [MODEL]  {model}")
    lines.append(f" [ENGINE] {engine}")
    lines.append("-" * 78)
    lines.append(" TELEMETRY & HARDWARE EFFICIENCY METRICS:")
    lines.append(f"   • Total Requests Handled:       {reqs:<12} (Compacted Multi-Turn: {compacted_reqs})")
    lines.append(f"   • Total Completion Tokens:      {format_number(compl_toks):<12} ({compl_toks} raw tokens)")
    lines.append(f"   • Net Prefill Tokens Served:    {format_number(prompt_toks):<12} ({prompt_toks} tokens)")
    lines.append(f"   • In-Flight Manifold Consensus: {interventions:<12} early exits triggered")
    lines.append("-" * 78)
    lines.append(" STATEFUL MULTI-TURN KV-CACHE SAVINGS:")
    lines.append(f"   • Tokens Evicted from Prefill:  {format_number(evicted):<12} ({evicted} transient tokens purged)")
    lines.append(f"   • Memory Reduction Ratio:       {render_bar(savings_pct, width=28)}")
    lines.append("-" * 78)
    lines.append(" [IDE INTEGRATION] Ready for Aider, Cursor, Cline, and Continue")
    lines.append(f"   -> Set OpenAI API Base: http://localhost:{port}/v1 (API key: subneutralize)")
    lines.append("=" * 78)
    return "\n".join(lines)


def run_terminal_monitor(host: str = "127.0.0.1", port: int = 8000, interval: float = 1.0, max_cycles: Optional[int] = None):
    """
    Launches an interactive live polling loop in the terminal.
    
    Args:
        host: Gateway host IP.
        port: Gateway port.
        interval: Polling frequency in seconds.
        max_cycles: Optional cycle ceiling for test/automation runners.
    """
    cycle = 0
    try:
        while True:
            data = fetch_telemetry(host=host, port=port)
            dashboard = generate_dashboard_text(data, host=host, port=port)

            # Clear terminal screen cleanly across OS
            if max_cycles is None:
                os.system("cls" if os.name == "nt" else "clear")

            print(dashboard)

            cycle += 1
            if max_cycles is not None and cycle >= max_cycles:
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[INFO] SubNeutralize Monitor stopped. Goodbye!")
