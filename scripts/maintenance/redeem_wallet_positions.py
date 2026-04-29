#!/usr/bin/env python3
"""
Standalone Polymarket redemption runner for wallet testing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRADER_PATH = PROJECT_ROOT / "trading bot" / "hourly_24h_trader.py"
ENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        parsed_value = value.strip()
        if len(parsed_value) >= 2 and parsed_value[0] == parsed_value[-1] and parsed_value[0] in {"'", '"'}:
            parsed_value = parsed_value[1:-1]
        os.environ[key] = parsed_value


def load_trader_module() -> Any:
    spec = importlib.util.spec_from_file_location("hourly_24h_trader_runtime", TRADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load trader module from {TRADER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preview_redeem_groups(trader: Any) -> int:
    wallet_address = trader.tournament.get_env_str("POLYMARKET_PROXY_WALLET") or trader.tournament.get_env_str("WALLET")
    if not wallet_address:
        raise RuntimeError("Missing POLYMARKET_PROXY_WALLET or WALLET environment variable.")

    positions = trader.fetch_auto_redeem_positions(wallet_address)
    redeem_groups = trader.build_auto_redeem_groups(positions)
    if not redeem_groups:
        print("[auto-redeem] No redeemable winner or zero-value cleanup markets found.", flush=True)
        return 0

    print(f"[auto-redeem] Previewing {len(redeem_groups)} redeem group(s).", flush=True)
    for group in redeem_groups:
        amount = Decimal(group["total_value"])
        action_parts: list[str] = []
        if group.get("has_winner_value"):
            action_parts.append(f"winner ${amount:.4f}")
        if group.get("has_zero_value_cleanup"):
            action_parts.append("zero-value cleanup")
        action_label = ", ".join(action_parts) if action_parts else "cleanup"
        print(
            f"[auto-redeem] {group['title']} | {action_label} | "
            f"positions={len(group.get('positions', []))} | negative_risk={bool(group.get('negative_risk'))}",
            flush=True,
        )
    return 0


def run_redeem(trader: Any) -> int:
    total_redeemed = trader.auto_redeem_winner_positions_at_start()
    print(f"[auto-redeem] Standalone finished | total redeemed ${Decimal(total_redeemed):.4f}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone wallet redemption tester.")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="List redeemable groups without executing any redemption transaction.",
    )
    return parser


def main() -> int:
    load_dotenv_file(ENV_PATH)
    args = build_parser().parse_args()
    trader = load_trader_module()
    if args.preview:
        return preview_redeem_groups(trader)
    return run_redeem(trader)


if __name__ == "__main__":
    raise SystemExit(main())
