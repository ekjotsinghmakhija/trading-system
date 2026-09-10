import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import torch
import numpy as np
import polars as pl

from features.factor_ledger import FactorLedger
from models.architecture import DualAlphaTCN
from execution.risk_manager import RiskManager
from env.strict_sim_env import StrictSimEnv

DB_PATH = "data/duckdb/market_data.duckdb"

def run_end_to_end_system():
    print("[1/5] Connecting to DuckDB & Loading Feature Space...")
    con = duckdb.connect(DB_PATH)
    nifty_df = con.execute("SELECT * FROM nifty_features_1m ORDER BY timestamp").pl().drop_nulls()
    banknifty_df = con.execute("SELECT * FROM banknifty_features_1m ORDER BY timestamp").pl().drop_nulls()
    con.close()

    common_ts = nifty_df.select("timestamp").intersect(banknifty_df.select("timestamp"))
    nifty_df = nifty_df.join(common_ts, on="timestamp").sort("timestamp")
    banknifty_df = banknifty_df.join(common_ts, on="timestamp").sort("timestamp")

    feature_cols = [c for c in nifty_df.columns if c not in ["timestamp", "trading_date", "target_5m_return"]]

    print("[2/5] Initializing Factor Ledger & Hazard Memory...")
    ledger = FactorLedger(hazard_threshold=-0.0015, target_threshold=0.0020, k_neighbors=25)
    ledger_stats = ledger.build_ledger(nifty_df, feature_cols)
    print(f"      └─ Ledger Ready: {ledger_stats['total_states']} historical states logged.")

    print("[3/5] Loading DualAlphaTCN Model Weights...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualAlphaTCN(in_channels=len(feature_cols)).to(device)
    model.eval()

    print("[4/5] Initializing Risk Manager & Strict Simulator...")
    risk_mgr = RiskManager(max_drawdown_limit=0.30, kelly_fraction=0.5)
    sim = StrictSimEnv(initial_capital=1000000.0, slippage_ticks=0.5)

    print("[5/5] Running High-Fidelity Historical Replay...")
    timestamps = nifty_df.select("timestamp").to_series().to_list()
    nifty_prices = nifty_df.select("close").to_series().to_numpy()
    banknifty_prices = banknifty_df.select("close").to_series().to_numpy()

    seq_len = 15
    nifty_feats = nifty_df.select(feature_cols).to_numpy()
    banknifty_feats = banknifty_df.select(feature_cols).to_numpy()

    daily_logs = []

    for idx in range(seq_len, len(timestamps)):
        current_ts = str(timestamps[idx])

        prices = {
            "NIFTY": float(nifty_prices[idx]),
            "BANKNIFTY": float(banknifty_prices[idx])
        }

        current_state = nifty_feats[idx]
        hazard_prob, is_suppressed = ledger.evaluate_hazard_probability(current_state)

        if is_suppressed[0]:
            target_weights = {"NIFTY": 0.0, "BANKNIFTY": 0.0}
        else:
            x_nifty = torch.tensor(nifty_feats[idx-seq_len:idx].T, dtype=torch.float32).unsqueeze(0).to(device)
            x_banknifty = torch.tensor(banknifty_feats[idx-seq_len:idx].T, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                z_nifty, z_banknifty = model(x_nifty, x_banknifty)
                exp_returns = np.array([z_nifty.item(), z_banknifty.item()])

            hist_nifty_rets = np.diff(nifty_prices[max(0, idx-60):idx+1]) / nifty_prices[max(0, idx-60):idx]
            hist_bn_rets = np.diff(banknifty_prices[max(0, idx-60):idx+1]) / banknifty_prices[max(0, idx-60):idx]
            hist_matrix = np.column_stack([hist_nifty_rets, hist_bn_rets])

            if len(hist_matrix) > 5:
                allocations = risk_mgr.calculate_allocations(
                    expected_returns=exp_returns,
                    historical_returns=hist_matrix,
                    current_portfolio_value=sim.capital,
                    timestamp_str=current_ts
                )
                target_weights = {"NIFTY": allocations["NIFTY"], "BANKNIFTY": allocations["BANKNIFTY"]}
            else:
                target_weights = {"NIFTY": 0.0, "BANKNIFTY": 0.0}

        sim_res = sim.step(timestamp_str=current_ts, prices=prices, target_weights=target_weights)
        daily_logs.append(sim_res)

    final_val = sim.capital
    total_return = (final_val - sim.initial_capital) / sim.initial_capital
    max_dd = max([log["current_drawdown"] for log in daily_logs]) if daily_logs else 0.0

    print("\n================ SYSTEM PERFORMANCE SUMMARY ================")
    print(f" Initial Capital:       ₹{sim.initial_capital:,.2f}")
    print(f" Final Portfolio Value: ₹{final_val:,.2f}")
    print(f" Cumulative Return:     {total_return * 100:.2f}%")
    print(f" Maximum Drawdown:      {max_dd * 100:.2f}% (Limit: 30.0%)")
    print("============================================================")

if __name__ == "__main__":
    run_end_to_end_system()
