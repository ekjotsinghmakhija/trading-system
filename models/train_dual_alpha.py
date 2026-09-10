import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import torch
import numpy as np
import polars as pl

from models.architecture import DualAlphaTCN
from models.train_engine import DifferentialSharpeLoss
from eval.evaluate_cpcv import CombinatorialPurgedCV
from features.factor_ledger import FactorLedger

DB_PATH = "data/duckdb/market_data.duckdb"

def prepare_tensors(df: pl.DataFrame, feature_cols: list, seq_len: int = 15):
    """Formats 2D feature table into 3D sliding sequence tensors [N, Channels, Seq_Len]."""
    feature_matrix = df.select(feature_cols).to_numpy()
    target_returns = df.select("target_5m_return").to_numpy()

    X_seq, Y_seq = [], []
    for i in range(seq_len, len(feature_matrix)):
        X_seq.append(feature_matrix[i - seq_len:i].T)
        Y_seq.append(target_returns[i])

    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(Y_seq), dtype=torch.float32)

def run_training_pipeline():
    print("[+] Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH)

    print("[+] Loading engineered feature tables...")
    nifty_df = con.execute("SELECT * FROM nifty_features_1m ORDER BY timestamp").pl().drop_nulls()
    banknifty_df = con.execute("SELECT * FROM banknifty_features_1m ORDER BY timestamp").pl().drop_nulls()
    con.close()

    # Align on common timestamps using inner join
    common_ts = nifty_df.select("timestamp").join(banknifty_df.select("timestamp"), on="timestamp", how="inner").unique()
    nifty_df = nifty_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")
    banknifty_df = banknifty_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")

    feature_cols = [col for col in nifty_df.columns if col not in ["timestamp", "trading_date", "target_5m_return"]]

    ledger = FactorLedger()
    stats = ledger.build_ledger(nifty_df, feature_cols)
    print(f"[+] Factor Ledger Built: {stats}")

    print("[+] Preparing Sequence Tensors...")
    X_nifty, Y_target = prepare_tensors(nifty_df, feature_cols, seq_len=15)
    X_banknifty, _ = prepare_tensors(banknifty_df, feature_cols, seq_len=15)

    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge_window=5, embargo_pct=0.01)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Training on Device: {device}")

    fold = 1
    for train_idx, test_idx in cpcv.split(nifty_df.slice(15)):
        print(f"\n--- CPCV Fold {fold} ---")

        X_n_tr, X_bn_tr, Y_tr = X_nifty[train_idx].to(device), X_banknifty[train_idx].to(device), Y_target[train_idx].to(device)
        X_n_te, X_bn_te, Y_te = X_nifty[test_idx].to(device), X_banknifty[test_idx].to(device), Y_target[test_idx].to(device)

        model = DualAlphaTCN(in_channels=len(feature_cols)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = DifferentialSharpeLoss()

        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            z_nifty, z_banknifty = model(X_n_tr, X_bn_tr)

            loss_nifty = criterion(z_nifty, Y_tr)
            loss_banknifty = criterion(z_banknifty, Y_tr)
            total_loss = loss_nifty + loss_banknifty

            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z_n_test, z_bn_test = model(X_n_te, X_bn_te)
            test_loss = criterion(z_n_test, Y_te).item()
            print(f"  └─ Fold {fold} Test Sharpe Loss: {test_loss:.4f}")

        fold += 1

if __name__ == "__main__":
    run_training_pipeline()
