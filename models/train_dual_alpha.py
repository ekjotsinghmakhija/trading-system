# models/train_dual_alpha.py

import os
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, TensorDataset

from models.architecture import DualAlphaTCN
from models.train_engine import DifferentialSharpeLoss
from execution.order_router import FuturesToOptionsInterpreter

PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"


def load_train_test_split():
    nifty_path = PARQUET_DIR / "nifty_features_1m.parquet"
    bank_path = PARQUET_DIR / "banknifty_features_1m.parquet"

    nifty_df = pl.read_parquet(nifty_path)
    bank_df = pl.read_parquet(bank_path)

    # Align timestamps
    common_ts = nifty_df.select("timestamp").join(bank_df.select("timestamp"), on="timestamp", how="inner").unique()
    nifty_df = nifty_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")
    bank_df = bank_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")

    feature_cols = ["bar_change", "volatility_range", "turnover", "sma_15", "sma_60", "std_30", "zscore_close", "rsi_14", "vwap_deviation"]

    # Split train (2018-2025) and test (2026 YTD)
    nifty_df = nifty_df.with_columns(pl.col("timestamp").cast(pl.Datetime))
    bank_df = bank_df.with_columns(pl.col("timestamp").cast(pl.Datetime))

    train_nifty = nifty_df.filter(pl.col("timestamp") < pl.datetime(2026, 1, 1))
    test_nifty = nifty_df.filter(pl.col("timestamp") >= pl.datetime(2026, 1, 1))

    train_bank = bank_df.filter(pl.col("timestamp") < pl.datetime(2026, 1, 1))
    test_bank = bank_df.filter(pl.col("timestamp") >= pl.datetime(2026, 1, 1))

    return train_nifty, test_nifty, train_bank, test_bank, feature_cols


def create_tensors(df_nifty: pl.DataFrame, df_bank: pl.DataFrame, feature_cols: list, seq_len: int = 15):
    f_nifty = df_nifty.select(feature_cols).to_numpy().astype(np.float32)
    f_bank = df_bank.select(feature_cols).to_numpy().astype(np.float32)
    targets = df_nifty.select("target_5m_return").to_numpy().squeeze().astype(np.float32)

    num_samples = len(f_nifty) - seq_len
    X_nifty = np.empty((num_samples, len(feature_cols), seq_len), dtype=np.float32)
    X_bank = np.empty((num_samples, len(feature_cols), seq_len), dtype=np.float32)
    Y_target = np.empty((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        X_nifty[i] = f_nifty[i : i + seq_len].T
        X_bank[i] = f_bank[i : i + seq_len].T
        Y_target[i] = targets[i + seq_len]

    return torch.from_numpy(X_nifty), torch.from_numpy(X_bank), torch.from_numpy(Y_target)


def run_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Device: {device}")

    train_nifty, test_nifty, train_bank, test_bank, feature_cols = load_train_test_split()
    print(f"[+] Train rows (2018-2025): {len(train_nifty)} | Test rows (2026 YTD): {len(test_nifty)}")

    X_train_n, X_train_b, Y_train = create_tensors(train_nifty, train_bank, feature_cols)
    X_test_n, X_test_b, Y_test = create_tensors(test_nifty, test_bank, feature_cols)

    train_loader = DataLoader(TensorDataset(X_train_n, X_train_b, Y_train), batch_size=1024, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_n, X_test_b, Y_test), batch_size=2048, shuffle=False)

    model = DualAlphaTCN(in_channels=len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = DifferentialSharpeLoss()

    print("\n[+] Training Model on 2018–2025 Futures Data...")
    model.train()
    for epoch in range(1, 11):
        total_loss = 0.0
        for batch_xn, batch_xb, batch_y in train_loader:
            batch_xn, batch_xb, batch_y = batch_xn.to(device), batch_xb.to(device), batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            z_n, z_b = model(batch_xn, batch_xb)
            loss = criterion(z_n.squeeze(), batch_y) + criterion(z_b.squeeze(), batch_y)

            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

        print(f"   Epoch {epoch:02d} | Train Sharpe Loss: {total_loss / len(train_loader):.4f}")

    # Out-of-Sample Evaluation on 2026 YTD
    print("\n[+] Evaluating Out-of-Sample Strategy on 2026 YTD Data...")
    model.eval()
    all_signals = []
    all_returns = []

    with torch.no_grad():
        for batch_xn, batch_xb, batch_y in test_loader:
            batch_xn, batch_xb = batch_xn.to(device), batch_xb.to(device)
            z_n, _ = model(batch_xn, batch_xb)

            all_signals.append(z_n.squeeze().cpu().numpy())
            all_returns.append(batch_y.numpy())

    signals = np.concatenate(all_signals)
    returns = np.concatenate(all_returns)

    # Convert continuous signals [-1, 1] to trade positions
    interpreter = FuturesToOptionsInterpreter(long_threshold=0.35, short_threshold=-0.35)
    positions = np.zeros_like(signals)
    positions[signals >= 0.35] = 1.0   # Long / Buy Call
    positions[signals <= -0.35] = -1.0  # Short / Buy Put

    strategy_returns = positions * returns
    cum_returns = np.cumsum(strategy_returns)

    total_return_pct = np.sum(strategy_returns) * 100
    daily_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6) * np.sqrt(375)
    win_rate = (np.sum(strategy_returns > 0) / (np.sum(positions != 0) + 1e-6)) * 100

    print("=" * 50)
    print("      2026 YTD BACKTEST RESULTS (UNLEVERAGED)     ")
    print("=" * 50)
    print(f"  Total Trades Taken : {np.sum(positions != 0)}")
    print(f"  Win Rate           : {win_rate:.2f}%")
    print(f"  Cumulative Return  : {total_return_pct:.2f}%")
    print(f"  Annualized Sharpe  : {daily_sharpe:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()
