import os
import sys
import gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import duckdb
import torch
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, TensorDataset

from models.architecture import DualAlphaTCN
from models.train_engine import DifferentialSharpeLoss
from eval.evaluate_cpcv import CombinatorialPurgedCV
from features.factor_ledger import FactorLedger

DB_PATH = "data/duckdb/market_data.duckdb"

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

def setup_hardware():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpus = os.cpu_count() or 4
    torch.set_num_threads(num_cpus)
    print(f"[Hardware Setup] Active Device: {device} | Max CPU Threads: {num_cpus}")
    if device.type == "cuda":
        print(f"                 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"                 Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device, num_cpus

def prepare_tensors(df: pl.DataFrame, feature_cols: list, seq_len: int = 15):
    feature_matrix = df.select(feature_cols).to_numpy().astype(np.float32)
    target_returns = df.select("target_5m_return").to_numpy().squeeze().astype(np.float32)

    X_seq, Y_seq = [], []
    for i in range(seq_len, len(feature_matrix)):
        X_seq.append(feature_matrix[i - seq_len:i].T)
        Y_seq.append(target_returns[i])

    return torch.tensor(np.array(X_seq), dtype=torch.float32), torch.tensor(np.array(Y_seq), dtype=torch.float32)

def load_and_preprocess_data(con):
    nifty_df = con.execute("SELECT * FROM nifty_features_1m ORDER BY timestamp").pl()
    banknifty_df = con.execute("SELECT * FROM banknifty_features_1m ORDER BY timestamp").pl()

    nifty_df = nifty_df.with_columns(pl.col("timestamp").cast(pl.Utf8))
    banknifty_df = banknifty_df.with_columns(pl.col("timestamp").cast(pl.Utf8))

    meta_cols = {"timestamp", "trading_date", "target_5m_return"}
    candidate_cols = sorted(list((set(nifty_df.columns) & set(banknifty_df.columns)) - meta_cols))

    feature_cols = []
    for col in candidate_cols:
        nifty_nulls = nifty_df.select(pl.col(col).null_count()).item() / len(nifty_df)
        bank_nulls = banknifty_df.select(pl.col(col).null_count()).item() / len(banknifty_df)
        if nifty_nulls < 0.20 and bank_nulls < 0.20:
            feature_cols.append(col)

    nifty_df = nifty_df.with_columns([pl.col(c).forward_fill().backward_fill() for c in feature_cols])
    banknifty_df = banknifty_df.with_columns([pl.col(c).forward_fill().backward_fill() for c in feature_cols])

    req_cols = feature_cols + ["target_5m_return"]
    nifty_df = nifty_df.drop_nulls(subset=req_cols)
    banknifty_df = banknifty_df.drop_nulls(subset=req_cols)

    common_ts = nifty_df.select("timestamp").join(banknifty_df.select("timestamp"), on="timestamp", how="inner").unique()
    nifty_df = nifty_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")
    banknifty_df = banknifty_df.join(common_ts, on="timestamp", how="inner").sort("timestamp")

    return nifty_df, banknifty_df, feature_cols

def run_training_pipeline():
    device, num_cpus = setup_hardware()

    print("[+] Connecting to DuckDB...")
    con = duckdb.connect(DB_PATH)
    nifty_df, banknifty_df, feature_cols = load_and_preprocess_data(con)
    con.close()

    print(f"[+] Aligned Dataset Size: {len(nifty_df)} rows | Feature Dimension: {len(feature_cols)}")

    ledger = FactorLedger(hazard_threshold=-0.0015, k_neighbors=25)
    stats = ledger.build_ledger(nifty_df, feature_cols)
    print(f"[+] Factor Ledger Built: {stats}")

    print("[+] Preparing Sequence Tensors...")
    X_nifty, Y_target = prepare_tensors(nifty_df, feature_cols, seq_len=15)
    X_banknifty, _ = prepare_tensors(banknifty_df, feature_cols, seq_len=15)

    del nifty_df, banknifty_df
    gc.collect()

    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=1, purge_window=5, embargo_pct=0.01)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    fold = 1
    for train_idx, test_idx in cpcv.split(X_nifty):
        print(f"\n--- CPCV Fold {fold} ---")

        train_dataset = TensorDataset(X_nifty[train_idx], X_banknifty[train_idx], Y_target[train_idx])
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(512, len(train_idx)),
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=min(4, num_cpus)
        )

        test_dataset = TensorDataset(X_nifty[test_idx], X_banknifty[test_idx], Y_target[test_idx])
        test_loader = DataLoader(
            test_dataset,
            batch_size=2048,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=min(4, num_cpus)
        )

        model = DualAlphaTCN(in_channels=len(feature_cols)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = DifferentialSharpeLoss()

        model.train()
        for epoch in range(10):
            for batch_xn, batch_xb, batch_y in train_loader:
                batch_xn = batch_xn.to(device, non_blocking=True)
                batch_xb = batch_xb.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad()

                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        z_nifty, z_banknifty = model(batch_xn, batch_xb)
                        loss = criterion(z_nifty, batch_y) + criterion(z_banknifty, batch_y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    z_nifty, z_banknifty = model(batch_xn, batch_xb)
                    loss = criterion(z_nifty, batch_y) + criterion(z_banknifty, batch_y)
                    loss.backward()
                    optimizer.step()

        model.eval()
        all_z_test = []
        all_y_test = []

        with torch.no_grad():
            for batch_xn, batch_xb, batch_y in test_loader:
                batch_xn = batch_xn.to(device, non_blocking=True)
                batch_xb = batch_xb.to(device, non_blocking=True)

                if device.type == "cuda":
                    with torch.amp.autocast('cuda'):
                        z_n_test, _ = model(batch_xn, batch_xb)
                else:
                    z_n_test, _ = model(batch_xn, batch_xb)

                all_z_test.append(z_n_test.cpu())
                all_y_test.append(batch_y.cpu())

        # Ensure 1D shape alignment to prevent (N, 1) * (N,) outer-product broadcasting
        full_z = torch.cat(all_z_test, dim=0).view(-1)
        full_y = torch.cat(all_y_test, dim=0).view(-1)

        test_loss = criterion(full_z, full_y).item()

        print(f"  └─ Fold {fold} Test Sharpe Loss: {test_loss:.4f}")

        del model, train_loader, test_loader, train_dataset, test_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        fold += 1

if __name__ == "__main__":
    run_training_pipeline()
