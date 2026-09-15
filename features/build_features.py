# features/build_features.py

from pathlib import Path
import polars as pl
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"


def compute_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Computes technical indicators and 5-minute forward returns without data leakage.
    """
    df = df.sort("date")

    # Basic Features
    df = df.with_columns([
        (pl.col("close") - pl.col("open")).alias("bar_change"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("volatility_range"),
        (pl.col("volume") * pl.col("close")).alias("turnover"),
    ])

    # Moving Averages & Z-Scores
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=15).alias("sma_15"),
        pl.col("close").rolling_mean(window_size=60).alias("sma_60"),
        pl.col("close").rolling_std(window_size=30).alias("std_30"),
    ])

    df = df.with_columns([
        ((pl.col("close") - pl.col("sma_15")) / (pl.col("std_30") + 1e-6)).alias("zscore_close"),
    ])

    # RSI (14)
    delta = df["close"].diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)

    avg_gain = gain.rolling_mean(window_size=14)
    avg_loss = loss.rolling_mean(window_size=14)
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - (100 / (1 + rs))
    df = df.with_columns(rsi.alias("rsi_14"))

    # VWAP (Cumulative per day)
    df = df.with_columns(pl.col("date").dt.date().alias("trading_date"))
    df = df.with_columns([
        ((pl.col("volume") * (pl.col("high") + pl.col("low") + pl.col("close")) / 3)
         .cum_sum().over("trading_date") /
         (pl.col("volume").cum_sum().over("trading_date") + 1e-6)).alias("vwap")
    ])

    df = df.with_columns([
        ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_deviation")
    ])

    # Target: 5-minute forward return
    df = df.with_columns([
        ((pl.col("close").shift(-5) - pl.col("close")) / pl.col("close")).alias("target_5m_return")
    ])

    # Rename date to timestamp for consistency
    df = df.rename({"date": "timestamp"})
    return df.drop_nulls()


def process_all_futures():
    for symbol in ["nifty", "banknifty"]:
        raw_file = PARQUET_DIR / f"{symbol}_futures_1m_2018_2026.parquet"
        if not raw_file.exists():
            print(f"[-] File not found: {raw_file}. Run data/fetch_zerodha.py first.")
            continue

        print(f"[+] Engineering features for {symbol.upper()}...")
        df = pl.read_parquet(raw_file)
        featured_df = compute_technical_indicators(df)

        out_file = PARQUET_DIR / f"{symbol}_features_1m.parquet"
        featured_df.write_parquet(out_file)
        print(f"[✓] Saved {len(featured_df)} feature rows -> {out_file}")


if __name__ == "__main__":
    process_all_futures()
