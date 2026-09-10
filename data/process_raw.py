import os
import duckdb
import polars as pl

RAW_DIR = "data/raw"
PARQUET_DIR = "data/parquet"
DB_PATH = "data/duckdb/market_data.duckdb"

SCHEMA_MAPPING = {
    "nifty_spot_raw.csv": ("nifty_spot_1m", "1-minute NIFTY 50 Spot Index"),
    "banknifty_spot_raw.csv": ("banknifty_spot_1m", "1-minute NIFTY BANK Spot Index"),
    "nifty_futures_daily_continuous_raw.csv": ("nifty_futures_daily", "Daily Continuous NIFTY Futures"),
    "nifty_futures_active_raw.csv": ("nifty_futures_active_1m", "1-minute Active NIFTY Futures")
}

def process_and_ingest():
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    con = duckdb.connect(DB_PATH)
    print(f"[+] Initialized DuckDB database at '{DB_PATH}'")

    for filename, (table_name, description) in SCHEMA_MAPPING.items():
        csv_path = os.path.join(RAW_DIR, filename)

        if not os.path.exists(csv_path):
            print(f"[!] Warning: File {csv_path} not found. Skipping...")
            continue

        print(f"[+] Processing {description} ({filename})...")

        # Load CSV using Polars
        df = pl.read_csv(csv_path)

        # Standardize timestamp column naming
        if "date" in df.columns:
            df = df.rename({"date": "timestamp"})

        # Convert timestamp strings with embedded offsets to Asia/Kolkata Datetime
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(time_zone="Asia/Kolkata")
        )

        # Enforce uniform schema columns
        if "oi" not in df.columns:
            df = df.with_columns(pl.lit(0).alias("oi"))

        # Deduplicate and sort chronologically before applying window functions
        df = df.sort("timestamp").unique(subset=["timestamp"], keep="first")

        # Intraday 1m specific cleaning
        if "1m" in table_name:
            # 1. Filter regular market hours (09:15:00 to 15:30:00 IST)
            df = df.filter(
                (pl.col("timestamp").dt.time() >= pl.time(9, 15, 0)) &
                (pl.col("timestamp").dt.time() <= pl.time(15, 30, 0))
            )

            # 2. Iteratively filter intraday bad ticks (>5% 1-min return) until zero remain
            while True:
                df = df.with_columns([
                    pl.col("close").shift(1).over(pl.col("timestamp").dt.date()).alias("prev_close")
                ])
                spike_mask = (
                    pl.col("prev_close").is_not_null() &
                    (((pl.col("close") - pl.col("prev_close")).abs() / pl.col("prev_close")) > 0.05)
                )
                num_spikes = df.filter(spike_mask).height
                if num_spikes == 0:
                    df = df.drop("prev_close")
                    break
                df = df.filter(~spike_mask).drop("prev_close")

        # Write compressed Parquet file
        parquet_path = os.path.join(PARQUET_DIR, f"{table_name}.parquet")
        df.write_parquet(parquet_path, compression="snappy")

        # Load directly into DuckDB
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")

        row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        min_date, max_date = con.execute(
            f"SELECT CAST(MIN(timestamp) AS VARCHAR), CAST(MAX(timestamp) AS VARCHAR) FROM {table_name}"
        ).fetchone()

        print(f"  └─ Created table '{table_name}': {row_count:,} rows [{str(min_date)[:10]} to {str(max_date)[:10]}]")

    con.close()
    print("[+] All raw datasets processed and DuckDB ingestion complete.")

if __name__ == "__main__":
    process_and_ingest()
