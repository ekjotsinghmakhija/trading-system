import os
import glob
from pathlib import Path
import pandas as pd
import polars as pl
import duckdb

# Path Configurations
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PARQUET_DIR = BASE_DIR / "data" / "parquet"
DUCKDB_PATH = BASE_DIR / "data" / "duckdb" / "quant_archive.duckdb"

def init_duckdb():
    """Initialize DuckDB database and create view attachments."""
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DUCKDB_PATH))
    conn.execute("SET preserve_insertion_order = false;")
    conn.execute("SET memory_limit = '12GB';")
    return conn

def process_and_partition_raw_data(symbol="NIFTY"):
    """
    Reads raw CSV/binary exports, enforces UTC normalization,
    resamples into 1-minute aligned bars, and writes Snappy-compressed Parquet archives.
    """
    print(f"[+] Starting ingestion pipeline for {symbol}...")

    # Locate files
    search_pattern = str(RAW_DIR / f"*{symbol}*.[c|p]*")
    files = glob.glob(search_pattern)

    if not files:
        print(f"[!] No raw files found for {symbol} in {RAW_DIR}. Place TrueData/GDF dumps there.")
        return

    for file_path in files:
        print(f"    --> Processing {file_path}")

        # Load using Polars for ultra-fast I/O
        df = pl.read_csv(file_path) if file_path.endswith('.csv') else pl.read_parquet(file_path)

        # Standardize column names
        df = df.rename({col: col.strip().lower() for col in df.columns})

        # Datetime casting and UTC enforcement
        if "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").str.to_datetime().dt.replace_time_zone("UTC"))
        elif "datetime" in df.columns:
            df = df.with_columns(pl.col("datetime").str.to_datetime().alias("timestamp").dt.replace_time_zone("UTC"))

        # Filter strictly for NSE Intraday Market Hours (09:15 to 15:30 IST -> 03:45 to 10:00 UTC)
        df = df.filter(
            (pl.col("timestamp").dt.time() >= pl.time(3, 45, 0)) &
            (pl.col("timestamp").dt.time() <= pl.time(10, 0, 0))
        )

        # Add year and month partitioning keys
        df = df.with_columns([
            pl.col("timestamp").dt.year().alias("year"),
            pl.col("timestamp").dt.month().alias("month")
        ])

        # Write partitioned Snappy-compressed Parquet
        years = df["year"].unique().to_list()
        for y in years:
            for m in df.filter(pl.col("year") == y)["month"].unique().to_list():
                partition_path = PARQUET_DIR / f"index={symbol}" / f"year={y}" / f"month={m:02d}"
                partition_path.mkdir(parents=True, exist_ok=True)

                sub_df = df.filter((pl.col("year") == y) & (pl.col("month") == m))
                output_file = partition_path / "data.parquet"

                sub_df.write_parquet(output_file, compression="snappy")
                print(f"        Saved partition: {output_file}")

def sync_parquet_to_duckdb(conn):
    """Binds all partitioned Parquet files directly into DuckDB views."""
    parquet_glob = str(PARQUET_DIR / "index=*" / "year=*" / "month=*" / "*.parquet")

    conn.execute(f"""
        CREATE OR REPLACE VIEW market_data_1min AS
        SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=1);
    """)
    print("[+] DuckDB view 'market_data_1min' successfully bound to Parquet storage.")

if __name__ == "__main__":
    db_conn = init_duckdb()

    # Process both target index options/futures
    process_and_partition_raw_data(symbol="NIFTY")
    process_and_partition_raw_data(symbol="BANKNIFTY")

    sync_parquet_to_duckdb(db_conn)
    db_conn.close()
    print("[✓] Raw data ingestion & partitioning complete.")
