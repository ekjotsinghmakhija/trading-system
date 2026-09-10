import os
import duckdb
import polars as pl

DB_PATH = "data/duckdb/market_data.duckdb"
PARQUET_DIR = "data/parquet"

def build_features_for_asset(primary_df: pl.DataFrame, secondary_df: pl.DataFrame, primary_name: str, secondary_name: str) -> pl.DataFrame:
    sec_close = secondary_df.select([
        pl.col("timestamp"),
        pl.col("close").alias(f"{secondary_name}_close")
    ])

    features_df = (
        primary_df.lazy()
        .join(sec_close.lazy(), on="timestamp", how="left")
        .sort("timestamp")
        # Step 1: Create trading_date prior to window evaluations
        .with_columns(
            pl.col("timestamp").dt.date().alias("trading_date")
        )
        # Step 2: Multi-horizon alpha factors
        .with_columns([
            # 1. Log Returns
            (pl.col("close") / pl.col("close").shift(1)).log().over("trading_date").alias("return_1m"),
            (pl.col("close") / pl.col("close").shift(5)).log().over("trading_date").alias("return_5m"),
            (pl.col("close") / pl.col("close").shift(15)).log().over("trading_date").alias("return_15m"),
            (pl.col("close") / pl.col("close").shift(60)).log().over("trading_date").alias("return_60m"),

            # 2. Intraday Bar Span
            ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("bar_span"),

            # 3. Rolling Volatility
            (pl.col("close") / pl.col("close").shift(1)).log()
                .rolling_std(window_size=15).over("trading_date").alias("volatility_15m"),
            (pl.col("close") / pl.col("close").shift(60)).log()
                .rolling_std(window_size=60).over("trading_date").alias("volatility_60m"),

            # 4. Intraday Cumulative VWAP
            (((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
                .cum_sum().over("trading_date") /
                pl.col("volume").cum_sum().over("trading_date").replace(0, None)
            ).alias("vwap"),

            # 5. Cross-Asset Ratio
            (pl.col("close") / pl.col(f"{secondary_name}_close")).alias(f"{primary_name}_{secondary_name}_ratio")
        ])
        # Step 3: VWAP distance & forward 5-minute target
        .with_columns([
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_distance"),
            (pl.col("close").shift(-5) / pl.col("close")).log().over("trading_date").alias("target_5m_return")
        ])
        .filter(pl.col("return_60m").is_not_null())
        .collect()
    )

    return features_df

def build_all_features():
    os.makedirs(PARQUET_DIR, exist_ok=True)
    print("[+] Connecting to DuckDB database...")
    con = duckdb.connect(DB_PATH)

    print("[+] Loading spot datasets...")
    nifty_df = con.execute("SELECT * FROM nifty_spot_1m").pl()
    banknifty_df = con.execute("SELECT * FROM banknifty_spot_1m").pl()

    targets = [
        (nifty_df, banknifty_df, "nifty", "banknifty", "nifty_features_1m"),
        (banknifty_df, nifty_df, "banknifty", "nifty", "banknifty_features_1m")
    ]

    for prim, sec, prim_name, sec_name, table_name in targets:
        print(f"[+] Engineering alpha features for {prim_name.upper()}...")
        feat_df = build_features_for_asset(prim, sec, prim_name, sec_name)

        parquet_path = os.path.join(PARQUET_DIR, f"{table_name}.parquet")
        feat_df.write_parquet(parquet_path, compression="snappy")

        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{parquet_path}')")

        row_cnt = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        col_cnt = len(con.execute(f"PRAGMA table_info('{table_name}')").fetchall())
        print(f"  └─ Created table '{table_name}': {row_cnt:,} rows × {col_cnt} columns.")

    con.close()
    print("[+] Dual feature engineering complete.")

if __name__ == "__main__":
    build_all_features()
