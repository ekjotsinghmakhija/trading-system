import os
import duckdb


def build_duckdb_from_csvs(
    raw_dir: str = "data/raw",
    db_path: str = "data/duckdb/market_data.duckdb"
):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(db_path)

    # Drop existing table if recreating schema
    conn.execute("DROP TABLE IF EXISTS ohlcv_bars")

    # Ingest train_1min.csv or all raw 1-min CSVs
    raw_csv = os.path.join(raw_dir, "train_1min.csv")
    if not os.path.exists(raw_csv):
        raw_csv = os.path.join(raw_dir, "nifty_1min_features.csv")

    print(f"Ingesting raw data from {raw_csv} into DuckDB...")

    query = f"""
        CREATE TABLE ohlcv_bars AS
        SELECT
            TRY_CAST(timestamp AS TIMESTAMP) AS timestamp,
            TRY_CAST(open AS DOUBLE) AS open,
            TRY_CAST(high AS DOUBLE) AS high,
            TRY_CAST(low AS DOUBLE) AS low,
            TRY_CAST(close AS DOUBLE) AS close,
            TRY_CAST(volume AS DOUBLE) AS volume,
            COALESCE(TRY_CAST(oi AS DOUBLE), 0.0) AS oi,
            'NIFTY' AS symbol
        FROM read_csv_auto('{raw_csv}')
        WHERE timestamp IS NOT NULL
        ORDER BY timestamp ASC;
    """
    conn.execute(query)
    count = conn.execute("SELECT COUNT(*) FROM ohlcv_bars").fetchone()[0]
    print(f"[SUCCESS] Populated {count:,} rows in DuckDB database: {db_path}")
    conn.close()


if __name__ == "__main__":
    build_duckdb_from_csvs()
