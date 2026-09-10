import duckdb
import pandas as pd
import torch


class DuckDBDataStreamer:
    """
    Zero-copy streaming engine fetching real historical options/underlying
    bars directly from DuckDB into PyTorch memory blocks.
    """
    def __init__(self, db_path: str = "data/duckdb/market_data.duckdb"):
        self.conn = duckdb.connect(db_path, read_only=True)

    def load_clean_bars(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        query = f"""
            SELECT timestamp, open, high, low, close, volume, oi
            FROM ohlcv_bars
            WHERE symbol = '{symbol}'
              AND timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY timestamp ASC
        """
        return self.conn.execute(query).df()
