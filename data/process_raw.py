import os
import pandas as pd


def process_and_standardize_data(
    input_file: str = "data/raw/nifty_futures_raw.csv",
    output_file: str = "data/raw/market_data.csv"
):
    """Sanitizes raw historical extracted data to comply with main.py schema."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found. Run fetch_zerodha.py first.")

    print("Sanitizing raw dataset...")
    df = pd.read_csv(input_file)

    # Standardize column naming
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Enforce strict NSE Market Hours (09:15 to 15:30 IST)
    df = df.set_index("timestamp").between_time("09:15", "15:30").reset_index()

    # Deduplicate and sort chronologically
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    # Filter to exact production schema
    standard_df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    standard_df.to_csv(output_file, index=False)
    print(f"Production dataset formatted and saved to {output_file} ({len(standard_df):,} rows).")


if __name__ == "__main__":
    process_and_standardize_data()
