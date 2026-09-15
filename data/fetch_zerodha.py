# data/fetch_zerodha.py

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from kiteconnect import KiteConnect

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

# Read API Key and Access Token
def get_kite_client():
    token_path = PROJECT_ROOT / "access_token.txt"
    if not token_path.exists():
        raise FileNotFoundError("access_token.txt not found. Ensure API token is saved.")

    with open(token_path, "r") as f:
        access_token = f.read().strip()

    api_key = os.getenv("KITE_API_KEY", "")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def fetch_continuous_futures(kite, symbol_name: str, instrument_token: int, start_date: str, end_date: str):
    print(f"[+] Fetching historical data for {symbol_name} ({start_date} to {end_date})...")

    current_start = datetime.strptime(start_date, "%Y-%m-%d")
    final_end = datetime.strptime(end_date, "%Y-%m-%d")

    records = []

    # Fetch in 60-day chunks (KiteConnect 1-min limit per call)
    while current_start < final_end:
        current_end = min(current_start + timedelta(days=60), final_end)

        try:
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=current_start.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=current_end.strftime("%Y-%m-%d %H:%M:%S"),
                interval="minute",
                continuous=True,
                oi=True
            )
            records.extend(data)
            time.sleep(0.35)  # Rate limiting
        except Exception as e:
            print(f"[-] Error fetching chunk {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}: {e}")
            time.sleep(1)

        current_start = current_end + timedelta(days=1)

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.drop_duplicates(subset=["date"], inplace=True)

        out_path = PARQUET_DIR / f"{symbol_name.lower()}_futures_1m_2018_2026.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[✓] Saved {len(df)} rows for {symbol_name} -> {out_path}")
    else:
        print(f"[!] No data retrieved for {symbol_name}.")


if __name__ == "__main__":
    try:
        kite = get_kite_client()
        # Replace instrument tokens with your specific Zerodha continuous futures tokens
        NIFTY_FUT_TOKEN = 256265   # Example NIFTY FUT continuous token
        BANKNIFTY_FUT_TOKEN = 260105 # Example BANKNIFTY FUT continuous token

        fetch_continuous_futures(kite, "NIFTY", NIFTY_FUT_TOKEN, "2018-01-01", "2026-09-16")
        fetch_continuous_futures(kite, "BANKNIFTY", BANKNIFTY_FUT_TOKEN, "2018-01-01", "2026-09-16")
    except Exception as e:
        print(f"[-] Fetch execution failed: {e}")
