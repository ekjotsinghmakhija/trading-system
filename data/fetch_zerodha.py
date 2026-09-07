import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Load environment variables from .env file
load_dotenv()


class ZerodhaDataFetcher:
    """Production-grade data extraction module for Zerodha Kite Connect API."""

    def __init__(self, api_key: str, api_secret: str, token_path: str = "access_token.txt"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.token_path = token_path
        self.kite = KiteConnect(api_key=self.api_key)

    def authenticate(self):
        """Authenticates using stored access token or prompts for new login."""
        if os.path.exists(self.token_path):
            with open(self.token_path, "r") as f:
                access_token = f.read().strip()
            self.kite.set_access_token(access_token)
            print("Successfully loaded access token from file.")
        else:
            print(f"Login URL: {self.kite.login_url()}")
            request_token = input("Enter request_token from redirected URL: ").strip()
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            with open(self.token_path, "w") as f:
                f.write(access_token)
            self.kite.set_access_token(access_token)
            print("Authentication complete. Token saved locally.")

    def get_current_nifty_future_token(self) -> int:
        """Fetches current month Nifty Future instrument token from active NFO list."""
        instruments = pd.DataFrame(self.kite.instruments("NFO"))
        nifty_futs = instruments[(instruments["name"] == "NIFTY") & (instruments["segment"] == "NFO-FUT")]
        current_fut = nifty_futs.sort_values(by="expiry").iloc[0]
        print(f"Selected Base Instrument: {current_fut['tradingsymbol']} | Token: {current_fut['instrument_token']}")
        return int(current_fut["instrument_token"])

    def fetch_continuous_data(
        self,
        instrument_token: int,
        start_date: datetime,
        end_date: datetime,
        output_path: str = "data/raw/nifty_futures_raw.csv"
    ):
        """Fetches 1-minute historical continuous data using 60-day window chunks."""
        full_records = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=60), end_date)
            print(f"Extracting: {current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}...")

            retry_count = 0
            while retry_count < 3:
                try:
                    records = self.kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=current_start,
                        to_date=current_end,
                        interval="minute",
                        continuous=True,
                        oi=True
                    )
                    full_records.extend(records)
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"API Warning: {e}. Retry {retry_count}/3 in 2 seconds...")
                    time.sleep(2)

            current_start = current_end + timedelta(days=1)
            time.sleep(0.4)  # Rate limit safety boundary

        df = pd.DataFrame(full_records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Data successfully saved to {output_path} ({len(df):,} total rows).")


if __name__ == "__main__":
    API_KEY = os.getenv("ZERODHA_API_KEY")
    API_SECRET = os.getenv("ZERODHA_API_SECRET")

    if not API_KEY or not API_SECRET:
        raise ValueError("Missing ZERODHA_API_KEY or ZERODHA_API_SECRET in environment variables or .env file.")

    fetcher = ZerodhaDataFetcher(api_key=API_KEY, api_secret=API_SECRET)
    fetcher.authenticate()
    token = fetcher.get_current_nifty_future_token()

    # Run historical pull
    fetcher.fetch_continuous_data(
        instrument_token=token,
        start_date=datetime(2018, 1, 1),
        end_date=datetime(2026, 8, 31)
    )
