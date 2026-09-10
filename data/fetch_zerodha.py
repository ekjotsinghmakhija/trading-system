import os
import time
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException
import requests

load_dotenv()


class ZerodhaDataFetcher:
    """Production-grade data extraction module with network retry logic & strict API limit handling."""

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
            print("[+] Successfully loaded access token from file.")
        else:
            print(f"[!] Login URL: {self.kite.login_url()}")
            request_token = input("Enter request_token from redirected URL: ").strip()
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            with open(self.token_path, "w") as f:
                f.write(access_token)
            self.kite.set_access_token(access_token)
            print("[+] Authentication complete. Token saved locally.")

    def _execute_with_retry(self, func, *args, max_retries=5, **kwargs):
        """Handles DNS/network drops with exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, NetworkException, Exception) as e:
                wait_time = attempt * 3
                print(f"[!] Network Error: {e}. Retrying {attempt}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
        raise ConnectionError(f"Failed {func.__name__} after {max_retries} retries due to connection drops.")

    def get_spot_token(self, symbol: str = "NIFTY 50") -> int:
        """Fetches spot index instrument token."""
        instruments = pd.DataFrame(self._execute_with_retry(self.kite.instruments, "NSE"))
        match = instruments[instruments["tradingsymbol"] == symbol]
        if match.empty:
            raise ValueError(f"Symbol {symbol} not found in NSE instrument list.")
        token = int(match.iloc[0]["instrument_token"])
        print(f"[+] Selected Spot Instrument: {symbol} | Token: {token}")
        return token

    def get_current_future_token(self, symbol: str = "NIFTY") -> int:
        """Fetches current active month future contract token from NFO."""
        instruments = pd.DataFrame(self._execute_with_retry(self.kite.instruments, "NFO"))
        futs = instruments[(instruments["name"] == symbol) & (instruments["segment"] == "NFO-FUT")]
        current_fut = futs.sort_values(by="expiry").iloc[0]
        print(f"[+] Selected Active Future: {current_fut['tradingsymbol']} | Token: {current_fut['instrument_token']}")
        return int(current_fut["instrument_token"])

    def fetch_historical_data(
        self,
        instrument_token: int,
        start_date: datetime,
        end_date: datetime,
        output_path: str,
        interval: str = "minute",
        continuous: bool = False,
        include_oi: bool = False
    ):
        """Fetches historical candles respecting Zerodha interval limits."""
        max_days = 2000 if interval == "day" else 60
        full_records = []
        current_start = start_date

        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days), end_date)
            print(f"Extracting [{interval}]: {current_start.strftime('%Y-%m-%d')} -> {current_end.strftime('%Y-%m-%d')}...")

            records = self._execute_with_retry(
                self.kite.historical_data,
                instrument_token=instrument_token,
                from_date=current_start,
                to_date=current_end,
                interval=interval,
                continuous=continuous,
                oi=include_oi
            )

            if records:
                full_records.extend(records)

            current_start = current_end + timedelta(days=1)
            time.sleep(0.35)

        if not full_records:
            print(f"[!] Warning: No records retrieved for token {instrument_token}.")
            return

        df = pd.DataFrame(full_records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[+] Saved to {output_path} ({len(df):,} total rows).")


if __name__ == "__main__":
    API_KEY = os.getenv("ZERODHA_API_KEY")
    API_SECRET = os.getenv("ZERODHA_API_SECRET")

    if not API_KEY or not API_SECRET:
        raise ValueError("Missing ZERODHA_API_KEY or ZERODHA_API_SECRET in environment variables.")

    fetcher = ZerodhaDataFetcher(api_key=API_KEY, api_secret=API_SECRET)
    fetcher.authenticate()

    START_2018 = datetime(2018, 1, 1)
    END_DATE = datetime(2026, 8, 31)

    # 1. Intraday Spot Indices 1-Minute Data (2018-2026, Includes 2020 Crash)
    spot_targets = [
        ("NIFTY 50", "data/raw/nifty_spot_raw.csv"),
        ("NIFTY BANK", "data/raw/banknifty_spot_raw.csv"),
    ]

    for symbol, path in spot_targets:
        token = fetcher.get_spot_token(symbol)
        fetcher.fetch_historical_data(
            instrument_token=token,
            start_date=START_2018,
            end_date=END_DATE,
            output_path=path,
            interval="minute",
            continuous=False,
            include_oi=False
        )

    # 2. Daily Continuous Futures Data (2018-2026)
    fut_token = fetcher.get_current_future_token("NIFTY")
    fetcher.fetch_historical_data(
        instrument_token=fut_token,
        start_date=START_2018,
        end_date=END_DATE,
        output_path="data/raw/nifty_futures_daily_continuous_raw.csv",
        interval="day",
        continuous=True,
        include_oi=True
    )

    # 3. Active Month NIFTY Futures Intraday 1-Minute Data
    fetcher.fetch_historical_data(
        instrument_token=fut_token,
        start_date=datetime(2026, 1, 1),
        end_date=END_DATE,
        output_path="data/raw/nifty_futures_active_raw.csv",
        interval="minute",
        continuous=False,
        include_oi=True
    )
