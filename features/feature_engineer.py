import numpy as np
import pandas as pd


class FeatureEngineer:
    def compute_18_alpha_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]
        volume = df["volume"]

        # Safe fallback for futures_close and oi when running on spot market data
        fut_close = df["futures_close"] if "futures_close" in df.columns else close
        oi = df["oi"] if "oi" in df.columns else pd.Series(0.0, index=df.index)

        # 1. Basis Alpha (Futures vs Spot spread)
        features["b_fut"] = (fut_close - close) / (close + 1e-8)

        # 2. RSI Scaled
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features["scaled_rsi"] = (100 - (100 / (1 + rs))) / 100.0

        # 3. Log Returns
        features["log_return"] = np.log(close / (close.shift(1) + 1e-8))

        # 4. Volatility Z-Score (20-period)
        vol_20 = features["log_return"].rolling(20).std()
        features["volatility_zscore"] = (vol_20 - vol_20.rolling(100).mean()) / (vol_20.rolling(100).std() + 1e-8)

        # 5. Volume Z-Score
        features["volume_zscore"] = (volume - volume.rolling(20).mean()) / (volume.rolling(20).std() + 1e-8)

        # 6. High-Low Spread
        features["hl_spread"] = (high - low) / (close + 1e-8)

        # 7. Open-Close Spread
        features["oc_spread"] = (close - open_p) / (open_p + 1e-8)

        # 8. VWAP Distance
        cum_vol = volume.cumsum()
        vwap = (close * volume).cumsum() / (cum_vol + 1e-8)
        features["vwap_dist"] = (close - vwap) / (vwap + 1e-8)

        # 9. Momentum 5-bar
        features["mom_5"] = close.pct_change(5)

        # 10. Momentum 20-bar
        features["mom_20"] = close.pct_change(20)

        # 11. OI Change
        features["oi_change"] = oi.pct_change().fillna(0.0)

        # 12. Upper Shadow Ratio
        features["upper_shadow"] = (high - np.maximum(close, open_p)) / (high - low + 1e-8)

        # 13. Lower Shadow Ratio
        features["lower_shadow"] = (np.minimum(close, open_p) - low) / (high - low + 1e-8)

        # 14. Exponential Moving Average 12/26 Spread
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features["ema_spread"] = (ema_12 - ema_26) / (close + 1e-8)

        # 15. Realized Volatility 10-bar
        features["realized_vol_10"] = features["log_return"].rolling(10).std()

        # 16. Volume-Price Trend
        features["vpt"] = (volume * (close.pct_change())).fillna(0.0)

        # 17. Bollinger Band Width
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features["bb_width"] = (2 * std_20) / (sma_20 + 1e-8)

        # 18. Normalized Range
        features["norm_range"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-8)

        # Preserve metadata columns
        features["timestamp"] = df.get("timestamp", df.index)
        features["symbol"] = df.get("symbol", "NIFTY")

        return features

    def generate_synthetic_raw_feed(self, rows: int = 500) -> pd.DataFrame:
        np.random.seed(42)
        price = 22000.0 + np.cumsum(np.random.randn(rows) * 10)
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="1min"),
            "symbol": "NIFTY",
            "open": price,
            "high": price + np.random.rand(rows) * 5,
            "low": price - np.random.rand(rows) * 5,
            "close": price + np.random.randn(rows) * 2,
            "volume": np.random.randint(100, 5000, size=rows),
            "futures_close": price + np.random.randn(rows) * 3,
            "oi": np.random.randint(1000, 50000, size=rows)
        })
