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

        fut_close = df["futures_close"] if "futures_close" in df.columns else close
        oi = df["oi"] if "oi" in df.columns else pd.Series(0.0, index=df.index)

        # 1. Basis Alpha
        features["b_fut"] = (fut_close - close) / (close + 1e-8)

        # 2. RSI Scaled
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        features["scaled_rsi"] = (100 - (100 / (1 + rs))) / 100.0

        # 3. Log Returns
        features["log_return"] = np.log(np.maximum(close, 1e-8) / np.maximum(close.shift(1), 1e-8))

        # 4. Volatility Z-Score
        vol_20 = features["log_return"].rolling(20).std().fillna(0.0)
        vol_std = vol_20.rolling(100).std().fillna(1e-8)
        features["volatility_zscore"] = (vol_20 - vol_20.rolling(100).mean().fillna(0.0)) / (vol_std + 1e-8)

        # 5. Volume Z-Score
        vol_m = volume.rolling(20).mean().fillna(0.0)
        vol_s = volume.rolling(20).std().fillna(1e-8)
        features["volume_zscore"] = (volume - vol_m) / (vol_s + 1e-8)

        # 6. Spreads
        features["hl_spread"] = (high - low) / (close + 1e-8)
        features["oc_spread"] = (close - open_p) / (open_p + 1e-8)

        # 7. VWAP Distance
        cum_vol = volume.cumsum()
        vwap = (close * volume).cumsum() / (cum_vol + 1e-8)
        features["vwap_dist"] = (close - vwap) / (vwap + 1e-8)

        # 8. Momentum
        features["mom_5"] = close.pct_change(5).fillna(0.0)
        features["mom_20"] = close.pct_change(20).fillna(0.0)

        # 9. OI Change
        features["oi_change"] = oi.pct_change().fillna(0.0)

        # 10. Shadow Ratios
        features["upper_shadow"] = (high - np.maximum(close, open_p)) / (high - low + 1e-8)
        features["lower_shadow"] = (np.minimum(close, open_p) - low) / (high - low + 1e-8)

        # 11. Moving Averages & Volatility
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        features["ema_spread"] = (ema_12 - ema_26) / (close + 1e-8)
        features["realized_vol_10"] = features["log_return"].rolling(10).std().fillna(0.0)
        features["vpt"] = (volume * close.pct_change().fillna(0.0)).fillna(0.0)

        # 12. Bollinger Band Width
        sma_20 = close.rolling(20).mean().fillna(0.0)
        std_20 = close.rolling(20).std().fillna(1e-8)
        features["bb_width"] = (2 * std_20) / (sma_20 + 1e-8)

        # 13. Normalized Range
        min_20 = low.rolling(20).min().fillna(0.0)
        max_20 = high.rolling(20).max().fillna(1e-8)
        features["norm_range"] = (close - min_20) / (max_20 - min_20 + 1e-8)

        features["timestamp"] = df.get("timestamp", df.index)
        features["symbol"] = df.get("symbol", "NIFTY")

        # Replace any residual inf/-inf with NaN and drop them cleanly
        features = features.replace([np.inf, -np.inf], np.nan)
        return features
