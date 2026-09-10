import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes high-frequency institutional alpha features based on financial machine learning research."""

    def compute_18_alpha_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        features = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_p = df["open"]
        volume = df["volume"]
        eps = 1e-8

        # 1. Multi-Horizon Stationary Log Returns (1m, 3m, 5m, 15m, 30m)
        log_c = np.log(np.maximum(close, eps))
        features["log_return"] = log_c.diff(1).fillna(0.0)  # Preserved for target reward computation
        features["log_ret_3m"] = log_c.diff(3).fillna(0.0)
        features["log_ret_5m"] = log_c.diff(5).fillna(0.0)
        features["log_ret_15m"] = log_c.diff(15).fillna(0.0)
        features["log_ret_30m"] = log_c.diff(30).fillna(0.0)

        # 2. Garman-Klass Microstructural Volatility (5m & 20m)
        log_hl = np.log(np.maximum(high, eps) / np.maximum(low, eps)) ** 2
        log_co = np.log(np.maximum(close, eps) / np.maximum(open_p, eps)) ** 2
        gk_vol = np.sqrt(np.maximum(0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co, eps))
        features["gk_vol_5m"] = gk_vol.rolling(5).mean().fillna(0.0)
        features["gk_vol_20m"] = gk_vol.rolling(20).mean().fillna(0.0)

        # 3. Parkinson Range Volatility (10m)
        parkinson = np.sqrt((1.0 / (4.0 * np.log(2.0))) * log_hl)
        features["parkinson_vol_10m"] = parkinson.rolling(10).mean().fillna(0.0)

        # 4. Order Flow / Volume Force Proxy (Directional Trade Pressure)
        body_ratio = (close - open_p) / (high - low + eps)
        vol_force_1m = body_ratio * np.log1p(volume)
        features["vol_force_1m"] = vol_force_1m
        features["vol_force_5m"] = vol_force_1m.rolling(5).mean().fillna(0.0)

        # 5. Amihud Price Impact / Illiquidity Measure
        abs_ret = np.abs(features["log_return"])
        features["amihud_illiquidity"] = (abs_ret / (np.log1p(volume) + eps)).rolling(10).mean().fillna(0.0)

        # 6. Multi-Timeframe Volume-Weighted Price (VWAP) Distances
        pv = close * volume
        vwap_5m = pv.rolling(5).sum() / (volume.rolling(5).sum() + eps)
        vwap_20m = pv.rolling(20).sum() / (volume.rolling(20).sum() + eps)
        features["vwap_dist_5m"] = (close - vwap_5m) / (vwap_5m + eps)
        features["vwap_dist_20m"] = (close - vwap_20m) / (vwap_20m + eps)

        # 7. Volatility-Adjusted Momentum (Rolling Sharpe Proxy)
        std_15m = features["log_return"].rolling(15).std().fillna(eps) + eps
        features["vol_adj_mom_15m"] = features["log_ret_15m"] / std_15m

        # 8. Normalized Candlestick Geometry
        features["hl_spread_norm"] = (high - low) / (close + eps)
        features["upper_wick_norm"] = (high - np.maximum(close, open_p)) / (high - low + eps)
        features["lower_wick_norm"] = (np.minimum(close, open_p) - low) / (high - low + eps)

        features["timestamp"] = df.get("timestamp", df.index)
        features["symbol"] = df.get("symbol", "NIFTY")

        # Replace residual infinite/NaN values cleanly
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features
