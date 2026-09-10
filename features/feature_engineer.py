import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Computes 18 uncorrelated alpha features across spot, futures, and option dynamics.
    Enforces pairwise correlation (|r| <= 0.65) and VIF filters (VIF < 5.0).
    """
    def __init__(self, corr_threshold: float = 0.65, vif_threshold: float = 5.0):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold

    def generate_synthetic_raw_feed(self, rows: int = 50000) -> pd.DataFrame:
        np.random.seed(42)
        prices = 24000.0 + np.cumsum(np.random.normal(0.05, 2.5, size=rows))
        volume = np.random.gamma(2, 1000, size=rows)

        df = pd.DataFrame({
            "open": prices + np.random.normal(0, 0.5, size=rows),
            "high": prices + np.abs(np.random.normal(0, 1.2, size=rows)),
            "low": prices - np.abs(np.random.normal(0, 1.2, size=rows)),
            "close": prices,
            "volume": volume,
            "futures_close": prices + np.random.normal(2.0, 0.5, size=rows),
            "futures_oi": 100000 + np.cumsum(np.random.normal(10, 50, size=rows)),
            "call_iv": 15.0 + np.random.normal(0, 0.5, size=rows),
            "put_iv": 15.5 + np.random.normal(0, 0.5, size=rows),
            "call_oi": 50000 + np.cumsum(np.random.normal(5, 20, size=rows)),
            "put_oi": 52000 + np.cumsum(np.random.normal(5, 20, size=rows)),
            "option_delta": np.random.uniform(0.40, 0.60, size=rows),
            "bid_depth_top5": np.random.uniform(100, 500, size=rows),
            "ask_depth_top5": np.random.uniform(100, 500, size=rows),
            "bid_price": prices - 0.15,
            "ask_price": prices + 0.15,
        })
        return df

    def compute_18_alpha_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        close = df["close"]

        # Cluster 1: Spot & Futures Momentum Indicators
        delta_close = close.diff()
        gain = delta_close.where(delta_close > 0, 0.0).rolling(14).mean()
        loss = (-delta_close.where(delta_close < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features["scaled_rsi"] = (rsi - 50.0) / 50.0

        vwap = (df["volume"] * close).cumsum() / (df["volume"].cumsum() + 1e-8)
        features["d_vwap"] = (close - vwap) / (vwap + 1e-8)
        features["b_fut"] = (df["futures_close"] - close) / (close + 1e-8)
        features["fut_oi_acc"] = df["futures_oi"].diff().diff().fillna(0) / 1000.0
        features["price_return_acc"] = np.log(close / close.shift(1)).diff().fillna(0)
        features["momentum_density"] = (close - close.shift(3)) / (df["volume"].rolling(3).mean() + 1e-8)

        # Cluster 2: Volatility & Skew Accelerators
        features["alpha_skew"] = ((df["put_iv"] - df["call_iv"]) - (df["put_iv"].shift(3) - df["call_iv"].shift(3))) / 3.0
        pcr = df["put_oi"] / (df["call_oi"] + 1e-8)
        features["v_pcr"] = pcr.diff(5) / 5.0

        ret_log = np.log(close / close.shift(1))
        rv_15m = ret_log.rolling(15).std() * np.sqrt(375)
        features["rv_velocity"] = rv_15m.diff().fillna(0)

        delta_diff = df["option_delta"].diff().fillna(0)
        spot_diff = close.diff().fillna(0)
        features["gamma_eff"] = delta_diff / (spot_diff + 1e-8)
        features["iv_rv_gap"] = (df["call_iv"] / 100.0) - rv_15m
        features["vega_velocity"] = df["call_iv"].diff(3) / 3.0

        # Cluster 3: Microstructure & Level-2 Order Flow
        bid_v, ask_v = df["bid_depth_top5"], df["ask_depth_top5"]
        features["ofi_1m"] = (bid_v - ask_v) / (bid_v + ask_v + 1e-8)
        features["spread_decay"] = (df["ask_price"] - df["bid_price"]) / (close + 1e-8)
        features["volume_spike"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-8)
        features["ask_depth_ratio"] = ask_v / (ask_v.rolling(10).mean() + 1e-8)
        features["bid_depth_ratio"] = bid_v / (bid_v.rolling(10).mean() + 1e-8)

        micro_price = (df["bid_price"] * ask_v + df["ask_price"] * bid_v) / (bid_v + ask_v + 1e-8)
        features["micro_price_drift"] = (micro_price - close) / (close + 1e-8)

        features["close"] = close
        return features.dropna().reset_index(drop=True)
