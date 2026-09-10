import numpy as np
import pandas as pd
import polars as pl

def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Safe division preventing division by zero and NaNs."""
    return (numerator / denominator.replace(0, np.nan)).fillna(fill_value)

class FeatureEngine:
    """
    Generates the complete 18-feature Alpha Matrix across 3 specialized clusters
    for Indian Index Options & Futures intraday data (1-min frequency).
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().sort_values("timestamp").reset_index(drop=True)

    def calculate_cluster_1_price_momentum(self) -> pd.DataFrame:
        """Cluster 1: Technical & Trend Momentum (6 Features)"""
        close = self.df["close"]
        high = self.df["high"]
        low = self.df["low"]
        volume = self.df["volume"]

        # 1. Log Returns (1-step)
        self.df["feat_log_return_1m"] = np.log(close / close.shift(1)).fillna(0.0)

        # 2. RSI (14-period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = safe_divide(gain, loss)
        self.df["feat_rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0) / 100.0  # Normalized [0, 1]

        # 3. Normalized Distance to VWAP
        cum_vol = volume.cumsum()
        cum_vol_price = (close * volume).cumsum()
        vwap = safe_divide(cum_vol_price, cum_vol, fill_value=close.iloc[0])
        self.df["feat_vwap_dist"] = safe_divide(close - vwap, vwap)

        # 4. Normalized Intraday Range (High-Low / Close)
        self.df["feat_intraday_range"] = safe_divide(high - low, close)

        # 5. Realized Volatility (20-bar rolling std of log returns)
        self.df["feat_realized_vol_20"] = self.df["feat_log_return_1m"].rolling(20).std().fillna(0.0)

        # 6. Volume Acceleration (5m SMA / 20m SMA)
        vol_sma_5 = volume.rolling(5).mean()
        vol_sma_20 = volume.rolling(20).mean()
        self.df["feat_vol_acceleration"] = safe_divide(vol_sma_5, vol_sma_20, fill_value=1.0)

        return self.df

    def calculate_cluster_2_options_microstructure(self) -> pd.DataFrame:
        """Cluster 2: Options Microstructure & Positioning (6 Features)"""
        call_oi = self.df.get("call_open_interest", pd.Series(1.0, index=self.df.index))
        put_oi = self.df.get("put_open_interest", pd.Series(1.0, index=self.df.index))
        call_iv = self.df.get("call_iv", pd.Series(0.15, index=self.df.index))
        put_iv = self.df.get("put_iv", pd.Series(0.15, index=self.df.index))

        # 7. PCR (Put-Call Ratio by Open Interest)
        pcr = safe_divide(put_oi, call_oi, fill_value=1.0)
        self.df["feat_pcr_oi"] = pcr

        # 8. PCR Velocity (5-bar change in PCR)
        self.df["feat_pcr_velocity"] = pcr.diff(5).fillna(0.0)

        # 9. IV Skew (Put IV - Call IV)
        self.df["feat_iv_skew"] = put_iv - call_iv

        # 10. IV Skew Velocity (5-bar diff in Skew)
        self.df["feat_iv_skew_velocity"] = self.df["feat_iv_skew"].diff(5).fillna(0.0)

        # 11. ATM Implied Volatility Level
        self.df["feat_atm_iv"] = (call_iv + put_iv) / 2.0

        # 12. Net Open Interest Change Rate (Total OI momentum)
        total_oi = call_oi + put_oi
        self.df["feat_oi_change_rate"] = safe_divide(total_oi.diff(5), total_oi.shift(5)).fillna(0.0)

        return self.df

    def calculate_cluster_3_greeks_orderflow(self) -> pd.DataFrame:
        """Cluster 3: Options Greeks & Order Flow Dynamics (6 Features)"""
        bid_qty = self.df.get("best_bid_qty", self.df["volume"] * 0.5)
        ask_qty = self.df.get("best_ask_qty", self.df["volume"] * 0.5)
        delta = self.df.get("net_delta", pd.Series(0.0, index=self.df.index))
        gamma = self.df.get("net_gamma", pd.Series(0.0, index=self.df.index))
        vega = self.df.get("net_vega", pd.Series(0.0, index=self.df.index))
        theta = self.df.get("net_theta", pd.Series(0.0, index=self.df.index))

        # 13. Level-2 Order Flow Imbalance (OFI)
        total_depth = bid_qty + ask_qty
        self.df["feat_ofi"] = safe_divide(bid_qty - ask_qty, total_depth).clip(-1.0, 1.0)

        # 14. Net Delta Exposure
        self.df["feat_net_delta"] = delta

        # 15. Delta Acceleration (5-bar diff in Net Delta)
        self.df["feat_delta_acceleration"] = delta.diff(5).fillna(0.0)

        # 16. Net Gamma Exposure
        self.df["feat_net_gamma"] = gamma

        # 17. Net Vega Exposure
        self.df["feat_net_vega"] = vega

        # 18. Theta Decay Velocity
        self.df["feat_theta_decay_rate"] = safe_divide(theta, self.df["close"]).fillna(0.0)

        return self.df

    def build_feature_matrix(self) -> pd.DataFrame:
        """Run full 18-feature generation pipeline."""
        self.calculate_cluster_1_price_momentum()
        self.calculate_cluster_2_options_microstructure()
        self.calculate_cluster_3_greeks_orderflow()

        # Drop warm-up NA rows caused by rolling windows
        self.df = self.df.dropna().reset_index(drop=True)
        return self.df

if __name__ == "__main__":
    dates = pd.date_range("2026-09-01 09:15:00", periods=100, freq="1min", tz="UTC")
    dummy_df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(100).cumsum() + 25000,
        "high": np.random.randn(100).cumsum() + 25020,
        "low": np.random.randn(100).cumsum() + 24980,
        "close": np.random.randn(100).cumsum() + 25000,
        "volume": np.random.randint(100, 5000, size=100)
    })

    engine = FeatureEngine(dummy_df)
    matrix = engine.build_feature_matrix()
    feature_cols = [c for c in matrix.columns if c.startswith("feat_")]
    print(f"[✓] Successfully generated {len(feature_cols)} features across {len(matrix)} rows.")
