import pandas as pd
import numpy as np


class FeatureEngine:
    """
    Computes 16 stationary, uncorrelated indicators with optional QR Gram-Schmidt
    orthogonalization to ensure strict input feature independence.
    """
    def __init__(self, apply_orthogonalization: bool = True):
        self.apply_orthogonalization = apply_orthogonalization

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        for col in ["close", "high", "low", "volume"]:
            if col not in data.columns:
                if col == "high": data["high"] = data["close"] * 1.001
                elif col == "low": data["low"] = data["close"] * 0.999
                elif col == "volume": data["volume"] = 100000.0

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        feat = pd.DataFrame(index=data.index)

        # --- 1. Return Dynamics & Log Ratios ---
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        feat["f01_log_ret_1"] = log_ret
        feat["f02_log_ret_5"] = np.log(close / close.shift(5)).fillna(0.0)
        feat["f03_log_ret_20"] = np.log(close / close.shift(20)).fillna(0.0)

        # --- 2. Volatility Regimes ---
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        feat["f04_norm_atr"] = (atr_14 / (close + 1e-8)).fillna(0.0)
        feat["f05_realized_vol_10"] = log_ret.rolling(10).std().fillna(0.0)
        feat["f06_parkinson_vol"] = (np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(high / low) ** 2)).rolling(10).mean()).fillna(0.0)

        # --- 3. Momentum & Oscillators ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        feat["f07_rsi_scaled"] = ((100.0 - (100.0 / (1.0 + rs))) / 100.0 - 0.5).fillna(0.0)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        feat["f08_macd_norm"] = (macd / (atr_14 + 1e-8)).fillna(0.0)

        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        stoch_k = (close - low_14) / (high_14 - low_14 + 1e-8)
        feat["f09_stoch_k"] = (stoch_k - 0.5).fillna(0.0)

        rolling_std = log_ret.rolling(10).std()
        feat["f10_vol_adj_mom"] = (log_ret.rolling(10).mean() / (rolling_std + 1e-8)).fillna(0.0)

        # --- 4. Mean Reversion & Spread ---
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        feat["f11_bollinger_pct_b"] = (((close - (sma_20 - 2 * std_20)) / (4 * std_20 + 1e-8)) - 0.5).fillna(0.0)
        ret_std = log_ret.rolling(20).std()
        feat["f12_return_zscore"] = ((log_ret - log_ret.rolling(20).mean()) / (ret_std + 1e-8)).fillna(0.0)
        feat["f13_close_to_sma_ratio"] = ((close - sma_20) / (sma_20 + 1e-8)).fillna(0.0)

        # --- 5. Volume & Flow Dynamics ---
        vol_sma_20 = volume.rolling(20).mean()
        feat["f14_volume_ratio"] = ((volume - vol_sma_20) / (vol_sma_20 + 1e-8)).fillna(0.0)

        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-8)
        feat["f15_cmf_surrogate"] = (mf_multiplier * volume).rolling(10).mean() / (vol_sma_20 + 1e-8)
        feat["f16_price_vol_corr"] = log_ret.rolling(10).corr(volume.pct_change().fillna(0.0)).fillna(0.0)

        # Normalize and clip
        feat_cols = list(feat.columns)
        feat = (feat - feat.mean()) / (feat.std() + 1e-8)
        feat = feat.clip(-5.0, 5.0).fillna(0.0)

        # Apply Orthogonalization
        if self.apply_orthogonalization and len(feat) > len(feat_cols):
            X = feat.values
            Q, _ = np.linalg.qr(X)
            feat = pd.DataFrame(Q, index=data.index, columns=feat_cols)

        feat["close"] = close
        return feat


def generate_rich_indicator_dataset(df: pd.DataFrame = None) -> pd.DataFrame:
    """Fallback generator function to preserve external API calls."""
    if df is None:
        dates = pd.date_range("2025-01-01", periods=20000, freq="1min")
        price = 100.0 + np.cumsum(np.random.normal(0, 0.2, 20000))
        df = pd.DataFrame({"close": price}, index=dates)

    engine = FeatureEngine(apply_orthogonalization=True)
    return engine.compute_features(df)
