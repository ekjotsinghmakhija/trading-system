import numpy as np
import pandas as pd


def compute_features_leak_free(
    df: pd.DataFrame, shift_signals: bool = True
) -> pd.DataFrame:
    """Computes technical features strictly using historical information.

    Prevents look-ahead bias by ensuring zero future-data leakage in feature
    calculations.
    """
    df = df.copy()

    # Ensure chronological sort
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required price/volume columns: {missing}")

    # --- 1. PAST-ONLY FEATURES (Computed strictly on current/past bars) ---

    # Log returns (t relative to t-1)
    df["feat_log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["feat_log_ret_5"] = np.log(df["close"] / df["close"].shift(5))

    # Rolling Volatility (strictly backward-looking window)
    df["feat_volatility_20"] = (
        df["feat_log_ret_1"].rolling(window=20, min_periods=20).std()
    )

    # Moving Average Ratio (Close relative to rolling mean)
    ma_20 = df["close"].rolling(window=20, min_periods=20).mean()
    df["feat_ma_ratio_20"] = df["close"] / ma_20 - 1.0

    # Rolling Z-Score of Close Price
    std_20 = df["close"].rolling(window=20, min_periods=20).std()
    df["feat_zscore_20"] = (df["close"] - ma_20) / (std_20 + 1e-8)

    # Average True Range (ATR)
    high_low = df["high"] - df["low"]
    high_cp = (df["high"] - df["close"].shift(1)).abs()
    low_cp = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["feat_atr_14"] = tr.rolling(window=14, min_periods=14).mean()

    # Relative Volume (Volume / 20-bar SMA Volume)
    vol_ma_20 = df["volume"].rolling(window=20, min_periods=20).mean()
    df["feat_rel_volume"] = df["volume"] / (vol_ma_20 + 1e-8)

    # --- 2. PREVENT SAME-BAR LOOK-AHEAD BIAS ---
    # If trades execute at bar 't' close or open based on features at 't',
    # features MUST be shifted by +1 so bar 't' only uses data up to 't-1'.
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if shift_signals:
        df[feature_cols] = df[feature_cols].shift(1)

    # --- 3. TARGET DEFINITION (Strict Forward Returns) ---
    # Target at bar 't' = Return from close(t) to close(t+1)
    # Execution occurs AFTER features at 't' are known.
    df["target_1b"] = (df["close"].shift(-1) / df["close"]) - 1.0

    # Drop NaNs created by rolling windows, feature shifting, and target shifting
    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":
    # Test stub
    sample_data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=100, freq="1h"),
            "open": np.random.randn(100).cumsum() + 100,
            "high": np.random.randn(100).cumsum() + 102,
            "low": np.random.randn(100).cumsum() + 98,
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(100, 1000, size=100),
        }
    )
    processed = compute_features_leak_free(sample_data)
    print(f"Features computed. Shape: {processed.shape}")
