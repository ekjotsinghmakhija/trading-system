import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class FeatureEngineer:
    """
    Zero-lookahead feature extraction pipeline.
    Transforms raw OHLCV data into stationary state vectors.
    """

    def __init__(self, return_horizons: List[int] = [1, 5, 15, 60], vol_window: int = 20, zscore_window: int = 200):
        self.return_horizons = return_horizons
        self.vol_window = vol_window
        self.zscore_window = zscore_window
        self.feature_columns: List[str] = []

    def _calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes rolling log returns for multiple time horizons."""
        for k in self.return_horizons:
            col_name = f"log_ret_{k}"
            df[col_name] = np.log(df["close"] / df["close"].shift(k))
            self.feature_columns.append(col_name)
        return df

    def _calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates Parkinson volatility using High and Low prices."""
        vol_raw = np.sqrt((1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2))
        df["parkinson_vol"] = vol_raw.ewm(span=self.vol_window, adjust=False).mean()
        self.feature_columns.append("parkinson_vol")
        return df

    def _calculate_volume_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes volume using a backward-looking rolling Z-score."""
        roll_mean = df["volume"].rolling(window=self.zscore_window, min_periods=1).mean()
        roll_std = df["volume"].rolling(window=self.zscore_window, min_periods=1).std()
        df["volume_zscore"] = (df["volume"] - roll_mean) / (roll_std + 1e-8)
        self.feature_columns.append("volume_zscore")
        return df

    def _fractional_differentiation(self, series: pd.Series, d: float, threshold: float = 1e-4) -> pd.Series:
        """
        Applies fractional differentiation. Calculates binomial coefficients (weights)
        and applies them to the time series to achieve stationarity while preserving memory.
        """
        # 1. Compute weights based on fraction d
        weights = [1.0]
        k = 1
        while True:
            weight = -weights[-1] * (d - k + 1) / k
            if abs(weight) < threshold:
                break
            weights.append(weight)
            k += 1

        weights = np.array(weights[::-1]) # Reverse to align with chronological order
        window_size = len(weights)

        # 2. Apply weights via rolling dot product
        frac_diff = pd.Series(index=series.index, dtype=np.float64)

        # Optimization: Use numpy stride tricks or rolling apply for speed
        # For a clean implementation, we use a loop over valid indices
        prices = series.values

        for i in range(window_size, len(prices)):
            window_data = prices[i - window_size : i]
            frac_diff.iloc[i] = np.dot(weights, window_data)

        # Add to feature columns list (assuming this is called within a pipeline loop)
        col_name = f"frac_diff_{str(d).replace('.', '_')}"
        return frac_diff

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline execution. Enforces strict ffill for missing data to prevent lookahead.
        """
        df = df.copy()

        # 1. Strict anti-lookahead forward fill for gaps
        df.ffill(inplace=True)

        # 2. Compute features
        df = self._calculate_log_returns(df)
        df = self._calculate_parkinson_volatility(df)
        df = self._calculate_volume_zscore(df)

        # 3. Drop NaNs created by rolling windows (removes early rows, zero lookahead impact)
        df.dropna(subset=self.feature_columns, inplace=True)

        return df[self.feature_columns]
