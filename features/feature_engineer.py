import numpy as np
import pandas as pd
from typing import List


class FeatureEngineer:
    """
    Zero-lookahead feature extraction pipeline.
    Transforms raw OHLCV data into stationary state vectors while retaining execution data.
    """

    def __init__(self, return_horizons: List[int] = [1, 5, 15, 60], vol_window: int = 20, zscore_window: int = 200, frac_d: float = 0.4):
        self.return_horizons = return_horizons
        self.vol_window = vol_window
        self.zscore_window = zscore_window
        self.frac_d = frac_d
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

    def _calculate_frac_diff(self, df: pd.DataFrame, threshold: float = 1e-4) -> pd.DataFrame:
        """Applies vectorized fractional differentiation on Close prices to preserve memory."""
        weights = [1.0]
        k = 1
        while True:
            w_k = -weights[-1] / k * (self.frac_d - k + 1)
            if abs(w_k) < threshold:
                break
            weights.append(w_k)
            k += 1

        weights = np.array(weights[::-1])
        res = np.convolve(df["close"].values, weights, mode="valid")

        # Pad initial steps with NaN to match series index alignment safely
        pad_size = len(df["close"]) - len(res)
        frac_diff_series = np.pad(res, (pad_size, 0), mode='constant', constant_values=np.nan)

        col_name = f"frac_diff_{str(self.frac_d).replace('.', '_')}"
        df[col_name] = frac_diff_series
        self.feature_columns.append(col_name)
        return df

    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline execution."""
        df = df.copy()

        # 1. Forward-fill gaps to eliminate lookahead bias
        df.ffill(inplace=True)

        # 2. Compute stationary state features
        df = self._calculate_log_returns(df)
        df = self._calculate_parkinson_volatility(df)
        df = self._calculate_volume_zscore(df)
        df = self._calculate_frac_diff(df)

        # 3. Drop early rows containing NaNs from rolling/convolution operations
        df.dropna(subset=self.feature_columns, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Return full dataset retaining execution columns (timestamp, close) alongside feature columns
        return df


if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/market_data.csv")
    fe = FeatureEngineer()
    featured_df = fe.process_data(raw_df)
    featured_df.to_csv("data/processed/featured_market_data.csv", index=False)
    print("Feature processing complete. Output shape:", featured_df.shape)
