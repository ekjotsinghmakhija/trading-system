import pandas as pd
import numpy as np


class FactorDiscoveryAnalyzer:
    """
    Parses Parquet factor ledger to extract Spearman Rank Information Coefficients across horizons.
    """
    def __init__(self, parquet_path: str = "logs/experiments/factor_ledger.parquet"):
        self.parquet_path = parquet_path

    def analyze_ic(self) -> pd.DataFrame:
        df = pd.read_parquet(self.parquet_path)
        horizons = ["fwd_ret_1m", "fwd_ret_5m", "fwd_ret_15m", "fwd_ret_30m", "fwd_ret_60m"]

        ignore_cols = set(horizons) | {
            "timestamp_step", "action_raw", "position", "raw_pnl",
            "friction_cost", "net_pnl", "equity", "hold_duration"
        }
        feature_cols = [c for c in df.columns if c not in ignore_cols]

        ic_results = []
        for feat in feature_cols:
            row = {"feature": feat}
            for h in horizons:
                if h in df.columns:
                    spearman_ic = df[feat].corr(df[h], method="spearman")
                    row[h] = spearman_ic
            ic_results.append(row)

        return pd.DataFrame(ic_results).sort_values(by="fwd_ret_5m", ascending=False)
