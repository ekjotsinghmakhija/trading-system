from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class FactorDiscoveryEngine:
    """
    Quarterly post-hoc factor discovery module.
    Evaluates feature impact on strategy returns recorded in factor_ledger.parquet.
    """
    def __init__(self, ledger_path: str = "logs/experiments/factor_ledger.parquet"):
        self.ledger_path = Path(ledger_path)

    def load_ledger_data(self) -> pd.DataFrame:
        if not self.ledger_path.exists():
            raise FileNotFoundError(f"[!] Factor ledger not found at {self.ledger_path}")
        df = pl.read_parquet(self.ledger_path).to_pandas()
        return df

    def analyze_factor_importance(self, target_col: str = "sharpe", top_k: int = 10) -> dict:
        df = self.load_ledger_data()
        feature_cols = [c for c in df.columns if c.startswith("feat_")]

        if not feature_cols:
            print("[!] No feature columns ('feat_') detected in the ledger.")
            return {}

        X = df[feature_cols].fillna(0.0)
        y = df[target_col].fillna(0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 1. ElasticNet Linear Factor Importance
        elastic = ElasticNet(l1_ratio=0.5, alpha=0.1, random_state=42)
        elastic.fit(X_scaled, y)
        elastic_imp = pd.Series(np.abs(elastic.coef_), index=feature_cols).sort_values(ascending=False)

        # 2. Random Forest Non-Linear Feature Importance
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
        rf.fit(X_scaled, y)
        rf_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

        print("\n=== TOP FACTOR IMPORTANCE RANKINGS ===")
        print("\n--- ElasticNet Top Factors ---")
        print(elastic_imp.head(top_k))
        print("\n--- Random Forest Top Factors ---")
        print(rf_imp.head(top_k))

        return {
            "elastic_net": elastic_imp.to_dict(),
            "random_forest": rf_imp.to_dict()
        }

if __name__ == "__main__":
    # Smoke Test
    ledger = Path("logs/experiments/factor_ledger.parquet")
    if not ledger.exists():
        ledger.parent.mkdir(parents=True, exist_ok=True)
        # Create dummy factor ledger for testing
        dummy_df = pl.DataFrame({
            "step": [100000, 200000, 300000],
            "sharpe": [1.2, 1.8, 2.1],
            "feat_rsi_14": [0.45, 0.62, 0.71],
            "feat_vwap_dist": [0.001, -0.002, 0.004],
            "feat_ofi": [0.12, -0.34, 0.55]
        })
        dummy_df.write_parquet(ledger, compression="snappy")

    discovery = FactorDiscoveryEngine()
    discovery.analyze_factor_importance()
