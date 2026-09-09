import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def prune_correlated_features(df: pd.DataFrame, feature_cols: list, correlation_threshold: float = 0.65) -> list:
    """
    Removes redundant features based on pairwise absolute correlation limit (|r| > 0.65).
    Keeps the feature with lower mean correlation to the remaining feature pool.
    """
    corr_matrix = df[feature_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper_tri.columns:
        # Find features highly correlated with this column
        high_corr = upper_tri.index[upper_tri[col] > correlation_threshold].tolist()
        if high_corr:
            # Drop the current column if it exceeds correlation threshold
            to_drop.add(col)

    selected = [col for col in feature_cols if col not in to_drop]
    print(f"[+] Pairwise Correlation Filter (|r| <= {correlation_threshold}): Retained {len(selected)}/{len(feature_cols)} features.")
    return selected

def enforce_vif_filter(df: pd.DataFrame, feature_cols: list, vif_threshold: float = 5.0) -> list:
    """
    Iteratively eliminates multicollinear features until all remaining features achieve VIF < 5.0.
    """
    features = list(feature_cols)
    while True:
        X = df[features].dropna()
        if X.shape[1] <= 1:
            break

        vif_data = pd.DataFrame()
        vif_data["feature"] = features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        max_vif = vif_data["VIF"].max()
        if max_vif > vif_threshold:
            max_feature = vif_data.sort_values("VIF", ascending=False).iloc[0]["feature"]
            features.remove(max_feature)
            print(f"    --> Dropped '{max_feature}' with VIF={max_vif:.2f}")
        else:
            break

    print(f"[+] VIF Filter (< {vif_threshold}): Retained {len(features)} orthogonal features.")
    return features

if __name__ == "__main__":
    from feature_engineer import FeatureEngine

    # Generate test matrix
    dates = pd.date_range("2026-09-01 09:15:00", periods=500, freq="1min", tz="UTC")
    dummy_df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(500).cumsum() + 25000,
        "high": np.random.randn(500).cumsum() + 25020,
        "low": np.random.randn(500).cumsum() + 24980,
        "close": np.random.randn(500).cumsum() + 25000,
        "volume": np.random.randint(100, 5000, size=500)
    })

    feat_engine = FeatureEngine(dummy_df)
    matrix = feat_engine.build_feature_matrix()
    raw_features = [c for c in matrix.columns if c.startswith("feat_")]

    # Run two-stage orthogonalization
    uncorrelated = prune_correlated_features(matrix, raw_features, correlation_threshold=0.65)
    final_features = enforce_vif_filter(matrix, uncorrelated, vif_threshold=5.0)
    print("Final Orthogonal Feature Set:", final_features)
