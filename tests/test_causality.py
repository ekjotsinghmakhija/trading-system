import numpy as np
from features.feature_engineer import FeatureEngineer


def test_zero_lookahead_leakage():
    engineer = FeatureEngineer()
    df_raw = engineer.generate_synthetic_raw_feed(rows=500)

    # Compute features on full series
    df_full = engineer.compute_18_alpha_matrix(df_raw.copy())

    # Compute features on truncated series (t = 350)
    df_truncated = engineer.compute_18_alpha_matrix(df_raw.iloc[:350].copy())

    # Get the last valid timestamp/index from truncated feature set post-dropna
    last_idx = df_truncated.index[-1]

    # Compare values at the exact same boundary index
    val_full = df_full.loc[last_idx, "scaled_rsi"]
    val_trunc = df_truncated.loc[last_idx, "scaled_rsi"]

    assert np.isclose(val_full, val_trunc, atol=1e-7), "Lookahead leakage detected in feature calculation!"
