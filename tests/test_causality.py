import numpy as np
import pytest
import torch
from features.feature_engineer import FeatureEngineer


def test_zero_lookahead_leakage():
    engineer = FeatureEngineer()
    df_raw = engineer.generate_synthetic_raw_feed(rows=500)

    # Compute features on full series
    df_full = engineer.compute_18_alpha_matrix(df_raw.copy())

    # Compute features on truncated series (t = 250)
    df_truncated = engineer.compute_18_alpha_matrix(df_raw.iloc[:250].copy())

    # Value at t=249 must match identically regardless of future data presence
    val_full = df_full.iloc[248]["rsi_14"]
    val_trunc = df_truncated.iloc[248]["rsi_14"]

    assert np.isclose(val_full, val_trunc, atol=1e-7), "Lookahead leakage detected in feature calculation!"
