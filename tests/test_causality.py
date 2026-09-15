# tests/test_causality.py

import numpy as np
import pandas as pd
from features.feature_engineer import TechnicalFeatureEngineer


def generate_synthetic_ohlcv(rows: int = 500) -> pd.DataFrame:
    """Generates synthetic minute-level OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01 09:15", periods=rows, freq="1min")
    close_prices = 25000.0 + np.cumsum(np.random.randn(rows) * 5.0)

    return pd.DataFrame({
        'open': close_prices - np.random.uniform(0, 2, rows),
        'high': close_prices + np.random.uniform(0, 5, rows),
        'low': close_prices - np.random.uniform(0, 5, rows),
        'close': close_prices,
        'volume': np.random.randint(100, 5000, rows)
    }, index=dates)


def test_zero_lookahead_leakage():
    engineer = TechnicalFeatureEngineer()
    df_raw = generate_synthetic_ohlcv(rows=500)

    # Compute features on full series
    df_full = engineer.compute_indicators(df_raw.copy())

    # Truncate at cutoff bar t = 350
    cutoff = 350
    df_truncated = engineer.compute_indicators(df_raw.iloc[:cutoff].copy())

    # Get boundary timestamp
    last_idx = df_truncated.index[-1]

    feature_cols = [
        c for c in df_truncated.columns
        if c not in ['open', 'high', 'low', 'close', 'volume']
    ]

    for col in feature_cols:
        val_full = df_full.loc[last_idx, col]
        val_trunc = df_truncated.loc[last_idx, col]

        # Skip if either is NaN due to fractional diff burn-in window variances
        if pd.isna(val_full) or pd.isna(val_trunc):
            continue

        assert np.isclose(val_full, val_trunc, atol=1e-7), \
            f"Lookahead leakage detected in feature '{col}' at index {last_idx}!"


def test_future_perturbation_causality():
    """Ensures modifying future rows T > 350 does not change features at T <= 350."""
    engineer = TechnicalFeatureEngineer()
    df_raw = generate_synthetic_ohlcv(rows=500)

    df_base = engineer.compute_indicators(df_raw.copy())

    # Perturb close prices far in the future at T >= 400
    df_perturbed = df_raw.copy()
    df_perturbed.iloc[400:, df_perturbed.columns.get_loc('close')] += 1000.0
    df_perturbed_fe = engineer.compute_indicators(df_perturbed)

    feature_cols = [
        c for c in df_base.columns
        if c not in ['open', 'high', 'low', 'close', 'volume']
    ]

    # Features up to T = 350 must be mathematically identical
    diff = np.abs(
        df_base.iloc[:350][feature_cols].values -
        df_perturbed_fe.iloc[:350][feature_cols].values
    )
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-7, f"Future price leakage detected! Max feature delta: {max_diff}"
