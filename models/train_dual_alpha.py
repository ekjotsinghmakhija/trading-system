import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def purged_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_1b",
    train_bars: int = 1000,
    test_bars: int = 200,
    purge_bars: int = 5,
    embargo_bars: int = 5,
) -> pd.DataFrame:
    """Executes Walk-Forward Validation with Purging and Embargoing to guarantee

    zero out-of-fold data leakage or cross-window contamination.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    total_samples = len(df)
    results = []

    start_idx = 0

    while start_idx + train_bars + purge_bars + test_bars <= total_samples:
        train_start = start_idx
        train_end = train_start + train_bars

        # Purge buffer eliminates target overlap between train and test sets
        test_start = train_end + purge_bars
        test_end = test_start + test_bars

        if test_end > total_samples:
            break

        train_df = df.iloc[train_start:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        # Fit Scaler strictly inside the current Training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols])
        y_train = train_df[target_col].values

        # Apply Training-derived parameters to Test set
        X_test = scaler.transform(test_df[feature_cols])
        y_test = test_df[target_col].values

        # Train Dual Alpha Models (e.g., Fast/Short-horizon vs. Slow/Long-horizon components)
        alpha_fast = Ridge(alpha=10.0)
        alpha_slow = Ridge(alpha=100.0)

        alpha_fast.fit(X_train, y_train)
        alpha_slow.fit(X_train, y_train)

        preds_fast = alpha_fast.predict(X_test)
        preds_slow = alpha_slow.predict(X_test)

        # Enforce ensemble combination strictly on test period
        combined_signal = 0.6 * preds_fast + 0.4 * preds_slow

        fold_res = test_df[["timestamp", "close", target_col]].copy()
        fold_res["pred_fast"] = preds_fast
        fold_res["pred_slow"] = preds_slow
        fold_res["alpha_signal"] = combined_signal
        fold_res["fold_id"] = start_idx

        results.append(fold_res)

        # Embargo step advances start_idx beyond immediate test boundary
        start_idx += test_bars + embargo_bars

    if not results:
        raise ValueError(
            "Insufficient dataset length for configured walk-forward parameters."
        )

    return pd.concat(results, ignore_index=True)


def calculate_leak_free_metrics(results_df: pd.DataFrame) -> dict:
    """Calculates backtest signal metrics after applying Zerodha-style transaction

    costs and slippage.
    """
    df = results_df.copy()

    # Signal position: Long if alpha_signal > 0, Short if < 0
    df["position"] = np.sign(df["alpha_signal"])

    # Returns = Position * Next bar return - Friction (Slippage + Brokerage/STT)
    cost_per_trade_bps = 0.0005  # ~5 bps friction for Indian markets
    df["position_change"] = df["position"].diff().abs().fillna(0)
    df["friction"] = df["position_change"] * cost_per_trade_bps

    df["strategy_ret"] = (df["position"] * df["target_1b"]) - df["friction"]

    cum_return = (1 + df["strategy_ret"]).prod() - 1
    sharpe = (
        (df["strategy_ret"].mean() / (df["strategy_ret"].std() + 1e-8))
        * np.sqrt(252 * 375)  # Minute bar annualized factor
    )

    return {
        "cumulative_return": cum_return,
        "sharpe_ratio": sharpe,
        "total_trades": int((df["position_change"] > 0).sum()),
    }


if __name__ == "__main__":
    from features.build_features import compute_features_leak_free

    # Generate test market series
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=3000, freq="1min")
    raw_df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 100 + np.random.randn(3000).cumsum(),
            "high": 101 + np.random.randn(3000).cumsum(),
            "low": 99 + np.random.randn(3000).cumsum(),
            "close": 100 + np.random.randn(3000).cumsum(),
            "volume": np.random.randint(500, 5000, size=3000),
        }
    )

    feat_df = compute_features_leak_free(raw_df)
    feats = [c for c in feat_df.columns if c.startswith("feat_")]

    oof_predictions = purged_walk_forward_cv(
        feat_df,
        feature_cols=feats,
        train_bars=1500,
        test_bars=300,
        purge_bars=10,
    )

    metrics = calculate_leak_free_metrics(oof_predictions)
    print("Dual Alpha Training Complete.")
    print(f"Out-of-Fold Samples: {len(oof_predictions)}")
    print(
        f"Sharpe: {metrics['sharpe_ratio']:.2f} | Cum Return: {metrics['cumulative_return']:.2%}"
    )
