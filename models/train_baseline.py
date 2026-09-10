import duckdb
import polars as pl
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr

DB_PATH = "data/duckdb/market_data.duckdb"

FEATURE_COLS = [
    "return_1m", "return_5m", "return_15m", "return_60m",
    "bar_span", "volatility_15m", "volatility_60m",
    "vwap_distance", "nifty_banknifty_ratio"
]
TARGET_COL = "target_5m_return"

def train_alpha_model():
    print("[+] Querying engineered feature matrix from DuckDB...")
    con = duckdb.connect(DB_PATH)

    # Load dataset filtered for valid target rows
    query = f"""
        SELECT timestamp, {', '.join(FEATURE_COLS)}, {TARGET_COL}
        FROM nifty_features_1m
        WHERE {TARGET_COL} IS NOT NULL
        ORDER BY timestamp ASC
    """
    df = con.execute(query).pl()
    con.close()

    # Temporal Splitting (No random cross-validation)
    train_df = df.filter(pl.col("timestamp") < pl.datetime(2024, 1, 1))
    val_df   = df.filter((pl.col("timestamp") >= pl.datetime(2024, 1, 1)) &
                         (pl.col("timestamp") < pl.datetime(2025, 6, 1)))
    test_df  = df.filter(pl.col("timestamp") >= pl.datetime(2025, 6, 1))

    print(f"[+] Dataset Split Summary:")
    print(f"  ├─ Train Set: {train_df.height:,} rows (2018 – 2023)")
    print(f"  ├─ Val Set:   {val_df.height:,} rows (2024 – Mid 2025)")
    print(f"  └─ Test Set:  {test_df.height:,} rows (Mid 2025 – 2026)")

    # Prepare Numpy arrays for LightGBM
    X_train, y_train = train_df.select(FEATURE_COLS).to_numpy(), train_df[TARGET_COL].to_numpy()
    X_val, y_val     = val_df.select(FEATURE_COLS).to_numpy(), val_df[TARGET_COL].to_numpy()
    X_test, y_test   = test_df.select(FEATURE_COLS).to_numpy(), test_df[TARGET_COL].to_numpy()

    # Model Initialization
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    print("[+] Training LightGBM Alpha Predictor...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    # Evaluate Out-Of-Sample Performance on Test Set
    preds = model.predict(X_test)

    # Calculate Quantitative Evaluation Metrics
    ic, _ = spearmanr(preds, y_test)
    directional_acc = np.mean(np.sign(preds) == np.sign(y_test))

    print("\n" + "=" * 60)
    print("           OUT-OF-SAMPLE TEST EVALUATION (2025-2026)")
    print("=" * 60)
    print(f"  ├─ Information Coefficient (Spearman IC): {ic:.4f}")
    print(f"  ├─ Directional Accuracy (Hit Rate)      : {directional_acc * 100:.2f}%")
    print("=" * 60)

    # Save model predictions back to DuckDB for backtesting
    test_results = test_df.select(["timestamp", TARGET_COL]).with_columns(
        pl.Series("pred_signal", preds)
    )

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS test_predictions")
    con.execute("CREATE TABLE test_predictions AS SELECT * FROM test_results")
    con.close()

    print("[+] Saved test predictions to DuckDB table 'test_predictions'.")

if __name__ == "__main__":
    train_alpha_model()
