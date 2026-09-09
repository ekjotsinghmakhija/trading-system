import os
import numpy as np
import pandas as pd
import logging

try:
    from models.train_ppo import train
    from eval.evaluate_cpcv import CPCVEvaluator, print_evaluation_report
except ImportError as e:
    logging.warning(f"Import error. Ensure you run this from the project root: {e}")

from features.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_dummy_data(filepath: str, rows: int = 100000):
    """Generates synthetic OHLCV data if authentic market data is missing for pipeline testing."""
    logging.info(f"Generating synthetic OHLCV dataset at {filepath}...")
    np.random.seed(42)

    returns = np.random.normal(loc=0.00001, scale=0.001, size=rows)
    close_prices = 10000 * np.cumprod(1 + returns)

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0005, rows)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0005, rows)))
    open_prices = np.roll(close_prices, shift=1)
    open_prices[0] = close_prices[0]

    volume = np.random.lognormal(mean=10, sigma=1, size=rows)
    timestamps = pd.date_range(start="2020-01-01", periods=rows, freq="1min")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logging.info("Synthetic data generation complete.")


def split_data_purged(raw_data_path: str, train_out: str, test_out: str, embargo_bars: int = 1875):
    """
    Computes feature engineering first, then splits into Train and Test sets
    with an embargo period to avoid lookahead bias.
    """
    logging.info("Loading raw data and running feature engineering pipeline...")
    df = pd.read_csv(raw_data_path)

    fe = FeatureEngineer()
    featured_df = fe.process_data(df)

    total_rows = len(featured_df)
    split_idx = int(total_rows * 0.7)

    train_end_idx = split_idx - embargo_bars
    if train_end_idx <= 0:
        raise ValueError("Dataset too small to support the required embargo period.")

    train_df = featured_df.iloc[:train_end_idx].copy()
    test_df = featured_df.iloc[split_idx:].copy()

    os.makedirs(os.path.dirname(train_out), exist_ok=True)
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logging.info(f"Data Splitting Complete. Embargo applied: {embargo_bars} bars dropped.")
    logging.info(f"Train Set: {len(train_df)} rows | Test Set: {len(test_df)} rows.")


def run_pipeline():
    """Executes the end-to-end Quant RL System v2 pipeline."""
    raw_path = "data/raw/market_data.csv"
    train_path = "data/processed/train_data.csv"
    test_path = "data/processed/test_data.csv"

    model_checkpoint = "models/checkpoints/ppo_strict_final.zip"
    vec_norm_checkpoint = "models/checkpoints/vec_normalize_final.pkl"

    # 1. Data Initialization & Processing
    logging.info("--- PHASE 1: DATA PIPELINE ---")
    if not os.path.exists(raw_path):
        logging.warning("Authentic market data not found. Falling back to synthetic generator.")
        generate_dummy_data(raw_path)

    split_data_purged(raw_path, train_path, test_path)

    # 2. Model Training
    logging.info("--- PHASE 2: POLICY TRAINING ---")
    try:
        train()
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}", exc_info=True)
        return

    # 3. CPCV Out-Of-Sample Evaluation
    logging.info("--- PHASE 3: OUT-OF-SAMPLE EVALUATION ---")
    if not os.path.exists(model_checkpoint):
        logging.error("Model checkpoint missing. Evaluation aborted.")
        return

    logging.info("Loading test data and evaluating strict policy parameters...")
    evaluator = CPCVEvaluator(model_path=model_checkpoint, vec_norm_path=vec_norm_checkpoint)
    raw_test_df = pd.read_csv(test_path)

    results = evaluator.evaluate_block(raw_test_df)
    print_evaluation_report(results)


if __name__ == "__main__":
    run_pipeline()
