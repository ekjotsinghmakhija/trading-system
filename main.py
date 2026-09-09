import sys
import argparse
from pathlib import Path
import duckdb

# Import repository pipeline modules
from data.process_raw import init_duckdb, process_and_partition_raw_data, sync_parquet_to_duckdb
from features.feature_engineer import FeatureEngine
from features.orthogonalizer import prune_correlated_features, enforce_vif_filter
from models.train_engine import train_ppo_12m_steps
from eval.factor_discovery import FactorDiscoveryEngine


def run_data_ingestion():
    """Phase 1: Ingests raw dumps, partitions Parquet stores, and binds DuckDB views."""
    print("\n==========================================================")
    print("      PHASE 1: DATA INGESTION & DUCKDB BINDING            ")
    print("==========================================================\n")
    conn = init_duckdb()
    process_and_partition_raw_data(symbol="NIFTY")
    process_and_partition_raw_data(symbol="BANKNIFTY")
    sync_parquet_to_duckdb(conn)
    conn.close()
    print("[✓] Data ingestion complete.")


def run_feature_pipeline():
    """Phase 2: Calculates 18-feature alpha matrix and runs orthogonalization filters."""
    print("\n==========================================================")
    print("      PHASE 2: FEATURE CALCULATIONS & ORTHOGONALIZATION   ")
    print("==========================================================\n")
    import pandas as pd
    import numpy as np

    # Example load check / smoke sample for verification
    dates = pd.date_range("2026-09-01 09:15:00", periods=500, freq="1min", tz="UTC")
    dummy_df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(500).cumsum() + 25000,
        "high": np.random.randn(500).cumsum() + 25020,
        "low": np.random.randn(500).cumsum() + 24980,
        "close": np.random.randn(500).cumsum() + 25000,
        "volume": np.random.randint(100, 5000, size=500)
    })

    engine = FeatureEngine(dummy_df)
    matrix = engine.build_feature_matrix()
    raw_features = [c for c in matrix.columns if c.startswith("feat_")]

    uncorrelated = prune_correlated_features(matrix, raw_features, correlation_threshold=0.65)
    final_features = enforce_vif_filter(matrix, uncorrelated, vif_threshold=5.0)

    print(f"[✓] Feature pipeline operational. Active Orthogonal Features: {len(final_features)}")
    return final_features


def run_training_pipeline():
    """Phase 3 & 4: Launches the 12M-step PPO walk-forward training run."""
    print("\n==========================================================")
    print("      PHASE 3 & 4: HIGH-THROUGHPUT RL MODEL TRAINING      ")
    print("==========================================================\n")
    train_ppo_12m_steps()


def run_factor_discovery():
    """Phase 5: Analyzes accumulated factor ledger importance via ElasticNet & RF."""
    print("\n==========================================================")
    print("      PHASE 5: POST-HOC FACTOR DISCOVERY ANALYSIS         ")
    print("==========================================================\n")
    ledger_path = Path("logs/experiments/factor_ledger.parquet")
    if not ledger_path.exists():
        print(f"[!] Ledger not found at {ledger_path}. Skipping factor discovery.")
        return

    discovery = FactorDiscoveryEngine(ledger_path=str(ledger_path))
    discovery.analyze_factor_importance()


def main():
    parser = argparse.ArgumentParser(description="Quant RL Pipeline Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["all", "ingest", "features", "train", "eval"],
        default="train",
        help="Pipeline phase execution mode (default: train)"
    )

    args = parser.parse_args()

    if args.mode == "ingest":
        run_data_ingestion()
    elif args.mode == "features":
        run_feature_pipeline()
    elif args.mode == "train":
        run_training_pipeline()
    elif args.mode == "eval":
        run_factor_discovery()
    elif args.mode == "all":
        run_data_ingestion()
        run_feature_pipeline()
        run_training_pipeline()
        run_factor_discovery()


if __name__ == "__main__":
    main()
