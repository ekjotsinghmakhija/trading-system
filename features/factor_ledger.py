import os
import pandas as pd


class FactorDiscoveryLedger:
    """
    Parquet factor-return discovery ledger for offline alpha attribution.
    """
    def __init__(self, output_path: str = "logs/experiments/factor_ledger.parquet"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.records = []

    def log_step(self, features: list, feature_names: list, state_record: dict):
        record = {name: float(val) for name, val in zip(feature_names, features)}
        record.update(state_record)
        self.records.append(record)

    def flush_to_disk(self):
        if not self.records:
            return
        df_new = pd.DataFrame(self.records)
        if os.path.exists(self.output_path):
            df_existing = pd.read_parquet(self.output_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(self.output_path, index=False, compression="snappy")
        else:
            df_new.to_parquet(self.output_path, index=False, compression="snappy")
        self.records.clear()
