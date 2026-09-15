import joblib
import pandas as pd
from features.feature_engineer import TechnicalFeatureEngineer
from models.train_baseline import FEATURE_COLS

def run_live_pipeline(data_path: str, model_path: str = "checkpoints/lgb_baseline_model.pkl"):
    # Load model checkpoint
    model = joblib.load(model_path)

    # Load recent market data slice
    df = pd.read_parquet(data_path)

    # Compute current indicators
    engineer = TechnicalFeatureEngineer()
    feature_df = engineer.compute_indicators(df)

    latest_bar = feature_df[FEATURE_COLS].iloc[[-1]]
    latest_close = feature_df['close'].iloc[-1]

    prediction = model.predict(latest_bar)[0]
    probabilities = model.predict_proba(latest_bar)[0]

    signal_map = {-1: "BUY_PUT", 0: "HOLD", 1: "BUY_CALL"}

    print("\n--------------------------------------------------")
    print(f"Index Price: {latest_close}")
    print(f"Signal:      {signal_map[prediction]}")
    print(f"Probabilities: PUT={probabilities[0]:.2f}, HOLD={probabilities[1]:.2f}, CALL={probabilities[2]:.2f}")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    run_live_pipeline("data/parquet/nifty_spot_1m.parquet")
