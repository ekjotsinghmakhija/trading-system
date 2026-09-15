import os
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
from features.feature_engineer import TechnicalFeatureEngineer

FEATURE_COLS = [
    'ema_9', 'ema_21', 'ema_50', 'ema_200', 'adx', 'supertrend', 'supertrend_dir',
    'psar', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'stoch_rsi_k', 'stoch_rsi_d',
    'cci', 'roc', 'willr', 'atr', 'bb_percent', 'bb_width', 'kc_upper', 'kc_lower',
    'donchian_high', 'donchian_low', 'typical_price'
]

def train_baseline_model(parquet_path: str, checkpoint_dir: str = "checkpoints"):
    # 1. Load Data
    df = pd.read_parquet(parquet_path)

    # 2. Engineer Features
    engineer = TechnicalFeatureEngineer(forward_bars=5, threshold_pct=0.003)
    feature_df = engineer.compute_indicators(df)
    labeled_df = engineer.create_categorical_target(feature_df)

    # Drop indicator warm-up NaNs and future target NaNs
    clean_df = labeled_df.dropna(subset=FEATURE_COLS + ['target']).reset_index(drop=True)

    X = clean_df[FEATURE_COLS]
    y = clean_df['target']

    # 3. Time-Series Train/Val Split (80/20 Chronological Cut)
    split_idx = int(len(clean_df) * 0.80)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 4. Train LightGBM Model
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=3,
        random_state=42,
        verbose=-1
    )

    model.fit(X_train, y_train)

    # 5. Out-of-Sample Evaluation
    y_pred = model.predict(X_val)
    print("=== Validation Classification Report ===")
    print(classification_report(y_val, y_pred, target_names=['PUT (-1)', 'NO TRADE (0)', 'CALL (1)']))

    # 6. Save Checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "lgb_baseline_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_baseline_model("data/parquet/nifty_spot_1m.parquet")
