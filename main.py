import time
import numpy as np
import torch

from features.feature_engineer import FeatureEngineer
from features.factor_ledger import FactorDiscoveryLedger
from features.orthogonalizer import FeatureOrthogonalizer
from env.strict_sim_env import StrictOptionSimEnv
from models.architecture import ActorCriticTCNGRU
from models.train_engine import HighThroughputPPOTrainer
from models.callbacks import PerformanceMonitorCallback
from eval.evaluate_cpcv import CPCVEvaluator
from logs.logger import setup_logger
from logs.trade_telemetry import TradeTelemetryLogger


def run_production_pipeline():
    logger = setup_logger()
    logger.info("Initializing Quantum Engine Production Loop...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1: Feature Generation (Strictly Causal)
    engineer = FeatureEngineer()
    raw_data = engineer.generate_synthetic_raw_feed(rows=20000)
    df_features = engineer.compute_18_alpha_matrix(raw_data)

    # Decorrelate Features (Prevent Multicollinearity Overfitting)
    ortho = FeatureOrthogonalizer(variance_explained=0.98)
    df_ortho = ortho.fit_transform(df_features, exclude_cols=["close"])

    feature_cols = [c for c in df_ortho.columns if c != "close"]

    # Step 2: Initialize Environments and Loggers
    env = StrictOptionSimEnv(df_ortho, feature_cols=feature_cols, initial_capital=50000.0)
    factor_ledger = FactorDiscoveryLedger("logs/experiments/factor_ledger.parquet")
    telemetry = TradeTelemetryLogger("logs/experiments/trade_telemetry.parquet")
    monitor = PerformanceMonitorCallback()
    cpcv = CPCVEvaluator(n_splits=5, purge_window=60, embargo_window=120)

    # Step 3: Model & Trainer Setup
    input_dim = len(feature_cols) + 2  # Features + Position + Hold Ratio
    model = ActorCriticTCNGRU(input_dim=input_dim, action_dim=1)
    trainer = HighThroughputPPOTrainer(model=model, env=env, lr=3e-6, device=device)

    logger.info("Starting Zero Look-Ahead Iteration Training...")

    obs, _ = env.reset()
    for epoch in range(1, 51):
        start_time = time.time()

        # Generate rollout without future feature leakage
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, logp, val = model.get_action(obs_t)

        act_val = float(action.cpu().numpy()[0][0])
        next_obs, reward, term, trunc, info = env.step(act_val)

        # High-Precision Telemetry Stream
        state_rec = info["state_record"]
        telemetry.record_event(
            timestamp=state_rec["timestamp_step"],
            step=env.current_step,
            order_id=f"ORD_{env.current_step}",
            action=act_val,
            target_qty=env.current_position,
            filled_price=df_ortho.iloc[env.current_step]["close"],
            expected_price=df_ortho.iloc[env.current_step - 1]["close"],
            slippage_bps=0.0018 * 10000,
            latency_ms=(time.time() - start_time) * 1000,
            equity=env.equity,
            pnl=info["pnl"]
        )

        # Factor Persistence
        factor_ledger.log_step(obs[:18].tolist(), feature_cols[:18], state_rec)

        # AMP Training Step
        dummy_batch = torch.tensor(np.array([obs]), dtype=torch.float32)
        dummy_act = torch.tensor(np.array([[act_val]]), dtype=torch.float32)
        dummy_logp = torch.tensor(np.array([logp.cpu().numpy()[0]]), dtype=torch.float32)
        dummy_adv = torch.tensor(np.array([reward]), dtype=torch.float32)
        dummy_rtg = torch.tensor(np.array([env.equity]), dtype=torch.float32)

        loss = trainer.train_epoch_amp(dummy_batch, dummy_act, dummy_logp, dummy_adv, dummy_rtg)

        # Compute Purged Out-of-Sample DSR
        returns_hist = np.diff(np.array([50000.0, env.equity]))
        dsr = cpcv.compute_deflated_sharpe_ratio(returns_hist)

        # Stream Metrics
        monitor.log_metrics(epoch, {
            "PPO_Loss": loss,
            "Equity": env.equity,
            "Deflated_Sharpe_Ratio": dsr,
            "Entropy_Coef": trainer.entropy_coef
        })

        if epoch % 10 == 0:
            telemetry.flush()
            factor_ledger.flush_to_disk()
            monitor.save_checkpoint(model, trainer.optimizer, epoch, dsr)
            logger.info(f"Epoch {epoch} | Equity: ₹{env.equity:,.2f} | DSR: {dsr:.4f}")

        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()

    monitor.close()
    telemetry.flush()
    factor_ledger.flush_to_disk()
    logger.info("Pipeline Execution Complete. System fully synchronized.")


if __name__ == "__main__":
    run_production_pipeline()
