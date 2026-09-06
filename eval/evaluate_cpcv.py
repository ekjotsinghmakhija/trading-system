import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from features.feature_engineer import FeatureEngineer
from env.strict_sim_env import StrictFrictionEnv


class CPCVEvaluator:
    """
    Evaluates trained policy checkpoints across purged and embargoed out-of-sample
    data blocks to verify performance against strict 20x return and 30% max drawdown targets.
    """

    def __init__(
        self,
        model_path: str,
        vec_norm_path: str,
        initial_balance: float = 50000.0,
        annualization_factor: int = 252 * 375  # 1-minute bars per trading year (NSE: 6.25 hrs/day)
    ):
        self.model_path = model_path
        self.vec_norm_path = vec_norm_path
        self.initial_balance = initial_balance
        self.annualization_factor = annualization_factor

    def _load_env_and_model(self, data: pd.DataFrame, features: pd.DataFrame) -> Tuple[PPO, DummyVecEnv]:
        """Instantiates environment with saved normalization statistics and loads PPO weights."""
        def _init():
            return StrictFrictionEnv(data=data, features=features, initial_balance=self.initial_balance)

        dummy_env = DummyVecEnv([_init])

        # Load empirical observation/reward normalization stats from training
        vec_env = VecNormalize.load(self.vec_norm_path, dummy_env)
        vec_env.training = False  # Freeze running mean/std updates during evaluation
        vec_env.norm_reward = False  # Evaluate raw financial returns

        model = PPO.load(self.model_path, env=vec_env)
        return model, vec_env

    def evaluate_block(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs deterministic policy inference on an out-of-sample data block."""
        engineer = FeatureEngineer()
        features_df = engineer.process_data(raw_df)
        aligned_data = raw_df.loc[features_df.index].reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)

        model, env = self._load_env_and_model(data=aligned_data, features=features_df)

        obs = env.reset()
        done = False

        balances = [self.initial_balance]
        drawdowns = [0.0]
        step_returns = []

        while not done:
            # Deterministic evaluation (zero entropy/noise)
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            info = infos[0]
            balances.append(info["balance"])
            drawdowns.append(info["drawdown"])

            if len(balances) > 1:
                ret = (balances[-1] - balances[-2]) / balances[-2]
                step_returns.append(ret)

            done = dones[0]

        return self.compute_metrics(np.array(balances), np.array(step_returns), np.array(drawdowns))

    def compute_metrics(
        self,
        balances: np.ndarray,
        step_returns: np.ndarray,
        drawdowns: np.ndarray
    ) -> Dict[str, Any]:
        """Computes comprehensive quantitative performance and risk metrics."""
        total_steps = len(step_returns)
        if total_steps == 0:
            raise ValueError("Evaluation sequence contained zero execution steps.")

        # 1. Total Cumulative Return
        total_return_mult = balances[-1] / balances[0]
        net_return_pct = (total_return_mult - 1.0) * 100.0

        # 2. Drawdown Metrics
        max_drawdown_pct = np.max(drawdowns) * 100.0

        # 3. Annualized Return & Calmar Ratio
        years = total_steps / self.annualization_factor
        cagr = (total_return_mult ** (1.0 / max(years, 1e-4))) - 1.0
        calmar_ratio = cagr / (max_drawdown_pct / 100.0) if max_drawdown_pct > 0 else 0.0

        # 4. Profit Factor
        gains = step_returns[step_returns > 0]
        losses = step_returns[step_returns < 0]
        gross_profit = np.sum(gains) if len(gains) > 0 else 0.0
        gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss

        # 5. Sharpe & Sortino Ratios (Annualized)
        mean_ret = np.mean(step_returns)
        std_ret = np.std(step_returns) + 1e-8
        downside_std = np.std(step_returns[step_returns < 0]) + 1e-8

        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(self.annualization_factor)
        sortino_ratio = (mean_ret / downside_std) * np.sqrt(self.annualization_factor)

        # 6. Target Threshold Verification
        meets_20x = total_return_mult >= 20.0
        meets_30_dd = max_drawdown_pct <= 30.0
        passed_all = meets_20x and meets_30_dd

        return {
            "initial_balance": balances[0],
            "final_balance": balances[-1],
            "total_return_multiple": total_return_mult,
            "net_return_pct": net_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "cagr_pct": cagr * 100.0,
            "calmar_ratio": calmar_ratio,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "target_20x_passed": meets_20x,
            "target_30dd_passed": meets_30_dd,
            "evaluation_passed": passed_all
        }


def print_evaluation_report(metrics: Dict[str, Any]):
    """Prints formatted evaluation report."""
    print("=========================================================")
    print("         QUANT RL SYSTEM V2 - EVALUATION REPORT          ")
    print("=========================================================")
    print(f" Initial Balance         : ₹{metrics['initial_balance']:,.2f}")
    print(f" Final Balance           : ₹{metrics['final_balance']:,.2f}")
    print(f" Total Multiple          : {metrics['total_return_multiple']:.2f}x")
    print(f" Net Return              : {metrics['net_return_pct']:.2f}%")
    print(f" Maximum Drawdown        : {metrics['max_drawdown_pct']:.2f}%")
    print(f" Annualized Return (CAGR): {metrics['cagr_pct']:.2f}%")
    print(f" Calmar Ratio            : {metrics['calmar_ratio']:.2f}")
    print(f" Profit Factor           : {metrics['profit_factor']:.2f}")
    print(f" Sharpe Ratio            : {metrics['sharpe_ratio']:.2f}")
    print(f" Sortino Ratio           : {metrics['sortino_ratio']:.2f}")
    print("---------------------------------------------------------")
    print(f" Target 20x Return       : [{'PASS' if metrics['target_20x_passed'] else 'FAIL'}]")
    print(f" Target <30% Drawdown    : [{'PASS' if metrics['target_30dd_passed'] else 'FAIL'}]")
    print(f" OVERALL SYSTEM STATUS   : [{'VERIFIED' if metrics['evaluation_passed'] else 'REJECTED'}]")
    print("=========================================================")


if __name__ == "__main__":
    # Test evaluation call on OOS dataset
    model_file = "models/checkpoints/ppo_strict_final.zip"
    norm_file = "models/checkpoints/vec_normalize_final.pkl"
    test_data_file = "data/processed/test_data.csv"

    if os.path.exists(model_file) and os.path.exists(test_data_file):
        evaluator = CPCVEvaluator(model_path=model_file, vec_norm_path=norm_file)
        raw_test_df = pd.read_csv(test_data_file)
        results = evaluator.evaluate_block(raw_test_df)
        print_evaluation_report(results)
    else:
        print("Model or test data file missing. Run training pipeline first.")
