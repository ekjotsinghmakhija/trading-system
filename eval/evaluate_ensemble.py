import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.strict_sim_env import AdvancedFrictionEnv
from features.feature_engineer import FeatureEngineer
from eval.ensemble_agent import EnsembleAgent


class EnsembleEvaluator:
    def __init__(self, leaderboard_csv: str = "models/checkpoints/leaderboard.csv", top_k: int = 5, initial_balance: float = 50000.0):
        self.initial_balance = initial_balance
        self.ensemble = EnsembleAgent(leaderboard_csv=leaderboard_csv, top_k=top_k)

        # Load vector normalization stats from the top checkpoint
        df_leader = pd.read_csv(leaderboard_csv)
        self.vec_norm_path = df_leader.iloc[0]['vec_norm']

    def _prepare_env(self, raw_df: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        fe = FeatureEngineer()

        if "parkinson_vol" not in raw_df.columns:
            processed_df = fe.process_data(raw_df)
        else:
            processed_df = raw_df.copy()

        execution_data = processed_df[['close']].copy()

        numeric_features = processed_df.select_dtypes(include=[np.number])
        for col in ['timestamp', 'date', 'datetime', 'time']:
            if col in numeric_features.columns:
                numeric_features = numeric_features.drop(columns=[col])

        def _init():
            return AdvancedFrictionEnv(
                data=execution_data,
                features=numeric_features,
                initial_balance=self.initial_balance
            )

        dummy_env = DummyVecEnv([_init])

        if self.vec_norm_path and os.path.exists(self.vec_norm_path):
            env = VecNormalize.load(self.vec_norm_path, dummy_env)
            env.training = False
            env.norm_reward = False
        else:
            env = dummy_env

        return env, processed_df

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        env, processed_df = self._prepare_env(test_df)

        obs = env.reset()
        done = False

        balances = [self.initial_balance]
        positions = []
        drawdowns = []

        num_models = len(self.ensemble.models)
        lstm_states = [None] * num_models
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = self.ensemble.predict_ensemble_action(
                obs,
                lstm_states_list=lstm_states,
                episode_starts=episode_starts
            )
            obs, rewards, dones, infos = env.step(action)

            info = infos[0]
            balances.append(info.get("balance", self.initial_balance))
            positions.append(float(action[0][0]))
            drawdowns.append(info.get("drawdown", 0.0))

            done = dones[0]
            episode_starts[0] = done

        returns = pd.Series(balances).pct_change().dropna()
        total_return = (balances[-1] - self.initial_balance) / self.initial_balance
        max_drawdown = max(drawdowns) if drawdowns else 0.0

        sharpe_ratio = 0.0
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 375)

        return {
            "initial_balance": self.initial_balance,
            "final_balance": balances[-1],
            "total_return_pct": total_return * 100,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(positions),
            "avg_position_exposure": float(np.mean(positions)),
            "max_short_exposure": float(np.min(positions)),
            "max_long_exposure": float(np.max(positions))
        }


def print_ensemble_report(results: Dict[str, Any]):
    print("\n" + "=" * 50)
    print("      ENSEMBLE OUT-OF-SAMPLE EVALUATION REPORT   ")
    print("=" * 50)
    print(f" Initial Balance        : ${results['initial_balance']:,.2f}")
    print(f" Final Balance          : ${results['final_balance']:,.2f}")
    print(f" Total Return (%)       : {results['total_return_pct']:.2f}%")
    print(f" Max Drawdown (%)       : {results['max_drawdown_pct']:.2f}%")
    print(f" Annualized Sharpe      : {results['sharpe_ratio']:.4f}")
    print(f" Total Steps Evaluated   : {results['total_trades']}")
    print("-" * 50)
    print(f" Avg Position Exposure  : {results['avg_position_exposure']:.2f} (Range [-1.0, 1.0])")
    print(f" Max Long Exposure      : {results['max_long_exposure']:.2f}")
    print(f" Max Short Exposure     : {results['max_short_exposure']:.2f}")
    print("=" * 50 + "\n")
