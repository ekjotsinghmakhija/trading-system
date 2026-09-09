import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.strict_sim_env import AdvancedFrictionEnv


class EnsembleAgent:
    """
    Ensemble Agent that averages position allocations across the Top-K checkpoints
    to maximize upside while suppressing variance and drawdowns.
    """
    def __init__(self, leaderboard_csv: str = "models/checkpoints/leaderboard.csv", top_k: int = 5):
        df_leader = pd.read_csv(leaderboard_csv)
        top_models = df_leader.head(top_k)

        self.models = []
        self.vec_norms = []

        for _, row in top_models.iterrows():
            ckpt_path = row['path']
            vec_norm_path = row['vec_norm']

            # Load model
            model = RecurrentPPO.load(ckpt_path, device="cpu")
            self.models.append(model)
            self.vec_norms.append(vec_norm_path)

    def predict_ensemble_action(self, obs_raw, lstm_states_list, episode_starts):
        actions = []
        new_lstm_states = []

        for idx, model in enumerate(self.models):
            # Predict position choice per model
            action, state = model.predict(
                obs_raw,
                state=lstm_states_list[idx],
                episode_start=episode_starts,
                deterministic=True
            )
            actions.append(action[0][0])
            new_lstm_states.append(state)

        # Average positions across all Top-K models
        ensemble_action = np.array([[np.mean(actions)]], dtype=np.float32)
        return ensemble_action, new_lstm_states
