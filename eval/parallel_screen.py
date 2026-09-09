import os
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
from eval.evaluate_cpcv import CPCVEvaluator


def evaluate_single_checkpoint(args):
    ckpt_path, test_csv_path = args
    vec_norm_file = ckpt_path.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vec_norm_file):
        vec_norm_file = "models/checkpoints/vec_normalize_final.pkl"

    test_df = pd.read_csv(test_csv_path)
    evaluator = CPCVEvaluator(model_path=ckpt_path, vec_norm_path=vec_norm_file)
    res = evaluator.evaluate_block(test_df)

    return {
        "checkpoint": os.path.basename(ckpt_path),
        "path": ckpt_path,
        "vec_norm": vec_norm_file,
        "final_balance": res["final_balance"],
        "total_return_pct": res["total_return_pct"],
        "max_drawdown_pct": res["max_drawdown_pct"],
        "sharpe_ratio": res["sharpe_ratio"],
        "avg_exposure": res["avg_position_exposure"]
    }


def run_parallel_screening(test_csv: str = "data/processed/test_data.csv"):
    checkpoints = glob.glob("models/checkpoints/ppo_checkpoint_*_steps.zip")
    print(f"--- Screening {len(checkpoints)} Checkpoints across CPU Cores ---")

    # Use up to 8 CPU cores for parallel out-of-sample backtesting
    num_workers = min(8, cpu_count())
    tasks = [(ckpt, test_csv) for ckpt in checkpoints]

    with Pool(processes=num_workers) as pool:
        results = pool.map(evaluate_single_checkpoint, tasks)

    df_results = pd.DataFrame(results).sort_values(by="final_balance", ascending=False)

    print("\n" + "=" * 70)
    print("                 TOP PERFORMING CHECKPOINTS LEADERBOARD               ")
    print("=" * 70)
    print(df_results.head(10)[['checkpoint', 'total_return_pct', 'max_drawdown_pct', 'sharpe_ratio']].to_string(index=False))
    print("=" * 70 + "\n")

    df_results.to_csv("models/checkpoints/leaderboard.csv", index=False)
    return df_results


if __name__ == "__main__":
    run_parallel_screening()
