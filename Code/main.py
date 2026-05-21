import os
import pandas as pd
from q_learning import training
from evaluate import evaluate
from random_policy import run_random_policy
from plots import save_plots
from config import RESULTS_DIR

def main():
    """Run training, evaluation, plotting, and summary export."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=== Training Q-Learning Agent ===")
    q_table, episode_rewards, episode_steps = training()

    print("\n=== Evaluating Q-Learning Agent ===")
    ql_results = evaluate(q_table)
    print(ql_results)

    print("\n=== Running Random Policy Baseline ===")
    rand_results = run_random_policy()
    print(rand_results)

    print("\n=== Saving Plots ===")
    save_plots(episode_rewards, episode_steps)

    print("\n=== Saving Summary CSV ===")
    summary = pd.DataFrame({
        "Metric": ["Avg Reward", "Avg Steps", "Success Rate (%)", "Avg Illegal Actions"],
        "Random Policy": [rand_results["avg_reward"], rand_results["avg_steps"],
                        rand_results["success_rate"], rand_results["avg_illegal"]],
        "Q-Learning": [ql_results["avg_reward"], ql_results["avg_steps"],
                    ql_results["success_rate"], ql_results["avg_illegal"]]
    })

    summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print("Summary saved to results/evaluation_summary.csv")
    print("\nAll done!")

if __name__ == "__main__":
    main()