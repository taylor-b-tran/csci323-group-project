from pathlib import Path

"""
Project configuration for Taxi-v3 Q-learning training and evaluation.
"""

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

ALPHA = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
EPISODES = 10000
EVAL_EPISODES = 500
SEED = 42
Q_TABLE_PATH = RESULTS_DIR / "q_table.npy"