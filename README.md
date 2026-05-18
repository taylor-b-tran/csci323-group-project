# CSCI323 — Drive a Taxi

**Course:** CSCI323 Artificial Intelligence — University of Wollongong, Autumn 2026

**Project:** Reinforcement Learning for Autonomous Taxi Navigation using Q-Learning in Taxi-v3

## Group Members

| Name | Student ID | Role |
|---|---|---|
| Thanh Binh Tran | tbt286 | Project Lead + Introduction |
| Hoang Thanh Truc Nguyen | httn639 | Background Theory |
| The Long Tran | tlt834 | Code Implementation |
| Sneha Akter Shomy | sas651 | Evaluation + Results |
| Ashraful Islam Bhuiyan | aib998 | Report Integration + Slides |

## Project Overview

The taxi agent learns to navigate a 5×5 grid-world using Q-Learning.
It must pick up a passenger and drop them off at the correct destination
using the fewest possible steps.

## Setup

```bash
cd Code
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Or run individually:

```bash
python q_learning.py   # train the agent
python evaluate.py     # evaluate trained agent
python plots.py        # generate graphs
```

Also, you can see how the taxi actually performs by changing:

```python
env = make_env(render=True)
```

## Results

Output files are saved to `Code/results/`:
- `training_rewards.png` — reward convergence curve
- `training_steps.png` — steps per episode over training
- `evaluation_summary.csv` — final metrics table
- `q_table.npy` — trained Q-table

## Algorithms

- **Random Policy** — baseline agent
- **Q-Learning** — main RL algorithm
- *(Optional)* **SARSA** — comparison if time allows

## Submission

Deadline: **29 May 2026**
