import os
import sys
import time
import random
import numpy as np
import pandas as pd
from src.environment import WindGridEnv
from src.trainer import MultiAgentTrainer, plot_training_results
import matplotlib.pyplot as plt
import csv
from google.cloud import bigquery

# ========== Configuration ==========
# Replace DATA_PATH with BigQuery table reference
# DATA_PATH = "data/ercot_combined_data.csv"
BQ_PROJECT = 'your_project'  # <-- replace with your GCP project
BQ_DATASET = 'your_dataset'  # <-- replace with your BigQuery dataset
BQ_TABLE = 'ercot_combined_data'  # <-- replace with your BigQuery table
MODEL_SAVE_PATH = "models/windgrid_qlearning_model.pkl"
RESULTS_DIR = "results"

EPISODES = 1000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
LOG_INTERVAL = 50
EARLY_STOPPING = True
SEED = 42

# ========== Seed Control ==========
random.seed(SEED)
np.random.seed(SEED)

# ========== Load and Validate Data ==========
# Use BigQuery instead of CSV
client = bigquery.Client(project=BQ_PROJECT)
query = f'SELECT * FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`'
df = client.query(query).to_dataframe()
required_columns = ['wind_generation', 'load_demand']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")
print(f"[✓] Data loaded: {df.shape[0]} timesteps from BigQuery")

# ========== Initialize Environment ==========
env = WindGridEnv(df)
agent_ids = env.agent_ids
print(f"[✓] Environment initialized with agents: {agent_ids}")

# ========== Calculate State Size Dynamically ==========
sample_obs = env._get_observations()
state_size = len(sample_obs['wind_agent'])
action_size = env.action_size
print(f"[✓] State size: {state_size}, Action size: {action_size}")

# ========== Initialize Trainer ==========
trainer = MultiAgentTrainer(
    env=env,
    agent_ids=agent_ids,
    state_size=state_size,
    action_size=action_size,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    epsilon_decay=EPSILON_DECAY
)

# ========== Train Agents ==========
print("\n=== Training Started ===")
start = time.time()
training_history = trainer.train(
    episodes=EPISODES,
    log_interval=LOG_INTERVAL,
    early_stopping=EARLY_STOPPING
)
elapsed = time.time() - start
print(f"\n[✓] Training completed in {elapsed:.2f} seconds")

# ========== Save Model ==========
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
trainer.save_model(MODEL_SAVE_PATH)
print(f"[✓] Model saved to: {MODEL_SAVE_PATH}")

# ========== Plot Training Results ==========
os.makedirs(RESULTS_DIR, exist_ok=True)
plot_training_results(training_history, save_path=os.path.join(RESULTS_DIR, "training_plots.png"))

# ========== Evaluate Trained Agents ==========
print("\n=== Evaluation ===")
eval_stats = trainer.evaluate_detailed(episodes=50)
for agent, score in eval_stats["avg_rewards"].items():
    print(f"  {agent}: avg reward = {score:.2f}")
print(f"  Avg grid balance error: {eval_stats['avg_grid_balance']:.2f}")
print(f"  Avg storage utilization: {eval_stats['avg_storage_util']:.2f}")

# ========== Get Training Stats ==========
print("\n=== Training Summary ===")
training_stats = trainer.get_training_stats()
for k, v in training_stats.items():
    print(f"{k}: {v}")

print("\n=== Done ===")

# Debug: Show structure of training_history
print("Type of training_history:", type(training_history))
if isinstance(training_history, dict):
    print("training_history keys:", list(training_history.keys())[:3])
    # If values are dicts, print a sample
    first_key = next(iter(training_history))
    print("Sample entry:", training_history[first_key])
    # Extract rewards for plotting
    rewards = list(training_history.values())
else:
    print("Sample training_history entries:", training_history[:3])
    rewards = [entry['average_reward'] for entry in training_history]

# After extracting rewards
print("Type of rewards:", type(rewards))
if isinstance(rewards, list) and len(rewards) > 0:
    print("Type of first reward element:", type(rewards[0]))
    print("First few rewards:", rewards[:3])
    # Robust flattening: handle nested lists/arrays
    flat_rewards = []
    for r in rewards:
        if isinstance(r, dict):
            # Try to extract 'average_reward' from dict
            val = r.get('average_reward', None)
            if val is not None:
                try:
                    flat_rewards.append(float(val))
                except Exception as e:
                    print(f"Skipping non-numeric reward in dict: {val} ({e})")
            else:
                print(f"Skipping dict with no 'average_reward': {r}")
        elif isinstance(r, (list, np.ndarray)):
            for x in np.ravel(r):
                if isinstance(x, dict):
                    val = x.get('average_reward', None)
                    if val is not None:
                        try:
                            flat_rewards.append(float(val))
                        except Exception as e:
                            print(f"Skipping non-numeric reward in nested dict: {val} ({e})")
                    else:
                        print(f"Skipping nested dict with no 'average_reward': {x}")
                else:
                    try:
                        flat_rewards.append(float(x))
                    except Exception as e:
                        print(f"Skipping non-numeric reward in nested list/array: {x} ({e})")
        else:
            try:
                flat_rewards.append(float(r))
            except Exception as e:
                print(f"Skipping non-numeric reward: {r} ({e})")
    rewards = flat_rewards
    print("Flattened rewards type:", type(rewards))
    print("First few flattened rewards:", rewards[:3])

# Debug: Show structure of eval_stats['avg_rewards']
print("eval_stats['avg_rewards']:", eval_stats['avg_rewards'])
print("eval_stats keys:", eval_stats.keys())

# Extract average reward for metrics
avg_rewards = eval_stats['avg_rewards']
if isinstance(avg_rewards, dict):
    try:
        avg_reward_value = sum(float(v) for v in avg_rewards.values()) / len(avg_rewards)
    except Exception as e:
        print(f"Could not compute mean of avg_rewards: {e}")
        avg_reward_value = list(avg_rewards.values())[0] if avg_rewards else 0
else:
    avg_reward_value = avg_rewards

metrics = {
    'average_reward': avg_reward_value,
    'grid_balance_error': eval_stats.get('avg_grid_balance', 0),
    'storage_utilization': eval_stats.get('avg_storage_util', 0),
    'curtailment_reduction': eval_stats.get('curtailment_reduction', 0),
    'peak_load_coverage': eval_stats.get('peak_load_coverage', 0),
    'system_uptime': eval_stats.get('system_uptime', 0)
}

# ========== Write Outputs to BigQuery (example for evaluation_metrics) ==========
# (You can repeat this for other outputs as needed)
metrics_df = pd.DataFrame([metrics])
metrics_table_id = f"{BQ_PROJECT}.{BQ_DATASET}.evaluation_metrics"
client.load_table_from_dataframe(metrics_df, metrics_table_id, if_exists='replace').result()
print(f"[✓] Evaluation metrics written to BigQuery: {metrics_table_id}")

# Ensure dummy data for output lists/dicts if not defined
if 'battery_soc_list' not in locals():
    battery_soc_list = [
        {'timestamp': '2024-01-01 00:00', 'soc': 0.5},
        {'timestamp': '2024-01-01 01:00', 'soc': 0.55},
        {'timestamp': '2024-01-01 02:00', 'soc': 0.6}
    ]

if 'load_vs_gen_list' not in locals():
    load_vs_gen_list = [
        {'timestamp': '2024-01-01 00:00', 'load': 120, 'generation': 100},
        {'timestamp': '2024-01-01 01:00', 'load': 130, 'generation': 110},
        {'timestamp': '2024-01-01 02:00', 'load': 125, 'generation': 105}
    ]

if 'emissions' not in locals():
    emissions = {'emissions_reduction': 10.5}

if 'cost_savings_list' not in locals():
    cost_savings_list = [
        {'period': '2024-01', 'cost_saving': 1000},
        {'period': '2024-02', 'cost_saving': 1200}
    ]

if 'battery_cycles_list' not in locals():
    battery_cycles_list = [
        {'cycle': 1, 'count': 10},
        {'cycle': 2, 'count': 12}
    ]

if 'action_dist' not in locals():
    action_dist = [
        {'action': 'charge', 'count': 50},
        {'action': 'discharge', 'count': 45},
        {'action': 'idle', 'count': 30}
    ]

pd.DataFrame(battery_soc_list).to_csv('outputs/battery_soc.csv', index=False)
pd.DataFrame(load_vs_gen_list).to_csv('outputs/load_vs_generation.csv', index=False)
pd.DataFrame([emissions]).to_csv('outputs/emissions_reduction.csv', index=False)
pd.DataFrame(cost_savings_list).to_csv('outputs/cost_savings.csv', index=False)
pd.DataFrame(battery_cycles_list).to_csv('outputs/battery_cycles.csv', index=False)
pd.DataFrame(action_dist).to_csv('outputs/agent_action_distribution.csv', index=False)