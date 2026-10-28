
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


log_path = "results/training_log.csv"
config_path = "results/config.json"
results_dir = "results"

# Load training data and configuration
for path in (log_path, config_path):
    if not os.path.exists(path):
        raise SystemExit(f"Error: {path} not found.")

df = pd.read_csv(log_path)

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

if df.empty:
    raise SystemExit("Error: Training log is empty.")


# Print training parameters
print("\nTraining Parameters")
print("=" * 30)

for key, value in config.items():
    print(f"{key}: {value}")


# Print training results
print("\nTraining Results")
print("=" * 30)

print(f"Episodes: {len(df)}")
print(f"Best Reward: {df['reward'].max()}")
print(f"Average Reward: {df['reward'].mean():.2f}")
print(f"Best Episode Length: {df['episode_length'].max()}")
print(f"Final Epsilon: {df['epsilon'].iloc[-1]:.3f}")


os.makedirs(results_dir, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename), dpi=150)
    plt.show()


# Reward
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["reward"], label="Reward")

window = 10
if len(df) >= window:
    moving_average = df["reward"].rolling(window).mean()
    plt.plot(
        df["episode"],
        moving_average,
        label=f"Moving Average ({window})"
    )

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Breakout Training Reward")
plt.legend()
plt.grid()
save_plot("reward.png")


# Episode length
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["episode_length"])
plt.xlabel("Episode")
plt.ylabel("Environment Steps")
plt.title("Episode Length")
plt.grid()
save_plot("episode_length.png")


# Exploration rate
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["epsilon"])
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Exploration Rate")
plt.grid()
save_plot("epsilon.png")


# Training loss
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["average_loss"])
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("DQN Training Loss")
plt.grid()
save_plot("loss.png")


# Replay buffer size
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["buffer_size"])
plt.xlabel("Episode")
plt.ylabel("Number of Experiences")
plt.title("Replay Buffer Size")
plt.grid()
save_plot("buffer_size.png")


# Remaining lives
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["lives"])
plt.xlabel("Episode")
plt.ylabel("Lives Remaining")
plt.title("Remaining Lives at End of Episode")
plt.grid()
save_plot("lives.png")


# Termination reasons
termination_counts = df["termination_reason"].value_counts()

plt.figure(figsize=(8, 5))
termination_counts.plot(kind="bar")
plt.xlabel("Termination Reason")
plt.ylabel("Number of Episodes")
plt.title("Episode Termination Reasons")
plt.xticks(rotation=0)
plt.grid(axis="y")
save_plot("termination_reasons.png")


# Summary
plot_files = [
    "reward.png",
    "episode_length.png",
    "epsilon.png",
    "loss.png",
    "buffer_size.png",
    "lives.png",
    "termination_reasons.png",
]

print("\nPlots saved:")
for filename in plot_files:
    print(os.path.join(results_dir, filename))

print("\nPlotting finished.")
