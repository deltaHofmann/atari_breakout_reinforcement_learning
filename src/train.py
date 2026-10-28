import csv
import json
import os
import random
import time

import ale_py
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ============================================================
# CONFIG
# ============================================================

NUM_ENVS = 25
NUM_EPISODES = 20000

BATCH_SIZE = 32 * 3
GAMMA = 0.985
LEARNING_RATE = 0.0001

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY_STEPS = 300000

LEARNING_STARTS = 80_000

# 1 vector step = 8 environment steps
# 2 updates per vector step correspond to approx. 1 update per 4 environment steps.
GRADIENT_UPDATES_PER_VECTOR_STEP = 2

TARGET_UPDATE_FREQUENCY = 10_000
BUFFER_CAPACITY = 100_000
CHECKPOINT_FREQUENCY = 200
MAX_EPISODE_STEPS = 10_000

IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
FIRE_ACTION = 1


# ============================================================
# DIRECTORIES & DEVICE
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# ENVIRONMENT
# ============================================================

def make_env():
    """Creates a single Breakout environment for AsyncVectorEnv."""
    def thunk():
        gym.register_envs(ale_py)
        env = gym.make("ALE/Breakout-v5")
        env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        return env
    return thunk


def create_vector_env():
    return AsyncVectorEnv(
        [make_env() for _ in range(NUM_ENVS)],
        shared_memory=True,
        autoreset_mode=AutoresetMode.SAME_STEP
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(frames):
    """Converts a batch of RGB Atari frames into binary 84x84 images."""
    brightness = frames.mean(axis=-1, dtype=np.float32)
    tensor = torch.from_numpy(brightness).unsqueeze(1)
    resized = F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest")
    binary = (resized[:, 0] > 100).to(torch.uint8)
    return binary.numpy()


def initialize_frame_stacks(frames):
    """Creates an initial stack of 4 identical frames: (N, 4, 84, 84)."""
    processed = preprocess(frames)
    return np.repeat(processed[:, None, :, :], FRAME_STACK_SIZE, axis=1)


def update_frame_stacks(old_stacks, processed_frames, reset_mask):
    """Adds the newest frame to every stack and resets stacks for finished episodes."""
    new_stacks = np.roll(old_stacks, shift=-1, axis=1)
    new_stacks[:, -1] = processed_frames

    if np.any(reset_mask):
        new_stacks[reset_mask] = np.repeat(
            processed_frames[reset_mask, None, :, :],
            FRAME_STACK_SIZE,
            axis=1
        )

    return new_stacks


# ============================================================
# DQN ARCHITECTURE
# ============================================================

class DQN(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, state):
        return self.network(state)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, states, actions, rewards, next_states, dones):
        for i in range(len(actions)):
            self.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


# ============================================================
# TRAINING HELPERS
# ============================================================

def optimize_model(
    policy_net,
    target_net,
    replay_buffer,
    optimizer,
    loss_function,
    batch_size,
    gamma,
    num_updates
):
    if len(replay_buffer) < LEARNING_STARTS:
        return None

    losses = []

    for _ in range(num_updates):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.from_numpy(states).float().to(device)
        next_states = torch.from_numpy(next_states).float().to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        # Current Q values
        q_values = policy_net(states)
        current_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = target_net(next_states).max(dim=1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Loss & Optimization
        loss = loss_function(current_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def get_epsilon(global_step):
    epsilon = max(
        EPSILON_MIN,
        EPSILON_START - (global_step / EPSILON_DECAY_STEPS) * (EPSILON_START - EPSILON_MIN)
    )
    return epsilon


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Create environments
    envs = create_vector_env()
    action_size = envs.single_action_space.n

    print("\nNumber of environments:", NUM_ENVS)
    print("Observation shape:", envs.single_observation_space.shape)
    print("Action space:", envs.single_action_space)
    print("Number of actions:", action_size)

    # Networks & Optimizer
    policy_net = DQN(action_size).to(device)
    target_net = DQN(action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_function = nn.SmoothL1Loss()

    # Replay buffer
    state_shape = (FRAME_STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY, state_shape)

    # Config & Logging setup
    config = {
        "num_envs": NUM_ENVS,
        "num_episodes": NUM_EPISODES,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "epsilon_start": EPSILON_START,
        "epsilon_min": EPSILON_MIN,
        "epsilon_decay_steps": EPSILON_DECAY_STEPS,
        "learning_starts": LEARNING_STARTS,
        "gradient_updates_per_vector_step": GRADIENT_UPDATES_PER_VECTOR_STEP,
        "target_update_frequency": TARGET_UPDATE_FREQUENCY,
        "buffer_capacity": BUFFER_CAPACITY,
        "frame_stack_size": FRAME_STACK_SIZE,
        "image_size": IMAGE_SIZE,
        "fire_action": FIRE_ACTION,
        "max_episode_steps": MAX_EPISODE_STEPS,
    }

    with open("results/config.json", "w") as f:
        json.dump(config, f, indent=4)

    log_file = "results/training_log.csv"
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "episode_length", "epsilon",
            "average_loss", "buffer_size", "lives", "termination_reason"
        ])

    # Environment initialization
    observations, infos = envs.reset()
    states = initialize_frame_stacks(observations)

    episode_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)
    lives = np.full(NUM_ENVS, 5, dtype=np.int32)
    min_lives = lives.copy()
    needs_fire = np.ones(NUM_ENVS, dtype=bool)

    global_step = 0
    completed_episodes = 0
    next_checkpoint = CHECKPOINT_FREQUENCY
    loss_window = []

    start_time = time.time()
    last_speed_time = start_time
    last_speed_step = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================
    try:
        while completed_episodes < NUM_EPISODES:
            epsilon = get_epsilon(global_step)

            # Action selection (Epsilon-Greedy)
            state_tensor = torch.from_numpy(states).float().to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                greedy_actions = q_values.argmax(dim=1).cpu().numpy()

            random_mask = np.random.random(NUM_ENVS) < epsilon
            random_actions = np.random.randint(0, action_size, size=NUM_ENVS)
            actions = np.where(random_mask, random_actions, greedy_actions).astype(np.int64)

            # Handle FIRE action after reset or life loss
            force_fire_mask = needs_fire.copy()
            actions[force_fire_mask] = FIRE_ACTION
            needs_fire[force_fire_mask] = False

            # Step environments
            next_observations, rewards, terminated, truncated, infos = envs.step(actions)
            global_step += NUM_ENVS

            episode_rewards += rewards
            episode_lengths += 1

            processed_next = preprocess(next_observations)
            done = terminated | truncated
            next_states = update_frame_stacks(states, processed_next, done)

            # Store transitions (exclude forced FIRE actions)
            store_mask = ~force_fire_mask
            replay_buffer.add_batch(
                states[store_mask],
                actions[store_mask],
                rewards[store_mask],
                next_states[store_mask],
                done[store_mask]
            )

            # Track lives
            new_lives = infos.get("lives", None)
            if new_lives is not None:
                new_lives = np.asarray(new_lives, dtype=np.int32)
                life_lost = new_lives < lives
                needs_fire |= (life_lost & (new_lives > 0) & (~done))
                lives = new_lives
                min_lives = np.minimum(min_lives, lives)

            needs_fire[done] = True
            states = next_states

            # Train model
            if len(replay_buffer) >= LEARNING_STARTS:
                loss = optimize_model(
                    policy_net, target_net, replay_buffer, optimizer,
                    loss_function, BATCH_SIZE, GAMMA, GRADIENT_UPDATES_PER_VECTOR_STEP
                )
                if loss is not None:
                    loss_window.append(loss)
                    if len(loss_window) > 1000:
                        loss_window.pop(0)

            # Target network update
            if global_step % TARGET_UPDATE_FREQUENCY == 0:
                target_net.load_state_dict(policy_net.state_dict())
                print(f"Target network updated at {global_step:,} steps")

            # Handle finished episodes
            done_indices = np.flatnonzero(done)
            for i in done_indices:
                completed_episodes += 1
                termination_reason = "terminated" if terminated[i] else "truncated"
                average_loss = float(np.mean(loss_window)) if loss_window else 0.0

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        completed_episodes,
                        episode_rewards[i],
                        episode_lengths[i],
                        epsilon,
                        average_loss,
                        len(replay_buffer),
                        min_lives[i],
                        termination_reason
                    ])

                print(
                    f"Episode {completed_episodes:4d} | Env {i} | "
                    f"Reward: {episode_rewards[i]:7.2f} | Length: {episode_lengths[i]:5d} | "
                    f"Epsilon: {epsilon:.3f} | Buffer: {len(replay_buffer):6d} | Loss: {average_loss:.5f}"
                )

                # Reset per-episode stats
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0
                lives[i] = 5
                min_lives[i] = 5
                needs_fire[i] = True

                # Checkpoint saving
                if completed_episodes >= next_checkpoint:
                    checkpoint = {
                        "episode": completed_episodes,
                        "global_step": global_step,
                        "model_state_dict": policy_net.state_dict(),
                        "target_model_state_dict": target_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epsilon": epsilon,
                        "config": config
                    }
                    torch.save(checkpoint, f"models/checkpoint_episode_{completed_episodes}.pth")
                    torch.save(checkpoint, "models/checkpoint_latest.pth")
                    print(f"Checkpoint saved at episode {completed_episodes}")
                    next_checkpoint += CHECKPOINT_FREQUENCY

            # Speed report
            current_time = time.time()
            if current_time - last_speed_time >= 10:
                elapsed = current_time - last_speed_time
                steps = global_step - last_speed_step
                steps_per_second = steps / elapsed
                print(
                    f"Speed: {steps_per_second:.1f} env-steps/s | "
                    f"{steps_per_second / NUM_ENVS:.1f} vector-steps/s"
                )
                last_speed_time = current_time
                last_speed_step = global_step

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        checkpoint = {
            "episode": completed_episodes,
            "global_step": global_step,
            "model_state_dict": policy_net.state_dict(),
            "target_model_state_dict": target_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epsilon": get_epsilon(global_step),
            "config": config
        }
        torch.save(checkpoint, "models/checkpoint_interrupted.pth")
        print("Emergency checkpoint saved.")

    finally:
        envs.close()

    # ========================================================
    # FINAL MODEL SAVE
    # ========================================================
    torch.save(policy_net.state_dict(), "models/breakout_dqn.pth")
    print("\nTraining finished.")
    print("Episodes:", completed_episodes)
    print("Environment steps:", f"{global_step:,}")
    print("Experiences in replay buffer:", f"{len(replay_buffer):,}")
    print("Model saved to: models/breakout_dqn.pth")