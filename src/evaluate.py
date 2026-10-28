
import random
from collections import deque
from pathlib import Path

import ale_py
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from dqn import DQN


gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

frame_stack = deque(maxlen=4)
frames = []


def preprocess_frame(frame):
    frame = torch.as_tensor(frame, dtype=torch.float32)
    frame = frame.mean(dim=2)
    frame = frame.unsqueeze(0).unsqueeze(0)

    frame = F.interpolate(
        frame,
        size=(84, 84),
        mode="nearest",
    )

    return (frame > 100).float().squeeze().numpy()


def get_state(frame):
    processed = preprocess_frame(frame)
    frame_stack.append(processed)

    while len(frame_stack) < 4:
        frame_stack.append(processed)

    return torch.tensor(np.array(frame_stack), dtype=torch.float32)


action_size = env.action_space.n
policy_net = DQN(action_size).to(device)

checkpoint = torch.load(
    "models/checkpoint_latest.pth",
    map_location=device,
    weights_only=False,
)

policy_net.load_state_dict(checkpoint["model_state_dict"])
policy_net.eval()

evaluation_epsilon = checkpoint["epsilon"]

print("Loaded checkpoint from episode:", checkpoint["episode"])
print("Evaluation epsilon:", evaluation_epsilon)


try:
    frame, info = env.reset()
    state = get_state(frame)
    total_reward = 0

    # Save the initial frame.
    frames.append(env.render())

    # Start the game.
    frame, reward, terminated, truncated, info = env.step(1)
    state = get_state(frame)
    total_reward += reward
    frames.append(env.render())

    while not (terminated or truncated):
        if random.random() < evaluation_epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = state.unsqueeze(0).to(device)

            with torch.no_grad():
                q_values = policy_net(state_tensor)

            action = q_values.argmax(dim=1).item()

        frame, reward, terminated, truncated, info = env.step(action)
        state = get_state(frame)
        total_reward += reward

        # Store the rendered frame for the GIF.
        frames.append(env.render())

    print("Reward:", total_reward)

    output_path = Path("results/breakout_demo.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imageio.mimsave(
        output_path,
        frames,
        fps=30,
        loop=0,
    )

    print(f"GIF saved to: {output_path.resolve()}")

finally:
    env.close()
