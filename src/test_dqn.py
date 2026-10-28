
import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from dqn import DQN


IMAGE_SIZE = 84
FRAME_STACK_SIZE = 4
BRIGHTNESS_THRESHOLD = 100

gym.register_envs(ale_py)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/Breakout-v5")


def preprocess(frame):
    """Convert an RGB frame to a binary 84x84 image."""
    brightness = frame.mean(axis=-1, dtype=np.float32)
    tensor = torch.from_numpy(brightness)[None, None, :, :]

    resized = F.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )

    binary_frame = (resized[0, 0] > BRIGHTNESS_THRESHOLD)
    return binary_frame.to(torch.uint8).numpy()


try:
    observation, info = env.reset()

    # Initialize the state with four copies of the first frame.
    frame = preprocess(observation)
    state = np.repeat(
        frame[None, :, :],
        FRAME_STACK_SIZE,
        axis=0,
    )

    # Expected input shape: (batch_size, 4, 84, 84).
    state_tensor = torch.from_numpy(state).float()
    state_tensor = state_tensor.unsqueeze(0).to(device)

    model = DQN(action_size=env.action_space.n).to(device)
    model.eval()

    with torch.no_grad():
        q_values = model(state_tensor)

    print("Device:", device)
    print("Input shape:", tuple(state_tensor.shape))
    print("Q-values:", q_values)
    print("Output shape:", tuple(q_values.shape))

finally:
    env.close()
