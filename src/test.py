
import ale_py
import gymnasium as gym


gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode="human")
state, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)

env.close()
