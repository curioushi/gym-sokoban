import gymnasium as gym
import gym_sokoban
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback

env = gym.make("Sokoban-small-v1")

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="A2C_MlpPolicy",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="runs/MlpPolicy_01")

model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
