from copy import deepcopy
import torch
import numpy as np
import gymnasium as gym
import gym_sokoban
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.logger import Video
import argparse


class VideoRecorderCallback(BaseCallback):
    def __init__(
        self,
        eval_env: gym.Env,
        render_freq: int,
    ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        """
        super().__init__()
        eval_env.reset(seed=1234)
        self._eval_env = eval_env
        self._render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            for deterministic in [True, False]:
                env = deepcopy(self._eval_env)
                screens = []
                observations = env.render()
                screens.append(observations.transpose(2, 0, 1))
                while True:
                    actions, _ = model.predict(
                        observations,
                        deterministic=deterministic,
                    )
                    new_observation, _, terminated, truncated, _ = env.step(int(actions))
                    screens.append(new_observation.transpose(2, 0, 1))
                    if terminated or truncated:
                        break
                    observations = new_observation
                self.logger.record(
                    "trajectory/video_" + ("deterministic" if deterministic else "stochastic"),
                    Video(torch.from_numpy(np.asarray([screens])), fps=60),
                    exclude=("stdout", "log", "json", "csv"),
                )
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--policy_name", type=str, default="CnnPolicy")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--observation_mode", type=str, default="rgb_array")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = gym.make("Sokoban-small-v1", observation_mode=args.observation_mode)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/",
        name_prefix=args.exp_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    video_callback = VideoRecorderCallback(
        gym.make("Sokoban-small-v1", observation_mode=args.observation_mode),
        render_freq=100_000,
    )

    callbacks = CallbackList([checkpoint_callback, video_callback])

    model = PPO(
        args.policy_name, env, verbose=1, tensorboard_log=f"runs/{args.exp_name}"
    )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

