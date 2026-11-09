"""Creates the Pong environment used by both train.py and play.py."""

import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

ENV_ID = "ALE/Pong-v5"
FRAME_STACK = 4

gym.register_envs(ale_py)


def make_env(render_mode=None):
    # frameskip=1 because AtariPreprocessing does the frame skipping itself.
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)

    # Convert each frame to greyscale, resize it to 84x84,
    # and repeat each chosen action for 4 frames.
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)

    # Give the network the last 4 frames so it can see how the ball is moving.
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    return env
