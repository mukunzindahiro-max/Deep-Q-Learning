import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

ENV_ID = "ALE/Freeway-v5"
FRAME_STACK = 4

gym.register_envs(ale_py)


def make_env(render_mode=None):
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    return env
