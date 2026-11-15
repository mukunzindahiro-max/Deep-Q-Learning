import ale_py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

ENV_ID = "ALE/Freeway-v5"
FRAME_STACK = 4

gym.register_envs(ale_py)


def make_env(render_mode=None):
    # frameskip=1 here because AtariPreprocessing does the frame skipping itself
    env = gym.make(ENV_ID, frameskip=1, render_mode=render_mode)
    # gray 84x84 frames, and each action is repeated for 4 frames
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    # stack the last 4 frames so the network can see which way the cars move
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    return env
