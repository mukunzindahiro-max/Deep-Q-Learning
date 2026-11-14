import argparse

import numpy as np
import torch

from atari_env import make_env
from model import QNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Atari Freeway.")
    parser.add_argument("--model", type=str, default="dqn_model.pth", help="path to the saved model")
    parser.add_argument("--episodes", type=int, default=1, help="number of games to play")
    return parser.parse_args()


def load_model(model_path, num_actions):
    q_network = QNetwork(num_actions)
    q_network.load_state_dict(torch.load(model_path, map_location="cpu"))
    q_network.eval()
    return q_network


def select_greedy_action(q_network, state):
    with torch.no_grad():
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        q_values = q_network(state_tensor)
    return int(q_values.argmax(dim=1).item())


def play_episode(env, q_network):
    state, _ = env.reset()
    episode_reward, episode_length = 0.0, 0
    done = False

    while not done:
        action = select_greedy_action(q_network, state)
        state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        episode_length += 1
        done = terminated or truncated

    return episode_reward, episode_length


def main():
    args = parse_args()

    env = make_env(render_mode="human")
    q_network = load_model(args.model, env.action_space.n)

    rewards = []
    for episode in range(1, args.episodes + 1):
        episode_reward, episode_length = play_episode(env, q_network)
        rewards.append(episode_reward)
        print(f"Episode {episode} | length {episode_length} | reward {episode_reward:.0f}", flush=True)

    env.close()
    print(f"Average reward over {len(rewards)} games: {np.mean(rewards):.1f}")


if __name__ == "__main__":
    main()
