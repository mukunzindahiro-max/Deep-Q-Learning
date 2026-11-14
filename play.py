import argparse

import torch

from atari_env import make_env
from model import QNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play Atari Freeway.")
    parser.add_argument("--model", type=str, default="dqn_model.pth", help="path to the saved model")
    parser.add_argument("--episodes", type=int, default=1, help="number of games to play")
    return parser.parse_args()


def select_greedy_action(q_network, state):
    with torch.no_grad():
        state_tensor = torch.as_tensor(state).unsqueeze(0)
        q_values = q_network(state_tensor)
    return int(q_values.argmax(dim=1).item())


def main():
    args = parse_args()

    env = make_env(render_mode="human")
    num_actions = env.action_space.n

    q_network = QNetwork(num_actions)
    q_network.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_network.eval()

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        crossings, episode_length = 0, 0
        done = False

        while not done:
            action = select_greedy_action(q_network, state)
            state, reward, terminated, truncated, _ = env.step(action)
            crossings += int(reward)
            episode_length += 1
            done = terminated or truncated

        print(f"Episode {episode} | length {episode_length} | crossings {crossings}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
