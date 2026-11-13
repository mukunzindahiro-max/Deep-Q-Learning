import argparse

import torch

from atari_env import make_env
from model import QNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="Play Atari Freeway with a trained DQN.")
    parser.add_argument("--model", default="dqn_model.pth", help="path to the saved model")
    parser.add_argument("--episodes", type=int, default=5, help="number of episodes to play")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode="human")

    try:
        network = QNetwork(env.action_space.n).to(device)
        state_dict = torch.load(args.model, map_location=device, weights_only=True)
        network.load_state_dict(state_dict)
        network.eval()

        for episode in range(1, args.episodes + 1):
            state, _ = env.reset(seed=args.seed + episode - 1)
            episode_reward = 0.0
            terminated = truncated = False

            while not (terminated or truncated):
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
                    action = int(network(state_tensor).argmax(dim=1).item())

                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            print(f"Episode {episode}: reward={episode_reward:.1f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
