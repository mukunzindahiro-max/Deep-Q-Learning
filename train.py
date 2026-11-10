import argparse
import csv
import json
import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from atari_env import make_env
from model import QNetwork
from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari Freeway.")

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=50_000,
                        help="number of steps to go from eps_start to eps_end")

    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--learning_starts", type=int, default=10_000,
                        help="steps of experience to collect before learning")
    parser.add_argument("--train_freq", type=int, default=4, help="update the network every N steps")
    parser.add_argument("--target_update", type=int, default=1_000,
                        help="copy online network to target network every N steps")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--member", type=str, default=None, help="group member name, e.g. James")
    parser.add_argument("--experiment", type=int, default=None, help="experiment number 1-10")
    return parser.parse_args()


def get_output_paths(args):
    if args.member is None:
        return "dqn_model.pth", "logs/dqn_model.csv", "plots/dqn_model.png"

    run_name = f"experiment_{args.experiment}"
    return (
        f"models/{args.member}/{run_name}.pth",
        f"logs/{args.member}/{run_name}.csv",
        f"plots/{args.member}/{run_name}.png",
    )


def get_epsilon(step, eps_start, eps_end, eps_decay):
    fraction = min(step / eps_decay, 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def select_action(online_net, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randrange(num_actions)

    with torch.no_grad():
        state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
        q_values = online_net(state_tensor)
    return int(q_values.argmax(dim=1).item())


def train_step(online_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.as_tensor(states, device=device)
    actions = torch.as_tensor(actions, device=device)
    rewards = torch.as_tensor(rewards, device=device)
    next_states = torch.as_tensor(next_states, device=device)
    dones = torch.as_tensor(dones, device=device)

    q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1).values
        target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = F.smooth_l1_loss(q_values, target_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10)
    optimizer.step()
    return loss.item()


def save_log(episode_log, log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_log[0].keys())
        writer.writeheader()
        writer.writerows(episode_log)


def save_plot(episode_log, plot_path, title):
    episodes = [row["episode"] for row in episode_log]
    rewards = [row["reward"] for row in episode_log]
    lengths = [row["episode_length"] for row in episode_log]
    total_steps = [row["total_steps"] for row in episode_log]

    window = min(10, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(episodes, rewards, alpha=0.4, label="reward")
    axes[0].plot(episodes[window - 1:], moving_avg, label=f"{window}-episode average")
    axes[0].set(title="Reward per episode", xlabel="Episode", ylabel="Reward")
    axes[0].legend()

    axes[1].plot(episodes, lengths)
    axes[1].set(title="Episode length", xlabel="Episode", ylabel="Steps in episode")

    axes[2].plot(total_steps, rewards)
    axes[2].set(title="Reward vs total steps", xlabel="Total steps", ylabel="Reward")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    args = parse_args()
    model_path, log_path, plot_path = get_output_paths(args)
    for path in (model_path, log_path, plot_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with settings: {vars(args)}")

    env = make_env()
    num_actions = env.action_space.n
    state_shape = env.observation_space.shape

    online_net = QNetwork(num_actions).to(device)
    target_net = QNetwork(num_actions).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)

    replay_buffer = ReplayBuffer(args.buffer_size, state_shape)

    episode_log = []
    episode_reward, episode_length, episode_losses = 0.0, 0, []
    state, _ = env.reset(seed=args.seed)

    for step in range(1, args.total_steps + 1):
        epsilon = get_epsilon(step, args.eps_start, args.eps_end, args.eps_decay)
        action = select_action(online_net, state, epsilon, num_actions, device)
        next_state, reward, terminated, truncated, _ = env.step(action)

        replay_buffer.add(state, action, reward, next_state, terminated)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if step > args.learning_starts and step % args.train_freq == 0:
            loss = train_step(online_net, target_net, optimizer, replay_buffer,
                              args.batch_size, args.gamma, device)
            episode_losses.append(loss)

        if step % args.target_update == 0:
            target_net.load_state_dict(online_net.state_dict())

        if terminated or truncated:
            episode_log.append({
                "episode": len(episode_log) + 1,
                "total_steps": step,
                "episode_length": episode_length,
                "reward": episode_reward,
                "epsilon": round(epsilon, 4),
                "avg_loss": round(float(np.mean(episode_losses)), 5) if episode_losses else 0.0,
            })
            print(f"Episode {len(episode_log):4d} | steps {step:7d} | length {episode_length:5d} | "
                  f"reward {episode_reward:5.1f} | epsilon {epsilon:.3f}", flush=True)
            episode_reward, episode_length, episode_losses = 0.0, 0, []
            state, _ = env.reset()

        if step % 50_000 == 0:
            torch.save(online_net.state_dict(), model_path)

    env.close()

    torch.save(online_net.state_dict(), model_path)
    with open(log_path.replace(".csv", "_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if episode_log:
        save_log(episode_log, log_path)
        save_plot(episode_log, plot_path, title=model_path)
        last_rewards = [row["reward"] for row in episode_log[-10:]]
        print(f"Average reward over the last {len(last_rewards)} episodes: {np.mean(last_rewards):.2f}")
    print(f"Saved model to {model_path}, log to {log_path}, plot to {plot_path}")


if __name__ == "__main__":
    main()
