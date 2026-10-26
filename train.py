import argparse
import os
import random
import numpy as np
import torch
from env.ev_braking_env import EVBrakingEnv
from agents.sac import SACAgent, ReplayBuffer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_seed(
    seed: int = 42,
    max_steps: int = 15000,
    batch_size: int = 256,
    start_steps: int = 1000,
    w_viol: float = 14.0,
    save_prefix: str = "sac_seed",
    log_file: str = "training_log.txt",
):
    set_seed(seed)
    env = EVBrakingEnv(mode="mixed", num_segments=4, w_viol=w_viol, seed=seed)

    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if observation_shape is None or action_shape is None:
        raise ValueError("Observation and action spaces must have defined shapes")
    state_dim = observation_shape[0]
    action_dim = action_shape[0]

    agent = SACAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

    state, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_count = 0

    log_entries = []
    print(f"\n[Seed {seed} | w_viol={w_viol}] Starting SAC Training on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("-" * 55)

    for step in range(max_steps):
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        dead = 1.0 if terminated else 0.0

        replay_buffer.add(state, action, reward, next_state, dead)

        state = next_state
        episode_reward += reward
        episode_steps += 1

        if step >= start_steps:
            agent.train(replay_buffer, batch_size)

        if done:
            episode_count += 1
            entry = f"Step: {step+1}/{max_steps} | Episode: {episode_count} | Reward: {episode_reward:.2f} | Length: {episode_steps}"
            log_entries.append(entry)
            if episode_count % 10 == 0 or episode_count == 1:
                print(f"[Seed {seed}] {entry}")
            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0

    os.makedirs("saved_models", exist_ok=True)
    save_path = f"saved_models/{save_prefix}_{seed}.pt"
    torch.save(agent.actor.state_dict(), save_path)
    print(f"[Seed {seed}] Model saved to {save_path}")

    # If default seed 42, also save standard sac_regen.pt for backward compatibility
    if seed == 42 and w_viol == 14.0:
        torch.save(agent.actor.state_dict(), "saved_models/sac_regen.pt")

    if log_file:
        mode = "a" if os.path.exists(log_file) else "w"
        with open(log_file, mode, encoding="utf-8") as f:
            f.write(f"\n--- Seed {seed} (w_viol={w_viol}) ---\n")
            for line in log_entries:
                f.write(line + "\n")

    return agent, log_entries


def main():
    parser = argparse.ArgumentParser(description="Train SAC on EVBrakingEnv with multi-seed and ablation support")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="List of seeds to train")
    parser.add_argument("--max_steps", type=int, default=15000, help="Total environment steps per seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--start_steps", type=int, default=1000, help="Random exploration steps")
    parser.add_argument("--w_viol", type=float, default=14.0, help="Penalty weight for stopping violations")
    parser.add_argument("--ablation", action="store_true", help="Run reward weight ablation (w_viol in [5, 14, 30])")
    parser.add_argument("--append", action="store_true", help="Append to training_log.txt instead of overwriting")
    args = parser.parse_args()

    if args.ablation:
        print("=== Running Reward-Weight Ablation (w_viol in [5.0, 14.0, 30.0]) ===")
        for w in [5.0, 14.0, 30.0]:
            train_single_seed(
                seed=42,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                start_steps=args.start_steps,
                w_viol=w,
                save_prefix=f"sac_wviol_{int(w)}",
                log_file="ablation_log.txt",
            )
    else:
        print(f"=== Multi-Seed SAC Training across {len(args.seeds)} seeds: {args.seeds} ===")
        if not args.append:
            with open("training_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Multi-Seed Training Log: Seeds {args.seeds}\n")

        for s in args.seeds:
            train_single_seed(
                seed=s,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                start_steps=args.start_steps,
                w_viol=args.w_viol,
                save_prefix="sac_seed",
                log_file="training_log.txt",
            )
        print("\nAll seeds trained successfully!")


if __name__ == "__main__":
    main()