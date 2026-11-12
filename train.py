"""Training entry point: python train.py [--total-steps N] [--num-envs N] ...

Do not import torch (or dqn_car_racing) at module level here: on Windows the
AsyncVectorEnv workers are spawned and re-import this file, and loading torch
with CUDA in every worker exhausts the paging file (WinError 1455).
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", "--num_envs", type=int, default=8)
    parser.add_argument("--total-steps", "--total_steps", type=int, default=1_000_000)
    parser.add_argument("--buffer-size", "--buffer_size", type=int, default=300_000)
    parser.add_argument("--updates-per-step", "--updates_per_step", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", "--no_render", action="store_true")
    args = parser.parse_args()

    from dqn_car_racing import train, render, device

    print("device:", device)
    train(num_envs=args.num_envs, total_steps=args.total_steps,
          buffer_size=args.buffer_size, updates_per_step=args.updates_per_step,
          seed=args.seed)
    if not args.no_render:
        render()


if __name__ == "__main__":
    main()
