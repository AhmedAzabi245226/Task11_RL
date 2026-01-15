from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


"""
This script evaluates a trained PPO model on the OT2GymEnv environment.

It:
- Loads a .zip model file.
- Runs a number of test episodes.
- Tracks the final distance to the goal each episode.
- Groups results into distance buckets (like <1mm, 1-5mm, etc.).
- Prints average final distance and bucket percentages.
"""


"""
Set up paths so imports like "envs.*" work even if we run this file from the /training folder.
"""
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ot2_gym_wrapper import OT2GymEnv  # noqa: E402


def parse_args():
    """Read command line arguments used for evaluation."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to .zip model (relative to project root or absolute)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")

    """These environment settings should match what was used during training."""
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--success_threshold", type=float, default=0.005)
    p.add_argument("--action_repeat", type=int, default=1)
    p.add_argument("--vel_max", type=float, default=0.3)
    p.add_argument(
        "--near_goal_slowdown",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "y", "t"),
        default=True,
    )
    return p.parse_args()


def resolve_model_path(user_path: str) -> Path:
    """Convert the user model path into a real absolute path to a .zip file."""
    p = Path(user_path)

    """If the user did not include .zip, add it once."""
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")

    """If the path is relative, make it relative to the project root."""
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    return p


def bucket_error(dist: float) -> str:
    """Return the bucket label for a given distance value."""
    if dist < 0.001:
        return "<1mm"
    if dist < 0.005:
        return "1-5mm"
    if dist < 0.01:
        return "5-10mm"
    return ">=10mm"


def evaluate(model, env: OT2GymEnv, episodes: int) -> Tuple[Dict[str, int], float]:
    """Run episodes, collect final distances, and return bucket counts plus average final distance."""
    buckets = {"<1mm": 0, "1-5mm": 0, "5-10mm": 0, ">=10mm": 0}
    final_dists = []

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

        dist = float(info.get("distance", np.nan))
        final_dists.append(dist)
        buckets[bucket_error(dist)] += 1

    avg_dist = float(np.nanmean(final_dists))
    return buckets, avg_dist


def main():
    """Load the model, create the environment, run evaluation, and print results."""
    args = parse_args()

    """Import Stable-Baselines3 after path setup so imports work correctly."""
    from stable_baselines3 import PPO  # noqa: E402

    model_path = resolve_model_path(args.model_path)

    print("CWD:", Path.cwd())
    print("[load] requested:", args.model_path)
    print("[load] resolved :", model_path)

    """Stop early if the model file does not exist."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    """Create the OT2 environment with the same settings used during training."""
    env = OT2GymEnv(
        render=args.render,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        seed=args.seed,
        debug=False,
        action_repeat=args.action_repeat,
        vel_max=args.vel_max,
        near_goal_slowdown=args.near_goal_slowdown,
    )

    """Print the environment settings being used for evaluation."""
    print(
        "Env settings:",
        "thr=", args.success_threshold,
        "max_steps=", args.max_steps,
        "action_repeat=", args.action_repeat,
        "vel_max=", args.vel_max,
        "slowdown=", args.near_goal_slowdown,
    )

    """Load the PPO model from disk and attach the environment."""
    model = PPO.load(str(model_path), env=env, device="cpu")

    """Run evaluation and close the environment after finishing."""
    buckets, avg_dist = evaluate(model, env, episodes=args.episodes)
    env.close()

    """Print the final summary results."""
    total = sum(buckets.values())
    print("\n EVALUATION RESULTS")
    print("episodes:", total)
    print("avg_final_distance_m:", avg_dist)
    for k, v in buckets.items():
        print(f"{k:>6}: {v}  ({(100.0 * v / max(1,total)):.1f}%)")

    """Compute a simple rubric-based score using the bucket counts."""
    score = 8 * buckets["<1mm"] + 6 * buckets["1-5mm"] + 4 * buckets["5-10mm"]
    print("rubric_points_sum_over_episodes:", score)
    


if __name__ == "__main__":
    main()
