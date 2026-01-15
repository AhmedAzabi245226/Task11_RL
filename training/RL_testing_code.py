from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ------------------------------------------------------------
# Path setup so "envs.*" works when running from /training
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Task11_RL/training
PROJECT_ROOT = THIS_DIR.parent                      # .../Task11_RL
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ot2_gym_wrapper import OT2GymEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Path to .zip model (relative to project root or absolute)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")

    # IMPORTANT: these MUST match training env/action space
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
    """
    Resolve model path robustly:
    - If user passes without .zip, add .zip
    - If relative, resolve relative to PROJECT_ROOT
    - Never create ".zip.zip"
    """
    p = Path(user_path)

    # Add ".zip" only if missing (prevents .zip.zip)
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")

    # Resolve relative paths relative to project root
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()

    return p


def bucket_error(dist: float) -> str:
    if dist < 0.001:
        return "<1mm"
    if dist < 0.005:
        return "1-5mm"
    if dist < 0.01:
        return "5-10mm"
    return ">=10mm"


def evaluate(model, env: OT2GymEnv, episodes: int) -> Tuple[Dict[str, int], float]:
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
    args = parse_args()

    # Import SB3 after path/env setup
    from stable_baselines3 import PPO  # noqa: E402

    model_path = resolve_model_path(args.model_path)

    print("CWD:", Path.cwd())
    print("[load] requested:", args.model_path)
    print("[load] resolved :", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

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

    print(
        "Env settings:",
        "thr=", args.success_threshold,
        "max_steps=", args.max_steps,
        "action_repeat=", args.action_repeat,
        "vel_max=", args.vel_max,
        "slowdown=", args.near_goal_slowdown,
    )

    model = PPO.load(str(model_path), env=env, device="cpu")

    buckets, avg_dist = evaluate(model, env, episodes=args.episodes)
    env.close()

    total = sum(buckets.values())
    print("\n==== EVALUATION RESULTS ====")
    print("episodes:", total)
    print("avg_final_distance_m:", avg_dist)
    for k, v in buckets.items():
        print(f"{k:>6}: {v}  ({(100.0 * v / max(1,total)):.1f}%)")

    # Simple “score” view based on your rubric
    # (This is not official scoring, just a quick summary.)
    score = 8 * buckets["<1mm"] + 6 * buckets["1-5mm"] + 4 * buckets["5-10mm"]
    print("rubric_points_sum_over_episodes:", score)
    print("============================\n")


if __name__ == "__main__":
    main()
