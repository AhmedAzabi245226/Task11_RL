from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

# -------------------------------------------------
# Ensure project root is on sys.path so "envs" imports work
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ot2_gym_wrapper import OT2GymEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on OT2GymEnv")

    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SB3 .zip model (e.g., models/ot2_stage3.../ppo_ot2_final.zip or ppo_ot2_finalst3.zip)",
    )

    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--success_threshold", type=float, default=0.001)  # Stage 3 = 1mm
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--render", action="store_true", help="Render the simulation (slow)")
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (recommended for evaluation)",
    )

    # Optional: force a fixed target for repeatable demonstration
    p.add_argument("--fixed_target", action="store_true", help="Force a fixed target (if env supports env.target)")
    p.add_argument("--tx", type=float, default=0.22)
    p.add_argument("--ty", type=float, default=-0.15)
    p.add_argument("--tz", type=float, default=0.27)

    return p.parse_args()


def force_target_if_supported(env: OT2GymEnv, target: np.ndarray, obs: np.ndarray) -> None:
    """
    Your earlier script assumes env.target and env.prev_distance exist.
    We keep that behavior but protect it with checks so it won't crash
    if the env implementation differs.
    """
    if not hasattr(env, "target"):
        print("[warn] env has no attribute 'target'. Skipping fixed target.")
        return

    env.target = target

    # Re-sync prev_distance if present
    if hasattr(env, "prev_distance"):
        pipette = obs[0:3]
        env.prev_distance = float(np.linalg.norm(env.target - pipette))


def extract_distance_and_success(info: dict) -> Tuple[Optional[float], Optional[bool]]:
    dist = None
    succ = None
    if isinstance(info, dict):
        if "distance" in info:
            try:
                dist = float(info["distance"])
            except Exception:
                dist = None
        if "is_success" in info:
            try:
                succ = bool(info["is_success"])
            except Exception:
                succ = None
    return dist, succ


def main():
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print("\nEVAL MODE — NO TRAINING")
    print("Model:", model_path)
    print("Episodes:", args.episodes)
    print("Max steps:", args.max_steps)
    print("Success threshold:", args.success_threshold)
    print("Action repeat:", args.action_repeat)
    print("Deterministic:", bool(args.deterministic))
    print("Render:", bool(args.render))
    print("Fixed target:", bool(args.fixed_target))

    env = OT2GymEnv(
        render=args.render,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        seed=args.seed,
        debug=False,
        action_repeat=args.action_repeat,
    )

    model = PPO.load(str(model_path), env=env)

    fixed_target = np.array([args.tx, args.ty, args.tz], dtype=np.float32)

    successes = 0
    final_errors = []
    episode_lengths = []
    last_distances = []

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)

        if args.fixed_target:
            force_target_if_supported(env, fixed_target, obs)

        done = False
        step = 0
        last_info = {}

        while not done and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            step += 1
            last_info = info if isinstance(info, dict) else {}

        dist, succ = extract_distance_and_success(last_info)

        # If env reports success, use it. Otherwise compute final error against env.target if available.
        if succ is None and hasattr(env, "target"):
            try:
                pipette = obs[0:3]
                succ = bool(np.linalg.norm(pipette - np.array(env.target)) <= float(args.success_threshold))
            except Exception:
                succ = False

        if succ:
            successes += 1

        # Track final error (best effort)
        if dist is not None:
            final_errors.append(dist)
            last_distances.append(dist)
        else:
            # fallback compute error if possible
            if hasattr(env, "target"):
                try:
                    pipette = obs[0:3]
                    err = float(np.linalg.norm(pipette - np.array(env.target)))
                    final_errors.append(err)
                except Exception:
                    pass

        episode_lengths.append(step)

        if ep % max(1, args.episodes // 10) == 0:
            sr = successes / ep
            mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
            mean_err = float(np.mean(final_errors)) if final_errors else float("nan")
            print(f"[{ep:04d}/{args.episodes}] success_rate={sr:.3f} mean_len={mean_len:.1f} mean_final_err={mean_err:.6f}")

    success_rate = successes / max(1, args.episodes)
    mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    mean_err = float(np.mean(final_errors)) if final_errors else float("nan")
    median_err = float(np.median(final_errors)) if final_errors else float("nan")

    print("\n========== EVAL SUMMARY ==========")
    print("Episodes:", args.episodes)
    print("Success threshold:", args.success_threshold)
    print("Successes:", successes)
    print("Success rate:", success_rate)
    print("Mean ep length:", mean_len)
    print("Mean final error:", mean_err)
    print("Median final error:", median_err)
    print("=================================\n")

    env.close()


if __name__ == "__main__":
    main()
