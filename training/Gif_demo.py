"""
rl_agent_gif_demo.py (Task 11 – RL Agent GIF Demo)

Loads a trained PPO model and demonstrates it reaching a few fixed targets.
Saves a GIF with an on-screen overlay showing target, pipette position, and error.

Run example:
python training/rl_agent_gif_demo.py --model_path training/ppo_ot2_finals4.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import imageio
import cv2

# ------------------------------------------------------------
# Path setup so "envs.*" works when running from /training
# ------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.ot2_gym_wrapper import OT2GymEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="training/ppo_ot2_finals4.zip")
    p.add_argument("--gif_name", type=str, default="rl_agent_demo.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--vel_max", type=float, default=0.3)
    p.add_argument("--action_repeat", type=int, default=1)
    p.add_argument("--near_goal_slowdown", type=str, default="true")
    return p.parse_args()


def resolve_model_path(user_path: str) -> Path:
    p = Path(user_path)
    if p.suffix.lower() != ".zip":
        p = p.with_suffix(".zip")
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


def overlay_text(frame, lines):
    img = np.array(frame)

    # RGBA -> RGB
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Ensure uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    img = img.copy()
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    dy = 12

    box_w = 360
    box_h = dy * (len(lines) + 1)

    x0 = max(0, (w - box_w) // 2)
    y1 = h - 3
    y0 = max(0, y1 - box_h)

    cv2.rectangle(img, (x0, y0), (x0 + box_w, y1), (0, 0, 0), thickness=-1)

    tx = x0 + 6
    ty = y0 + 14
    for i, t in enumerate(lines):
        y = ty + i * dy
        cv2.putText(img, t, (tx + 1, y + 1), font, scale, (0, 0, 0),
                    thickness + 1, cv2.LINE_8)
        cv2.putText(img, t, (tx, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_8)

    return img


def main():
    args = parse_args()
    near_goal_slowdown = args.near_goal_slowdown.strip().lower() in ("1", "true", "yes", "y", "t")

    from stable_baselines3 import PPO  # import after sys.path setup

    model_path = resolve_model_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Render must be True to show motion; rgb_array must be True if sim provides frames
    env = OT2GymEnv(
        render=True,
        max_steps=args.max_steps,
        success_threshold=0.005,          # demo threshold (does not affect model)
        seed=0,
        debug=False,
        action_repeat=args.action_repeat,
        vel_max=args.vel_max,
        near_goal_slowdown=near_goal_slowdown,
    )

    model = PPO.load(str(model_path), env=env, device="cpu")

    # Fixed demo targets (safe inside envelope)
    targets = [
        [0.00, 0.00, 0.27],
        [-0.15, -0.15, 0.20],
        [0.22, -0.15, 0.27],
        [-0.15, 0.20, 0.20],
        [0.22, 0.20, 0.20]
        ]

    frames = []

    for ti, target in enumerate(targets, 1):
        obs, info = env.reset(options={"target": target})
        terminated = truncated = False
        step = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            pip = info.get("pipette", [np.nan, np.nan, np.nan])
            dist = float(info.get("distance", np.nan))

            # Pull the latest rendered frame from the underlying sim (like Task 9/10)
            sim = getattr(env, "sim", None)
            frame = getattr(sim, "current_frame", None) if sim is not None else None
            if frame is not None:
                lines = [
                    f"Target {ti}/{len(targets)}: [{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}]",
                    f"Pip: [{pip[0]:+.3f},{pip[1]:+.3f},{pip[2]:+.3f}]",
                    f"Err(m): {dist:.6f}  step:{step}",
                ]
                frames.append(overlay_text(frame, lines))

            step += 1

    env.close()

    if not frames:
        raise RuntimeError(
            "No frames captured. If your sim does not provide env.sim.current_frame, "
            "enable rgb_array in Simulation inside OT2GymEnv or capture frames another way."
        )

    imageio.mimsave(args.gif_name, frames, fps=args.fps)
    print(f"GIF saved as {args.gif_name}")


if __name__ == "__main__":
    main()
