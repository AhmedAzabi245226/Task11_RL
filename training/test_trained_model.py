import os
import sys
import numpy as np
from stable_baselines3 import PPO

# -------------------------------------------------
# Ensure project root is on sys.path so "envs" imports work
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.ot2_gym_wrapper import OT2GymEnv


def main():
    print("TEST MODE — NO TRAINING")

    env = OT2GymEnv(render=True, max_steps=400, success_threshold=0.03, debug=False)

    model_path = os.path.join(PROJECT_ROOT, "models", "ppo_ot2_50k.zip")
    print("Loading model from:", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = PPO.load(model_path, env=env)

    # Fixed test target (for PID comparison later)
    target = np.array([0.22, -0.15, 0.27], dtype=np.float32)

    # Reset first
    obs, info = env.reset()

    # Force target AFTER reset
    env.target = target

    # IMPORTANT: re-sync prev_distance to the forced target
    pipette = obs[0:3]
    env.prev_distance = float(np.linalg.norm(env.target - pipette))

    print("Target forced to:", target)

    done = False
    step = 0

    while not done and step < 400:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        # Print every 25 steps
        if step % 25 == 0:
            print(
                f"step={step:03d} reward={reward:.4f} "
                f"distance={info['distance']:.6f} success={info['is_success']}"
            )

    final_pipette = obs[0:3]
    final_error = float(np.linalg.norm(final_pipette - target))

    print("\nEpisode finished")
    print("Steps:", step)
    print("Final pipette position:", final_pipette)
    print("Final target:", target)
    print("Final error (m):", final_error)

    env.close()


if __name__ == "__main__":
    main()
