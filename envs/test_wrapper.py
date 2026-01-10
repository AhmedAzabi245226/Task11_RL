"""
test_wrapper.py (Task 11)

Runs the OT2GymEnv for 1000 steps using random actions.
This verifies the Gym wrapper works correctly (reset/step/termination).
"""

import os
import sys


# -------------------------------------------------
# Ensure project root is on sys.path so "envs" imports work
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.ot2_gym_wrapper import OT2GymEnv


TOTAL_STEPS = 1000


def main():
    # NOTE: render=False is safer for remote/server execution (no GUI needed)
    env = OT2GymEnv(render=False, max_steps=200, success_threshold=0.01, debug=False)

    obs, info = env.reset()
    print("Reset -> obs:", obs)
    print("Target:", info.get("target", None))

    episodes = 0
    successes = 0

    for step in range(TOTAL_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Print useful telemetry every 25 steps
        if step % 25 == 0:
            target = info.get("target", None)
            pip = info.get("pipette", None)
            dist = info.get("distance", None)
            print(
                f"step={step:04d} reward={reward:.4f} "
                f"pip={pip} target={target} dist={dist}"
            )

        if terminated or truncated:
            episodes += 1
            if info.get("is_success", False):
                successes += 1

            print(
                f"Episode done at step {step} "
                f"(terminated={terminated}, truncated={truncated}, success={info.get('is_success', False)})"
            )

            obs, info = env.reset()
            print("Reset -> new target:", info.get("target", None))

    env.close()
    print("\nFinished 1000-step random test.")
    print(f"Episodes completed: {episodes}")
    print(f"Successes: {successes}")


if __name__ == "__main__":
    main()
