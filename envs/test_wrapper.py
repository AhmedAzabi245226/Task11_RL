"""
test_wrapper.py (Task 11)

This script performs a simple smoke test for OT2GymEnv.
It runs the environment for 1000 steps with random actions to confirm that:
- reset() returns a valid observation + info dict
- step() returns (obs, reward, terminated, truncated, info) in the correct format
- episodes end correctly (terminated or truncated) and the env can be reset again
"""

import os
import sys

# Add the project root to sys.path so "envs" can be imported when running this file directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from envs.ot2_gym_wrapper import OT2GymEnv


TOTAL_STEPS = 1000


def main():
    # render=False avoids opening a GUI (safer on servers / remote machines)
    env = OT2GymEnv(render=False, max_steps=200, success_threshold=0.01, debug=False)

    obs, info = env.reset()
    print("Reset -> obs:", obs)
    print("Target:", info.get("target", None))

    episodes = 0
    successes = 0

    for step in range(TOTAL_STEPS):
        # Random valid action from the environment action space
        action = env.action_space.sample()

        # Take one step in the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Print some values every 25 steps to confirm the env is updating normally
        if step % 25 == 0:
            target = info.get("target", None)
            pipette = info.get("pipette", None)
            dist = info.get("distance", None)
            print(
                f"step={step:04d} reward={reward:.4f} "
                f"pipette={pipette} target={target} dist={dist}"
            )

        # If the episode ends, count it and reset to start a new episode
        if terminated or truncated:
            episodes += 1
            if info.get("is_success", False):
                successes += 1

            print(
                f"Episode done at step {step} "
                f"(terminated={terminated}, truncated={truncated}, "
                f"success={info.get('is_success', False)})"
            )

            obs, info = env.reset()
            print("Reset -> new target:", info.get("target", None))

    env.close()
    print("\nFinished 1000-step random test.")
    print(f"Episodes completed: {episodes}")
    print(f"Successes: {successes}")


if __name__ == "__main__":
    main()
