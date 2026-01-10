"""
ot2_gym_wrapper.py (Task 11 – RL Controller)

Gymnasium-compatible environment wrapper for the Opentrons OT-2 PyBullet
digital twin (Task 9 sim_class.py).

Key design:
- Action: pipette velocity commands [vx, vy, vz, drop] (drop unused)
- Observation: pipette position, target position, and error vector
- Reward: progress-based shaping + distance shaping + success bonus
          + small action penalty + boundary penalty
- Termination: success when error < success_threshold, truncation at max_steps

This wrapper is compatible with Stable Baselines 3.
"""

import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# -------------------------------------------------
# Task 9 path setup (portable: Task 9 is inside Task11_RL)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../Task11_RL
TASK9_PATH = (PROJECT_ROOT / "task09_robotics_environment").resolve()

if not TASK9_PATH.exists():
    raise FileNotFoundError(
        f"Task 9 folder not found at: {TASK9_PATH}\n"
        "Expected layout:\n"
        "  Task11_RL/\n"
        "    envs/ot2_gym_wrapper.py\n"
        "    task09_robotics_environment/\n"
        "      sim_class.py, URDFs, meshes/, textures/\n"
    )

# Make Task 9 importable (for sim_class.py)
if str(TASK9_PATH) not in sys.path:
    sys.path.append(str(TASK9_PATH))

from sim_class import Simulation


class OT2GymEnv(gym.Env):
    """Gym wrapper around the OT-2 simulation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render: bool = False,
        max_steps: int = 200,
        seed=None,
        success_threshold: float = 0.01,
        debug: bool = False,
        action_repeat: int = 1,   # optional: repeat each action N sim steps
    ):
        super().__init__()

        # Task 9 uses relative asset paths -> ensure cwd points to Task 9 during sim lifetime
        self._old_cwd = os.getcwd()
        self._cwd_set = False
        self._set_cwd_for_assets()

        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action: vx, vy, vz, drop (drop unused but kept for compatibility)
        self.action_space = spaces.Box(
            low=np.array([-0.3, -0.3, -0.3, 0.0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: pipette_xyz (3) + target_xyz (3) + error_xyz (3) = 9
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        # Workspace boundaries from Task 9 (meters)
        self.X_MIN, self.X_MAX = -0.187, 0.253
        self.Y_MIN, self.Y_MAX = -0.1706, 0.2196
        self.Z_MIN, self.Z_MAX = 0.1195, 0.2896

        self.margin = 0.02
        self.max_steps = int(max_steps)
        self.step_count = 0

        self.success_threshold = float(success_threshold)
        self.debug = bool(debug)

        self.action_repeat = max(1, int(action_repeat))

        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(3, dtype=np.float32)

        # For progress reward shaping
        self.prev_distance = None

        if self.debug:
            print("ENV INIT success_threshold =", self.success_threshold)

    def _set_cwd_for_assets(self):
        """Ensure CWD is Task 9 folder so relative assets (textures, URDF, etc.) resolve correctly."""
        if not self._cwd_set:
            os.chdir(str(TASK9_PATH))
            self._cwd_set = True

    def _sample_target(self):
        """Sample a random reachable target within a safe subset of the workspace."""
        x = self.rng.uniform(self.X_MIN + self.margin, self.X_MAX - self.margin)
        y = self.rng.uniform(self.Y_MIN + self.margin, self.Y_MAX - self.margin)

        # Conservative global Z floor (highest observed z_min across corners)
        Z_SAFE_MIN = 0.1695
        z_low = max(self.Z_MIN + self.margin, Z_SAFE_MIN + self.margin)
        z_high = self.Z_MAX - self.margin
        z = self.rng.uniform(z_low, z_high)

        return np.array([x, y, z], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Ensure assets still resolve even if caller changed cwd
        self._set_cwd_for_assets()

        # Gymnasium seeding convention
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.sim.reset(num_agents=1)
        self.step_count = 0

        # New random target each episode
        self.target = self._sample_target()

        # Compute obs ONCE, then derive prev_distance from it
        obs = self._get_obs()
        pipette = obs[0:3]
        self.prev_distance = float(np.linalg.norm(self.target - pipette))

        info = {"target": self.target.copy()}
        return obs, info

    def step(self, action):
        # Ensure assets still resolve even if caller changed cwd
        self._set_cwd_for_assets()

        self.step_count += 1

        action = np.array(action, dtype=np.float32)
        action[0:3] = np.clip(action[0:3], -0.3, 0.3)
        action[3] = 0.0  # drop not used

        # Apply action for N sim steps (action repeat)
        self.sim.run([action.tolist()], num_steps=self.action_repeat)

        obs = self._get_obs()
        pipette = obs[0:3]
        error_vec = self.target - pipette
        distance = float(np.linalg.norm(error_vec))

        # -------------------------
        # Reward shaping
        # -------------------------
        # Progress reward: positive when moving closer
        progress = float(self.prev_distance - distance) if self.prev_distance is not None else 0.0
        reward = 10.0 * progress

        # Dense distance shaping
        reward -= 0.2 * distance

        # Small action penalty
        reward -= 0.01 * float(np.linalg.norm(action[0:3]))

        # Boundary penalty: discourage saturating at workspace edges/corners
        x, y, z = float(pipette[0]), float(pipette[1]), float(pipette[2])
        edge_margin = 0.01  # 1 cm band near edges
        edge_pen = 0.0

        if x < self.X_MIN + edge_margin or x > self.X_MAX - edge_margin:
            edge_pen += 1.0
        if y < self.Y_MIN + edge_margin or y > self.Y_MAX - edge_margin:
            edge_pen += 1.0
        if z < self.Z_MIN + edge_margin or z > self.Z_MAX - edge_margin:
            edge_pen += 1.0

        reward -= 0.5 * edge_pen

        # Update for next step
        self.prev_distance = distance

        success = distance < self.success_threshold

        # Success bonus
        if success:
            reward += 10.0

        if self.debug and (self.step_count % 50 == 0 or success):
            print(
                "DEBUG step", self.step_count,
                "distance", distance,
                "threshold", self.success_threshold,
                "success", success,
                "reward", reward,
                "edge_pen", edge_pen
            )

        terminated = success
        truncated = self.step_count >= self.max_steps

        info = {
            "target": self.target.copy(),
            "pipette": pipette.copy(),
            "distance": distance,
            "is_success": success,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        states = self.sim.get_states()
        robot_state = next(iter(states.values()))

        pipette = np.array(robot_state["pipette_position"], dtype=np.float32)
        error = self.target - pipette

        obs = np.concatenate([pipette, self.target, error]).astype(np.float32)
        return obs

    def render(self):
        return None

    def close(self):
        try:
            self.sim.close()
        finally:
            # Restore original working directory even if sim.close() errors
            if self._old_cwd is not None:
                os.chdir(self._old_cwd)
            self._cwd_set = False
