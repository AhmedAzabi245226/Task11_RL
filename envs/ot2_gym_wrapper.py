"""
ot2_gym_wrapper.py (Task 11 – RL Controller)

Gymnasium-compatible environment wrapper for the Opentrons OT-2 PyBullet
digital twin (Task 9 sim_class.py).

Key design:
- Action: bounded Cartesian delta commands [dx, dy, dz, drop] (drop unused)
- Observation: pipette position, target position, and error vector
- Reward: progress shaping + distance shaping + success bonus
          + small action penalty + boundary penalty
- Termination: success when error < success_threshold
- Truncation: max_steps

Compatible with Stable Baselines 3.

Critical stability fix:
- close() aggressively disconnects PyBullet clients to prevent native memory creep
  leading to std::bad_alloc / exit code 139.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# PyBullet only used for cleanup safety
try:
    import pybullet as p  # type: ignore
except Exception:
    p = None


# -------------------------------------------------
# Paths
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # Task11_RL/
TASK9_PATH = (PROJECT_ROOT / "task09_robotics_environment").resolve()

if str(TASK9_PATH) not in sys.path:
    sys.path.insert(0, str(TASK9_PATH))

from sim_class import Simulation  # noqa: E402


class OT2GymEnv(gym.Env):
    """Gym wrapper around the OT-2 simulation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render: bool = False,
        max_steps: int = 200,
        seed: Optional[int] = None,
        success_threshold: float = 0.01,
        debug: bool = False,
        action_repeat: int = 1,
    ):
        super().__init__()

        # Validate Task 9 path AT RUNTIME (never at import time)
        if not TASK9_PATH.exists():
            raise RuntimeError(f"Task 9 folder not found at runtime: {TASK9_PATH}")

        self._old_cwd = os.getcwd()
        self._cwd_set = False
        self._set_cwd_for_assets()

        # Create simulation
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        # Action: dx, dy, dz, drop (drop unused)
        self.action_space = spaces.Box(
            low=np.array([-0.3, -0.3, -0.3, 0.0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: pipette_xyz (3) + target_xyz (3) + error_xyz (3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        # Workspace boundaries (meters)
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
        self.prev_distance: Optional[float] = None

    # ---------------------------
    # Helpers
    # ---------------------------
    def _set_cwd_for_assets(self) -> None:
        if not self._cwd_set:
            os.chdir(str(TASK9_PATH))
            self._cwd_set = True

    def _sample_target(self) -> np.ndarray:
        x = self.rng.uniform(self.X_MIN + self.margin, self.X_MAX - self.margin)
        y = self.rng.uniform(self.Y_MIN + self.margin, self.Y_MAX - self.margin)

        Z_SAFE_MIN = 0.1695
        z_low = max(self.Z_MIN + self.margin, Z_SAFE_MIN + self.margin)
        z_high = self.Z_MAX - self.margin
        z = self.rng.uniform(z_low, z_high)

        return np.array([x, y, z], dtype=np.float32)

    def _get_pipette(self) -> np.ndarray:
        states = self.sim.get_states()
        robot_state = next(iter(states.values()))
        return np.array(robot_state["pipette_position"], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        pipette = self._get_pipette()
        error = self.target - pipette
        return np.concatenate([pipette, self.target, error]).astype(np.float32)

    # ---------------------------
    # Gymnasium API
    # ---------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._set_cwd_for_assets()

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.sim.reset(num_agents=1)
        self.step_count = 0

        if options and options.get("target") is not None:
            self.target = np.array(options["target"], dtype=np.float32)
        else:
            self.target = self._sample_target()

        obs = self._get_obs()
        self.prev_distance = float(np.linalg.norm(self.target - obs[:3]))

        return obs, {
            "target": self.target.copy(),
            "pipette": obs[:3].copy(),
            "distance": self.prev_distance,
            "is_success": False,
        }

    def step(self, action: np.ndarray):
        self._set_cwd_for_assets()
        self.step_count += 1

        action = np.array(action, dtype=np.float32)
        action[:3] = np.clip(action[:3], -0.3, 0.3)
        action[3] = 0.0

        self.sim.run([action.tolist()], num_steps=self.action_repeat)

        obs = self._get_obs()
        pipette = obs[:3]
        distance = float(np.linalg.norm(self.target - pipette))

        progress = (self.prev_distance - distance) if self.prev_distance else 0.0
        reward = 10.0 * progress - 0.2 * distance - 0.01 * np.linalg.norm(action[:3])

        edge_pen = 0.0
        if not (self.X_MIN < pipette[0] < self.X_MAX):
            edge_pen += 1
        if not (self.Y_MIN < pipette[1] < self.Y_MAX):
            edge_pen += 1
        if not (self.Z_MIN < pipette[2] < self.Z_MAX):
            edge_pen += 1
        reward -= 0.5 * edge_pen

        self.prev_distance = distance
        success = distance < self.success_threshold
        if success:
            reward += 10.0

        terminated = success
        truncated = self.step_count >= self.max_steps

        return obs, float(reward), terminated, truncated, {
            "distance": distance,
            "is_success": success,
            "edge_pen": edge_pen,
        }

    def close(self):
        try:
            if hasattr(self, "sim"):
                self.sim.close()
        except Exception:
            pass

        try:
            if p is not None:
                p.disconnect()
        except Exception:
            pass

        try:
            os.chdir(self._old_cwd)
        except Exception:
            pass

            self._cwd_set = False
