"""
ot2_gym_wrapper.py (Task 11 – RL Controller)

Gymnasium-compatible environment wrapper for the Opentrons OT-2 PyBullet digital twin
(Task 9 sim_class.py).

Key design:
- Action: pipette velocity commands [vx, vy, vz, drop] (drop unused)
- Observation: pipette position, target position, and error vector (9D)
- Reward: progress shaping + distance shaping + success bonus + small action penalty + boundary penalty
- Termination: success when error < success_threshold
- Truncation: max_steps

Compatible with Stable Baselines 3.

Critical stability fix:
- close() aggressively disconnects PyBullet clients to prevent native memory creep.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# IMPORTANT: pybullet is used ONLY for cleanup safety
try:
    import pybullet as p  # type: ignore
except Exception:
    p = None  # close() will handle


# -------------------------------------------------
# Task 9 path setup (portable: Task 9 is inside this repo)
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

if str(TASK9_PATH) not in sys.path:
    sys.path.insert(0, str(TASK9_PATH))

from sim_class import Simulation  # noqa: E402


class OT2GymEnv(gym.Env):
    """Gym wrapper around the OT-2 simulation."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        render: bool = False,
        max_steps: int = 400,
        seed: Optional[int] = None,
        success_threshold: float = 0.01,
        debug: bool = False,
        action_repeat: int = 1,
        vel_max: float = 0.08,
        near_goal_slowdown: bool = True,
    ):
        super().__init__()

        self._old_cwd = os.getcwd()
        self._cwd_set = False
        self._set_cwd_for_assets()

        # Create simulation (PyBullet lives inside Simulation)
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

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
        self.vel_max = float(vel_max)
        self.near_goal_slowdown = bool(near_goal_slowdown)

        # Action: vx, vy, vz, drop (drop unused)
        self.action_space = spaces.Box(
            low=np.array([-self.vel_max, -self.vel_max, -self.vel_max, 0.0], dtype=np.float32),
            high=np.array([ self.vel_max,  self.vel_max,  self.vel_max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: pipette_xyz (3) + target_xyz (3) + error_xyz (3) = 9
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(3, dtype=np.float32)
        self.prev_distance: Optional[float] = None

        if self.debug:
            print(
                "ENV INIT:",
                "success_threshold=", self.success_threshold,
                "max_steps=", self.max_steps,
                "action_repeat=", self.action_repeat,
                "vel_max=", self.vel_max,
                "near_goal_slowdown=", self.near_goal_slowdown,
            )

    # ---------------------------
    # Helpers
    # ---------------------------
    def _set_cwd_for_assets(self) -> None:
        """Ensure CWD is Task 9 folder so relative assets resolve correctly."""
        if not self._cwd_set:
            os.chdir(str(TASK9_PATH))
            self._cwd_set = True

    def _sample_target(self) -> np.ndarray:
        """Sample a random reachable target within a safe subset of the workspace."""
        x = self.rng.uniform(self.X_MIN + self.margin, self.X_MAX - self.margin)
        y = self.rng.uniform(self.Y_MIN + self.margin, self.Y_MAX - self.margin)

        # Conservative global Z floor
        Z_SAFE_MIN = 0.1695
        z_low = max(self.Z_MIN + self.margin, Z_SAFE_MIN + self.margin)
        z_high = self.Z_MAX - self.margin
        z = self.rng.uniform(z_low, z_high)

        return np.array([x, y, z], dtype=np.float32)

    def _get_pipette(self) -> np.ndarray:
        """Read pipette XYZ from sim."""
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

        if options and "target" in options and options["target"] is not None:
            self.target = np.array(options["target"], dtype=np.float32)
        else:
            self.target = self._sample_target()

        obs = self._get_obs()
        pipette = obs[0:3]
        self.prev_distance = float(np.linalg.norm(self.target - pipette))

        info = {
            "target": self.target.copy(),
            "pipette": pipette.copy(),
            "distance": float(self.prev_distance),
            "is_success": False,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._set_cwd_for_assets()
        self.step_count += 1

        action = np.array(action, dtype=np.float32)
        action[0:3] = np.clip(action[0:3], -self.vel_max, self.vel_max)
        action[3] = 0.0  # drop unused

        # Optional near-goal slowdown to help fine precision
        if self.near_goal_slowdown and (self.prev_distance is not None):
            if self.prev_distance < 0.02:  # inside 2 cm
                scale = float(np.clip(self.prev_distance / 0.02, 0.15, 1.0))
                action[0:3] *= scale

        # Apply action for N sim steps (action repeat)
        self.sim.run([action.tolist()], num_steps=self.action_repeat)

        obs = self._get_obs()
        pipette = obs[0:3]
        error_vec = self.target - pipette
        distance = float(np.linalg.norm(error_vec))

        # Reward shaping
        progress = (self.prev_distance - distance) if self.prev_distance is not None else 0.0
        reward = 10.0 * float(progress)
        reward -= 0.2 * distance
        reward -= 0.01 * float(np.linalg.norm(action[0:3]))

        # Boundary penalty
        x, y, z = float(pipette[0]), float(pipette[1]), float(pipette[2])
        edge_margin = 0.01
        edge_pen = 0.0
        if x < self.X_MIN + edge_margin or x > self.X_MAX - edge_margin:
            edge_pen += 1.0
        if y < self.Y_MIN + edge_margin or y > self.Y_MAX - edge_margin:
            edge_pen += 1.0
        if z < self.Z_MIN + edge_margin or z > self.Z_MAX - edge_margin:
            edge_pen += 1.0
        reward -= 0.5 * edge_pen

        self.prev_distance = distance

        success = distance < self.success_threshold
        if success:
            reward += 10.0

        terminated = success
        truncated = self.step_count >= self.max_steps

        info = {
            "target": self.target.copy(),
            "pipette": pipette.copy(),
            "distance": distance,
            "is_success": success,
            "edge_pen": edge_pen,
            "action_repeat": self.action_repeat,
            "vel_max": self.vel_max,
            "near_goal_slowdown": self.near_goal_slowdown,
        }

        if self.debug and (self.step_count % 50 == 0 or success):
            print(
                "DEBUG step", self.step_count,
                "distance", distance,
                "success", success,
                "reward", float(reward),
                "edge_pen", edge_pen,
            )

        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        """
        Critical: aggressively release PyBullet native memory.
        """
        try:
            if hasattr(self, "sim") and self.sim is not None:
                try:
                    self.sim.close()
                except Exception:
                    pass
        finally:
            try:
                if p is not None:
                    for cid in range(0, 32):
                        try:
                            info = p.getConnectionInfo(physicsClientId=cid)
                            if info and info.get("isConnected", 0) == 1:
                                p.disconnect(physicsClientId=cid)
                        except Exception:
                            pass
                    try:
                        p.disconnect()
                    except Exception:
                        pass
            except Exception:
                pass

        try:
            if getattr(self, "_old_cwd", None) is not None:
                os.chdir(self._old_cwd)
        except Exception:
            pass

        self._cwd_set = False
