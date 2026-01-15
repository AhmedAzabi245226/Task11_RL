"""
ot2_gym_wrapper.py (Task 11 – RL Controller)

Gymnasium-compatible environment wrapper for the Opentrons OT-2 PyBullet digital twin
(Task 9 sim_class.py).

Important note for resuming SB3 PPO models:
- The action_space must NOT change when loading an old checkpoint.
- Stage 3 used: Box([-0.3,-0.3,-0.3,0], [0.3,0.3,0.3,1])
- Keep those limits fixed and scale internally using vel_max.

Design:
- Action: [ax, ay, az, drop]
  - ax, ay, az are always in [-0.3, 0.3] (fixed policy range)
  - Executed velocity = (a / 0.3) * vel_max
- Observation: pipette position (3) + target position (3) + error vector (3) = 9
- Reward: progress + distance shaping + success bonus + action penalty + boundary penalty
- Done (terminated): distance < success_threshold
- Truncated: max_steps reached
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Used only for safe disconnect/cleanup
try:
    import pybullet as p  # type: ignore
except Exception:
    p = None


# Task 9 path setup
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
    metadata = {"render_modes": ["human"]}

    # Fixed action bound (do not change if you want to resume older PPO models)
    ACTION_MAX = 0.3

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

        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)

        self.max_steps = int(max_steps)
        self.step_count = 0
        self.success_threshold = float(success_threshold)
        self.debug = bool(debug)
        self.action_repeat = max(1, int(action_repeat))

        # vel_max controls the real speed (precision stages use smaller vel_max)
        self.vel_max = float(vel_max)
        self.near_goal_slowdown = bool(near_goal_slowdown)

        a = self.ACTION_MAX
        self.action_space = spaces.Box(
            low=np.array([-a, -a, -a, 0.0], dtype=np.float32),
            high=np.array([a, a, a, 1.0], dtype=np.float32),
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

        self.rng = np.random.default_rng(seed)
        self.target = np.zeros(3, dtype=np.float32)
        self.prev_distance: Optional[float] = None

        if self.debug:
            print(
                "ENV INIT:",
                "thr=", self.success_threshold,
                "max_steps=", self.max_steps,
                "action_repeat=", self.action_repeat,
                "ACTION_MAX=", self.ACTION_MAX,
                "vel_max=", self.vel_max,
                "slowdown=", self.near_goal_slowdown,
            )

    def _set_cwd_for_assets(self) -> None:
        """Switches cwd to Task 9 so URDF/mesh paths resolve correctly."""
        if not self._cwd_set:
            os.chdir(str(TASK9_PATH))
            self._cwd_set = True

    def _sample_target(self) -> np.ndarray:
        """Samples a safe random target inside the robot workspace."""
        x = self.rng.uniform(self.X_MIN + self.margin, self.X_MAX - self.margin)
        y = self.rng.uniform(self.Y_MIN + self.margin, self.Y_MAX - self.margin)

        Z_SAFE_MIN = 0.1695
        z_low = max(self.Z_MIN + self.margin, Z_SAFE_MIN + self.margin)
        z_high = self.Z_MAX - self.margin
        z = self.rng.uniform(z_low, z_high)

        return np.array([x, y, z], dtype=np.float32)

    def _get_pipette(self) -> np.ndarray:
        """Reads the current pipette XYZ from the simulator state."""
        states = self.sim.get_states()
        robot_state = next(iter(states.values()))
        return np.array(robot_state["pipette_position"], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Builds the 9D observation: pipette, target, and target error."""
        pipette = self._get_pipette()
        error = self.target - pipette
        return np.concatenate([pipette, self.target, error]).astype(np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets simulation and chooses a new target location."""
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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Runs one environment step:
        - Clips the policy action to fixed bounds
        - Scales it to a real velocity using vel_max
        - Steps the simulator, then computes reward and done flags
        """
        self._set_cwd_for_assets()
        self.step_count += 1

        action = np.array(action, dtype=np.float32)

        a = self.ACTION_MAX
        action[0:3] = np.clip(action[0:3], -a, a)
        action[3] = 0.0  # drop unused

        vel = (action[0:3] / a) * self.vel_max

        if self.near_goal_slowdown and (self.prev_distance is not None) and self.prev_distance < 0.02:
            scale = float(np.clip(self.prev_distance / 0.02, 0.15, 1.0))
            vel *= scale

        cmd = np.array([vel[0], vel[1], vel[2], 0.0], dtype=np.float32)

        for _ in range(self.action_repeat):
            self.sim.run([cmd.tolist()], num_steps=1)

        obs = self._get_obs()
        pipette = obs[0:3]
        distance = float(np.linalg.norm(self.target - pipette))

        progress = (self.prev_distance - distance) if self.prev_distance is not None else 0.0
        reward = 10.0 * float(progress)
        reward -= 0.2 * distance
        reward -= 0.01 * float(np.linalg.norm(vel))

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
            "ACTION_MAX": self.ACTION_MAX,
        }

        return obs, float(reward), terminated, truncated, info

    def render(self):
        """Rendering is handled by the simulator when render=True."""
        return None

    def close(self):
        """Closes the simulator and safely disconnects any PyBullet connections."""
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
