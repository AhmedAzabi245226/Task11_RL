from __future__ import annotations

# =================================================
# ABSOLUTE TOP: only stdlib imports here
# =================================================
import os
import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Do not expose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")


# -------------------------------------------------
# Paths (do NOT import env/sim here)
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Task11_RL/training
PROJECT_ROOT = THIS_DIR.parent                      # .../Task11_RL
REQ_FILE = PROJECT_ROOT / "requirements.txt"


# =================================================
# ClearML helper
# =================================================
def _is_worker() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"))


def _pip_install(args: List[str]) -> None:
    print("[pip]", " ".join(args))
    subprocess.check_call(args)


def _restart_self() -> None:
    print("[restart] restarting process ...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _attach_or_get_task():
    from clearml import Task

    task = Task.current_task()
    if task is None:
        tid = os.environ.get("CLEARML_TASK_ID")
        if tid:
            task = Task.get_task(task_id=tid)
    return task


def _add_requirements_to_task(task) -> None:
    """
    ClearML 2.0.2 does NOT support set_packages_requirements().
    Use add_requirements() and keep it best-effort.
    """
    if not REQ_FILE.exists():
        print(f"[ClearML] requirements.txt not found at: {REQ_FILE} (skipping)")
        return

    try:
        # common signature
        task.add_requirements(str(REQ_FILE))
        print(f"[ClearML] Added requirements file: {REQ_FILE}")
        return
    except TypeError:
        # alternative signature in some versions
        try:
            task.add_requirements(requirements_file=str(REQ_FILE))
            print(f"[ClearML] Added requirements file: {REQ_FILE}")
            return
        except Exception as e:
            print("[ClearML] add_requirements failed:", repr(e))
    except Exception as e:
        print("[ClearML] add_requirements failed:", repr(e))


def clearml_setup_and_exit_if_local(args) -> Optional[str]:
    """
    If --use_clearml:
      - Local: create task -> enqueue -> EXIT (no env imports, no pybullet needed locally)
      - Worker: attach and continue
    If not --use_clearml:
      - Run locally (then you DO need pybullet locally)
    """
    if not args.use_clearml and not _is_worker():
        return None

    from clearml import Task

    if _is_worker():
        task = _attach_or_get_task()
        if task:
            task.connect(vars(args), name="cli_args")
            print("[ClearML] Worker attached:", task.id)
            return task.id
        return None

    # Local enqueue path
    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task.connect(vars(args), name="cli_args")
    task.set_base_docker(args.docker)

    # IMPORTANT: correct method for ClearML 2.0.2
    _add_requirements_to_task(task)

    print("[ClearML] Enqueuing to queue:", args.queue)
    task.execute_remotely(queue_name=args.queue)
    raise SystemExit(0)


# =================================================
# Runtime dependency enforcement (WORKER ONLY)
# =================================================
def ensure_cpu_torch_on_worker() -> None:
    """
    Fix libcupti.so.12 by forcing CPU torch inside the venv before SB3 import.
    """
    if not _is_worker():
        return

    need_install = False
    try:
        import torch as _t
        if _t.version.cuda is not None:
            need_install = True
    except Exception as e:
        # common when CUDA torch exists but CUDA libs are missing
        print("[torch-check] torch import failed:", repr(e))
        need_install = True

    if not need_install:
        return

    print("[FIX] Installing CPU-only torch to avoid CUDA/CUPTI issues...")
    _pip_install([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--force-reinstall", "--upgrade",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.4.1+cpu",
    ])
    _restart_self()


def ensure_pybullet_on_worker() -> None:
    """
    Ensure pybullet + pybullet_data exist on the worker.
    """
    if not _is_worker():
        return

    try:
        import pybullet  # noqa
        import pybullet_data  # noqa
        return
    except Exception as e:
        print("[pybullet-check] missing:", repr(e))

    print("[FIX] Installing pybullet on worker...")
    _pip_install([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--upgrade",
        "pybullet==3.2.6",
    ])
    _restart_self()


# =================================================
# Args
# =================================================
def parse_args():
    p = argparse.ArgumentParser()

    # ClearML
    p.add_argument("--use_clearml", action="store_true")
    p.add_argument("--project_name", type=str, default="Mentor Group - Alican/Group 1")
    p.add_argument("--task_name", type=str, default="OT2_PPO_Train")
    p.add_argument("--queue", type=str, default="default")
    p.add_argument("--docker", type=str, default="deanis/2023y2b-rl:latest")

    # Run
    p.add_argument("--run_name", type=str, default="")
    p.add_argument("--seed", type=int, default=0)

    # Env
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--success_threshold", type=float, default=0.03)
    p.add_argument("--action_repeat", type=int, default=5)
    p.add_argument("--render", action="store_true")

    # PPO
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_steps", type=int, default=512)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)

    # Timesteps
    p.add_argument("--total_timesteps", type=int, default=800_000)

    return p.parse_args()


# =================================================
# Main
# =================================================
def main():
    args = parse_args()

    # Ensure project root import works (but do NOT import env/sb3 yet)
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 1) If local + --use_clearml: enqueue and exit BEFORE importing env/sb3
    clearml_setup_and_exit_if_local(args)

    # 2) Worker only: force CPU torch (prevents libcupti crash)
    ensure_cpu_torch_on_worker()

    # 3) Worker only: ensure pybullet exists
    ensure_pybullet_on_worker()

    # 4) Now safe to import heavy deps
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import BaseCallback

    from envs.ot2_gym_wrapper import OT2GymEnv

    class ClearMLScalarCallback(BaseCallback):
        def __init__(self, report_every: int):
            super().__init__()
            self.report_every = int(report_every)
            self._logger = None

        def _on_training_start(self) -> None:
            try:
                from clearml import Task
                t = Task.current_task()
                self._logger = t.get_logger() if t else None
            except Exception:
                self._logger = None

        def _on_step(self) -> bool:
            if self._logger and (self.num_timesteps % self.report_every == 0):
                self._logger.report_scalar("time", "timesteps", self.num_timesteps, iteration=self.num_timesteps)
            return True

    def make_env():
        def _make():
            env = OT2GymEnv(
                render=args.render,
                max_steps=args.max_steps,
                success_threshold=args.success_threshold,
                action_repeat=args.action_repeat,
                seed=args.seed,
            )
            return Monitor(env)

        return VecMonitor(DummyVecEnv([_make]))

    if not args.run_name:
        args.run_name = f"ot2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_dir = PROJECT_ROOT / "models" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        device="cpu",  # hard force CPU
    )

    cb = ClearMLScalarCallback(report_every=args.n_steps)
    model.learn(total_timesteps=int(args.total_timesteps), callback=cb)

    final_path = model_dir / "ppo_ot2_final"
    model.save(final_path)
    env.close()

    print("Training finished successfully.")
    print("Saved to:", final_path)


if __name__ == "__main__":
    main()
