from __future__ import annotations

# =================================================
# CRITICAL FIX — MUST BE AT VERY TOP (before SB3)
# =================================================
import os
import sys
import subprocess

# Disable GPU visibility immediately
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# If running on ClearML worker, force CPU-only torch
if "CLEARML_TASK_ID" in os.environ or "CLEARML_WORKER_ID" in os.environ:
    try:
        import torch  # noqa
        if torch.version.cuda is not None:
            print("[FIX] CUDA torch detected. Reinstalling CPU-only torch...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "torch==2.4.1+cpu",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu"
            ])
            print("[FIX] CPU torch installed. Restarting process...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
    except Exception:
        pass

# =================================================
# NORMAL IMPORTS (SAFE NOW)
# =================================================
import argparse
import shutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# -------------------------------------------------
# Force correct project root
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ot2_gym_wrapper import OT2GymEnv

# =================================================
# ARGS
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
    p.add_argument("--checkpoint_freq", type=int, default=102_400)

    return p.parse_args()


# =================================================
# ClearML helpers
# =================================================
def _is_worker():
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"))


def clearml_setup(args):
    if not args.use_clearml and not _is_worker():
        return None

    from clearml import Task

    if _is_worker():
        task = Task.current_task()
        task.connect(vars(args), name="cli_args")
        print("[ClearML] Worker attached:", task.id)
        return task.id

    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task.connect(vars(args), name="cli_args")
    task.set_base_docker(args.docker)
    task.execute_remotely(queue_name=args.queue)
    raise SystemExit(0)


# =================================================
# Callbacks
# =================================================
class ClearMLScalarCallback(BaseCallback):
    def __init__(self, report_every=2048):
        super().__init__()
        self.report_every = report_every
        self.logger = None

    def _on_training_start(self):
        try:
            from clearml import Task
            self.logger = Task.current_task().get_logger()
        except Exception:
            self.logger = None

    def _on_step(self):
        if self.logger and self.num_timesteps % self.report_every == 0:
            self.logger.report_scalar("time", "timesteps", self.num_timesteps, self.num_timesteps)
        return True


# =================================================
# VecEnv
# =================================================
def make_env(args):
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


# =================================================
# MAIN
# =================================================
def main():
    args = parse_args()
    clearml_setup(args)

    if not args.run_name:
        args.run_name = f"ot2_stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_dir = PROJECT_ROOT / "models" / args.run_name
    model_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)

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
        device="cpu",   # EXTRA SAFETY
    )

    cb = ClearMLScalarCallback(args.n_steps)

    model.learn(total_timesteps=args.total_timesteps, callback=cb)

    final_path = model_dir / "ppo_ot2_final"
    model.save(final_path)
    env.close()

    print("Training finished successfully.")
    print("Saved to:", final_path)


if __name__ == "__main__":
    main()
