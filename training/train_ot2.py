from __future__ import annotations

# =================================================
# ABSOLUTE FIRST LINES — NO CUDA EVER
# =================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# =================================================
# STANDARD IMPORTS
# =================================================
import sys
import argparse
from datetime import datetime
from pathlib import Path

# -------------------------------------------------
# Force correct project root
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
os.chdir(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

# =================================================
# THIRD-PARTY (SAFE NOW)
# =================================================
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from envs.ot2_gym_wrapper import OT2GymEnv

# =================================================
# ARGUMENTS
# =================================================
def parse_args():
    p = argparse.ArgumentParser()

    # ClearML
    p.add_argument("--use_clearml", action="store_true")
    p.add_argument("--project_name", type=str, default="Mentor Group - Alican/Group 1")
    p.add_argument("--task_name", type=str, default="OT2_STAGE1_CPU")
    p.add_argument("--queue", type=str, default="default")

    # 🚨 CRITICAL FIX — CPU-ONLY CONTAINER
    p.add_argument("--docker", type=str, default="python:3.10-slim")

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
# CLEARML SETUP
# =================================================
def _is_worker():
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"))


def clearml_setup(args):
    if not args.use_clearml and not _is_worker():
        return

    from clearml import Task

    if _is_worker():
        task = Task.current_task()
        task.connect(vars(args), name="cli_args")
        print("[ClearML] Worker attached:", task.id)
        return

    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task.connect(vars(args), name="cli_args")
    task.set_base_docker(args.docker)

    print("[ClearML] Enqueuing to queue:", args.queue)
    task.execute_remotely(queue_name=args.queue)
    raise SystemExit(0)


# =================================================
# CALLBACK
# =================================================
class ClearMLScalarCallback(BaseCallback):
    def __init__(self, report_every: int):
        super().__init__()
        self.report_every = report_every
        self.logger = None

    def _on_training_start(self):
        try:
            from clearml import Task
            self.logger = Task.current_task().get_logger()
        except Exception:
            self.logger = None

    def _on_step(self) -> bool:
        if self.logger and self.num_timesteps % self.report_every == 0:
            self.logger.report_scalar(
                title="time",
                series="timesteps",
                value=self.num_timesteps,
                iteration=self.num_timesteps,
            )
        return True


# =================================================
# ENV FACTORY
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
        device="cpu",  # HARD CPU LOCK
    )

    callback = ClearMLScalarCallback(args.n_steps)

    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    final_path = model_dir / "ppo_ot2_final"
    model.save(final_path)
    env.close()

    print("Training finished successfully.")
    print("Saved to:", final_path)


if __name__ == "__main__":
    main()
