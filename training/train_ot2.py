from __future__ import annotations

# =================================================
# ABSOLUTE TOP: only stdlib imports here
# =================================================
import os
import sys
import subprocess
import argparse
import shutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Do not expose GPU (prevents CUDA torch from trying to use CUPTI)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

# -------------------------------------------------
# Paths (do NOT import env/sim here)
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Task11_RL/training
PROJECT_ROOT = THIS_DIR.parent                      # .../Task11_RL


# =================================================
# ClearML / Worker detection
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
    try:
        from clearml import Task
    except Exception:
        return None

    task = Task.current_task()
    if task is None:
        tid = os.environ.get("CLEARML_TASK_ID")
        if tid:
            task = Task.get_task(task_id=tid)
    return task


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

    # IMPORTANT:
    # Do NOT call task.add_requirements() with requirements.txt if it contains pip options
    # like --index-url / --extra-index-url (ClearML SDK will crash parsing them).
    # Let the agent do its normal requirements/package analysis from the repo.

    print("[ClearML] Enqueuing to queue:", args.queue)
    task.execute_remotely(queue_name=args.queue)
    raise SystemExit(0)


# =================================================
# Runtime dependency enforcement (WORKER ONLY)
# =================================================
def ensure_cpu_torch_on_worker() -> None:
    """
    Fix libcupti.so.12 by forcing CPU torch inside the venv BEFORE SB3 import.
    """
    if not _is_worker():
        return

    need_install = False
    try:
        import torch as _t  # noqa
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
# Memory cleanup (helps long runs)
# =================================================
def _cleanup_memory(tag: str = "") -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # noqa
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if tag:
        print(f"[mem] cleanup done: {tag}")


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

    # Timesteps & saving
    p.add_argument("--total_timesteps", type=int, default=800_000)
    p.add_argument("--checkpoint_freq", type=int, default=102_400)

    # Artifacts upload controls
    p.add_argument("--upload_final", action="store_true", help="Upload final model to ClearML (default ON).")
    p.add_argument("--upload_run_zip", action="store_true", help="Upload run folder zip to ClearML (default OFF).")
    p.set_defaults(upload_final=True, upload_run_zip=False)

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

    # -------- ClearML logging callback (Scalars) --------
    class ClearMLScalarCallback(BaseCallback):
        """
        Logs:
          - rollout/ep_rew_mean_est (rolling mean over last N episodes)
          - rollout/success_rate_est (episodes with info["is_success"])
          - time/timesteps
        """
        def __init__(self, report_every_steps: int, window: int = 50):
            super().__init__()
            self.report_every_steps = int(report_every_steps)
            self.window = int(window)
            self._logger = None

            self._ep_return = 0.0
            self._ep_returns: list[float] = []
            self._episodes = 0
            self._successes = 0

        def _on_training_start(self) -> None:
            try:
                task = _attach_or_get_task()
                self._logger = task.get_logger() if task else None
                print("[ClearML] scalar logger active:", bool(self._logger))
            except Exception:
                self._logger = None
                print("[ClearML] scalar logger active: False")

        def _on_step(self) -> bool:
            # VecEnv has shape (n_envs,) even when n_envs=1
            reward = float(self.locals["rewards"][0])
            self._ep_return += reward

            done = bool(self.locals["dones"][0])
            info = self.locals["infos"][0]

            if done:
                self._episodes += 1
                self._ep_returns.append(self._ep_return)
                if bool(info.get("is_success", False)):
                    self._successes += 1
                self._ep_return = 0.0

            if self._logger and (self.num_timesteps % self.report_every_steps == 0):
                recent = self._ep_returns[-self.window:] if self._ep_returns else []
                ep_rew_mean = (sum(recent) / len(recent)) if recent else 0.0
                success_rate = (self._successes / self._episodes) if self._episodes > 0 else 0.0
                it = int(self.num_timesteps)

                self._logger.report_scalar("rollout", "ep_rew_mean_est", ep_rew_mean, iteration=it)
                self._logger.report_scalar("rollout", "success_rate_est", success_rate, iteration=it)
                self._logger.report_scalar("time", "timesteps", it, iteration=it)

            return True

    # -------- VecEnv --------
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

    cb = ClearMLScalarCallback(report_every_steps=args.n_steps, window=50)

    # -------- Training loop with optional checkpoints --------
    total = int(args.total_timesteps)
    ckpt_freq = int(args.checkpoint_freq) if int(args.checkpoint_freq) > 0 else total

    trained = 0
    while trained < total:
        this_chunk = min(ckpt_freq, total - trained)

        model.learn(
            total_timesteps=this_chunk,
            callback=cb,
            progress_bar=not _is_worker(),  # progress bar locally only
            reset_num_timesteps=False,
        )
        trained += this_chunk

        # Save checkpoint (local filesystem)
        ckpt_base = model_dir / f"ppo_ot2_{trained}_steps"
        model.save(str(ckpt_base))
        _cleanup_memory(tag=f"after save {trained}")

    # -------- Final save --------
    final_base = model_dir / "ppo_ot2_final"
    model.save(str(final_base))
    final_zip = str(final_base) + ".zip"

    # Close env BEFORE uploads (most stable with pybullet)
    try:
        env.close()
    except Exception:
        pass
    _cleanup_memory(tag="after env.close()")

    print("Training finished successfully.")
    print("Saved to:", final_base)

    # -------- Upload artifacts to ClearML (Artifacts tab) --------
    task = _attach_or_get_task()
    if task is not None:
        # Upload final model
        if args.upload_final and os.path.exists(final_zip):
            try:
                print("[ClearML] Uploading artifact: ppo_final_model ->", final_zip)
                task.upload_artifact(
                    name="ppo_final_model",
                    artifact_object=final_zip,
                    wait_on_upload=True,
                )
                try:
                    task.flush(wait_for_uploads=True)
                except Exception:
                    pass
            except Exception as e:
                print("[ClearML] final model upload failed:", repr(e))

        # Optional: zip the run folder and upload
        if args.upload_run_zip:
            try:
                zip_base = str(model_dir)  # creates <model_dir>.zip
                zip_path = shutil.make_archive(base_name=zip_base, format="zip", root_dir=str(model_dir))
                print("[ClearML] Uploading artifact: run_folder_zip ->", zip_path)
                task.upload_artifact(
                    name="run_folder_zip",
                    artifact_object=zip_path,
                    wait_on_upload=True,
                )
                try:
                    task.flush(wait_for_uploads=True)
                except Exception:
                    pass
            except Exception as e:
                print("[ClearML] run folder zip upload failed:", repr(e))

    _cleanup_memory(tag="final")


if __name__ == "__main__":
    main()
