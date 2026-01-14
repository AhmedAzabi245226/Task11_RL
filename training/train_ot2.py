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
from typing import Any, Optional, List

# Do not expose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

# -------------------------------------------------
# Paths (do NOT import env/sim here)
# -------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent          # .../Task11_RL/training
PROJECT_ROOT = THIS_DIR.parent                      # .../Task11_RL


# =================================================
# Helpers
# =================================================
def _is_worker() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"))


def _pip_install(args: List[str]) -> None:
    print("[pip]", " ".join(args))
    subprocess.check_call(args)


def _restart_self() -> None:
    print("[restart] restarting process ...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def _cleanup_memory(tag: str = "") -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if tag:
        print(f"[mem] cleanup done: {tag}")


def _attach_or_get_task():
    try:
        from clearml import Task
        task = Task.current_task()
        if task is None:
            tid = os.environ.get("CLEARML_TASK_ID")
            if tid:
                task = Task.get_task(task_id=tid)
        return task
    except Exception:
        return None


# =================================================
# ClearML enqueue / attach
# =================================================
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

    # Timesteps + checkpoints
    p.add_argument("--total_timesteps", type=int, default=800_000)
    p.add_argument("--checkpoint_freq", type=int, default=102_400)

    # Resume (Stage training)
    p.add_argument("--resume_path", type=str, default="", help="Local path to .zip (worker-local or repo-relative).")
    p.add_argument("--resume_task_id", type=str, default="", help="ClearML task id to pull model artifact from.")
    p.add_argument("--resume_artifact", type=str, default="ppo_final_model", help="Artifact name in resume task.")
    p.add_argument("--reset_timesteps", action="store_true", help="Reset SB3 timestep counter when resuming.")

    # Upload controls
    p.add_argument("--upload_final", action="store_true", help="Upload final model artifact (default ON).")
    p.add_argument("--upload_run_zip", action="store_true", help="Upload run folder zip (default OFF).")
    p.set_defaults(upload_final=True, upload_run_zip=False)

    return p.parse_args()


# =================================================
# Resume resolver
# =================================================
def resolve_resume_local_path(args) -> str:
    """
    Priority:
      1) --resume_path (must exist either as-is or relative to PROJECT_ROOT)
      2) --resume_task_id + --resume_artifact (download via ClearML)
      else: "" (fresh training)
    """
    if args.resume_path:
        # reject URLs explicitly
        if args.resume_path.startswith("http://") or args.resume_path.startswith("https://"):
            raise RuntimeError("Do not pass an http(s) URL as --resume_path. Use --resume_task_id instead.")

        if os.path.exists(args.resume_path):
            return args.resume_path

        candidate = str(PROJECT_ROOT / args.resume_path)
        if os.path.exists(candidate):
            return candidate

        raise FileNotFoundError(f"--resume_path not found: {args.resume_path} (also tried {candidate})")

    if args.resume_task_id:
        from clearml import Task
        t = Task.get_task(task_id=args.resume_task_id)
        if args.resume_artifact not in t.artifacts:
            raise RuntimeError(
                f"Artifact '{args.resume_artifact}' not found in task {args.resume_task_id}. "
                f"Available: {list(t.artifacts.keys())}"
            )
        local = t.artifacts[args.resume_artifact].get_local_copy()
        print("[resume] Pulled from ClearML artifact:", local)
        return local

    return ""


# =================================================
# Upload artifact (safe)
# =================================================
def upload_artifact(name: str, filepath: str) -> None:
    try:
        task = _attach_or_get_task()
        if task is None:
            print(f"[upload] no ClearML task, skip {name}")
            return
        if not filepath or not os.path.exists(filepath):
            print(f"[upload] missing file for {name}: {filepath}")
            return

        print(f"[ClearML] Uploading artifact: {name} -> {filepath}")
        task.upload_artifact(name=name, artifact_object=filepath, wait_on_upload=True)
        try:
            task.flush(wait_for_uploads=True)
        except Exception:
            pass
    except Exception as e:
        print(f"[upload] FAILED {name}: {e}")


# =================================================
# Main
# =================================================
def main():
    args = parse_args()

    # Ensure project root import works (but do NOT import env/sb3 yet)
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # 1) Local + --use_clearml => enqueue and exit BEFORE importing env/sb3
    clearml_setup_and_exit_if_local(args)

    # 2) Worker only: fix torch + pybullet before importing SB3 / env
    ensure_cpu_torch_on_worker()
    ensure_pybullet_on_worker()

    # 3) Now safe to import heavy deps
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

    # Resume or fresh
    resume_local = resolve_resume_local_path(args)
    if resume_local:
        print("[resume] Loading model:", resume_local)
        model = PPO.load(resume_local, env=env, device="cpu")
    else:
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
            device="cpu",
        )

    cb = ClearMLScalarCallback(report_every=args.n_steps)

    # Optionally save periodic checkpoints (no upload by default)
    total = int(args.total_timesteps)
    ckpt_freq = int(args.checkpoint_freq)
    trained = 0
    ckpt_idx = 0

    while trained < total:
        chunk = min(ckpt_freq, total - trained)

        model.learn(
            total_timesteps=chunk,
            callback=cb,
            reset_num_timesteps=bool(args.reset_timesteps),
            progress_bar=not _is_worker(),
        )
        args.reset_timesteps = False
        trained += chunk
        ckpt_idx += 1

        ckpt_base = model_dir / f"ppo_ot2_{trained}_steps"
        model.save(str(ckpt_base))
        _cleanup_memory(tag=f"after save {trained}")

    # Final save + upload
    final_base = model_dir / "ppo_ot2_final"
    model.save(str(final_base))
    env.close()
    _cleanup_memory(tag="after env.close()")

    final_zip = str(final_base) + ".zip"
    print("Training finished successfully.")
    print("Saved to:", final_base)

    if args.upload_final:
        upload_artifact("ppo_final_model", final_zip)

    if args.upload_run_zip:
        try:
            zip_base = str(model_dir)
            zip_file = shutil.make_archive(base_name=zip_base, format="zip", root_dir=str(model_dir))
            upload_artifact("run_folder_zip", zip_file)
        except Exception as e:
            print("[zip] FAILED:", e)

    _cleanup_memory(tag="final")


if __name__ == "__main__":
    main()
