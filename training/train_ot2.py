from __future__ import annotations

"""
train.py — Task 11 RL Controller (Stable Baselines 3 + ClearML)

Single training script for all curriculum stages (A/B/C) using CLI arguments.

Key improvements included:
1) Proper boolean CLI parsing (near_goal_slowdown true/false)
2) Stage A-friendly defaults (easier reaching to get success > 0 quickly)
3) ClearML remote execution without importing heavy deps locally
4) Worker fixes: force CPU-only torch, install pybullet if missing
5) Optional VecNormalize support (recommended) with save/load of stats
6) Resume-from-ClearML-artifact support (Stage B/C fine-tuning)
7) Chunked training with checkpointing and final model upload

Recommended curriculum:
- Stage A: thr=0.02, vel_max=0.15, slowdown=false
- Stage B: thr=0.005, vel_max=0.10, slowdown=true (resume from Stage A)
- Stage C: thr=0.001, vel_max=0.08, slowdown=true (resume from Stage B)
"""

# ============================================================
# ABSOLUTE TOP (stdlib only) — do not import SB3 / torch / envs
# ============================================================
import os
import sys
import argparse
import subprocess
import shutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force CPU (also avoids CUDA/CUPTI headaches on shared workers)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


# ============================================================
# Helpers
# ============================================================
def _is_worker() -> bool:
    return bool(os.environ.get("CLEARML_TASK_ID") or os.environ.get("CLEARML_WORKER_ID"))


def _pip_install(cmd: list[str]) -> None:
    print("[pip]", " ".join(cmd))
    subprocess.check_call(cmd)


def _restart_self() -> None:
    print("[restart] Restarting process ...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


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


def round_up_to_multiple(x: int, m: int) -> int:
    x = int(x)
    m = int(m)
    return int(((x + m - 1) // m) * m)


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean true/false, got: {v!r}")


# ============================================================
# Args
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    # ClearML
    p.add_argument("--use_clearml", action="store_true", help="Enqueue remotely via ClearML")
    p.add_argument("--project_name", type=str, default="Mentor Group - Alican/Group 1")
    p.add_argument("--task_name", type=str, default="OT2_PPO_Train")
    p.add_argument("--queue", type=str, default="default")
    p.add_argument("--docker", type=str, default="deanis/2023y2b-rl:latest")

    # Run bookkeeping
    p.add_argument("--run_name", type=str, default="", help="models/<run_name>/...")
    p.add_argument("--seed", type=int, default=0)

    # --------------------------------------------------------
    # Env defaults: Stage A (easy reaching; get success > 0)
    # --------------------------------------------------------
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--success_threshold", type=float, default=0.02)  # 2 cm
    p.add_argument("--action_repeat", type=int, default=1)
    p.add_argument("--vel_max", type=float, default=0.15)
    p.add_argument("--near_goal_slowdown", type=str2bool, default=False)
    p.add_argument("--render", action="store_true", help="GUI render (avoid on workers)")

    # PPO hyperparameters (strong baseline for continuous control)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.005)

    # Timesteps & checkpointing
    p.add_argument("--total_timesteps", type=int, default=800_000)
    p.add_argument("--checkpoint_freq", type=int, default=204_800)

    # Resume from ClearML
    p.add_argument("--resume_task_id", type=str, default="", help="ClearML task id to resume from")
    p.add_argument("--resume_artifact", type=str, default="ppo_final_model", help="Artifact name")
    p.add_argument("--reset_timesteps", type=str2bool, default=False, help="Reset SB3 timestep counter")

    # VecNormalize (recommended for stability and precision)
    p.add_argument("--use_vecnormalize", type=str2bool, default=True)
    p.add_argument("--norm_reward", type=str2bool, default=True)
    p.add_argument("--clip_obs", type=float, default=10.0)

    # Upload controls
    p.add_argument("--upload_checkpoints", type=str2bool, default=False)
    p.add_argument("--upload_every_n_checkpoints", type=int, default=1)
    p.add_argument("--upload_final", type=str2bool, default=True)
    p.add_argument("--upload_run_zip", type=str2bool, default=False)

    return p.parse_args()


# ============================================================
# ClearML setup
# ============================================================
def _get_task():
    try:
        from clearml import Task
        t = Task.current_task()
        if t is None:
            tid = os.environ.get("CLEARML_TASK_ID")
            if tid:
                t = Task.get_task(task_id=tid)
        return t
    except Exception:
        return None


def clearml_setup_and_exit_if_local(args) -> Optional[str]:
    """
    - Local + --use_clearml: create task, set docker, enqueue, then exit immediately.
      This avoids importing SB3/torch/pybullet locally.
    - Worker: attach to task and continue.
    - No ClearML: run locally.
    """
    if not args.use_clearml and not _is_worker():
        return None

    from clearml import Task

    if _is_worker():
        task = _get_task()
        if task:
            task.connect(vars(args), name="cli_args")
            print("[ClearML] Worker attached:", task.id)
            return task.id
        print("[ClearML] Worker detected but no task attached")
        return None

    task = Task.init(project_name=args.project_name, task_name=args.task_name)
    task.connect(vars(args), name="cli_args")
    task.set_base_docker(args.docker)
    print("[ClearML] Enqueuing to queue:", args.queue)
    task.execute_remotely(queue_name=args.queue)
    raise SystemExit(0)


# ============================================================
# Worker-only dependency enforcement
# ============================================================
def ensure_cpu_torch_on_worker() -> None:
    if not _is_worker():
        return

    need_install = False
    try:
        import torch as _t
        if _t.version.cuda is not None:
            print("[torch-check] CUDA torch detected:", _t.__version__, "cuda:", _t.version.cuda)
            need_install = True
    except Exception as e:
        print("[torch-check] torch import failed:", repr(e))
        need_install = True

    if not need_install:
        print("[torch-check] CPU torch already OK")
        return

    print("[FIX] Installing CPU-only torch ...")
    _pip_install([
        sys.executable, "-m", "pip", "install",
        "--no-cache-dir", "--force-reinstall", "--upgrade",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.4.1+cpu",
    ])
    _restart_self()


def ensure_pybullet_on_worker() -> None:
    if not _is_worker():
        return
    try:
        import pybullet  # noqa
        import pybullet_data  # noqa
        print("[pybullet-check] pybullet OK")
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


# ============================================================
# Resume helper
# ============================================================
def resolve_resume_local_path(args) -> str:
    if not args.resume_task_id:
        return ""
    from clearml import Task
    t = Task.get_task(task_id=args.resume_task_id)

    if args.resume_artifact not in t.artifacts:
        raise RuntimeError(
            f"Artifact '{args.resume_artifact}' not found in task {args.resume_task_id}. "
            f"Available: {list(t.artifacts.keys())}"
        )

    local_path = t.artifacts[args.resume_artifact].get_local_copy()
    print("[resume] Pulled model artifact to:", local_path)
    return local_path


# ============================================================
# Upload helper
# ============================================================
def upload_artifact(name: str, filepath: str) -> None:
    try:
        task = _get_task()
        if task is None:
            print(f"[upload] No ClearML task; skipping {name}")
            return
        if not filepath or not os.path.exists(filepath):
            print(f"[upload] Missing file for {name}: {filepath}")
            return

        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[ClearML] Uploading artifact: {name} -> {filepath} ({size_mb:.2f} MB)")
        task.upload_artifact(name=name, artifact_object=filepath, wait_on_upload=True)
        try:
            task.flush(wait_for_uploads=True)
        except Exception:
            pass
        _cleanup_memory(tag=f"after upload {name}")
    except Exception as e:
        print(f"[upload] FAILED {name}: {e}")


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()

    # Ensure project root on path for env imports (worker will clone repo)
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    current_task_id = clearml_setup_and_exit_if_local(args)

    # Worker-only: enforce dependencies before importing SB3/env
    ensure_cpu_torch_on_worker()
    ensure_pybullet_on_worker()

    # Heavy imports (safe now)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
    from envs.ot2_gym_wrapper import OT2GymEnv

    class ClearMLScalarCallback(BaseCallback):
        def __init__(self, report_every_steps: int):
            super().__init__()
            self.report_every_steps = int(report_every_steps)
            self._logger = None
            self._ep_return = 0.0
            self._ep_returns: list[float] = []
            self._episodes = 0
            self._successes = 0

        def _on_training_start(self) -> None:
            try:
                task = _get_task()
                self._logger = task.get_logger() if task else None
                print("[ClearML] scalar logger attached:", bool(self._logger))
            except Exception as e:
                self._logger = None
                print("[ClearML] scalar logger attach failed:", repr(e))

        def _on_step(self) -> bool:
            reward = float(self.locals["rewards"][0])
            done = bool(self.locals["dones"][0])
            info = self.locals["infos"][0] if self.locals.get("infos") else {}

            self._ep_return += reward

            if done:
                self._episodes += 1
                self._ep_returns.append(self._ep_return)
                if bool(info.get("is_success", False)):
                    self._successes += 1
                self._ep_return = 0.0

            if self._logger and (self.num_timesteps % self.report_every_steps == 0):
                it = int(self.num_timesteps)
                window = 50
                recent = self._ep_returns[-window:] if self._ep_returns else []
                ep_rew_mean_est = (sum(recent) / len(recent)) if recent else 0.0
                success_rate_est = (self._successes / self._episodes) if self._episodes > 0 else 0.0

                self._logger.report_scalar("time", "timesteps", it, iteration=it)
                self._logger.report_scalar("rollout", "ep_rew_mean_est", ep_rew_mean_est, iteration=it)
                self._logger.report_scalar("rollout", "success_rate_est", success_rate_est, iteration=it)
            return True

    # Run naming
    if not args.run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"ot2_run_{stamp}"

    # Round timesteps to multiples of rollout length (SB3 requirement-ish)
    rollout = int(args.n_steps)
    args.total_timesteps = round_up_to_multiple(args.total_timesteps, rollout)
    args.checkpoint_freq = round_up_to_multiple(args.checkpoint_freq, rollout)

    model_root = PROJECT_ROOT / "models" / args.run_name
    model_root.mkdir(parents=True, exist_ok=True)

    total = int(args.total_timesteps)
    chunk = int(args.checkpoint_freq) if int(args.checkpoint_freq) > 0 else total

    # Vec env factory
    def make_vec_env():
        def _make():
            env = OT2GymEnv(
                render=args.render,
                max_steps=args.max_steps,
                success_threshold=args.success_threshold,
                seed=args.seed,
                debug=False,
                action_repeat=args.action_repeat,
                vel_max=args.vel_max,
                near_goal_slowdown=args.near_goal_slowdown,
            )
            return Monitor(env)

        venv = VecMonitor(DummyVecEnv([_make]))
        if args.use_vecnormalize:
            venv = VecNormalize(
                venv,
                norm_obs=True,
                norm_reward=bool(args.norm_reward),
                clip_obs=float(args.clip_obs),
            )
        return venv

    def safe_close(vec_env) -> None:
        try:
            if vec_env is not None:
                vec_env.close()
        except Exception:
            pass
        _cleanup_memory(tag="after env.close()")

    # Resume (model + VecNormalize stats if present)
    resume_local = resolve_resume_local_path(args)
    resume_vecnorm = ""
    if resume_local:
        # If previous run folder exists locally, user might copy vecnormalize.pkl alongside model
        # For ClearML resume: you can optionally upload vecnormalize.pkl as artifact too.
        # Here we simply look in the same directory as the downloaded model.
        try:
            cand = Path(resume_local).resolve().parent / "vecnormalize.pkl"
            if cand.exists():
                resume_vecnorm = str(cand)
        except Exception:
            resume_vecnorm = ""

    # Print config
    print("\n========== TRAINING CONFIG ==========")
    print("Worker:", _is_worker())
    print("ClearML task id:", current_task_id if current_task_id else "(none)")
    print("Run name:", args.run_name)
    print("Total timesteps (rounded):", total)
    print("Checkpoint freq (rounded):", chunk)
    print("PPO: n_steps=", args.n_steps, "batch=", args.batch_size, "epochs=", args.n_epochs,
          "lr=", args.learning_rate, "gamma=", args.gamma, "ent_coef=", args.ent_coef)
    print("Env: max_steps=", args.max_steps,
          "thr=", args.success_threshold,
          "vel_max=", args.vel_max,
          "slowdown=", args.near_goal_slowdown,
          "action_repeat=", args.action_repeat)
    print("VecNormalize:", bool(args.use_vecnormalize), "norm_reward:", bool(args.norm_reward), "clip_obs:", args.clip_obs)
    print("Resume model:", resume_local if resume_local else "(none)")
    print("Resume vecnorm:", resume_vecnorm if resume_vecnorm else "(none)")
    print("Save dir:", str(model_root))
    print("====================================\n")

    # Build env (load VecNormalize stats if resuming and available)
    vec_env = make_vec_env()
    if args.use_vecnormalize and resume_vecnorm:
        try:
            # VecNormalize.load replaces wrapper; keep same env instance structure
            from stable_baselines3.common.vec_env import VecNormalize
            vec_env = VecNormalize.load(resume_vecnorm, vec_env)
            vec_env.training = True
            vec_env.norm_reward = bool(args.norm_reward)
            print("[vecnorm] Loaded stats:", resume_vecnorm)
        except Exception as e:
            print("[vecnorm] Failed to load stats:", repr(e))

    # Create / Load model
    if resume_local:
        print("[resume] Loading PPO from:", resume_local)
        model = PPO.load(resume_local, env=vec_env, device="cpu")
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
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

    cb = ClearMLScalarCallback(report_every_steps=max(512, int(args.n_steps)))

    # Chunked training + checkpointing
    trained = 0
    ckpt_index = 0

    while trained < total:
        this_chunk = min(chunk, total - trained)

        model.learn(
            total_timesteps=this_chunk,
            progress_bar=not _is_worker(),
            reset_num_timesteps=bool(args.reset_timesteps),
            callback=cb,
        )
        args.reset_timesteps = False

        trained += this_chunk
        ckpt_index += 1

        ckpt_base = model_root / f"ppo_ot2_{trained}_steps"
        ckpt_zip = str(ckpt_base) + ".zip"
        print("[save] checkpoint:", str(ckpt_base))
        model.save(str(ckpt_base))

        # Save VecNormalize stats alongside checkpoints (critical for eval reproducibility)
        if args.use_vecnormalize:
            try:
                vec_path = model_root / "vecnormalize.pkl"
                vec_env.save(str(vec_path))
            except Exception as e:
                print("[vecnorm] save failed:", repr(e))

        if args.upload_checkpoints and (ckpt_index % max(1, int(args.upload_every_n_checkpoints)) == 0):
            upload_artifact(name=f"ppo_checkpoint_{trained}_steps", filepath=ckpt_zip)
            if args.use_vecnormalize:
                vec_path = str(model_root / "vecnormalize.pkl")
                upload_artifact(name=f"vecnormalize_{trained}_steps", filepath=vec_path)

        _cleanup_memory(tag=f"after save {trained}")

    # Final save
    final_base = model_root / "ppo_ot2_final"
    final_zip = str(final_base) + ".zip"
    model.save(str(final_base))
    print("Training finished successfully.")
    print("Saved to:", str(final_base))

    # Save final VecNormalize stats
    if args.use_vecnormalize:
        try:
            vec_path = model_root / "vecnormalize.pkl"
            vec_env.save(str(vec_path))
            print("[vecnorm] Saved final stats:", vec_path)
        except Exception as e:
            print("[vecnorm] final save failed:", repr(e))

    safe_close(vec_env)

    # Upload final artifacts
    if args.upload_final:
        upload_artifact(name="ppo_final_model", filepath=final_zip)
        if args.use_vecnormalize:
            upload_artifact(name="vecnormalize_final", filepath=str(model_root / "vecnormalize.pkl"))

    if args.upload_run_zip:
        try:
            zip_file = shutil.make_archive(
                base_name=str(PROJECT_ROOT / "models" / args.run_name),
                format="zip",
                root_dir=str(model_root),
            )
            upload_artifact(name="run_folder_zip", filepath=zip_file)
        except Exception as e:
            print("[zip] FAILED:", e)

    _cleanup_memory(tag="final")


if __name__ == "__main__":
    main()
