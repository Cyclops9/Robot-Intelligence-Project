# File: eureka_baseline/train_candidate.py
import argparse
import sys
import os
import importlib.util
import json
from isaaclab.app import AppLauncher
import traceback
# ---------------------------------------------------------
# 1. ARGUMENT PARSING
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Train EUREKA Candidate with RSL-RL.")

# -- Eureka Specific --
parser.add_argument("--candidate_id", type=int, required=True, help="ID of the candidate (0-3)")

# -- Standard RSL-RL / Isaac Lab Arguments --
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")

# -- Agent / Run Management (Previously in cli_args) --
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run to load.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load.")
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")

# Append AppLauncher args
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

# Enable cameras if video is requested
if args_cli.video:
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. IMPORTS (Must be after AppLauncher)
# ---------------------------------------------------------
import gymnasium as gym
import logging
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

# Use the official wrapper from Isaac Lab
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 3. EUREKA UTILS
# ---------------------------------------------------------
@configclass
class CustomRewardsCfg:
    """Empty container for reward terms if original env has none."""
    pass

def load_reward_function(candidate_id):
    """Loads the generated reward function from the file system."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "generated_rewards", f"reward_iter0_candidate{candidate_id}.py")
    
    print(f"\n[EUREKA] Loading Candidate Reward from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("eureka_candidate", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_reward

# ---------------------------------------------------------
# 4. MAIN TRAINING LOOP
# ---------------------------------------------------------
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    
    # Define result_path early so we can write to it in the 'except' block
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    result_path = os.path.join(results_dir, f"candidate_{args_cli.candidate_id}.json")
    os.makedirs(results_dir, exist_ok=True)

    try:
        # --- A. Apply CLI Overrides ---
        if args_cli.run_name: agent_cfg.run_name = args_cli.run_name
        if args_cli.max_iterations: agent_cfg.max_iterations = args_cli.max_iterations
        if args_cli.resume: agent_cfg.resume = args_cli.resume
        if args_cli.load_run: agent_cfg.load_run = args_cli.load_run
        if args_cli.checkpoint: agent_cfg.load_checkpoint = args_cli.checkpoint

        # --- B. Eureka Reward Injection ---
        custom_reward_fn = load_reward_function(args_cli.candidate_id)
        if hasattr(env_cfg, "rewards"): env_cfg.rewards = CustomRewardsCfg()
        if hasattr(env_cfg, "curriculum"): env_cfg.curriculum = CustomRewardsCfg()
        env_cfg.rewards.eureka_term = RewTerm(func=custom_reward_fn, weight=1.0)

        # --- C. Environment Setup & Training ---
        if args_cli.num_envs is not None: env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.seed = agent_cfg.seed
        
        # ... (Logging setup code you already have) ...
        log_root_path = os.path.join("logs", "eureka_baseline", f"candidate_{args_cli.candidate_id}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(log_root_path, log_dir)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        
        print(f"\n[EUREKA] Starting Training for Candidate {args_cli.candidate_id}...")
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        
        # --- SUCCESS RESULT ---
        result_data = {
            "status": "completed",
            "log_dir": log_dir,
            "reward_code_path": f"reward_iter0_candidate{args_cli.candidate_id}.py"
        }

    except Exception as e:
        # --- CRASH HANDLER ---
        print(f"\n[EUREKA ERROR] Candidate {args_cli.candidate_id} crashed!")
        print(e)
        
        # Capture the full traceback (tells the LLM exactly line number and error)
        error_msg = traceback.format_exc()
        
        # Limit error size to prevent context overflow, but keep the end (most relevant)
        if len(error_msg) > 2000:
            error_msg = "...(truncated)...\n" + error_msg[-2000:]

        result_data = {
            "status": "failed",
            "error_trace": error_msg
        }

    # Write the result (Success OR Failure) to the JSON file
    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=4)
        print(f"[EUREKA] Result receipt written to: {result_path}")
    print("[EUREKA] Force exiting to prevent shutdown hang...")
    os._exit(0) 
if __name__ == "__main__":
    main()
    simulation_app.close()