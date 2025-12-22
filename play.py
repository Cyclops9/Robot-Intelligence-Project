# File: eval_candidate.py

import argparse
import sys
import os
import torch
import glob
import importlib.util
from isaaclab.app import AppLauncher

# ---------------------------------------------------------
# 1. SETUP & ARGS
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate a Trained EUREKA Candidate.")
parser.add_argument("--candidate_id", type=int, required=True, help="ID of the candidate to evaluate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model.pt.")
parser.add_argument("--record_video", action="store_true", help="Record video instead of live interactive view.")

# AppLauncher handles --headless automatically
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Force cameras on if we are watching live
if not args_cli.headless and not args_cli.record_video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Launch App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------
# 2. IMPORTS
# ---------------------------------------------------------
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab.utils import configclass
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

@configclass
class CustomRewardsCfg:
    pass

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def load_reward_function(candidate_id):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "generated_rewards", f"reward_iter0_candidate{candidate_id}.py")
    if not os.path.exists(file_path):
        return None
    spec = importlib.util.spec_from_file_location("eureka_candidate", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_reward

def find_latest_checkpoint(candidate_id):
    base_log_dir = os.path.join("logs", "eureka_baseline", f"candidate_{candidate_id}")
    if not os.path.exists(base_log_dir):
        raise FileNotFoundError(f"No logs found for Candidate {candidate_id}")
    runs = glob.glob(os.path.join(base_log_dir, "*"))
    runs.sort(key=os.path.getmtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No run folders found.")
    latest_run = runs[0]
    
    # Try standard location first, then checkpoints folder
    models = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not models:
        models = glob.glob(os.path.join(latest_run, "checkpoints", "model_*.pt"))
        
    if not models:
        raise FileNotFoundError(f"No models found in {latest_run}")
        
    def get_step(path):
        try: return int(os.path.basename(path).replace("model_", "").replace(".pt", ""))
        except: return 0
    models.sort(key=get_step, reverse=True)
    return models[0], latest_run

# ---------------------------------------------------------
# 4. EVALUATION LOOP
# ---------------------------------------------------------
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    env = None
    runner = None
    
    try:
        # 1. Locate Checkpoint
        if args_cli.checkpoint:
            ckpt_path = args_cli.checkpoint
            log_root = os.path.dirname(ckpt_path)
        else:
            print(f"[INFO] Auto-locating latest checkpoint for Candidate {args_cli.candidate_id}...")
            ckpt_path, log_root = find_latest_checkpoint(args_cli.candidate_id)
        
        print(f"[INFO] Loading Checkpoint: {ckpt_path}")

        # 2. Patch Environment
        custom_reward = load_reward_function(args_cli.candidate_id)
        if custom_reward:
            if hasattr(env_cfg, "rewards"): env_cfg.rewards = CustomRewardsCfg()
            if hasattr(env_cfg, "curriculum"): env_cfg.curriculum = CustomRewardsCfg()
            env_cfg.rewards.eureka_term = RewTerm(func=custom_reward, weight=1.0)

        # 3. Configure Env
        env_cfg.scene.num_envs = args_cli.num_envs
        render_mode = "rgb_array" if args_cli.record_video else None
        
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        
        if args_cli.record_video:
            video_file = os.path.join(log_root, "eval_video")
            print(f"[INFO] Recording video to {video_file}...")
            env = gym.wrappers.RecordVideo(env, video_folder=video_file, step_trigger=lambda x: x == 0, video_length=500)

        # RSL-RL Wrapper (This converts 5 return values -> 4 return values)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # 4. Load Agent
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device=agent_cfg.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=env.device)

        # 5. Run Simulation
        obs, _ = env.reset()
        print(f"\n[INFO] Simulation Started. Check the window.")
        
        while simulation_app.is_running():
            with torch.inference_mode():
                action = policy(obs)
                
                # FIXED LINE: Unpack 4 values instead of 5
                # RslRlVecEnvWrapper returns: obs, reward, dones, extras
                obs, reward, dones, extras = env.step(action)
                
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        print("[INFO] Cleaning up environment...")
        if env is not None:
            env.close()
        del runner
        del env
        print("[INFO] Cleanup done.")

if __name__ == "__main__":
    main()
    print("[INFO] Closing Simulation App...")
    simulation_app.close()