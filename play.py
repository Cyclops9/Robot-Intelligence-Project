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
parser.add_argument("--reward_file", type=str, default=None, help="Path to specific reward function file.") # <--- NEW ARGUMENT
parser.add_argument("--record_video", action="store_true", help="Record video instead of live interactive view.")
parser.add_argument("--video_length", type=int, default=250, help="Length of recorded video in steps.")
parser.add_argument("--max_episodes", type=int, default=None, help="Exit after N episodes (useful for automated testing).")

# AppLauncher handles --headless automatically
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# =========================================================================
# CRITICAL FIX: Enable cameras if we are recording, even if headless!
# =========================================================================
if args_cli.record_video or not args_cli.headless:
    args_cli.enable_cameras = True
# =========================================================================

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
def load_reward_function(candidate_id, reward_file_path=None):
    if reward_file_path:
        file_path = reward_file_path
        print(f"[INFO] Loading custom reward file: {file_path}")
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "generated_rewards", f"reward_iter0_candidate{candidate_id}.py")
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Reward file not found: {file_path}")
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
        custom_reward = load_reward_function(args_cli.candidate_id, reward_file_path=args_cli.reward_file)
        if custom_reward:
            if hasattr(env_cfg, "rewards"): env_cfg.rewards = CustomRewardsCfg()
            if hasattr(env_cfg, "curriculum"): env_cfg.curriculum = CustomRewardsCfg()
            env_cfg.rewards.eureka_term = RewTerm(func=custom_reward, weight=1.0)

        # 3. Configure Env
        env_cfg.scene.num_envs = args_cli.num_envs
        
        # --- FIX: SET CAMERA CLOSE TO ROBOT ---
        if hasattr(env_cfg, "viewer"):
            # 'eye': Camera Position (X, Y, Z)
            # 'lookat': Where the camera points (X, Y, Z)
            
            # This view is ~2 meters away, looking at the table center
            env_cfg.viewer.eye = (1.6, 1.2, 1.0)  
            env_cfg.viewer.lookat = (0.0, 0.0, 0.1) 
            
            print(f"[INFO] Camera set to Eye: {env_cfg.viewer.eye}, Lookat: {env_cfg.viewer.lookat}")
        # --------------------------------------

        # --- FIX: REMOVE ARROWS/MARKERS ---
        # 1. Disable End-Effector Frame Arrows
        if hasattr(env_cfg.scene, "ee_frame") and env_cfg.scene.ee_frame is not None:
             # Setting visualizer_cfg to None disables the frame markers
             env_cfg.scene.ee_frame.visualizer_cfg = None
             print("[INFO] Disabled End-Effector Frame Arrows.")

        # 2. Disable Target Object Pose Arrows
        if hasattr(env_cfg.commands, "object_pose") and env_cfg.commands.object_pose is not None:
             env_cfg.commands.object_pose.debug_vis = False
             print("[INFO] Disabled Object Pose Debug Arrows.")
        # ----------------------------------

        render_mode = "rgb_array" if args_cli.record_video else None
        
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
        
        if args_cli.record_video:
            video_file = os.path.join(log_root, "eval_video")
            print(f"[INFO] Recording video to {video_file}...")
            
            # --- FIX: USE DYNAMIC VIDEO LENGTH ---
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder=video_file, 
                step_trigger=lambda x: x == 0, 
                video_length=args_cli.video_length  # <--- Use argument here
            )
        # RSL-RL Wrapper (This converts 5 return values -> 4 return values)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # 4. Load Agent
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root, device=agent_cfg.device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=env.device)

        # 5. Run Simulation
        obs, _ = env.reset()
        print(f"\n[INFO] Simulation Started. Check the window.")
        
        # --- ADDED: Initialize Step Counter ---
        step_count = 0 
        success_buffer = [] 
        
        while simulation_app.is_running():
            with torch.inference_mode():
                action = policy(obs)
                
                # RslRlVecEnvWrapper returns: obs, reward, dones, extras
                obs, reward, dones, extras = env.step(action)
            
            # --- ADDED: Termination Logic ---
            step_count += 1
            
            # --- ADDED: Success Tracking ---
            if dones.any():
                # Check for success in extras
                # RSL-RL usually puts episode stats in extras["log"] or extras["episode"]
                current_success = None
                
                # Check 1: Direct 'log' (RSL-RL standard)
                if "log" in extras:
                    if "metrics/success" in extras["log"]:
                        current_success = extras["log"]["metrics/success"]
                    elif "GPT/success" in extras["log"]:
                        current_success = extras["log"]["GPT/success"]
                    elif "GPT/Success" in extras["log"]:
                        current_success = extras["log"]["GPT/Success"]
                    elif "GPT/gt_success" in extras["log"]:
                        current_success = extras["log"]["GPT/gt_success"]
                
                # Check 2: Direct 'episode' (IsaacLab/Eureka standard)
                if current_success is None and "episode" in extras:
                     if "metrics/success" in extras["episode"]:
                        current_success = extras["episode"]["metrics/success"]
                     elif "GPT/success" in extras["episode"]:
                        current_success = extras["episode"]["GPT/success"]
                     elif "GPT/Success" in extras["episode"]:
                        current_success = extras["episode"]["GPT/Success"]
                     elif "GPT/gt_success" in extras["episode"]:
                        current_success = extras["episode"]["GPT/gt_success"]

                if current_success is not None:
                    # If it's a tensor, get item
                    if isinstance(current_success, torch.Tensor):
                        val = current_success.mean().item()
                    else:
                        val = current_success
                    
                    print(f"[INFO] Episode finished. Success: {val:.2f}")
                    success_buffer.append(val)

            if args_cli.record_video and step_count >= args_cli.video_length:
                print(f"[INFO] Video recording complete ({step_count} steps). Exiting...")
                break
                
            if args_cli.max_episodes and len(success_buffer) >= args_cli.max_episodes:
                 print(f"[INFO] Completed {len(success_buffer)} episodes. Exiting...")
                 break
                
        if success_buffer:
            avg_success = sum(success_buffer) / len(success_buffer)
            print(f"\n[RESULTS] Average Success Rate over {len(success_buffer)} episodes: {avg_success:.2f}")
        else:
            print("\n[RESULTS] No episodes completed to measure success.")
                
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