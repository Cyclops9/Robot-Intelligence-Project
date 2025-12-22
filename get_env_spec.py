"""
IVSR Reflection Script: Generates the 'Cheat Sheet' for Gemini
"""
import argparse
import sys
import os
from isaaclab.app import AppLauncher

# 1. SETUP APP LAUNCHER (MUST BE FIRST)
parser = argparse.ArgumentParser(description="Generate Env Spec")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. SAFE IMPORTS
import gymnasium as gym
import isaaclab.envs
from isaaclab_tasks.utils import parse_env_cfg

# --- UPDATE THIS TO YOUR CORRECT TASK NAME ---
# Use "Isaac-Lift-Cube-Franka-v0" or whatever you found in the list
TASK_NAME = "Isaac-Stack-Cube-Franka-v0" 

def generate_cheat_sheet():
    print(f"Inspecting {TASK_NAME}...")
    
    try:
        env_cfg = parse_env_cfg(TASK_NAME)
        # Headless=True is faster
        env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not load task '{TASK_NAME}'.")
        print(f"Error details: {e}")
        print("Run the 'list tasks' command I gave you earlier to find the exact name!")
        return

    output = []
    output.append(f"TASK: {TASK_NAME}")
    output.append("--- SCENE ASSETS (Entities you can access) ---")
    
    # --- FIX: ROBUST ASSET INSPECTION ---
    # We try to get keys, or fall back to dir() if keys() is missing
    scene = env.unwrapped.scene
    
    if hasattr(scene, "keys"):
        # The standard way in modern Isaac Lab
        asset_names = scene.keys()
    else:
        # Fallback: Look at the configuration keys
        asset_names = [k for k in env.unwrapped.cfg.scene.__dict__.keys() 
                       if not k.startswith("_") and k not in ["num_envs", "env_spacing"]]

    for asset_name in asset_names:
        try:
            # Access the object from the scene
            asset_obj = getattr(scene, asset_name, None)
            if asset_obj is None and hasattr(scene, "get"):
                 asset_obj = scene.get(asset_name)
            
            if asset_obj:
                class_name = asset_obj.__class__.__name__
                output.append(f"  - env.scene['{asset_name}'] -> Type: {class_name}")
                
                # Check for articulation data (Robot stuff)
                if hasattr(asset_obj, "data"):
                    output.append(f"      (Available Data: .data.joint_pos, .data.eef_pos_w, .data.body_pos_w...)")
        except Exception as e:
            output.append(f"  - env.scene['{asset_name}'] (Could not inspect details: {e})")

    output.append("\n--- OBSERVATIONS (What the robot sees) ---")
    
    # Inspect Observations
    try:
        obs_manager = env.unwrapped.observation_manager
        # Handle different Isaac Lab versions of ObservationManager
        if hasattr(obs_manager, "group_obs_dim"):
            for group_name, dim in obs_manager.group_obs_dim.items():
                output.append(f"  Group '{group_name}' (dim: {dim}):")
                # Try to list active terms
                if hasattr(obs_manager, "active_terms") and group_name in obs_manager.active_terms:
                    for term in obs_manager.active_terms[group_name]:
                        output.append(f"    - {term}")
    except Exception as e:
        output.append(f"  (Could not inspect observations: {e})")

    output.append("\n--- ACTION SPACE ---")
    output.append(f"  Shape: {env.action_space.shape}")

    # Save to file
    cheat_sheet = "\n".join(output)
    print("\n" + "="*40)
    print(cheat_sheet)
    print("="*40)
    
    output_path = os.path.join(os.path.dirname(__file__), "env_spec.txt")
    with open(output_path, "w") as f:
        f.write(cheat_sheet)
    print(f"\nSaved to '{output_path}'. Copy this content to your prompt!")
    
    env.close()

if __name__ == "__main__":
    generate_cheat_sheet()
    simulation_app.close()