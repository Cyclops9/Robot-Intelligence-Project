# File: eureka_baseline/debug_keys.py
import argparse
import sys
from isaaclab.app import AppLauncher

# 1. LAUNCH THE APP (Mandatory Step)
parser = argparse.ArgumentParser(description="Debug Keys")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. NOW it is safe to import Isaac Lab
from isaaclab_tasks.utils import parse_env_cfg

# Use the task name you found earlier
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"

def check_keys():
    print(f"--- INSPECTING CONFIG FOR: {TASK_NAME} ---")
    try:
        # Load the configuration blueprint
        cfg = parse_env_cfg(TASK_NAME)
        
        print("\n[SUCCESS] Found these Scene Assets:")
        # Filter out hidden keys (starting with _)
        found_keys = [k for k in cfg.scene.__dict__.keys() if not k.startswith("_")]
        
        for key in found_keys:
            print(f"  - env.scene['{key}']")
            
    except Exception as e:
        print(f"\n[ERROR] Could not load config: {e}")

if __name__ == "__main__":
    check_keys()
    simulation_app.close()