
import os
import glob
import subprocess
import argparse
import sys

# CONFIG
PYTHON_EXEC = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISAAC_LAB_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # Assumes c:/Robot/IsaacLab/eureka_baseline/ -> c:/Robot/IsaacLab
SCRIPT_DIR = os.path.join(ISAAC_LAB_ROOT, "IsaacLab", "eureka_baseline")

def find_latest_checkpoint(run_dir):
    # Standard location
    models = glob.glob(os.path.join(run_dir, "model_*.pt"))
    if not models:
        models = glob.glob(os.path.join(run_dir, "checkpoints", "model_*.pt"))
        
    if not models:
        return None
        
    # Sort by step count (model_100.pt vs model_500.pt)
    def get_step(path):
        try: return int(os.path.basename(path).replace("model_", "").replace(".pt", ""))
        except: return 0
        
    models.sort(key=get_step, reverse=True)
    return models[0] # Return newest model

def get_specific_checkpoint(run_dir, model_name):
    possible_paths = [
        os.path.join(run_dir, model_name),
        os.path.join(run_dir, "checkpoints", model_name)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, default="logs/eureka_baseline", help="Root dir to search for runs")
    parser.add_argument("--model_name", type=str, default=None, help="Specific model filename (e.g., model_749.pt)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes per run")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default 1 for recording)")
    parser.add_argument("--headless", action="store_true", help="Run headless (no UI)")
    parser.add_argument("--record", action="store_true", help="Record video")
    args = parser.parse_args()

    abs_log_root = os.path.join(ISAAC_LAB_ROOT, args.log_root)
    print(f"--> Searching for runs in: {abs_log_root}")

    # Find all candidate folders
    candidate_dirs = glob.glob(os.path.join(abs_log_root, "candidate_*"))
    
    for c_dir in candidate_dirs:
        c_name = os.path.basename(c_dir)
        # Extract ID (e.g. candidate_0 -> 0)
        try:
            c_id = c_name.split("_")[-1]
        except:
            c_id = "0"
            
        # Find all timestamp runs inside
        run_dirs = glob.glob(os.path.join(c_dir, "*"))
        # Filter for directories only
        run_dirs = [d for d in run_dirs if os.path.isdir(d)]
        # Sort by time
        run_dirs.sort(key=os.path.getmtime)
        
        print(f"\n=== Found {len(run_dirs)} runs for {c_name} (ID: {c_id}) ===")
        
        for i, run in enumerate(run_dirs):
            run_name = os.path.basename(run)
            
            if args.model_name:
                ckpt = get_specific_checkpoint(run, args.model_name)
                if not ckpt:
                     print(f"  [SKIP] {run_name} (File {args.model_name} not found)")
                     continue
            else:
                ckpt = find_latest_checkpoint(run)
            
            if not ckpt:
                print(f"  [SKIP] {run_name} (No checkpoints found)")
                continue
                
            print(f"  > Processing: {run_name}")
            print(f"    Checkpoint: {os.path.basename(ckpt)}")
            
            # Construct Command
            cmd = [
                PYTHON_EXEC, os.path.join(SCRIPT_DIR, "play.py"),
                "--task", "Isaac-Lift-Cube-Franka-v0",
                "--candidate_id", str(c_id),
                "--checkpoint", ckpt,
                "--num_envs", str(args.num_envs),
                "--max_episodes", str(args.episodes)
            ]
            
            if args.headless:
                cmd.append("--headless")
            if args.record:
                cmd.append("--record_video")
                cmd.extend(["--video_length", "250"])

            # Execute
            try:
                print(f"    [CMD] {' '.join(cmd)}")
                subprocess.run(cmd, check=True, cwd=ISAAC_LAB_ROOT)
            except subprocess.CalledProcessError:
                print(f"    [ERROR] play.py failed for {run_name}")
            except Exception as e:
                print(f"    [ERROR] {e}")

if __name__ == "__main__":
    main()
