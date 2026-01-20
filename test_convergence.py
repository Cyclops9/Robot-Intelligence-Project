
import subprocess
import os
import argparse
import glob
import sys
import shutil

# CONFIG
PYTHON_EXEC = sys.executable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ISAAC_LAB_ROOT = os.path.dirname(os.path.dirname(BASE_DIR)) # Assuming c:/Robot/IsaacLab/eureka_baseline/ -> c:/Robot/IsaacLab
SCRIPT_DIR = os.path.join(ISAAC_LAB_ROOT, "IsaacLab/eureka_baseline")

def find_latest_checkpoint(candidate_id):
    """
    Finds the latest model for a candidate in the logs directory.
    """
    base_log_dir = os.path.join(ISAAC_LAB_ROOT, "logs", "eureka_baseline", f"candidate_{candidate_id}")
    if not os.path.exists(base_log_dir):
        return None
        
    runs = glob.glob(os.path.join(base_log_dir, "*"))
    runs.sort(key=os.path.getmtime, reverse=True)
    
    if not runs:
        return None
        
    latest_run = runs[0]
    
    # Standard location
    models = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not models:
        models = glob.glob(os.path.join(latest_run, "checkpoints", "model_*.pt"))
        
    if not models:
        return None
        
    # Sort by step count (model_100.pt vs model_500.pt)
    def get_step(path):
        try: return int(os.path.basename(path).replace("model_", "").replace(".pt", ""))
        except: return 0
        
    models.sort(key=get_step, reverse=True)
    return models[0] # Return newest model

def run_test(candidate_id, iterations, episodes, checkpoint=None, reward_file=None, skip_train=False):
    print(f"==================================================")
    print(f" TESTING CANDIDATE {candidate_id}")
    print(f"==================================================")

    # 1. TRAINING (Optional)
    if not skip_train:
        print(f"\n[1] Starting Training ({iterations} steps)...")
        train_cmd = [
            PYTHON_EXEC, os.path.join(SCRIPT_DIR, "train_candidate.py"),
            "--task", "Isaac-Lift-Cube-Franka-v0",
            "--candidate_id", str(candidate_id),
            "--max_iterations", str(iterations),
            "--headless",
            "--video" # Optional: turn off if slower
        ]
        
        try:
            subprocess.run(train_cmd, check=True, cwd=ISAAC_LAB_ROOT)
        except subprocess.CalledProcessError:
            print("[ERROR] Training failed / crashed.")
            return
    else:
        print("\n[1] Skipping Training (Using provided checkpoint)...")

    # 2. FIND CHECKPOINT
    print(f"\n[2] Locating Checkpoint...")
    if checkpoint:
        ckpt = checkpoint
        if not os.path.exists(ckpt):
             print(f"[ERROR] Provided checkpoint not found: {ckpt}")
             return
    else:
        ckpt = find_latest_checkpoint(candidate_id)
        
    if not ckpt:
        print("[ERROR] No checkpoint found.")
        return
    print(f" -> Found: {ckpt}")

    # 3. EVALUATION
    print(f"\n[3] Running Evaluation ({episodes} episodes)...")
    play_cmd = [
        PYTHON_EXEC, os.path.join(SCRIPT_DIR, "play.py"),
        "--task", "Isaac-Lift-Cube-Franka-v0",
        "--candidate_id", str(candidate_id),
        "--checkpoint", ckpt,
        "--num_envs", "50",
        "--headless",      
        "--max_episodes", str(episodes)
    ]
    
    if reward_file:
        play_cmd.extend(["--reward_file", reward_file])
    
    try:
        subprocess.run(play_cmd, check=True, cwd=ISAAC_LAB_ROOT)
    except subprocess.CalledProcessError:
        print("[ERROR] Play/Eval script crashed.")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_id", type=int, default=3, help="Candidate ID to test")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    
    # New Arguments for Manual Testing
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model.pt to test")
    parser.add_argument("--reward_file", type=str, default=None, help="Path to specific reward file to load")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and just evaluate the checkpoint")

    args = parser.parse_args()

    # Smart Default: If checkpoint is provided, assume we want to skip training unless forced otherwise
    if args.checkpoint and not args.skip_train:
        print("[INFO] Checkpoint provided. Enabling --skip_train automatically.")
        args.skip_train = True

    run_test(args.candidate_id, args.iterations, args.episodes, args.checkpoint, args.reward_file, args.skip_train)
