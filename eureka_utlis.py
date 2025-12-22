import os
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# -----------------------------------------------------------------------------
# 1. LOG PARSING (The "Eyes" of Eureka)
# -----------------------------------------------------------------------------
# File: eureka_baseline/eureka_utils.py

def get_feedback_stats(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents*"))
    if not event_files:
        print(f"[WARNING] No event file found in {log_dir}")
        return {}

    event_acc = EventAccumulator(event_files[0], size_guidance={"scalars": 0})
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    stats = {}

    # --- DEBUG: PRINT ALL TAGS FOUND ---
    print(f"\n[DEBUG LOGS] Tags found in {os.path.basename(log_dir)}:")
    found_gpt = False
    for t in tags:
        # Only print relevant ones to keep console clean
        if "GPT" in t or "Reward" in t or "metrics" in t.lower():
            print(f"  - {t}")
        if "GPT" in t: found_gpt = True
    
    if not found_gpt:
        print("  [WARNING] No 'GPT/...' tags found! The reward function is not logging correctly.")
    # -----------------------------------

    # Define targets to look for
    targets = ["GPT", "Episode_Reward", "Episode_Termination", "Metrics", "metrics"]
    tags = event_acc.Tags()['scalars']
    
    print("\n[DEBUG CHECKS]")
    if "Episode_Termination/time_out" in tags:
        print("✅ Found Timeouts in logs")
    else:
        print("❌ Timeouts MISSING from logs")
    # We use 'x in t' so 'GPT/reach' matches 'GPT'
    keys_to_track = [t for t in tags if any(x in t for x in targets)]

    for key in keys_to_track:
        events = event_acc.Scalars(key)
        values = [e.value for e in events]
        if len(values) == 0: continue

        stats[key] = {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "final": float(values[-1]), 
        }

    return stats
def safe_extract(stats, primary_key, secondary_key):
    # 1. Find Primary Key (fuzzy match)
    found_primary = None
    for k in stats.keys():
        if primary_key.lower() in k.lower():
            found_primary = k
            break
            
    if not found_primary:
        return 0.0
    
    # 2. Find Secondary Key (fuzzy match)
    section = stats[found_primary]
    for k in section.keys():
        if secondary_key.lower() in k.lower():
            return section[k] # Return the first match found

    return 0.0
def format_feedback(stats, candidate_id):
    """
    Formats the stats into a clear report for the LLM.
    """
    if not stats:
        return f"Candidate {candidate_id}: Failed to run.\n"

    # 1. Top Level Success Metrics (The "Ground Truth")
    feedback = f"Feedback for Candidate {candidate_id}:\n"
    
    # Try to find standard Isaac Lab success/fail metrics
    # Note: Keys might vary slightly by task, but these are standard
    success = safe_extract(stats, "metrics/success", "final")
    timeout = safe_extract(stats, "Episode_Termination/time_out", "mean")
    
    feedback += f"--- GROUND TRUTH ---\n"
    feedback += f"Success Rate: {success:.2f} (1.0 = 100% success)\n"
    feedback += f"Timeout Rate: {timeout:.2f} (High timeout = robot is stuck)\n"
    
    # 2. The Reward Score
    total_score = stats.get("Episode_Reward/eureka_term", {}).get("max", 0.0)
    feedback += f"Total Reward: {total_score:.2f}\n"

    # 3. Component Breakdown
    feedback += "--- REWARD COMPONENTS ---\n"
    for key, val in stats.items():
        if "GPT/" in key:
            clean_name = key.replace("GPT/", "")
            feedback += (
                f"  - {clean_name}: "
                f"Mean={val['mean']:.2f}, "
                f"Max={val['max']:.2f}, "
                f"Trend_End={val['final']:.2f}\n"
            )
            
    return feedback

# -----------------------------------------------------------------------------
# 3. PROMPT CONSTRUCTION (The "Instruction")
# -----------------------------------------------------------------------------
def get_eureka_prompt(task_description, previous_code, feedback_text):
    """
    Constructs the prompt to send to the LLM for the NEXT generation.
    """
    prompt = f"""
You are an expert Reinforcement Learning engineer. 
Your goal is to write a reward function for the task: {task_description}.

### PREVIOUS ATTEMPT
Here is the code you wrote in the last generation:
```python
{previous_code}```
FEEDBACK
Here is how that code performed during training: {feedback_text}

INSTRUCTIONS
Analyze the feedback. Look for components that are too small (near 0) or negative (penalties that are too harsh).

Write an IMPROVED reward function.

You must return the Total Reward Tensor.

You MUST log individual components to env.extras['GPT/component_name'] as shown in the previous code.

Retusrn only the Python code block. """ 
    return prompt