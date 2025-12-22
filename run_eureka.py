import os
import subprocess
import json
import re
import time
import google.generativeai as genai
from eureka_utlis import get_feedback_stats, format_feedback
# -----------------------------------------------------------------------------
# CONFIGURATIOsN
# -----------------------------------------------------------------------------
NUM_GENERATIONS = 5
NUM_CANDIDATES = 4       # Gemini generates 4 variations per generation
TRAINING_ITERATIONS = 1500
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"
PYTHON_EXEC = "python"   # Or path to your conda python

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EUREKA_ROOT = os.path.join(BASE_DIR, "eureka_baseline")
REWARD_OUTPUT_DIR = os.path.join(BASE_DIR, "generated_rewards")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------------------------------
# 1. GEMINI SETUP & PROMPTING
# -----------------------------------------------------------------------------
API_KEY = 'AIzaSyDceYS8RslgPXDw5E8vqBbEXu6yZjUqtZ0'
if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-3-pro-image-preview") # Or "gemini-1.5-pro"

# This context ensures Gemini remembers the variable names in every generation
ENV_SPEC_CONTEXT = """
TASK: Isaac-Lift-Cube-Franka-v0

--- SCENE ASSETS (Correct Names) ---
  - env.scene['robot'] (Articulation)   -> The Franka Robot
  - env.scene['object'] (RigidObject)   -> The Cube (Target to lift)
  - env.scene['ee_frame'] (Frame)       -> End Effector Frame (The Gripper)
  - env.scene['table'] (RigidObject)    -> Table Surface

--- ACTION SPACE ---
  - Size: 8 Dimensions
  - [0-6]: Arm Joint Positions (7 DOF)
  - [7]:   Gripper Position (1 DOF)


--- CRITICAL RULES ---
1. You MUST return a single tensor 'total_reward'.
2. You MUST log components to env.extras['GPT/name'] = value.mean().
3. Use torch operations. Tensors must be on env.device.
4. The system deletes old rewards. You MUST re-implement 'action_rate' or 'smoothness' penalties to prevent shaking.

--- COMMON PITFALLS ---
1. DO NOT access 'env.actions'. The 'env' object is a ManagerBasedRLEnv.
2. To access the robot's current joint positions, use: 
   env.scene['robot'].data.joint_pos
3. To access the last applied action, use:
   env.action_manager.action
"""
def get_improvement_prompt(task_name, previous_code, feedback):
    """
    Constructs the prompt asking Gemini to improve the previous code.
    Updated to ask for SINGLE candidate generation.
    """
    return f"""
You are an expert RL Reward Designer.
Your goal is to IMPROVE the reward function for: {task_name}.

--- ENVIRONMENT CONTEXT ---
{ENV_SPEC_CONTEXT}

--- PREVIOUS CODE ---
```python
{previous_code}
--- FEEDBACK (Performance of Previous Code) --- {feedback}

--- INSTRUCTIONS ---

Analyze the feedback. Identify which components are zero (not working) or negative (penalties too high).

Generate ONE improved reward function.

CRITICAL: Return ONLY the Python code block. Do not add conversational text like "Here is the code".

The function signature must be def compute_reward(env):. """

def query_gemini(prompt): 
    """Sends prompt to Gemini and extracts code blocks.""" 
    print("\n[GEMINI] Generating new reward functions...") 
    try: 
        response = model.generate_content(prompt) 
        raw_text = response.text
        # Split by the separator we asked for
        # If Gemini fails to split, we might get one big block, so we handle that.
        if "### NEXT CANDIDATE ###" in raw_text:
            candidates = raw_text.split("### NEXT CANDIDATE ###")
        else:
            # Fallback: Just treat the whole response as one candidate (or regex split)
            candidates = [raw_text]

        # Clean up code blocks
        clean_codes = []
        for code in candidates:
            # Remove Markdown ```python ... ```
            code = re.sub(r"```python", "", code)
            code = re.sub(r"```", "", code)
            code = code.strip()
            if "def compute_reward" in code:
                clean_codes.append(code)
                
        # Pad with the last code if we didn't get enough
        while len(clean_codes) < NUM_CANDIDATES:
            clean_codes.append(clean_codes[-1] if clean_codes else "")
            
        return clean_codes[:NUM_CANDIDATES]

    except Exception as e:
        print(f"[GEMINI ERROR] {e}")
        return []
def query_gemini_single(prompt):
    """
    Sends a prompt to Gemini and gets a single valid Python code block.
    """
    print(".", end="", flush=True) # Progress dot
    try:
        response = model.generate_content(prompt)
        raw_text = response.text
        
        # Cleanup Markdown
        code = re.sub(r"```python", "", raw_text)
        code = re.sub(r"```", "", code)
        code = code.strip()
        
        # Validation: Ensure it looks like a function
        if "def compute_reward" not in code:
            print(f"\n[GEMINI WARNING] Generated text might not be code. Retrying...")
            return None
            
        return code

    except Exception as e:
        print(f"\n[GEMINI ERROR] {e}")
        return None

def generate_candidates_sequentially(prompt, n=4):
    """
    Calls Gemini 'n' times to get 'n' distinct candidates.
    """
    print(f"\n[GEMINI] Generating {n} candidates sequentially", end="")
    candidates = []
    attempts = 0
    max_attempts = n * 2 # Allow some failures
    
    while len(candidates) < n and attempts < max_attempts:
        # We can add a "temperature" note or slight variation to prompt if needed,
        # but usually Gemini is stochastic enough on its own.
        code = query_gemini_single(prompt)
        
        if code:
            candidates.append(code)
            # SAFETY: Sleep 2s to avoid hitting "Requests per minute" limits
            time.sleep(2.0) 
            
        attempts += 1
        
    if len(candidates) < n:
        print(f"\n[WARNING] Only generated {len(candidates)}/{n} candidates.")
        # Fill the rest with the last successful one to prevent crash
        while len(candidates) < n and candidates:
             candidates.append(candidates[-1])
             
    print(" Done!")
    return candidates
def run_training(candidate_id, generation): 
    print(f"--> [TRAIN] Gen {generation} | Candidate {candidate_id} starting...") 
    cmd = [ PYTHON_EXEC, os.path.join(BASE_DIR, "train_candidate.py"), "--task", TASK_NAME, "--candidate_id", str(candidate_id), "--max_iterations", str(TRAINING_ITERATIONS), "--headless" ] 
    try: 
        subprocess.run(cmd, check=True) 
        time.sleep(3)
        return True 
    except subprocess.CalledProcessError: 
        print(f"[ERROR] Candidate {candidate_id} crashed.") 
        return False
def main(): 
    os.makedirs(REWARD_OUTPUT_DIR, exist_ok=True) 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- STEP 0: ESTABLISH BASELINE ---
    seed_file = os.path.join(REWARD_OUTPUT_DIR, "reward_iter0_candidate0.py")

    if not os.path.exists(seed_file):
        print(f"[ERROR] Seed file not found: {seed_file}")
        print("Please save your 'reward_iter0_candidate0.py' code first!")
        return

    print("--- STARTING EUREKA WITH SEED CODE ---")
    with open(seed_file, "r") as f:
        best_code_so_far = f.read()

    # Initial Run
    print("[INIT] Training Baseline (Generation 0)...")
    run_training(0, 0) # We ignore return val, we check receipt

    # Initial Stats
    best_reward_so_far = -10000.0
    feedback_history = "Baseline Run Failed."
    
    # Check Baseline Receipt
    receipt_path = os.path.join(RESULTS_DIR, "candidate_0.json")
    if os.path.exists(receipt_path):
        with open(receipt_path, "r") as f: data = json.load(f)
        
        if data.get("status") == "completed":
            stats = get_feedback_stats(data["log_dir"])
            best_reward_so_far = stats.get("Episode_Reward/eureka_term", {}).get("max", -100.0)
            feedback_history = format_feedback(stats, 0)
            print(f"[BASELINE] Initial Score: {best_reward_so_far}")
        else:
            print(f"[BASELINE] Failed: {data.get('error_trace', 'Unknown Error')}")
    else:
        print("[CRITICAL] Baseline failed to generate a receipt. Exiting.")
        return

    # --- EVOLUTION LOOP (Generations 1 to N) ---
    for gen in range(1, NUM_GENERATIONS):
        print(f"\n\n================ GENERATION {gen} ================")
        
        # 1. Ask Gemini to improve
        prompt = get_improvement_prompt(TASK_NAME, best_code_so_far, feedback_history)
        # print(prompt) # Optional: Print prompt to debug
        new_codes = generate_candidates_sequentially(prompt, n=NUM_CANDIDATES)
        
        if not new_codes:
            print("[ERROR] Gemini produced no code. Retrying next loop.")
            continue

        # 2. Save and Train all Candidates
        gen_scores = {}
        
        for i, code in enumerate(new_codes):
            # Save Code
            filename = os.path.join(REWARD_OUTPUT_DIR, f"reward_iter0_candidate{i}.py")
            with open(filename, "w") as f: f.write(code)
            
            # Train
            # We don't rely solely on 'success' bool, we check the JSON receipt
            run_training(i, gen)
            
            # Evaluate using JSON Receipt
            receipt = os.path.join(RESULTS_DIR, f"candidate_{i}.json")
            
            if os.path.exists(receipt):
                try:
                    with open(receipt, "r") as f: data = json.load(f)
                    
                    # --- SCENARIO A: SUCCESS ---
                    if data.get("status") == "completed":
                        stats = get_feedback_stats(data["log_dir"])
                        
                        # Robust Scoring
                        reward_term = stats.get("Episode_Reward/eureka_term", {}).get("max", 0.0)
                        success_term = stats.get("GPT/gt_success", {}).get("mean", 0.0)
                        
                        # HEURISTIC: If success is > 0, boost score massively
                        total_score = reward_term + (success_term * 100.0)
                        feedback = format_feedback(stats, i)
                        
                        gen_scores[i] = {"score": total_score, "code": code, "feedback": feedback}
                        print(f"  > Candidate {i} Score: {total_score:.2f}")

                    # --- SCENARIO B: CODE ERROR (AttributeError, SyntaxError, etc) ---
                    else:
                        error_trace = data.get("error_trace", "Unknown Error")
                        print(f"  > Candidate {i} Crashed (Python Error).")
                        
                        # We save the Error Trace as feedback!
                        fail_feedback = f"EXECUTION ERROR:\n{error_trace}\nFix this error in the next code."
                        gen_scores[i] = {"score": -10000.0, "code": code, "feedback": fail_feedback}

                except Exception as e:
                    print(f"  > Candidate {i} JSON Corrupt: {e}")
                    gen_scores[i] = {"score": -10000.0, "code": code, "feedback": "JSON Read Error"}
            else:
                # --- SCENARIO C: SYSTEM CRASH (Segfault, C++ Error) ---
                print(f"  > Candidate {i} Failed (No Receipt Found).")
                gen_scores[i] = {"score": -10000.0, "code": code, "feedback": "System/Simulator Crash."}

        # 3. Selection (Runs AFTER all candidates are done)
        if not gen_scores:
            print("[ERROR] gen_scores empty. Skipping selection.")
            continue
            
        best_id = max(gen_scores, key=lambda x: gen_scores[x]["score"])
        best_gen_data = gen_scores[best_id]
        
        print(f"--> Best in Gen {gen}: Candidate {best_id} (Score: {best_gen_data['score']:.2f})")
        
        # 4. Update Global Best
        # We only update if the score is better AND it wasn't a crash (-10000)
        if best_gen_data["score"] > best_reward_so_far and best_gen_data["score"] > -9000.0:
            print("--> NEW GLOBAL BEST! Updating Seed.")
            best_reward_so_far = best_gen_data["score"]
            best_code_so_far = best_gen_data["code"]
            feedback_history = best_gen_data["feedback"]
        else:
            print("--> Failed to improve. Keeping previous best code.")
            # OPTIONAL: If the best of this generation was a crash, we append that error 
            # to the history so Gemini knows not to do it again.
            if best_gen_data["score"] <= -9000.0:
                print("    (Appending Error Trace to Feedback for next Prompt)")
                feedback_history += f"\n\n[WARNING] Attempted code crashed:\n{best_gen_data['feedback']}"
if __name__ == "__main__": 
    main()