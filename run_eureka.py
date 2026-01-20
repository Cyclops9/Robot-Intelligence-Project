import os
import subprocess
import json
import re
import time
import google.generativeai as genai
from eureka_utlis import get_feedback_stats, format_feedback
from vlm_interface import get_vlm_analysis, get_vlm_score
import glob
import torch
import argparse

# -----------------------------------------------------------------------------
# CONFIGURATIOsN
# -----------------------------------------------------------------------------
NUM_GENERATIONS = 4
NUM_CANDIDATES = 4       # Gemini generates 4 variations per generation
TRAINING_ITERATIONS = 750
TASK_NAME = "Isaac-Lift-Cube-Franka-v0"
PYTHON_EXEC = "python"   # Or path to your conda python

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EUREKA_ROOT = os.path.join(BASE_DIR, "eureka_baseline_vlm_1")
REWARD_OUTPUT_DIR = os.path.join(BASE_DIR, "generated_rewards")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------------------------------
# 1. GEMINI SETUP & PROMPTING
# -----------------------------------------------------------------------------
API_KEY = ''
if not API_KEY:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-3-pro-preview") # Or "gemini-1.5-pro"

# This context ensures Gemini remembers the variable names in every generatio
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
5. You MUST log the success rate to env.extras['GPT/success'] = is_success.mean().
6. DO NOT use 'success' as a key, use 'GPT/success'.

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
def find_best_video(iteration, candidate_id):
    # Matches: logs/eureka_baseline/candidate_0/2024-05-20_14-00/eval_video/*.mp4
    # The '**' searches inside all timestamp folders
    search_pattern = os.path.join("logs", "eureka_baseline", f"candidate_{candidate_id}", "**", "eval_video", "*.mp4")
    
    videos = glob.glob(search_pattern, recursive=True)
    if videos:
        # Returns the file with the most recent modification time
        return max(videos, key=os.path.getmtime)
    return None

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
    # CHANGED: Use 'base_reward.py' as the seed file if it exists, else fallback
    seed_file = os.path.join(REWARD_OUTPUT_DIR, "base_reward.py")
    if not os.path.exists(seed_file):
        seed_file = os.path.join(REWARD_OUTPUT_DIR, "reward_iter0_candidate0.py")
    
    # NEW: Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_gen", type=int, default=0, help="Generation to resume from (default: 0)")
    parser.add_argument("--resume_candidate", type=int, default=0, help="Candidate index to resume from (default: 0)")
    parser.add_argument("--vlm_mode", action="store_true", help="Enable VLM-based selection and critique.")
    args = parser.parse_args()

    # If resuming from gen > 0, we verify previous prompt exists into order to restore state
    if args.resume_gen > 0:
        prev_prompt_file = os.path.join(REWARD_OUTPUT_DIR, f"prompt_iter{args.resume_gen}.txt")
        if not os.path.exists(prev_prompt_file):
             # Try previous generation if current one missing (maybe we crashed before saving it?)
             # Actually consistency requires prompt_iter{gen} to be the one that generated the code for {gen}
             # Wait, prompt_iter{gen} is used to generate candidates for generation {gen}. 
             # So if we are at gen 5, we need prompt_iter5.txt (which was created from best of gen 4).
             print(f"[RESUME] Looking for prompt file to restore state: {prev_prompt_file}")
             if not os.path.exists(prev_prompt_file):
                 raise FileNotFoundError(f"Cannot resume from Gen {args.resume_gen}: missing {prev_prompt_file}")
    
    # -------------------------------------------------------------------------
    # RESTORE STATE OR INITIALIZE BASELINE
    # -------------------------------------------------------------------------
    if args.resume_gen == 0:
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
                # --- FIX: INITIALIZE BASELINE SCORE BASED ON MODE ---
                if args.vlm_mode:
                    # Will be updated below in the VLM block
                    best_reward_so_far = -10000.0 
                else:
                    # Base Mode: Use Success Rate * 100
                    s_term = 0.0
                    if "GPT/success" in stats: s_term = stats.get("GPT/success", {}).get("mean", 0.0)
                    elif "GPT/Success" in stats: s_term = stats.get("GPT/Success", {}).get("mean", 0.0)
                    elif "GPT/gt_success" in stats: s_term = stats.get("GPT/gt_success", {}).get("mean", 0.0)
                    
                    best_reward_so_far = s_term * 100.0
                    print(f"[BASELINE] Initial Score (Success Rate): {best_reward_so_far}")
                
                # feedback_history is still text description
                feedback_history = format_feedback(stats, 0)
                
                # =========================================================
                # NEW: VERIFY VIDEO RECORDING (Baseline Test)
                # =========================================================
                print("--> [BASELINE] Recording video to verify setup...")
                try:
                    # Run play.py to generate the video
                    subprocess.run([
                        PYTHON_EXEC, "eureka_baseline/play.py", 
                        "--candidate_id", "0",
                        "--headless", 
                        "--record_video", 
                        "--num_envs", "1",
                        "--video_length", "250" # Keep it short for the test
                    ], check=False)
                    
                    # Check if file was created
                    # (Ensure you have the 'find_best_video' helper function defined!)
                    test_video_path = find_best_video(0, 0)
                    
                    if test_video_path and os.path.exists(test_video_path):
                        print(f"--> [BASELINE] SUCCESS: Video recorded at: {test_video_path}")
                        
                        # OPTIONAL: Test VLM here too if you want to be 100% sure
                        # from vlm_interface import get_vlm_analysis
                        # print(f"--> [BASELINE] VLM Test Critique: {get_vlm_analysis(test_video_path)}")
                    else:
                        print("--> [BASELINE] WARNING: Video recording failed. No file found.")
                except Exception as e:
                    print(f"--> [BASELINE] ERROR: Failed to run play.py: {e}")
                # =========================================================

                # NEW: VLM BASELINE CRITIQUE (If VLM Mode is Active)
                if args.vlm_mode:
                    print("--> [BASELINE] VLM Mode Active. Critiquing Baseline...")
                    try:
                        # Record video for baseline (Candidate 0)
                        subprocess.run([
                            PYTHON_EXEC, "eureka_baseline/play.py", 
                            "--candidate_id", "0",
                            "--headless", 
                            "--record_video", 
                            "--num_envs", "1",
                            "--video_length", "250" 
                        ], check=False)
                        
                        test_video_path = find_best_video(0, 0)
                        if test_video_path and os.path.exists(test_video_path):
                            vlm_score, vlm_rationale = get_vlm_score(test_video_path)
                            print(f"[BASELINE] VLM Score: {vlm_score}")
                            
                            # NEW: Set Global Best to VLM Score
                            best_reward_so_far = float(vlm_score)
                            
                            # Append VLM critique to feedback history for the first prompt
                            feedback_history += f"\n\n[BASELINE VLM CRITIQUE]:\n{vlm_rationale}"
                        else:
                             print("[BASELINE] VLM Critique failed (no video).")
                    except Exception as e:
                        print(f"[BASELINE] VLM Error: {e}")
                # =========================================================

            else:
                print(f"[BASELINE] Failed: {data.get('error_trace', 'Unknown Error')}")
    else:
        # RESUMING
        print(f"--- RESUMING FROM GENERATION {args.resume_gen} ---")
        
        # Restore best_code_so_far and feedback_history from prompt_iter{args.resume_gen}.txt
        # This prompt contains the "PREVIOUS CODE" (which was best of Gen N-1) and "FEEDBACK"
        load_path = os.path.join(REWARD_OUTPUT_DIR, f"prompt_iter{args.resume_gen}.txt")
        print(f"[RESUME] Loading state from {load_path}...")
        
        with open(load_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Extract Code
        # We look for: --- PREVIOUS CODE ---\n```python\n(content)
        # It usually ends before --- FEEDBACK (Performance of Previous Code) ---
        code_pattern = r"--- PREVIOUS CODE ---\s+```python\s+(.*?)(?=\n.*?--- FEEDBACK)" 
        match_code = re.search(code_pattern, content, re.DOTALL)
        
        if match_code:
            best_code_so_far = match_code.group(1).strip()
            print("  > recovered 'best_code_so_far'")
        else:
             raise ValueError("Could not parse PREVIOUS CODE from prompt file")
             
        # Extract Feedback
        # We look for: --- FEEDBACK (Performance of Previous Code) --- (content)
        # It goes until --- INSTRUCTIONS ---
        feedback_pattern = r"--- FEEDBACK \(Performance of Previous Code\) ---\s+(.*?)(?=\n.*?--- INSTRUCTIONS)"
        match_fb = re.search(feedback_pattern, content, re.DOTALL)
        
        if match_fb:
            feedback_history = match_fb.group(1).strip()
            print("  > recovered 'feedback_history'")
        else:
            # Fallback for robustness
            print("  > [WARNING] Could not parse feedback, setting to empty.")
            feedback_history = ""
            
        # We also need 'best_reward_so_far'. 
        # Unfortunately, the prompt doesn't strictly contain the scalar score, just the feedback text.
        # We can try to parse it (Format: "Score: 123.45") inside feedback or just set it to a safe low value
        # so that any valid new candidate beats it.
        # Or we could scan previous result jsons, but that's expensive.
        # Let's simple check if the feedback content has "Max Reward: X"
        score_match = re.search(r"Max Reward:\s*([\d\.-]+)", feedback_history)
        if score_match:
            best_reward_so_far = float(score_match.group(1))
            print(f"  > recovered 'best_reward_so_far' = {best_reward_so_far}")
        else:
            best_reward_so_far = -10000.0
            print(f"  > [WARNING] Could not find numeric max reward in feedback, resetting to -10000.0")

    # --- EVOLUTION LOOP (Generations 1 to N) ---
    for gen in range(1, NUM_GENERATIONS):
        # SKIP GENERATIONS BEFORE RESUME_GEN
        if gen < args.resume_gen:
            continue
            
        print(f"\n\n================ GENERATION {gen} ================")
        
        # 1. Ask Gemini to improve (Only if we don't already have candidates/files OR if we are generating)
        # If we are resuming `gen == args.resume_gen`, we might use existing candidates.
        
        prompt_filename = f"prompt_iter{gen}.txt"
        prompt_path = os.path.join(REWARD_OUTPUT_DIR, prompt_filename)
        
        if gen == args.resume_gen and os.path.exists(prompt_path):
            # We already have the prompt from the crashed/previous run
             with open(prompt_path, "r", encoding="utf-8") as f:
                 prompt = f.read()
             print(f"--> [RESUME] Loaded existing prompt: {prompt_path}")
        else:
             # Fresh generation
             prompt = get_improvement_prompt(TASK_NAME, best_code_so_far, feedback_history)
             try:
                 with open(prompt_path, "w", encoding="utf-8") as f:
                     f.write(prompt)
                 print(f"--> Saved improvement prompt to: {prompt_path}")
             except Exception as e:
                 print(f"[Warning] Failed to save prompt: {e}")

        # GENERATION / LOADING CANDIDATES
        current_candidates = [None] * NUM_CANDIDATES
        is_new_candidate = [False] * NUM_CANDIDATES
        
        # If resuming this generation, check if we have existing candidate files
        # NOTE: User overwrites files as 'reward_iter0_candidate{i}.py' regardless of generation.
        # So we always look for 'iter0' files if we are resuming.
        if gen == args.resume_gen:
             print(f"--> [RESUME] Checking for existing candidate files (looking for iter0 files)...")
             existing_count = 0
             for i in range(NUM_CANDIDATES):
                 # ALWAYS READ FROM iter0
                 cand_file = os.path.join(REWARD_OUTPUT_DIR, f"reward_iter0_candidate{i}.py")
                 if os.path.exists(cand_file):
                     with open(cand_file, "r") as f: current_candidates[i] = f.read()
                     existing_count += 1
             
             missing_count = NUM_CANDIDATES - existing_count
             if missing_count > 0:
                 print(f"--> [RESUME] Found {existing_count} files. Generating {missing_count} more...")
                 new_codes = generate_candidates_sequentially(prompt, n=missing_count)
                 
                 # Fill missing slots
                 new_idx = 0
                 for i in range(NUM_CANDIDATES):
                     if current_candidates[i] is None:
                         if new_idx < len(new_codes):
                             current_candidates[i] = new_codes[new_idx]
                             is_new_candidate[i] = True
                             new_idx += 1
                         else:
                             # Fallback if generation failed to produce enough
                             current_candidates[i] = new_codes[-1] if new_codes else ""
                             is_new_candidate[i] = True
             else:
                 print(f"--> [RESUME] All {NUM_CANDIDATES} candidates found on disk.")
        
        else:
            # Normal Flow
            start_gen = generate_candidates_sequentially(prompt, n=NUM_CANDIDATES)
            for i, c in enumerate(start_gen):
                if i < NUM_CANDIDATES:
                    current_candidates[i] = c
                    is_new_candidate[i] = True

        # Sanity check
        current_candidates = [c if c else "" for c in current_candidates]

        if not any(current_candidates):
             print("[ERROR] No candidates available.")
             continue

        # 2. Save and Train all Candidates
        gen_scores = {}
        
        for i, code in enumerate(current_candidates):
            # Save Code (Overwrite or Ensure it exists)
            # ALWAYS SAVE AS iter0 per user requirement
            filename = os.path.join(REWARD_OUTPUT_DIR, f"reward_iter0_candidate{i}.py")
            with open(filename, "w") as f: f.write(code)
            
            # --- SKIPPING CANDIDATES IF RESUMING ---
            # We skip ONLY if:
            # 1. We are in the resume_gen
            # 2. The index is below the resume_candidate threshold
            # 3. AND the candidate is NOT new (i.e. it was loaded from disk)
            # If it is new, we MUST train it, even if index < resume_candidate (which shouldn't happen logically if user sets flags right, but for safety).
            if gen == args.resume_gen and i < args.resume_candidate and not is_new_candidate[i]:
                print(f"--> [RESUME] Skipping Training for Candidate {i} (already done?)")
                # We MUST Attempt to load its result to populate 'gen_scores'
                receipt = os.path.join(RESULTS_DIR, f"candidate_{i}.json")
                if os.path.exists(receipt):
                    try:
                        with open(receipt, "r") as f: data = json.load(f)
                        if data.get("status") == "completed":
                            stats = get_feedback_stats(data["log_dir"])
                            r_term = stats.get("Episode_Reward/eureka_term", {}).get("max", 0.0)
                            s_term = stats.get("GPT/gt_success", {}).get("mean", 0.0)
                            score = s_term * 100.0
                            feedback = format_feedback(stats, i)
                            gen_scores[i] = {"score": score, "code": code, "feedback": feedback}
                            print(f"     > Loaded Score: {score:.2f}")
                        else:
                            print(f"     > Previous run failed/crashed. Score: -10000")
                            gen_scores[i] = {"score": -10000.0, "code": code, "feedback": "Previous Crash"}
                    except:
                        print(f"     > Failed to read receipt. Score: -10000")
                        gen_scores[i] = {"score": -10000.0, "code": code, "feedback": "Receipt Read Error"}
                else:
                    print(f"     > No receipt found. Score: -10000")
                    gen_scores[i] = {"score": -10000.0, "code": code, "feedback": "Missing Receipt"}
                continue
            # ---------------------------------------

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
                        # 1. Total Reward
                        # Tries 'Episode_Reward/eureka_term' (standard) -> 'Episode_Reward/mean' (RLGames) -> 'Episode_Reward'
                        reward_term = 0.0
                        if "Episode_Reward/eureka_term" in stats:
                            reward_term = stats.get("Episode_Reward/eureka_term", {}).get("max", 0.0)
                        elif "Episode_Reward/mean" in stats: # Fallback for some RL Libs
                             reward_term = stats.get("Episode_Reward/mean", {}).get("max", 0.0)
                        elif "Episode_Reward" in stats:
                             reward_term = stats.get("Episode_Reward", {}).get("max", 0.0)
                        
                        # 2. Success Rate
                        # Tries 'GPT/success' (Enforced) -> 'GPT/Success' (Common) -> 'GPT/gt_success' (Legacy)
                        success_term = 0.0
                        if "GPT/success" in stats:
                            success_term = stats.get("GPT/success", {}).get("mean", 0.0)
                        elif "GPT/Success" in stats:
                             success_term = stats.get("GPT/Success", {}).get("mean", 0.0)
                        elif "GPT/gt_success" in stats:
                            success_term = stats.get("GPT/gt_success", {}).get("mean", 0.0)
                        
                        # HEURISTIC: Use ONLY success rate (User Request)
                        total_score = success_term * 100.0
                        feedback = format_feedback(stats, i)
                        
                        gen_scores[i] = {
                            "score": total_score, 
                            "reward_term": reward_term, # Store dense reward for tie-breaking
                            "code": code, 
                            "feedback": feedback
                        }
                        print(f"  > Candidate {i} Score: {total_score:.2f} (Dense: {reward_term:.2f})")

                    # --- SCENARIO B: CODE ERROR (AttributeError, SyntaxError, etc) ---
                    else:
                        error_trace = data.get("error_trace", "Unknown Error")
                        print(f"  > Candidate {i} Crashed (Python Error).")
                        
                        # We save the Error Trace as feedback!
                        fail_feedback = f"EXECUTION ERROR:\n{error_trace}\nFix this error in the next code."
                        gen_scores[i] = {"score": -10000.0, "reward_term": -10000.0, "code": code, "feedback": fail_feedback}

                except Exception as e:
                    print(f"  > Candidate {i} JSON Corrupt: {e}")
                    gen_scores[i] = {"score": -10000.0, "reward_term": -10000.0, "code": code, "feedback": "JSON Read Error"}
            else:
                # --- SCENARIO C: SYSTEM CRASH (Segfault, C++ Error) ---
                print(f"  > Candidate {i} Failed (No Receipt Found).")
                gen_scores[i] = {"score": -10000.0, "reward_term": -10000.0, "code": code, "feedback": "System/Simulator Crash."}

        # 3. Selection (Runs AFTER all candidates are done)
        if not gen_scores:
            print("[ERROR] gen_scores empty. Skipping selection.")
            continue
            
        # Select Best Candidate
        if args.vlm_mode:
            # We must first execute the VLM block to get scores, then select.
            # (Logic handled below in VLM block)
            pass 
        else:
            # BASE MODE SELECTION
            # Primary: 'score' (Success Rate)
            # Secondary: 'reward_term' (Dense Reward) - Pick highest reward if success is 0
            best_id = max(gen_scores, key=lambda x: (gen_scores[x]["score"], gen_scores[x]["reward_term"]))
        
        # If VLM mode is off, 'best_id' is now set.
        # If VLM mode is on, we'll override 'best_id' inside the VLM block.
        
        # We temporarily grab data for the "Base Winner" (or placeholder if VLM)
        # Note: If VLM mode is ON, we iterate ALL again below, so this is just for initialization.
        best_id_temp = max(gen_scores, key=lambda x: gen_scores[x]["score"])
        best_gen_data = gen_scores[best_id_temp]
        
        print(f"--> Best in Gen {gen} (Pre-VLM): Candidate {best_id_temp} (Score: {best_gen_data['score']:.2f})")
        
        # =================================================================
        # NEW: RECORDING WITH ERROR LOGGING ENABLED (AND VLM SELECTION)
        # =================================================================
        vlm_feedback = ""
        
        if args.vlm_mode:
            print("--> [VLM MODE] Evaluating ALL candidates with VLM...")
            vlm_scores = {} # id -> (score, rationale)
            
            # Loop through ALL valid candidates
            for idx, data in gen_scores.items():
                if data["score"] <= -9000.0: continue # Skip crashes
                
                print(f"    > Processing Candidate {idx}...")
                
                # 1. Record Video
                try:
                    subprocess.run([
                        "python", "eureka_baseline/play.py", 
                        "--candidate_id", str(idx),
                        "--headless", 
                        "--record_video", 
                        "--num_envs", "1",
                        "--video_length", "250" 
                    ], check=False)
                except Exception as e:
                    print(f"      [Warning] play.py failed: {e}")
                    continue

                # 2. Get VLM Score
                video_path = find_best_video(gen, idx)
                if video_path and os.path.exists(video_path):
                     # Crucial: clean memory (import inside loop if needed or rely on func)
                    torch.cuda.empty_cache()
                    
                    score, rationale = get_vlm_score(video_path)
                    vlm_scores[idx] = (score, rationale)
                else:
                    print(f"      [Warning] No video found for Candidate {idx}")

            # 3. SELECT BEST CANDIDATE BASED ON VLM SCORE
            if vlm_scores:
                best_id = max(vlm_scores, key=lambda x: vlm_scores[x][0])
                best_score, best_rationale = vlm_scores[best_id]
                
                print(f"--> [VLM SELECTION] Best Candidate is {best_id} with VLM Score {best_score}")
                
                # OVERWRITE best_gen_data to match the VLM winner
                best_gen_data = gen_scores[best_id]
                # Update its 'score' field so the "Global Best" check uses VLM score?
                # The user said: "best score should be chosen that will guide the next generations code"
                # So we treat VLM score as the primary metric.
                best_gen_data["score"] = float(best_score) 
                
                # Retrieve video path again for detailed analysis
                video_path = find_best_video(gen, best_id)
                
                # 4. GET DETAILED ANALYSIS (User Request)
                detailed_analysis = ""
                if video_path:
                    print(f"--> [VLM] Getting detailed analysis for Winner (Candidate {best_id})...")
                    torch.cuda.empty_cache()
                    detailed_analysis = get_vlm_analysis(video_path)
                
                # Append Rationale + Detailed Analysis
                vlm_feedback = f"\n\n[VLM SCORING RATIONALE]:\n{best_rationale}\n\n[VLM DETAILED ANALYSIS]:\n{detailed_analysis}"
            else:
                print("--> [VLM] Failed to score any candidates. Falling back to success rate.")
        
        else:
            # --- BASE MODE ---
            # We already selected `best_id` using (Success, Dense) logic above.
            best_gen_data = gen_scores[best_id]
            pass
            
        # =================================================================

        # 4. Update Global Best (Standard Logic)
        update_global = False
        
        if best_gen_data["score"] > best_reward_so_far and best_gen_data["score"] > -9000.0:
            print(f"--> NEW GLOBAL BEST! (Score: {best_gen_data['score']:.2f} > {best_reward_so_far:.2f})")
            update_global = True
            
        elif not args.vlm_mode and best_gen_data["score"] == 0.0 and best_reward_so_far == 0.0:
            # EXPLORATION CLAUSE:
            # If we are stuck at 0 success, we ALWAYS adopt the new best candidate (highest dense reward).
            # This prevents sticking to the baseline forever.
            print(f"--> 0% Success Tie. Adopting new candidate (Highest Dense Reward) to force exploration.")
            update_global = True
            
        if update_global:
            best_reward_so_far = best_gen_data["score"]
            best_code_so_far = best_gen_data["code"]
            
            # Combine stats + visual feedback
            feedback_history = best_gen_data["feedback"] + vlm_feedback
            
        else:
            print("--> Failed to improve. Keeping previous best code.")
            
            # CRITICAL: Even if we didn't improve, we pass the VLM feedback 
            # to the next generation so it knows WHY it failed.
            if vlm_feedback:
                 print("    (Appending VLM feedback to help next generation)")
                 feedback_history += vlm_feedback

            # If it was a crash, append the error trace
            if best_gen_data["score"] <= -9000.0:
                print("    (Appending Error Trace to Feedback)")
                feedback_history += f"\n\n[WARNING] The previous best attempt crashed:\n{best_gen_data['feedback']}"

if __name__ == "__main__": 
    main()