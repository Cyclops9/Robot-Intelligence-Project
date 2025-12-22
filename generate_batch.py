import google.generativeai as genai
import time
import os
import re

# --- CONFIGURATION ---
API_KEY = "AIzaSyDceYS8RslgPXDw5E8vqBbEXu6yZjUqtZ0"  # Paste your key
MODEL_NAME = "gemini-2.5-flash" # The model that works for you
TASK_NAME = "Isaac-Lift-Cube-Franka-v0" # Your target task
BATCH_SIZE = 4 # K=4 to be safe with rate limits
OUTPUT_FOLDER = 'eureka_baseline/generated_rewards'
# --- SETUP ---
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
os.makedirs("generated_rewards", exist_ok=True)

env_spec_context = """
TASK: Isaac-Lift-Cube-Franka-v0

--- SCENE ASSETS (Entities you can access) ---
  - env.scene['robot'] (Articulation) -> The Franka Robot
  - env.scene['cube_1'] (RigidObject) -> First Cube
  - env.scene['cube_2'] (RigidObject) -> Second Cube
  - env.scene['cube_3'] (RigidObject) -> Third Cube
  - env.scene['ee_frame'] (Frame)     -> End Effector Frame
  - env.scene['table'] (RigidObject)  -> Table Surface

--- OBSERVATIONS (What the robot sees) ---
  Group 'policy':
    - actions
    - joint_pos
    - joint_vel
    - object
    - cube_positions
    - cube_orientations
    - eef_pos
    - eef_quat
    - gripper_pos

--- ACTION SPACE ---
  Shape: (4096, 8)
"""
# --- THE EUREKA PROMPT (Adapted for Isaac Lab) ---
# We give it the 'signature' it must match.
EUREKA_PROMPT = """
You are an expert Reward Engineer for NVIDIA Isaac Lab.
Your goal is to write a reward function for the task: {task_name}.

--- ENVIRONMENT SPECIFICATION ---
{env_spec_context}
---------------------------------

Your Instructions:
1. Write a Python function named `compute_reward(env)`.
2. The input `env` is the `ManagerBasedRLEnv` object.
3. You can access the robot and objects using the names listed in the Environment Specification above (e.g., `env.scene['robot']`).
4. The function must return a single `torch.Tensor` of shape (num_envs,) representing the reward for each environment.
5. Do NOT use `@torch.jit.script`.
6. Use standard PyTorch operations. Ensure all new tensors are on `env.device`.

Example Signature:
```python
def compute_reward(env):
    # Access data
    # ...
    return reward_tensor"""

def generate_rewards():
    print(f"--- STARTING EUREKA GENERATION (Batch Size: {BATCH_SIZE}) ---")
    print(f"Saving to: {OUTPUT_FOLDER}")
    for i in range(BATCH_SIZE):
        print(f"Generating Candidate {i+1}/{BATCH_SIZE}...", end=" ", flush=True)
        
        try:
            # 1. Ask Gemini
            response = model.generate_content(
                EUREKA_PROMPT.format(
                    task_name=TASK_NAME, 
                    env_spec_context=env_spec_context
                )
            )
            code = response.text
            
            # 2. Clean up Markdown (Gemini loves ```python)
            code = re.sub(r"```python", "", code)
            code = re.sub(r"```", "", code)
            
            # 3. Save to file
            filename = os.path.join(OUTPUT_FOLDER, f"reward_iter0_candidate{i}.py")
            with open(filename, "w") as f:
                f.write(code)
                
            print(f"SAVED: reward_iter0_candidate{i}.py")
            
        except Exception as e:
            print(f"FAILED: {e}")

        # 4. RATE LIMIT SAFETY
        # Wait 15 seconds between calls to stay under 5 RPM
        if i < BATCH_SIZE - 1:
            print("Waiting 15s for rate limit...")
            time.sleep(15) 

    print("\nGeneration Complete! You can now run the training script.")
if __name__ == "__main__":
    generate_rewards()