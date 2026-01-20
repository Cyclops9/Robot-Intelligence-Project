import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP (Gripper) Position [Envs, 3]
    # Accessing the first body in the end-effector frame
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    
    # Cube Height (Z)
    cube_height = cube_pos[..., 2]
    
    # Joint Velocities [Envs, Num_Joints]
    joint_vel = env.scene["robot"].data.joint_vel
    
    # Current Actions [Envs, Action_Dim]
    actions = env.action_manager.action

    # ----------------------------------------------------------------------
    # 2. Definitions & Constants
    # ----------------------------------------------------------------------
    target_height = 0.5
    # Threshold to determine if object is on the table or lifted
    # We assume table/ground is near Z=0, with a small buffer.
    table_height_threshold = 0.04  
    
    # Euclidean distance from TCP to Cube
    dist = torch.norm(tcp_pos - cube_pos, dim=-1)
    
    # ----------------------------------------------------------------------
    # 3. Reward Components
    # ----------------------------------------------------------------------
    
    # --- A. Approach Reward ---
    # We use a Tanh kernel: 1 - tanh(scale * dist).
    # This is bounded [0, 1] and provides smooth gradients both near and far.
    # Scale 5.0 implies:
    #   at dist=0.2m (20cm), reward ~ 0.24
    #   at dist=0.05m (5cm), reward ~ 0.75
    #   at dist=0.01m (1cm), reward ~ 0.95
    reward_approach = 1.0 - torch.tanh(5.0 * dist)
    
    # --- B. Lift Reward ---
    # We reward height above the table threshold.
    # We scale this by 4.0 so that lifting becomes the dominant strategy over just hovering.
    # e.g., lifting to 0.25m gives ~0.84 reward, comparable to the max approach reward.
    # lifting to target 0.5m gives ~1.84 reward.
    reward_lift = torch.clamp(cube_height - table_height_threshold, min=0.0) * 4.0
    
    # --- C. Grasp/Holding Bonus ---
    # If the robot is close to the object AND the object is slightly lifted off the table,
    # we provide a constant bonus. This helps bridge the transition from "touching" to "lifting".
    # It tells the agent: "You have it, don't let go."
    is_lifted_slightly = (cube_height > (table_height_threshold + 0.01)).float()
    is_close = (dist < 0.04).float()
    reward_grasp = is_lifted_slightly * is_close * 1.0

    # --- D. Success Bonus ---
    # A large sparse reward for reaching the target height.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 5.0
    
    # --- E. Penalties ---
    # 1. Action Regularization: Penalize large action commands to prevent "banging" against limits.
    reward_penalty_action = -0.01 * torch.sum(torch.square(actions), dim=-1)
    
    # 2. Smoothness: Penalize high joint velocities to reduce jitter.
    reward_penalty_smoothness = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)

    # ----------------------------------------------------------------------
    # 4. Total Reward Calculation
    # ----------------------------------------------------------------------
    total_reward = (
        reward_approach + 
        reward_lift + 
        reward_grasp + 
        reward_success + 
        reward_penalty_action + 
        reward_penalty_smoothness
    )

    # ----------------------------------------------------------------------
    # 5. Logging
    # ----------------------------------------------------------------------
    env.extras["GPT/reward_approach"] = reward_approach.mean()
    env.extras["GPT/reward_lift"] = reward_lift.mean()
    env.extras["GPT/reward_grasp"] = reward_grasp.mean()
    env.extras["GPT/reward_success"] = reward_success.mean()
    env.extras["GPT/dist_tcp_cube"] = dist.mean()
    env.extras["GPT/cube_height"] = cube_height.mean()
    
    # REQUIRED: Success Rate Log
    env.extras["GPT/success"] = is_success.mean()

    return total_reward