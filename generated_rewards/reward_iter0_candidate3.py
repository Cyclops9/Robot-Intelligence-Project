import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP (End Effector) Position [Envs, 3]
    # Accessing the first body index (0) in the end-effector frame
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # TCP Orientation (Quaternion w, x, y, z) [Envs, 4]
    tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Joint Velocities [Envs, Num_Joints]
    joint_vel = env.scene["robot"].data.joint_vel
    
    # Actions [Envs, 8] (7 Arm + 1 Gripper)
    # We use this to incentivize specific gripper commands
    actions = env.action_manager.action

    # ----------------------------------------------------------------------
    # 2. Definitions & Metrics
    # ----------------------------------------------------------------------
    target_height = 0.5
    table_height = 0.02
    
    # Distance from TCP to Cube
    dist = torch.norm(tcp_pos - cube_pos, dim=-1)
    
    # Orientation Metric:
    # Calculate alignment of EE Z-axis with World -Z (Downwards).
    # From quaternion (w, x, y, z), the Z component of the local Z axis is:
    # z_local_z_world = 1 - 2(x^2 + y^2).
    # We want this to be -1 (pointing down).
    # We compute dot_product with (0, 0, -1), which simplifies to:
    # dot = -1 * (1 - 2(x^2 + y^2)) = 2(x^2 + y^2) - 1.
    qx = tcp_quat[..., 1]
    qy = tcp_quat[..., 2]
    dot_prod = 2.0 * (torch.square(qx) + torch.square(qy)) - 1.0
    # Map from [-1, 1] to [0, 1] where 1 is perfect downward alignment
    orientation_score = torch.clamp((dot_prod + 1.0) / 2.0, 0.0, 1.0)

    # ----------------------------------------------------------------------
    # 3. Reward Components
    # ----------------------------------------------------------------------
    
    # --- A. Approach (Reach) ---
    # Strong, shaped reward to guide the robot to the object.
    # tanh(5*dist) scales 0.0 -> 0.0, 0.1 -> 0.46, 0.5 -> 0.98
    # 1 - tanh gives 1.0 at dist=0.
    reward_approach = 1.0 - torch.tanh(5.0 * dist)
    
    # --- B. Orientation Bonus ---
    # Only useful if somewhat close, but good to have globally to prep grasp.
    # Weight: 0.5
    reward_orient = orientation_score * 0.5
    
    # --- C. Grasp Action Incentive ---
    # If the gripper is very close to the cube (dist < 0.04), reward commanding the gripper to close.
    # We assume action index 7 is gripper, and negative values imply closing (typical for Franka).
    is_close = (dist < 0.04).float()
    gripper_action = actions[..., 7]
    # Check if command is closing (e.g. < -0.2)
    is_closing = (gripper_action < -0.2).float()
    reward_grasp_action = is_close * is_closing * 1.0
    
    # --- D. Lift Reward ---
    # The most important part. If the cube rises above the table, reward it heavily.
    # Threshold 0.04 avoids noise for cube sitting on table (0.02).
    # This reward scales linearly with height to encourage lifting higher.
    # At target height (0.5), reward ~ (0.5 - 0.02) * 30 = 14.4.
    is_lifted = (cube_height > (table_height + 0.02)).float()
    reward_lift = is_lifted * (cube_height - table_height) * 30.0

    # --- E. Success Bonus ---
    # Sparse reward for completing the task.
    # Slightly relaxed threshold (target - 0.1) to ensure trigger.
    is_success = (cube_height > (target_height - 0.1)).float()
    reward_success = is_success * 10.0
    
    # --- F. Penalties ---
    # 1. Action Regularization: prevent extreme bang-bang controls
    reward_penalty_action = -0.01 * torch.sum(torch.square(actions), dim=-1)
    
    # 2. Velocity Smoothness: prevent jitter
    reward_penalty_vel = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)

    # ----------------------------------------------------------------------
    # 4. Total Reward Calculation
    # ----------------------------------------------------------------------
    total_reward = (
        (2.0 * reward_approach) +
        reward_orient +
        reward_grasp_action +
        reward_lift +
        reward_success +
        reward_penalty_action +
        reward_penalty_vel
    )

    # ----------------------------------------------------------------------
    # 5. Logging
    # ----------------------------------------------------------------------
    env.extras["GPT/reward_approach"] = reward_approach.mean()
    env.extras["GPT/reward_lift"] = reward_lift.mean()
    env.extras["GPT/reward_orient"] = reward_orient.mean()
    env.extras["GPT/reward_grasp_action"] = reward_grasp_action.mean()
    env.extras["GPT/success"] = is_success.mean()
    env.extras["GPT/dist"] = dist.mean()
    env.extras["GPT/cube_height"] = cube_height.mean()

    return total_reward