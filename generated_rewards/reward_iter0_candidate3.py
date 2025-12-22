import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP Position (Gripper) [Envs, 3]
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Joint Velocities [Envs, 7]
    joint_vel = env.scene['robot'].data.joint_vel
    
    # Get current actions [Envs, 8]. Last dim [7] is gripper.
    # Gripper action: range usually [-1, 1], where -1 is closing/closed.
    actions = env.action_manager.action
    gripper_action = actions[..., 7]

    # Goal/Target Parameters
    target_height = 0.5
    table_height = 0.0  # Assuming table is at z=0 in world frame
    lift_threshold = 0.03 # Height considered "lifted" off the table

    # ----------------------------------------------------------------------
    # 2. Calculate Reward Components
    # ----------------------------------------------------------------------
    
    # Distance between TCP and Cube surface (approximate center)
    dist_tcp_cube = torch.norm(tcp_pos - cube_pos, dim=-1)

    # --- A. Reaching Reward ---
    # Encourage the gripper (TCP) to move close to the cube.
    # Used a slightly softer tanh kernel than before to provide gradients further away.
    reward_reach = 1.0 - torch.tanh(5.0 * dist_tcp_cube)

    # --- B. Continuous Grasping Reward ---
    # Instead of a binary state, encourage the gripper to be closed ONLY when near the cube.
    # Define a continuous "closeness" factor [0, 1]
    is_close_continuous = 1.0 - torch.tanh(10.0 * dist_tcp_cube)
    
    # Define a target gripper action based on closeness.
    # If close (1.0), target is -1.0 (closed). If far (0.0), target is 1.0 (open).
    target_gripper_action = 1.0 - 2.0 * is_close_continuous
    
    # Penalty for deviation from the target gripper action.
    # This shapes the agent to open the gripper when approaching and close it when near.
    reward_grasp = -torch.square(gripper_action - target_gripper_action)

    # --- C. Lifting Reward (Proximity Conditioned) ---
    # Encourage lifting the cube towards the target height.
    # Crucially, this reward is scaled by proximity to ensure the robot is actually holding it.
    
    # Vertical distance to target height
    dist_to_target_h = torch.abs(target_height - cube_height)
    
    # Base lifting reward: increases as cube gets closer to target height.
    # Only active if the cube is physically lifted off the table surface.
    is_lifted = (cube_height > table_height + lift_threshold).float()
    reward_lift_base = (1.0 - torch.tanh(3.0 * dist_to_target_h)) * is_lifted

    # Scale by proximity factor. If TCP moves away from cube, reward drops rapidly.
    # This implicitly enforces maintaining a grasp while lifting.
    proximity_factor = torch.exp(-10.0 * dist_tcp_cube)
    reward_lift = reward_lift_base * proximity_factor

    # --- D. Success Bonus (Sparse) ---
    # A large bonus for achieving the target height.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 50.0

    # --- E. Penalties ---
    # Penalize high joint velocities for smoothness. 
    # Coefficient reduced to prevent overly cautious behavior causing timeouts.
    reward_joint_vel = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)
    
    # Removed action magnitude penalty to encourage faster exploration and movement.

    # ----------------------------------------------------------------------
    # 3. Compute Total Reward
    # ----------------------------------------------------------------------
    # Combine components with adjusted weights.
    # Priority: Lift > Reach > Grasp Shaping
    total_reward = (
        2.0 * reward_reach
        + 0.5 * reward_grasp      # Smaller weight as it's a shaping term
        + 15.0 * reward_lift      # Main task reward
        + 1.0 * reward_success    # Sparse bonus
        + 1.0 * reward_joint_vel  # Small smoothness penalty
    )

    # ----------------------------------------------------------------------
    # 4. Ground Truth & Logging
    # ----------------------------------------------------------------------
    try:
        # Log reward components
        env.extras["episode"]["GPT/reward_reach"] = reward_reach.mean().item()
        env.extras["episode"]["GPT/reward_grasp"] = reward_grasp.mean().item()
        env.extras["episode"]["GPT/reward_lift"] = reward_lift.mean().item()
        env.extras["episode"]["GPT/reward_success"] = reward_success.mean().item()
        env.extras["episode"]["GPT/reward_joint_vel"] = reward_joint_vel.mean().item()
        
        # Log physical metrics
        env.extras["episode"]["GPT/dist_tcp_cube"] = dist_tcp_cube.mean().item()
        env.extras["episode"]["GPT/cube_height"] = cube_height.mean().item()
        env.extras["episode"]["GPT/gt_success"] = is_success.mean().item()

    except Exception:
        pass # Logging failures shouldn't stop training

    # ----------------------------------------------------------------------
    # 5. Return Final Tensor
    # ----------------------------------------------------------------------
    return total_reward