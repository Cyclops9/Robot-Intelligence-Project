import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP Position (Gripper) [Envs, 3]
    # Using target_pos_w on the ee_frame is the standard way to get TCP pos in Isaac Lab
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Joint Velocities [Envs, 7] for smoothness penalty
    joint_vel = env.scene['robot'].data.joint_vel
    
    # Get current actions [Envs, 8]. Last dim [7] is gripper position command.
    actions = env.action_manager.action
    gripper_action = actions[..., 7]

    # Task Parameters
    target_height = 0.5
    table_height = 0.02  # Approximate table height
    
    # Thresholds for defining states
    grasp_dist_threshold = 0.04
    # Gripper action is usually [-1, 1]. Negative is closing.
    gripper_close_action_threshold = -0.3 
    # Height at which we consider the lift stage to have started
    lift_start_height = table_height + 0.02

    # ----------------------------------------------------------------------
    # 2. Calculate Reward Components
    # ----------------------------------------------------------------------
    
    # --- Distance ---
    dist_tcp_cube = torch.norm(tcp_pos - cube_pos, dim=-1)

    # --- A. Reaching Reward ---
    # Standard shaped reward to bring TCP to the cube center.
    # Using tanh(5.0 * dist) provides a good gradient over roughly 40cm.
    reward_reach = 1.0 - torch.tanh(5.0 * dist_tcp_cube)

    # --- B. Grasping Reward (Continuous) ---
    # Encourage closing the gripper, but ONLY when very close to the cube.
    # This prevents the robot from just closing the gripper far away.
    
    # 1. Define a very sharp continuous "closeness" factor [0, 1].
    # High only when dist < ~5cm.
    is_very_close = 1.0 - torch.tanh(20.0 * dist_tcp_cube)
    
    # 2. Define a "closing intention" factor based on action.
    # Assuming action range [-1, 1], where -1 is closed.
    # This factor becomes positive as action becomes negative (closing).
    closing_intention = torch.clamp(-gripper_action, min=0.0)
    
    # Combine: Reward = Being Close * Trying to Close
    reward_grasp = is_very_close * closing_intention

    # --- C. Lifting Reward (Staged) ---
    # We want to reward lifting towards the target height, but we must ensure
    # it's due to a grasp, not just pushing the cube up.
    
    # Define a rigorous proxy condition for a successful grasp:
    # 1. TCP is geometrically close to the cube.
    # 2. Gripper is actively being commanded to close.
    # 3. The cube has actually moved up off the table surface.
    has_grasp_proxy = (dist_tcp_cube < grasp_dist_threshold) & \
                      (gripper_action < gripper_close_action_threshold) & \
                      (cube_height > lift_start_height)
    
    # Shaping towards target height.
    dist_to_target = torch.abs(target_height - cube_height)
    lift_shaping = 1.0 - torch.tanh(3.0 * dist_to_target)
    
    # Apply lifting reward only when the rigorous grasp proxy is met.
    # This gate prevents "batting" rewards.
    reward_lift = has_grasp_proxy.float() * lift_shaping

    # --- D. Success Bonus (Sparse) ---
    # High reward for reaching the final target height.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 25.0

    # --- E. Penalties ---
    # Penalize high joint velocities for smooth motion.
    # Small penalty on action magnitude to encourage efficient control.
    reward_joint_vel = -0.002 * torch.sum(torch.square(joint_vel), dim=-1)
    reward_action_mag = -0.005 * torch.sum(torch.square(actions), dim=-1)

    # ----------------------------------------------------------------------
    # 3. Compute Total Reward
    # ----------------------------------------------------------------------
    # Compose the total reward with carefully chosen weights to guide the stages.
    # Reach -> Grasp -> Lift -> Success.
    total_reward = (
        2.0 * reward_reach      # Stage 1: Get to object
        + 5.0 * reward_grasp    # Stage 2: Initiate grasp when close
        + 15.0 * reward_lift    # Stage 3: Lift while holding (primary driver)
        + reward_success        # Stage 4: Final goal achievement
        + reward_joint_vel      # Regularization
        + reward_action_mag     # Regularization
    )

    # ----------------------------------------------------------------------
    # 4. Ground Truth & Logging
    # ----------------------------------------------------------------------
    try:
        # Log individual reward components for debugging purposes
        env.extras["episode"]["GPT/reward_reach"] = reward_reach.mean().item()
        env.extras["episode"]["GPT/reward_grasp"] = reward_grasp.mean().item()
        env.extras["episode"]["GPT/reward_lift"] = reward_lift.mean().item()
        env.extras["episode"]["GPT/reward_success"] = reward_success.mean().item()
        
        # Log key physical metrics to track progress independent of rewards
        env.extras["episode"]["GPT/dist_tcp_cube"] = dist_tcp_cube.mean().item()
        env.extras["episode"]["GPT/cube_height"] = cube_height.mean().item()
        env.extras["episode"]["GPT/gt_success"] = is_success.mean().item()

    except Exception:
        # Ensure training doesn't crash if logging fails
        pass

    # ----------------------------------------------------------------------
    # 5. Return Final Tensor
    # ----------------------------------------------------------------------
    return total_reward