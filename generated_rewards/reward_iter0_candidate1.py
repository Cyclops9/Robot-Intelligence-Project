import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP Position (Gripper Center) [Envs, 3]
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Joint States
    # The Franka robot typically has 9 joints in its state vector (7 arm + 2 gripper fingers).
    joint_pos = env.scene['robot'].data.joint_pos
    joint_vel = env.scene['robot'].data.joint_vel
    
    # Extract gripper finger positions (last 2 joints, indices 7 and 8)
    # Typical range for Franka fingers: ~0.04m (fully open) to ~0.0m (fully closed) per finger.
    finger_pos = joint_pos[..., 7:9]
    mean_finger_pos = torch.mean(finger_pos, dim=-1)

    # Get current actions [Envs, 8] for penalty calculation.
    actions = env.action_manager.action

    # Goal/Target Parameters
    target_height = 0.5
    table_height = 0.02
    
    # ----------------------------------------------------------------------
    # 2. Define Helper Signals (Continuous 0 to 1)
    # ----------------------------------------------------------------------
    # Distance from TCP to Cube
    dist_tcp_cube = torch.norm(tcp_pos - cube_pos, dim=-1)
    
    # Signal for being close to the cube. 1 when close, 0 when far.
    # Using a sharp tanh kernel.
    is_close = 1.0 - torch.tanh(15.0 * dist_tcp_cube)
    
    # Signal for gripper being closed based on physical state. 1 when closed, 0 when open.
    # Map mean finger pos from [0, 0.04] to roughly [1, 0].
    # 0.04 * 60 = 2.4 -> tanh(2.4) ~= 0.98. 1 - 0.98 = 0.02 (Open)
    # 0.00 * 60 = 0.0 -> tanh(0.0) = 0.00. 1 - 0.00 = 1.00 (Closed)
    is_gripper_closed = 1.0 - torch.tanh(mean_finger_pos * 60.0)
    
    # Combined "soft grasp" signal: close to cube AND gripper physically closed.
    # This is crucial for gating the lifting reward.
    soft_grasp = is_close * is_gripper_closed

    # ----------------------------------------------------------------------
    # 3. Calculate Reward Components
    # ----------------------------------------------------------------------
    
    # --- A. Reaching Reward ---
    # Encourage TCP to get close to the cube. This is the first stage.
    reward_reach = is_close

    # --- B. Grasping Reward ---
    # Encourage having the gripper closed, but ONLY when near the cube.
    # This guides the agent to connect the act of closing with proximity to the target.
    reward_grasp = soft_grasp

    # --- C. Lifting Reward ---
    # Encourage raising the cube. This reward is heavily gated by the `soft_grasp`
    # signal to prevent rewarding "batting" or knocking the cube up without control.
    
    # Reward for absolute height above table.
    height_above_table = torch.clamp(cube_height - table_height, min=0.0)
    
    # Reward for getting closer to the specific target height.
    dist_to_target = torch.abs(target_height - cube_height)
    close_to_target_height = 1.0 - torch.tanh(5.0 * dist_to_target)

    # Composite lifting reward, scaled up as it's the main task.
    # We reward both absolute height and proximity to target height, gated by grasp.
    reward_lift = soft_grasp * (10.0 * height_above_table + 5.0 * close_to_target_height)

    # --- D. Success Bonus ---
    # A large sparse bonus for achieving the target height within a tolerance.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 50.0

    # --- E. Penalties ---
    # Penalize high joint velocities and action magnitudes to encourage smooth, efficient motion.
    # Weights are kept low to not discourage exploration.
    reward_joint_vel = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)
    reward_action_mag = -0.001 * torch.sum(torch.square(actions), dim=-1)

    # ----------------------------------------------------------------------
    # 4. Compute Total Reward
    # ----------------------------------------------------------------------
    # Combine components with weights adjusted to prioritize the staged learning.
    total_reward = (
        3.0 * reward_reach
        + 5.0 * reward_grasp
        + 1.0 * reward_lift  # Note: reward_lift has internal scaling up to ~15
        + 1.0 * reward_success
        + 1.0 * reward_joint_vel
        + 1.0 * reward_action_mag
    )

    # ----------------------------------------------------------------------
    # 5. Logging
    # ----------------------------------------------------------------------
    try:
        # Log individual reward components for debugging
        env.extras["episode"]["GPT/r_reach"] = reward_reach.mean().item()
        env.extras["episode"]["GPT/r_grasp"] = reward_grasp.mean().item()
        env.extras["episode"]["GPT/r_lift"] = reward_lift.mean().item()
        env.extras["episode"]["GPT/r_success"] = reward_success.mean().item()
        
        # Log useful physical metrics
        env.extras["episode"]["GPT/dist_tcp_cube"] = dist_tcp_cube.mean().item()
        env.extras["episode"]["GPT/mean_finger_pos"] = mean_finger_pos.mean().item()
        env.extras["episode"]["GPT/soft_grasp_signal"] = soft_grasp.mean().item()
        env.extras["episode"]["GPT/cube_height"] = cube_height.mean().item()
    except Exception:
        pass # Ensure training doesn't crash if logging fails

    return total_reward