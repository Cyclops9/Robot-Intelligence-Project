import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # TCP (Gripper) Position & Orientation
    # Shape: [Num_Envs, 3] and [Num_Envs, 4]
    # We access index 0 for the body dimension as per convention for single-body frames
    tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    
    # Cube Position
    cube_pos = env.scene["object"].data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Joint States
    # Franka has 7 arm joints + 2 gripper fingers = 9 joints
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    
    # ----------------------------------------------------------------------
    # 2. Derived Metrics & Definitions
    # ----------------------------------------------------------------------
    target_height = 0.5
    table_height = 0.02
    
    # Euclidean distance from TCP to Cube
    dist = torch.norm(cube_pos - tcp_pos, dim=-1)
    
    # -- Gripper Orientation --
    # Calculate the TCP Z-axis vector in World Frame from Quaternion (w, x, y, z)
    # We want the gripper Z-axis (pointing out of palm) to align with World -Z (Down)
    w, x, y, z = tcp_quat[..., 0], tcp_quat[..., 1], tcp_quat[..., 2], tcp_quat[..., 3]
    
    # Formula for Z-axis of rotation matrix column 2
    # z_x = 2(xz + wy)
    # z_y = 2(yz - wx)
    # z_z = 1 - 2(xx + yy)
    tcp_z_z = 1.0 - 2.0 * (x**2 + y**2)
    
    # We want tcp_z_z to be -1.0 (pointing down). 
    # Alignment metric: range [-1, 1]. We want 1.
    # -1 * (-1) = 1.
    alignment = -tcp_z_z
    
    # -- Gripper Width --
    # Sum of last two joints (fingers)
    # Max width approx 0.08, Min width 0.0
    finger_width = torch.sum(joint_pos[..., -2:], dim=-1)

    # ----------------------------------------------------------------------
    # 3. Reward Components
    # ----------------------------------------------------------------------
    
    # --- A. Reach Reward ---
    # Encourages getting close to the object.
    # Tanh kernel gives nice gradients [0, 1]
    reward_reach = 1.0 - torch.tanh(5.0 * dist)
    
    # --- B. Orientation Reward ---
    # Encourage vertical approach. Only positive if somewhat aligned.
    reward_orient = torch.clamp(alignment, min=0.0)
    
    # --- C. Gripper Action Shaping ---
    # This is critical for success. We must teach the robot to OPEN on approach
    # and CLOSE when grabbing.
    
    # 1. Open Incentive: If far (>10cm), reward having fingers open.
    #    This prevents pushing the object away with a closed fist.
    is_far = dist > 0.10
    is_open = finger_width > 0.07 # Approx max width
    reward_open = is_far.float() * is_open.float() * 0.5
    
    # 2. Grasp Incentive: If near (<4cm), reward closing fingers.
    #    We reward the reduction of finger width.
    is_near = dist < 0.04
    # Reward scales as width decreases (0.08 -> 0.0)
    # Max reward when width is 0 (or clamped on object) is 0.08 * 10 = 0.8
    reward_grasp = is_near.float() * (0.08 - finger_width) * 10.0
    
    # --- D. Lift Reward ---
    # The primary task reward.
    # Only active if object is lifted slightly off table to avoid noise.
    is_lifted = cube_height > (table_height + 0.01)
    # Scale by 20 to dominate reaching once achieved.
    reward_lift = is_lifted.float() * (cube_height - table_height) * 20.0
    
    # --- E. Success Bonus ---
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 5.0
    
    # --- F. Penalties ---
    # 1. Smoothness: Penalize high joint velocities
    reward_smooth = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)
    
    # 2. Distance Linear Penalty: Ensures gradient exists even when tanh flattens
    reward_dist_penalty = -0.1 * dist

    # ----------------------------------------------------------------------
    # 4. Total Reward
    # ----------------------------------------------------------------------
    total_reward = (
        (2.0 * reward_reach) +
        (0.5 * reward_orient) +
        reward_open +
        reward_grasp +
        reward_lift +
        reward_success +
        reward_smooth + 
        reward_dist_penalty
    )

    # ----------------------------------------------------------------------
    # 5. Logging
    # ----------------------------------------------------------------------
    env.extras["GPT/reward_reach"] = reward_reach.mean()
    env.extras["GPT/reward_orient"] = reward_orient.mean()
    env.extras["GPT/reward_open"] = reward_open.mean()
    env.extras["GPT/reward_grasp"] = reward_grasp.mean()
    env.extras["GPT/reward_lift"] = reward_lift.mean()
    env.extras["GPT/reward_success"] = reward_success.mean()
    env.extras["GPT/cube_height"] = cube_height.mean()
    env.extras["GPT/dist"] = dist.mean()
    
    # REQUIRED
    env.extras["GPT/success"] = is_success.mean()

    return total_reward