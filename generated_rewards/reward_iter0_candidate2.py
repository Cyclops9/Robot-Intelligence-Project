import torch

def compute_reward(env):
    # ----------------------------------------------------------------------
    # 1. Retrieve Data
    # ----------------------------------------------------------------------
    # Robot and Object
    robot = env.scene["robot"]
    cube = env.scene["object"]
    ee_frame = env.scene["ee_frame"]
    
    # Tensors on device
    # TCP Position [Envs, 3] and Orientation [Envs, 4] (w, x, y, z)
    tcp_pos = ee_frame.data.target_pos_w[..., 0, :]
    tcp_quat = ee_frame.data.target_quat_w[..., 0, :]
    
    # Cube Position [Envs, 3]
    cube_pos = cube.data.root_pos_w[..., 0:3]
    cube_height = cube_pos[..., 2]
    
    # Robot Joint Data for penalties
    joint_vel = robot.data.joint_vel
    
    # ----------------------------------------------------------------------
    # 2. Definitions & Constants
    # ----------------------------------------------------------------------
    target_height = 0.5
    table_height = 0.02
    dist_scale = 10.0  # Controls the sharpness of the reach reward
    
    # Distance from TCP to Cube
    dist = torch.norm(tcp_pos - cube_pos, dim=-1)
    
    # ----------------------------------------------------------------------
    # 3. Reward Components
    # ----------------------------------------------------------------------
    
    # --- A. Approach/Reach Reward ---
    # We use a Tanh kernel which is bounded [0, 1] and provides good gradients near 0.
    # At dist=0.1 (10cm), tanh(1.0) ~0.76 -> reward ~0.24
    # At dist=0.02 (2cm), tanh(0.2) ~0.20 -> reward ~0.80
    reward_reach = 1.0 - torch.tanh(dist_scale * dist)
    
    # --- B. Orientation Reward (Top-Down Prior) ---
    # To successfully grasp a cube on a table, a top-down approach is most stable.
    # We encourage the TCP Z-axis to align with the World Down vector (0, 0, -1).
    # Rotation Matrix Z-axis from Quat (w, x, y, z):
    # z_z = 1 - 2(x^2 + y^2)
    x = tcp_quat[..., 1]
    y = tcp_quat[..., 2]
    tcp_z_z = 1.0 - 2.0 * (x**2 + y**2)
    
    # We want tcp_z_z to be -1 (pointing down).
    # We define alignment as -tcp_z_z (range -1 to 1).
    # We clamp to 0 to treat all upward/sideways orientations as 0.
    reward_orient = torch.clamp(-tcp_z_z, min=0.0)
    
    # --- C. Lift Reward ---
    # This is the primary driver once the object is grasped.
    # We calculate progress from table to target height.
    lift_height = cube_height - table_height
    lift_progress = torch.clamp(lift_height / (target_height - table_height), min=0.0, max=1.0)
    
    # We scale this reward so that a full lift is significantly valuable.
    # Weighting rationale: If lift is achieved, this should dominate the reach reward
    # to prevent the robot from diving back down to minimize 'dist' if the cube shifts in hand.
    reward_lift = lift_progress * 3.0
    
    # --- D. Success Bonus ---
    # A large sparse reward for achieving the target height.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 5.0
    
    # --- E. Penalties ---
    # 1. Smoothness: Penalize joint velocities to prevent shaking.
    #    Reduced weight to -0.001 to allow fast movements required for lifting.
    reward_penalty_smoothness = -0.001 * torch.sum(torch.square(joint_vel), dim=-1)
    
    # 2. Linear Distance Penalty:
    #    Ensures a gradient exists even when far from the object (where tanh is saturated).
    reward_penalty_dist = -0.2 * dist

    # ----------------------------------------------------------------------
    # 4. Total Reward Calculation
    # ----------------------------------------------------------------------
    # Total Reward Construction
    # Reach (1.0) + Orient (0.5) guides the approach.
    # Lift (3.0) takes over as the object rises.
    # Success (5.0) provides the final satisfaction.
    total_reward = (
        reward_reach + 
        0.5 * reward_orient + 
        reward_lift + 
        reward_success + 
        reward_penalty_smoothness + 
        reward_penalty_dist
    )

    # ----------------------------------------------------------------------
    # 5. Logging
    # ----------------------------------------------------------------------
    env.extras["GPT/reward_reach"] = reward_reach.mean()
    env.extras["GPT/reward_orient"] = reward_orient.mean()
    env.extras["GPT/reward_lift"] = reward_lift.mean()
    env.extras["GPT/reward_success"] = reward_success.mean()
    env.extras["GPT/dist_tcp_cube"] = dist.mean()
    env.extras["GPT/cube_height"] = cube_height.mean()
    
    # REQUIRED: Success Rate Log
    env.extras["GPT/success"] = is_success.mean()

    return total_reward