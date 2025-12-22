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
    actions = env.action_manager.action
    gripper_action = actions[..., 7]

    # Constants
    table_height = 0.02
    target_height = 0.5
    
    # ----------------------------------------------------------------------
    # 2. Calculate Reward Components
    # ----------------------------------------------------------------------
    
    # --- Distance Calculations ---
    dist_tcp_cube = torch.norm(tcp_pos - cube_pos, dim=-1)

    # --- A. Reaching Reward ---
    # Encourage the gripper to get close to the cube.
    # Using a slightly wider kernel to provide a smoother gradient.
    reward_reach = 1.0 - torch.tanh(5.0 * dist_tcp_cube)

    # --- B. Grasping Incentive ---
    # Encourage closing the gripper when near the object.
    # This is not gated by a strict grasp check, but rather provides an incentive.
    # Gripper action space is usually [-1, 1], where -1 is closed.
    # We define 'near' as within 6cm.
    is_near_cube = (dist_tcp_cube < 0.06).float()
    # Reward pushing action towards -1.0 (closed) when near.
    # (gripper_action + 1.0) is in range [0, 2]. Ideally we want it to be 0.
    reward_pre_grasp = is_near_cube * (1.0 - torch.tanh(2.0 * (gripper_action + 1.0)))

    # --- C. Lifting Reward (Shaped) ---
    # Reward lifting the cube above the table.
    # Crucially, scale this reward by how close the gripper is to the cube.
    # This implicitly encourages holding the object while lifting, as opposed to batting it.
    lift_height = torch.clamp(cube_height - table_height, min=0.0)
    # The closer the gripper is to the cube, the more reward for height.
    grasp_proxy = 1.0 - torch.tanh(5.0 * dist_tcp_cube)
    reward_lift = grasp_proxy * lift_height

    # --- D. Success Bonus (Sparse) ---
    # A large bonus for achieving the target height.
    is_success = (cube_height > (target_height - 0.05)).float()
    reward_success = is_success * 50.0

    # --- E. Penalties ---
    # Penalize high joint velocities for smoothness.
    reward_joint_vel = -0.005 * torch.sum(torch.square(joint_vel), dim=-1)
    
    # Penalize large actions to encourage efficient movement.
    reward_action_mag = -0.005 * torch.sum(torch.square(actions), dim=-1)

    # ----------------------------------------------------------------------
    # 3. Compute Total Reward
    # ----------------------------------------------------------------------
    # Combine components with tuned weights.
    # Prioritize reaching, then lifting while grasping.
    total_reward = (
        2.0 * reward_reach
        + 1.0 * reward_pre_grasp
        + 5.0 * reward_lift
        + 1.0 * reward_success
        + 1.0 * reward_joint_vel
        + 1.0 * reward_action_mag
    )

    # ----------------------------------------------------------------------
    # 4. Logging
    # ----------------------------------------------------------------------
    try:
        # Log reward components
        env.extras["episode"]["GPT/reward_reach"] = reward_reach.mean().item()
        env.extras["episode"]["GPT/reward_pre_grasp"] = reward_pre_grasp.mean().item()
        env.extras["episode"]["GPT/reward_lift"] = reward_lift.mean().item()
        env.extras["episode"]["GPT/reward_success"] = reward_success.mean().item()
        env.extras["episode"]["GPT/reward_joint_vel"] = reward_joint_vel.mean().item()
        env.extras["episode"]["GPT/reward_action_mag"] = reward_action_mag.mean().item()
        
        # Log physical metrics
        env.extras["episode"]["GPT/dist_tcp_cube"] = dist_tcp_cube.mean().item()
        env.extras["episode"]["GPT/cube_height"] = cube_height.mean().item()
        env.extras["episode"]["GPT/gt_success"] = is_success.mean().item()

    except Exception:
        pass

    # ----------------------------------------------------------------------
    # 5. Return Final Tensor
    # ----------------------------------------------------------------------
    return total_reward