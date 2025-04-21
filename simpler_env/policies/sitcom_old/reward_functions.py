import numpy as np

def reward_for_put_carrot_on_plate(state, action=None):
    """Reward function for putting a carrot on a plate."""
    # breakpoint()
    # Get evaluation results from parent class
    eval_results = state.evaluate(success_require_src_completely_on_target=True)
    
    # Extract important information
    is_grasped = eval_results['is_src_obj_grasped']
    consecutive_grasp = eval_results['consecutive_grasp']
    src_on_target = eval_results['src_on_target']
    
    # Calculate distances
    tcp_pose = state.get_obs()['extra']['tcp_pose']
    gripper_pos = tcp_pose[:3]
    source_pos = state.source_obj_pose.p
    target_pos = state.target_obj_pose.p
    
    gripper_to_source_dist = np.linalg.norm(gripper_pos - source_pos)
    source_to_target_dist = np.linalg.norm(source_pos - target_pos)
    
    # Build reward
    reward = 0
    
    if not is_grasped:
        # Phase 1: Encourage approaching and grasping carrot
        reward = -gripper_to_source_dist * 2  # Negative distance as reward
    else:
        # Phase 2: Encourage moving carrot to plate
        reward = 2.0  # Bonus for grasping
        reward -= source_to_target_dist * 2  # Penalty based on distance to target
        
        if consecutive_grasp:
            reward += 1.0  # Additional reward for stable grasp
            
        if src_on_target:
            reward += 10.0  # Large bonus for successful placement
            
    # print different things
    # print(f"Reward: {reward}")
    # print(f"Is grasped: {is_grasped}")
    # print(f"gripper_to_source_dist: {gripper_to_source_dist}")
    
    return reward
        
    # except Exception as e:
    #     return -10.0  # Default negative reward


def reward_for_stack_green_on_yellow(state, action=None):
    """Reward function for stacking green cube on yellow cube."""
    # Get evaluation results from parent class
    eval_results = state.evaluate(success_require_src_completely_on_target=True)
    
    # Extract important information
    is_grasped = eval_results['is_src_obj_grasped']
    consecutive_grasp = eval_results['consecutive_grasp']
    src_on_target = eval_results['src_on_target']
    moved_wrong_obj = eval_results['moved_wrong_obj']
    
    # Calculate distances
    tcp_pose = state.get_obs()['extra']['tcp_pose']
    gripper_pos = tcp_pose[:3]
    source_pos = state.source_obj_pose.p
    target_pos = state.target_obj_pose.p
    
    gripper_to_source_dist = np.linalg.norm(gripper_pos - source_pos)
    source_to_target_xy_dist = np.linalg.norm(source_pos[:2] - target_pos[:2])
    
    # Build reward
    reward = 0
    
    # Penalty for moving the wrong object (yellow cube)
    if moved_wrong_obj:
        reward -= 5.0
    
    if not is_grasped:
        # Phase 1: Encourage approaching and grasping green cube
        reward = -gripper_to_source_dist * 2
    else:
        # Phase 2: Encourage stacking
        reward = 3.0  # Bonus for grasping
        
        # For cubes, xy-alignment is critical
        reward -= source_to_target_xy_dist * 4  # Higher penalty for misalignment
        
        if consecutive_grasp:
            reward += 1.0  # Additional reward for stable grasp
            
        if src_on_target:
            reward += 10.0  # Large bonus for successful stacking
    
    return reward
        


def reward_for_put_spoon_on_tablecloth(state, action=None):
    """Reward function for putting a spoon on a tablecloth."""
    # Note: For spoon on tablecloth, the evaluate uses success_require_src_completely_on_target=False
    eval_results = state.evaluate(success_require_src_completely_on_target=False)
    
    # Extract important information
    is_grasped = eval_results['is_src_obj_grasped']
    consecutive_grasp = eval_results['consecutive_grasp']
    src_on_target = eval_results['src_on_target']
    
    # Calculate distances
    tcp_pose = state.get_obs()['extra']['tcp_pose']
    gripper_pos = tcp_pose[:3]
    source_pos = state.source_obj_pose.p
    target_pos = state.target_obj_pose.p
    
    gripper_to_source_dist = np.linalg.norm(gripper_pos - source_pos)
    source_to_target_dist = np.linalg.norm(source_pos[:2] - target_pos[:2])  # Only xy distance matters
    
    # Build reward
    reward = 0
    
    if not is_grasped:
        # Phase 1: Encourage approaching and grasping spoon
        reward = -gripper_to_source_dist * 2
    else:
        # Phase 2: Encourage placing on tablecloth
        reward = 2.0  # Bonus for grasping
        
        # For tablecloth, xy-position matters more than z-height
        reward -= source_to_target_dist * 2
        
        if consecutive_grasp:
            reward += 1.0
            
        if src_on_target:
            reward += 8.0  # Bonus for successful placement
    
    return reward
        

def reward_for_put_eggplant_in_basket(state, action=None):
    """Reward function for putting an eggplant in a basket."""
    # Note: For eggplant in basket, evaluate uses success_require_src_completely_on_target=False
    # and z_flag_required_offset=0.06
    eval_results = state.evaluate(success_require_src_completely_on_target=False, 
                                    z_flag_required_offset=0.06)
    
    # Extract important information
    is_grasped = eval_results['is_src_obj_grasped']
    consecutive_grasp = eval_results['consecutive_grasp']
    src_on_target = eval_results['src_on_target']
    
    # Calculate distances
    tcp_pose = state.get_obs()['extra']['tcp_pose']
    gripper_pos = tcp_pose[:3]
    source_pos = state.source_obj_pose.p
    target_pos = state.target_obj_pose.p
    
    gripper_to_source_dist = np.linalg.norm(gripper_pos - source_pos)
    source_to_target_xy_dist = np.linalg.norm(source_pos[:2] - target_pos[:2])
    
    # For basket, we also care about height (eggplant should be below basket rim)
    height_diff = source_pos[2] - target_pos[2]
    
    # Build reward
    reward = 0
    
    if not is_grasped:
        # Phase 1: Encourage approaching and grasping eggplant
        reward = -gripper_to_source_dist * 2
    else:
        # Phase 2: Encourage putting in basket
        reward = 2.0  # Bonus for grasping
        
        # For basket, xy-alignment and height are important
        reward -= source_to_target_xy_dist * 2
        
        # Encourage lowering the eggplant into the basket
        # If eggplant is above target, give penalty proportional to height
        if height_diff > 0:
            reward -= height_diff * 3
        
        if consecutive_grasp:
            reward += 1.0
            
        if src_on_target:
            reward += 10.0  # Large bonus for successful placement
    
    return reward