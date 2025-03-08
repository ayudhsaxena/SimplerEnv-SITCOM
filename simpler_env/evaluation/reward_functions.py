def reward_for_put_carrot_on_plate(state, action=None):
    """Reward function for putting a carrot on a plate."""
    try:
        # Get gripper position
        tcp_pose = state.get_obs()['extra']['tcp_pose']
        gripper_pos = tcp_pose[:3]
        
        # Get source and target object positions
        source_pos = state.source_obj_pose.p  # carrot position
        target_pos = state.target_obj_pose.p  # plate position
        
        # Check if source object is grasped
        is_grasped = state.agent.check_grasp(state.episode_source_obj)
        
        # Calculate distances
        gripper_to_source = np.linalg.norm(gripper_pos - source_pos)
        source_to_target = np.linalg.norm(source_pos[:2] - target_pos[:2])  # XY distance
        
        # Calculate height difference for z-positioning
        height_diff = source_pos[2] - target_pos[2]
        
        # Check if carrot is on plate using bounding box and contact information
        on_target = False
        try:
            # Using bounding box approach
            tgt_half_bbox = state.episode_target_obj_bbox_world / 2
            src_half_bbox = state.episode_source_obj_bbox_world / 2
            
            xy_dist = np.linalg.norm(source_pos[:2] - target_pos[:2])
            xy_threshold = np.linalg.norm(tgt_half_bbox[:2]) + 0.003
            
            on_target_xy = xy_dist <= xy_threshold
            on_target_z = (height_diff > 0) and (
                height_diff - tgt_half_bbox[2] - src_half_bbox[2] <= 0.02
            )
            
            on_target = on_target_xy and on_target_z
        except:
            # Fallback to simple distance check
            on_target = source_to_target < 0.1 and height_diff > 0
        
        # Build reward
        reward = 0
        
        if not is_grasped:
            # Phase 1: Reach and grasp carrot
            reward = -gripper_to_source  # Negative distance as reward
        else:
            # Phase 2: Move carrot to plate
            reward = 2.0  # Bonus for grasping
            reward -= source_to_target * 0.5  # Reward for moving toward target
            
            # Height positioning reward
            ideal_height = target_pos[2] + 0.05  # Estimated good height above plate
            height_reward = -abs(source_pos[2] - ideal_height)
            reward += height_reward * 0.3
            
            if on_target:
                reward += 5.0  # Large bonus for successful placement
        
        return reward
        
    except Exception as e:
        return -10.0  # Default penalty for errors

def reward_for_stack_green_on_yellow(state, action=None):
    """Reward function for stacking green cube on yellow cube."""
    try:
        # Get gripper position
        tcp_pose = state.get_obs()['extra']['tcp_pose']
        gripper_pos = tcp_pose[:3]
        
        # Get source and target object positions (green and yellow cubes)
        source_pos = state.source_obj_pose.p  # green cube
        target_pos = state.target_obj_pose.p  # yellow cube
        
        # Check if source object is grasped
        is_grasped = state.agent.check_grasp(state.episode_source_obj)
        
        # Calculate distances
        gripper_to_source = np.linalg.norm(gripper_pos - source_pos)
        source_to_target_xy = np.linalg.norm(source_pos[:2] - target_pos[:2])
        
        # For cubes, precise alignment is important
        # Get bounding box information
        try:
            tgt_half_bbox = state.episode_target_obj_bbox_world / 2
            src_half_bbox = state.episode_source_obj_bbox_world / 2
            
            # Ideal height would be target's top surface
            ideal_height = target_pos[2] + tgt_half_bbox[2] + src_half_bbox[2]
            height_diff = source_pos[2] - target_pos[2]
            
            # Check if green cube is on yellow cube
            xy_aligned = source_to_target_xy <= tgt_half_bbox[0] * 0.8  # Stricter for cubes
            z_aligned = height_diff > tgt_half_bbox[2] * 0.8 and abs(source_pos[2] - ideal_height) < 0.02
            
            on_target = xy_aligned and z_aligned
        except:
            # Fallback
            ideal_height = target_pos[2] + 0.06  # Estimated cube height
            on_target = source_to_target_xy < 0.05 and abs(source_pos[2] - ideal_height) < 0.03
        
        # Build reward
        reward = 0
        
        if not is_grasped:
            # Phase 1: Reach and grasp green cube
            reward = -gripper_to_source
            
            # Avoid knocking down the target cube
            target_moved = np.linalg.norm(target_pos[:2] - state.episode_target_obj_xyz_after_settle[:2]) > 0.03
            if target_moved:
                reward -= 5.0  # Penalty for disturbing target
        else:
            # Phase 2: Stack green cube on yellow cube
            reward = 2.0  # Grasping bonus
            
            # Horizontal alignment reward
            reward -= source_to_target_xy * 2.0  # Stronger penalty for misalignment
            
            # Vertical alignment reward
            height_error = abs(source_pos[2] - ideal_height)
            reward -= height_error * 2.0
            
            if on_target:
                reward += 8.0  # Large bonus for successful stacking
        
        return reward
        
    except Exception as e:
        return -10.0

def reward_for_put_spoon_on_tablecloth(state, action=None):
    """Reward function for putting a spoon on a tablecloth."""
    try:
        # Get gripper position
        tcp_pose = state.get_obs()['extra']['tcp_pose']
        gripper_pos = tcp_pose[:3]
        
        # Get source and target positions
        source_pos = state.source_obj_pose.p  # spoon
        target_pos = state.target_obj_pose.p  # tablecloth
        
        # Check if spoon is grasped
        is_grasped = state.agent.check_grasp(state.episode_source_obj)
        
        # Calculate distances
        gripper_to_source = np.linalg.norm(gripper_pos - source_pos)
        source_to_target = np.linalg.norm(source_pos[:2] - target_pos[:2])  # XY distance
        
        # Get bounding box info
        try:
            tgt_half_bbox = state.episode_target_obj_bbox_world / 2
            src_half_bbox = state.episode_source_obj_bbox_world / 2
            
            # For tablecloth, we care mainly about xy position and minimal z height
            # The spoon should be on the tablecloth (within xy bounds and touching)
            on_target_xy = source_to_target <= np.linalg.norm(tgt_half_bbox[:2])
            
            # Check if spoon is touching tablecloth through contacts
            contacts = state._scene.get_contacts()
            spoon_touching_cloth = False
            
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                if ((actor_0.name == state.episode_source_obj.name and 
                     actor_1.name == state.episode_target_obj.name) or
                    (actor_1.name == state.episode_source_obj.name and 
                     actor_0.name == state.episode_target_obj.name)):
                    spoon_touching_cloth = True
                    break
            
            on_target = on_target_xy and spoon_touching_cloth
        except:
            # Fallback to simple distance check
            on_target = source_to_target < 0.15 and abs(source_pos[2] - target_pos[2]) < 0.05
        
        # Build reward
        reward = 0
        
        if not is_grasped:
            # Phase 1: Reach and grasp spoon
            reward = -gripper_to_source
        else:
            # Phase 2: Move spoon to tablecloth
            reward = 2.0  # Grasping bonus
            
            # Position reward
            reward -= source_to_target * 0.7
            
            # Height reward - encourage placing, not hovering
            height_error = max(0, source_pos[2] - (target_pos[2] + 0.03))
            reward -= height_error * 1.5
            
            if on_target:
                reward += 5.0  # Success bonus
        
        return reward
        
    except Exception as e:
        return -10.0

def reward_for_put_eggplant_in_basket(state, action=None):
    """Reward function for putting an eggplant in a basket."""
    try:
        # Get gripper position
        tcp_pose = state.get_obs()['extra']['tcp_pose']
        gripper_pos = tcp_pose[:3]
        
        # Get source and target positions
        source_pos = state.source_obj_pose.p  # eggplant
        target_pos = state.target_obj_pose.p  # basket
        
        # Check if eggplant is grasped
        is_grasped = state.agent.check_grasp(state.episode_source_obj)
        
        # Calculate distances
        gripper_to_source = np.linalg.norm(gripper_pos - source_pos)
        source_to_target_xy = np.linalg.norm(source_pos[:2] - target_pos[:2])
        
        # For basket, we want object to be inside (xy position and lower z than rim)
        try:
            tgt_half_bbox = state.episode_target_obj_bbox_world / 2
            
            # Check xy alignment (within basket bounds)
            xy_aligned = source_to_target_xy <= tgt_half_bbox[0] * 0.8
            
            # For basket, object should be inside (slightly lower z than the top)
            basket_top_height = target_pos[2] + tgt_half_bbox[2]
            in_basket_z = source_pos[2] < basket_top_height
            
            # Check if eggplant is in basket via contacts
            contacts = state._scene.get_contacts()
            touching_basket = False
            touching_other_objects = False
            
            robot_link_names = [x.name for x in state.agent.robot.get_links()]
            
            for contact in contacts:
                actor_0, actor_1 = contact.actor0, contact.actor1
                
                # Check if eggplant is touching basket
                if ((actor_0.name == state.episode_source_obj.name and 
                     actor_1.name == state.episode_target_obj.name) or
                    (actor_1.name == state.episode_source_obj.name and 
                     actor_0.name == state.episode_target_obj.name)):
                    touching_basket = True
                
                # Check if eggplant is touching other objects
                if actor_0.name == state.episode_source_obj.name and actor_1.name not in [state.episode_target_obj.name] + robot_link_names:
                    touching_other_objects = True
                elif actor_1.name == state.episode_source_obj.name and actor_0.name not in [state.episode_target_obj.name] + robot_link_names:
                    touching_other_objects = True
            
            in_basket = xy_aligned and in_basket_z and touching_basket and not touching_other_objects
        except:
            # Fallback
            in_basket = source_to_target_xy < 0.1 and source_pos[2] < target_pos[2] + 0.1
        
        # Build reward
        reward = 0
        
        if not is_grasped:
            # Phase 1: Reach and grasp eggplant
            reward = -gripper_to_source
        else:
            # Phase 2: Put eggplant in basket
            reward = 2.0  # Grasping bonus
            
            # Position reward - getting to basket center
            reward -= source_to_target_xy * 0.8
            
            # Height reward - encourage lowering into basket
            basket_center_height = target_pos[2]  # Approximate basket center height
            height_above_center = max(0, source_pos[2] - basket_center_height)
            reward -= height_above_center * 0.5
            
            if in_basket:
                reward += 6.0  # Large bonus for successful placement in basket
        
        return reward
        
    except Exception as e:
        return -10.0