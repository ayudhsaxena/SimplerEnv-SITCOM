
import numpy as np
from simpler_env.reward_scripts import calculate_affordance_metrics
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import os
import tempfile
import cv2
import time
import shutil
from pathlib import Path
import numpy as np
import os
import tempfile
import cv2
from simpler_env.reward_scripts import calculate_affordance_metrics
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import uuid
import datetime

def compute_image_metrics(image, target_object, gripper_name, visualize=False, save_metrics=False):
    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        # Convert RGB to BGR for cv2.imwrite
        if image.ndim == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        cv2.imwrite(tmp_path, image_bgr)
    
    try:
        # Calculate affordance metrics for both carrot and plate
        metrics = calculate_affordance_metrics(
            img_path=tmp_path,
            target_object=target_object,
            gripper_name=gripper_name,
            visualize=visualize,
            save_metrics=save_metrics,
        )
        
        return metrics
    except Exception as e:
         # If there's an error, delete the temporary file and return a default reward
        # Save the error image to a specific directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        error_dir = "/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/outputs/error_images"
        os.makedirs(error_dir, exist_ok=True)
        error_filename = f"error_image_{timestamp}_{Path(tmp_path).stem}.png"
        error_path = os.path.join(error_dir, error_filename)
        if os.path.exists(tmp_path):
            shutil.copy(tmp_path, error_path)
        print(f"Error in affordance-based reward calculation for {Path(tmp_path).stem}: {str(e)}")
        return None  # Default negative reward
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compute_trajectory_statistics(image_history, window_sz=5):
    """
    Compute statistics for a trajectory based on image history.
    
    Args:
        image_history: List of images in the trajectory
        
    Returns:
        dict: Dictionary containing trajectory statistics
    """
    window_sz = min(window_sz, len(image_history))
    assert len(image_history) >= window_sz, "Window size must be less than or equal to image history length"
    
    image_history = image_history[-window_sz:]  # Use only the last 'window_sz' images
    carrot_positions = []
    image_metrics = []
    for img in image_history:
        metrics = compute_image_metrics(
            img,
            target_object="carrot",
            gripper_name="gripper",
            visualize=False,
            save_metrics=False
        )
        
        if metrics is None:
            continue
        image_metrics.append(metrics)
        carrot_positions.append(metrics['target_affordance_position'])

    window_sz = min(window_sz, len(carrot_positions))
    if window_sz == 0:
        print(f"Unable to compute imasge metrics for the last {window_sz} images")
        return None
    positions_array = np.array(carrot_positions)
    x_variance = np.var(positions_array[-window_sz:, 0])
    y_variance = np.var(positions_array[-window_sz:, 1])
    total_variance = x_variance + y_variance

    
    return {
        'carrot_positions': carrot_positions,
        'total_variance': total_variance,
        'target_image_metrics': image_metrics,  
    }
def reward_for_put_carrot_on_plate_with_image(trajectory_images=None):
    """
    Reward function for putting a carrot on a plate using affordance metrics and reward history.
    
    Args:
        state: The environment state
        action: The action taken (optional)
        image: The image directly provided (optional)
        rewards_history: History of rewards with carrot positions and distances
        
    Returns:
        float: The calculated reward
    """
    if trajectory_images is None:
        print("Trajectory images cant be none")
        return None
    stat = compute_trajectory_statistics(trajectory_images,window_sz=5)
    if stat is None:
        print("image metrics for all images are None")
        return None
    carrot_metrics = stat['target_image_metrics'][-1] # take the last image metrics as representative

    # Extract metrics
    carrot_distance = carrot_metrics['normalized_distance']
    carrot_position = carrot_metrics['target_affordance_position']  # Assuming this is available
    

    # Determine if carrot has been picked up based on position history
    use_carrot_in_reward = True  # Default to focusing on carrot
    
    total_variance = stat['total_variance']
    variance_threshold = 100  # This threshold would need tuning
    print("Total variance of carrot positions: ", total_variance)
    print("Carrot positions: ", stat['carrot_positions'])
    if total_variance > variance_threshold and carrot_distance < 0.2: #also want the carrot-to-gripper distance to remain small
        use_carrot_in_reward = False  # Carrot has moved, so focus on plate
    
    # Calculate reward based on whether to use carrot or plate
    reward = 0.0
    plate_metrics = None
    if use_carrot_in_reward:
        # Phase 1: Approaching and grasping carrot
        # Reward for being close to optimal carrot affordance position
        reward = 5.0 * (1.0 - carrot_distance)
        
        
        # Bonus for being very close to optimal position
        if carrot_distance < 0.08:
            reward += 5.0
    else:
        print("USING PLATE IN REWARD")
        # Phase 2: Moving carrot to plate
        # Base reward for having picked up carrot
        for img in reversed(trajectory_images):
            plate_metrics = compute_image_metrics(
                img,
                target_object="plate",
                gripper_name="gripper",
                visualize=False,
                save_metrics=False
            )
            if plate_metrics is not None:
                break
        if plate_metrics is None:
            print("Unable to compute plate metrics for any image")
            return None
        plate_distance = plate_metrics['normalized_distance']
        plate_affordance = plate_metrics['relative_affordance']
        reward = 10
        
        # Reward for being close to optimal plate affordance position
        reward += 5.0 * (1.0 - plate_distance)
        
        # Additional reward for being at high plate affordance value
        reward += 3.0 * plate_affordance
        
        # Bonus for being very close to optimal position
        if plate_distance < 0.1:
            reward += 8.0  # Higher bonus for successful placement
    

    # Store current carrot information in rewards_history
    metrics ={
        'carrot_position': carrot_position,
        'carrot_distance': carrot_distance,
        'reward': reward,
        'use_carrot_in_reward': use_carrot_in_reward,
    }

    if plate_metrics is not None:
        metrics.update({
            'plate_distance': plate_distance,
        })
    
    return metrics

def reward_for_put_carrot_on_plate(state):
    """Reward function for putting a carrot on a plate."""
    
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
    metrics ={
        'carrot_position': source_pos,
        'carrot_distance': gripper_to_source_dist,
        'reward': reward,
        'use_carrot_in_reward': is_grasped,
    }
    
    return metrics
        
    # except Exception as e:
    #     return -10.0  # Default negative reward


def reward_for_stack_green_on_yellow(state):
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
    
    metrics ={
        'obj_position': source_pos,
        'obj_distance': gripper_to_source_dist,
        'reward': reward,
        'use_obj_in_reward': is_grasped,
    }
    
    return metrics
        


def reward_for_put_spoon_on_tablecloth(state):
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
    
    metrics ={
        'obj_position': source_pos,
        'obj_distance': gripper_to_source_dist,
        'reward': reward,
        'use_obj_in_reward': is_grasped,
    }
    
    return metrics
        

def reward_for_put_eggplant_in_basket(state):
    """Reward function for putting an eggplant in a basket."""
    # Note: For eggplant in basket, evaluate uses success_require_src_completely_on_target=False
    # and z_flag_required_offset=0.06
    eval_results = state.evaluate()
    
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
    
    metrics ={
        'obj_position': source_pos,
        'obj_distance': gripper_to_source_dist,
        'reward': reward,
        'use_obj_in_reward': is_grasped,
    }
    
    return metrics