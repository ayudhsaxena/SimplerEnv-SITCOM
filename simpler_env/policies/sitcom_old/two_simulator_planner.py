from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from simpler_env.policies.sitcom.simulation_node import SimulationNode
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
from PIL import Image


from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)

# import re
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import os
# import uuid
# import torch

class TwoSimulatorPlanner:
    """Planning algorithm using two simulators."""
    
    def __init__(
        self,
        env_name,
        saved_model_path: str = "openvla/openvla-7b",
        reward_function=None,
        num_initial_actions=10,  # A parameter
        horizon_per_action=5,    # Horizon parameter
        num_steps_ahead=3,       # h parameter
        num_candidates=5,        # Number of candidate actions to sample
        num_best_actions=3,      # Number of best actions to select
        temperature=1.0,         # Temperature for sampling
        render_tree=False,       # Whether to render the tree
        logging_dir="./results/planning",
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        verbose=False,           # Control logging verbosity
    ):
        """
        Initialize the planner.
        
        Args:
            saved_model_path: Path to the OpenVLA model
            reward_function: Function to compute reward (state, action) -> reward
            num_initial_actions: Number of initial actions to sample (A)
            horizon_per_action: Number of actions to consider for each state (Horizon)
            num_steps_ahead: Number of simulation steps to look ahead (h)
            num_candidates: Number of candidate actions to sample
            num_best_actions: Number of best actions to select
            temperature: Temperature for sampling
            render_tree: Whether to render the tree
            logging_dir: Directory for logging
            policy_setup: Policy setup for the OpenVLA model
            action_scale: Scaling factor for actions
        """
        self.env_name = env_name
        # Initialize the OpenVLA model for action sampling
        self.model = OpenVLAInference(
            saved_model_path=saved_model_path, 
            policy_setup=policy_setup, 
            action_scale=action_scale,
            unnorm_key='simpler_rlds',
            # unnorm_key='bridge_orig',
        )
        
        
        # Set up reward function or use default
        if reward_function is None:
            self.reward_function = self._default_reward_function
        else:
            self.reward_function = reward_function
        
        # Hyperparameters
        self.num_initial_actions = num_initial_actions  # A
        self.horizon_per_action = horizon_per_action    # Horizon
        self.num_steps_ahead = num_steps_ahead          # h
        self.num_candidates = num_candidates
        self.num_best_actions = num_best_actions
        self.temperature = temperature
        
        # Visualization settings
        self.render_tree = render_tree
        self.logging_dir = logging_dir
        self.verbose = verbose
        
        # Task description
        self.task_description = None
        
        # Reset internal state
        self.reset()
        
    
         # For better performance with multi-image comparisons, you might want to use:
        # self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2-VL-7B-Instruct",
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        #     device_map="auto"
        # )
        
        # self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Initialized with parameters:")
            print(f"  - Model: {saved_model_path}")
            print(f"  - Initial actions (A): {num_initial_actions}")
            print(f"  - Horizon per action: {horizon_per_action}")
            print(f"  - Steps ahead (h): {num_steps_ahead}")
            print(f"  - Candidates: {num_candidates}")
            print(f"  - Best actions: {num_best_actions}")
            print(f"  - Temperature: {temperature}")
            print(f"  - Policy setup: {policy_setup}")
            print(f"  - Action scale: {action_scale}")
    
    def _default_reward_function(self, state, action=None):
        """
        Default reward function based on distances between objects.
        
        Args:
            state: The environment state
            action: The action (optional)
            
        Returns:
            reward: The computed reward
        """
        # This is a simplified example. Real implementations would extract positions from state
        # For example from robot state, object positions, etc.
        
        # Extract positions (implementation depends on environment)
        try:
            # Get positions from environment
            gripper_pos = np.array([0, 0, 0])  # Placeholder, replace with actual implementation
            object_pos = np.array([0, 0, 0])   # Placeholder, replace with actual implementation
            plate_pos = np.array([0, 0, 0])    # Placeholder, replace with actual implementation
            
            # Check if object is grabbed
            is_grabbed = False  # Placeholder, replace with actual implementation
            
            # Calculate distance reward
            if is_grabbed:
                # If object is grabbed, reward is based on distance to target
                distance = np.linalg.norm(gripper_pos - plate_pos)
            else:
                # If object is not grabbed, reward is based on distance to object
                distance = np.linalg.norm(gripper_pos - object_pos)
            
            # Convert distance to reward (closer is better)
            reward = -distance
            
            return reward
        except:
            # If we can't compute the reward, return a default value
            return 0.0
    
    def reset(self, task_description=None):
        """
        Reset the planner.
        
        Args:
            task_description: Optional task description
        """
        self.simulation_tree = None
        self.best_trajectory = None
        self.best_reward = float('-inf')
        
        if task_description is not None:
            self.task_description = task_description
            self.model.reset(task_description)
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Reset with task: {task_description}")
                
    def get_to_state(self, env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs):
        env = build_maniskill2_env(
            env_name,
            **additional_env_build_kwargs,
            **kwargs,
        )
        obs, _ = env.reset(options=env_reset_options)
        
        start_image = get_image_from_maniskill2_obs_dict(env, obs)
        
        for action in action_list:
            obs, reward, done, truncated, info = env.step(
                np.concatenate(
                    [action["world_vector"], action["rot_axangle"], action["gripper"]]
                ),
            )
        
        return env, obs, start_image
        

    def copy_state(self, state, kwargs, additional_env_build_kwargs):
        if self.verbose:
            print("[TwoSimulatorPlanner] Creating environment copy")

        episode_id = None
        if hasattr(state, 'episode_id'):
            episode_id = state.episode_id
        
        # Create a new environment instance with the same configuration
        from simpler_env.utils.env.env_builder import build_maniskill2_env
        
        # Create the new environment instance
        state_copy = build_maniskill2_env(
            self.env_name,
             **additional_env_build_kwargs,
            **kwargs,
        )
        
        # Reset with the same options/state
        reset_options = {
            'seed': state._episode_seed if hasattr(state, '_episode_seed') else None,
            'options': {
                'episode_id': episode_id
            }
        }
        
        # Reset the environment
        state_copy.reset(**reset_options)
        
        # If the environment supports state saving/loading
        if hasattr(state, 'get_state') and hasattr(state_copy, 'set_state'):
            current_state = state.get_state()
            state_copy.set_state(current_state)
    
        return state_copy

    def sample_actions_from_model(self, image, task_description, num_samples, temperature=None):
        """
        Sample actions from the model.
        
        Args:
            image: The current image observation
            task_description: The task description
            num_samples: Number of actions to sample
            temperature: Temperature for sampling (override default if provided)
            
        Returns:
            List of sampled actions
        """
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Sampling {num_samples} actions")
        
        actions = []
        temperature = temperature if temperature is not None else self.temperature
        
        # Sample actions from the model
        for i in range(num_samples):
            if self.verbose:
                print(f"  - Sampling action {i+1}/{num_samples} (temp={temperature:.2f})")
            
            raw_action, action = self.model.step(
                image, 
                task_description, 
                temperature=temperature
            )
            actions.append(action)
            
            if self.verbose:
                print(f"    - Action vector: {action['world_vector']}")
                print(f"    - Rotation: {action['rot_axangle']}")
                print(f"    - Gripper: {action['gripper']}")
        
        return actions
    
    def simulate_action(self, state, action, kwargs, additional_env_build_kwargs):
        """
        Simulate an action using the second simulator.
        
        Args:
            state: The current state
            action: The action to simulate
            
        Returns:
            next_state, reward, image, done
        """
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Simulating action in second simulator:")
            print(f"  - World vector: {action['world_vector']}")
            print(f"  - Rotation: {action['rot_axangle']}")
            print(f"  - Gripper: {action['gripper']}")
        
        # Create a copy of the state to avoid modifying the original
        # state_copy = self.copy_state(state, kwargs, additional_env_build_kwargs)
        
        # Simulate the action using the model (second simulator)
        # For ManiSkill2, we need to concatenate action components
        action_array = np.concatenate([
            action["world_vector"], 
            action["rot_axangle"], 
            action["gripper"]
        ])
        
        # Step the environment with the action
        obs, reward, done, truncated, info = state.step(action_array)
        
        if self.verbose:
            print(f"  - Reward: {reward}")
            print(f"  - Done: {done}")
            if done:
                print(f"  - Task completed in simulation!")
        
        # Extract the image from the observation
        image = get_image_from_maniskill2_obs_dict(state, obs)
        
        return state, obs, reward, image, done
    
    def compute_reward(self, state, action=None):
        """
        Compute reward for a state-action pair.
        
        Args:
            state: The current state
            action: The action (optional)
            
        Returns:
            reward: The computed reward
        """
        return self.reward_function(state, action)
    
    def _save_debug_images(self, start_image, image_list):
        """Save start image and image list for debugging.
        
        Args:
            start_image: Starting state image
            image_list: List of images to evaluate
        """
        
        
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Save start image
        self._save_single_image(start_image, os.path.join(debug_dir, "start_image.jpg"))
        
        # Save each image in the image list
        for i, img in enumerate(image_list):
            self._save_single_image(img, os.path.join(debug_dir, f"candidate_image_{i}.jpg"))
                
        print(f"Debug images saved to: {debug_dir}")

    def _save_single_image(self, img, save_path):
        """Save a single image to disk.
        
        Handles different image formats (PIL.Image, numpy array, file path, etc.)
        """
        try:
            # If it's already a PIL Image
            if hasattr(img, 'save'):
                img.save(save_path)
                return
            
            # If it's a file path
            if isinstance(img, str) and (img.startswith('file:///') or os.path.exists(img)):
                img_path = img.replace('file:///', '')
                if os.path.exists(img_path):
                    Image.open(img_path).save(save_path)
                    return
            
            # If it's a numpy array
            if isinstance(img, np.ndarray):
                Image.fromarray(img).save(save_path)
                return
            
            # If it's a PyTorch tensor
            try:
                if isinstance(img, torch.Tensor):
                    # Convert tensor to numpy and then to PIL Image
                    array = img.cpu().numpy()
                    # Handle different tensor formats
                    if array.ndim == 4:  # [batch, channels, height, width]
                        array = array[0].transpose(1, 2, 0)  # -> [height, width, channels]
                    elif array.ndim == 3 and array.shape[0] in [1, 3, 4]:  # [channels, height, width]
                        array = array.transpose(1, 2, 0)  # -> [height, width, channels]
                    
                    # Normalize if needed
                    if array.max() <= 1.0:
                        array = (array * 255).astype(np.uint8)
                    
                    Image.fromarray(array).save(save_path)
                    return
            except ImportError:
                pass
            
            # If we get here, we couldn't determine the image type
            print(f"Warning: Could not save image to {save_path}. Unknown image type.")
            
        except Exception as e:
            print(f"Error saving image to {save_path}: {str(e)}")

    def find_best_image(self, task, image_list, start_image):
        """Returns the index of the best image in the list based on the task.
        
        Args:
            task: The task description
            image_list: List of images to evaluate
            
        Returns:
            tuple: (best_image_index, best_image_reward)
        """
        # save start image and image list 
        self._save_debug_images(start_image, image_list)
        
        
        if not image_list:
            return None, 0
            
        if len(image_list) == 1:
            # If there's only one image, evaluate it against the task
            reward = self._evaluate_single_image(task, image_list[0], start_image)
            return 0, reward
        
        # Tournament approach
        current_winners = list(range(len(image_list)))
        current_rewards = [0] * len(image_list)
        
        # Continue until we have only one winner
        while len(current_winners) > 1:
            next_winners = []
            next_rewards = []
            
            # Create pairs and compare them
            i = 0
            while i < len(current_winners) - 1:
                idx1 = current_winners[i]
                idx2 = current_winners[i+1]
                winner_idx, winner_reward = self._compare_images(
                    task, 
                    image_list[idx1], 
                    image_list[idx2],
                    start_image
                )
                
                print(f"Comparing images {idx1} and {idx2}: Winner={winner_idx}, Reward={winner_reward}")
                
                # Add the winner to the next round
                next_winners.append(current_winners[winner_idx])
                next_rewards.append(winner_reward)
                i += 2
                
            # If we have an odd number of images, the last one advances automatically
            if i == len(current_winners) - 1:
                next_winners.append(current_winners[i])
                next_rewards.append(current_rewards[i])
                
            current_winners = next_winners
            current_rewards = next_rewards
        
        # breakpoint()
        
        # Return the final winner
        return current_winners[0], current_rewards[0]

    def _compare_images(self, task, image1, image2, start_image):
        """Compare two images and return the index of the better one (0 or 1) and its reward.
        
        Args:
            task: The task description
            image1: First image to evaluate
            image2: Second image to evaluate
            start_image: Starting state image for reference
            
        Returns:
            tuple: (winner_index (0 or 1), winner_reward)
        """
        # Construct a prompt for Qwen VL to compare the two images with context from the start image
        prompt = f"""Task: {task}
        
    You are given three images related to a robotic task:
    - The STARTING IMAGE shows the initial state of the environment.
    - IMAGE 1 and IMAGE 2 show different potential end states of the environment.

    Compare IMAGE 1 and IMAGE 2 and determine which one represents a more successful completion of the task relative to the STARTING IMAGE.

    Return your evaluation as follows:
    1. Score for IMAGE 1 (0-10): [score]
    2. Brief explanation for IMAGE 1: [explanation]
    3. Score for IMAGE 2 (0-10): [score]
    4. Brief explanation for IMAGE 2: [explanation]
    5. Final verdict: The better image is [IMAGE 1 or IMAGE 2]
    """
        
        # Pass all three images to Qwen VL model for comparison
        # Order: starting image first, then the two images to compare
        comparison_result = self._call_qwen_vl(prompt, [start_image, image1, image2])
        
        # Parse the comparison result to determine the winner
        winner_idx, winner_reward = self._parse_comparison_result(comparison_result)
        
        
        return winner_idx, winner_reward

    def _parse_comparison_result(self, comparison_result):
        """Parse the result from Qwen VL model to determine which image is better.
        
        Returns:
            tuple: (winner_index (0 or 1), winner_reward)
        """
        # Implement parsing logic based on your Qwen VL model output format
        # This is a placeholder implementation
        if "IMAGE 1" in comparison_result and "IMAGE 2" in comparison_result:
            # Extract scores
            try:
                # Look for patterns like "Score for Image 1: 8/10"
                score1 = float(re.search(r'Image 1:?\s*(\d+(?:\.\d+)?)', comparison_result, re.IGNORECASE).group(1))
                score2 = float(re.search(r'Image 2:?\s*(\d+(?:\.\d+)?)', comparison_result, re.IGNORECASE).group(1))
                
                if score1 > score2:
                    return 0, score1
                else:
                    return 1, score2
            except (AttributeError, ValueError):
                # If score extraction fails, fall back to keyword analysis
                image1_keywords = ["image 1", "first image", "image #1"]
                image2_keywords = ["image 2", "second image", "image #2"]
                
                for keyword in image1_keywords:
                    if f"better is {keyword}" in comparison_result.lower():
                        return 0, 1.0
                        
                for keyword in image2_keywords:
                    if f"better is {keyword}" in comparison_result.lower():
                        return 1, 1.0
                
                # If explicit winner isn't found, default to the first image
                return 0, 0.5
        
        # Default return if parsing fails
        return 0, 0.5

    def _evaluate_single_image(self, task, image, start_image):
        """Evaluate a single image against the task description and starting image.
        
        Args:
            task: The task description
            image: Image to evaluate
            start_image: Starting state image for reference
            
        Returns:
            float: Reward score for the image
        """
        prompt = f"""Task: {task}
        
    You are given two images related to a robotic task:
    - The STARTING IMAGE shows the initial state of the environment.
    - The RESULT IMAGE shows a potential end state of the environment.

    Evaluate how successfully the RESULT IMAGE represents completion of the task relative to the STARTING IMAGE.

    Return your evaluation as follows:
    1. Score (0-10): [score]
    2. Explanation: [detailed reasoning for your score, mentioning specific aspects of task completion]
    """
        
        # Pass both the starting image and the image to evaluate
        result = self._call_qwen_vl(prompt, [start_image, image])
        
        # Parse the result to extract the score
        try:
            # Look for patterns like "Score: 8" or "Score (0-10): 8"
            score = float(re.search(r'Score(?:\s+\(0-10\))?:\s*(\d+(?:\.\d+)?)', result, re.IGNORECASE).group(1))
            return score
        except (AttributeError, ValueError):
            # If that pattern fails, try looking for a score out of 10
            try:
                score = float(re.search(r'(\d+(?:\.\d+)?)/10', result).group(1))
                return score
            except (AttributeError, ValueError):
                # Default score if parsing fails
                return 5.0

    def _call_qwen_vl(self, prompt, images):
        """Call Qwen VL model with the given prompt and images.
        
        Args:
            prompt (str): The text prompt to send to the model
            images (list): List of image objects/paths to evaluate
            
        Returns:
            str: The model's response
        """
        
        import tempfile
        
        # Create a temporary directory for storing image files
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        
        # Prepare the messages in the format expected by Qwen
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # Add all images to the message content
        for i, image in enumerate(images):
            # Handle different image formats
            if isinstance(image, str) and (image.startswith('file:///') or os.path.exists(image)):
                # It's already a file path
                if image.startswith('file:///'):
                    image_path = image
                else:
                    image_path = f"file:///{os.path.abspath(image)}"
            else:
                # Convert numpy array or other formats to PIL Image and save
                temp_file = os.path.join(temp_dir, f"temp_image_{i}_{uuid.uuid4()}.jpg")
                
                if hasattr(image, 'save'):  # PIL Image
                    image.save(temp_file)
                elif 'numpy' in str(type(image)):  # NumPy array
                    Image.fromarray(image).save(temp_file)
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                    
                image_path = f"file:///{os.path.abspath(temp_file)}"
                temp_files.append(temp_file)
            
            # Add to message content
            messages[0]["content"].append({"type": "image", "image": image_path})
        
        # Add the prompt text
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        try:
            # Preparation for inference
            text = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            
            # Process vision information (this function expects file paths)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Create inputs for the model
            inputs = self.qwen_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to the same device as the model
            device = next(self.qwen_model.parameters()).device
            inputs = inputs.to(device)
            
            # Inference: Generate response
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=256)
            
            # Trim the input tokens to get only the generated output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the generated tokens to text
            output_text = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Return the first (and only) generated response
            return output_text[0]
        
        finally:
            # Clean up temporary files
            for file_path in temp_files:
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {file_path}: {e}")
            
            # Remove the temporary directory
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory {temp_dir}: {e}")
    
    def select_best_actions(self, state, candidate_actions, num_best, kwargs, additional_env_build_kwargs):
        """
        Select the best actions based on the reward function.
        
        Args:
            state: The current state
            candidate_actions: List of candidate actions
            num_best: Number of best actions to select
            
        Returns:
            best_actions: List of best actions
        """
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Selecting best {num_best} actions from {len(candidate_actions)} candidates")
        
        # Compute rewards for all candidate actions
        action_rewards = []
        
        for i, action in enumerate(candidate_actions):
            if self.verbose:
                print(f"  - Evaluating action {i+1}/{len(candidate_actions)}")
            
            # Simulate the action to get the next state
            next_state, _, _, _, _ = self.simulate_action(state, action, kwargs, additional_env_build_kwargs)
            
            # Compute the reward for the resulting state
            reward = self.compute_reward(next_state)
            
            if self.verbose:
                print(f"    - Reward: {reward}")
            
            # Store the action and its reward
            action_rewards.append((action, reward))
        
        # Sort by reward (descending) and select the best actions
        action_rewards.sort(key=lambda x: x[1], reverse=True)
        best_actions = [action for action, _ in action_rewards[:num_best]]
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Selected {len(best_actions)} best actions:")
            for i, (action, reward) in enumerate(action_rewards[:num_best]):
                print(f"  - Action {i+1}: reward={reward}")
        
        return best_actions
    
    def build_simulation_tree(self, root_state, root_image, task_description, kwargs, additional_env_build_kwargs):
        """
        Build a simulation tree by exploring possible actions.
        
        Args:
            root_state: The initial state
            root_image: The initial image
            task_description: The task description
            
        Returns:
            best_action: The best action to take
        """
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Building simulation tree for task: {task_description}")
            print(f"  - Sampling {self.num_initial_actions} initial actions (A parameter)")
        
        # Create the root node
        root_node = SimulationNode(root_state, root_image)
        
        # Sample initial actions (A = num_initial_actions)
        initial_actions = self.sample_actions_from_model(
            root_image, 
            task_description, 
            self.num_initial_actions,
            temperature=self.temperature
        )
        
        best_leaf_node = None
        best_reward = float('-inf')
        
        # For each initial action, simulate and build a subtree
        for i, action in enumerate(initial_actions):
            if self.verbose:
                print(f"\n[TwoSimulatorPlanner] Exploring initial action {i+1}/{len(initial_actions)}")
            
            # Simulate the action to get the next state
            next_state, obs, reward, next_image, done = self.simulate_action(root_state, action, kwargs, additional_env_build_kwargs)
            
            # Create a child node
            child_node = SimulationNode(
                next_state, 
                next_image, 
                parent=root_node, 
                action=action, 
                reward=reward,
                depth=1
            )
            child_node.original_action_idx = i  # Keep track of which initial action this is
            root_node.add_child(child_node)
            
            if self.verbose:
                print(f"  - Initial simulation reward: {reward}")
                print(f"  - Task completed: {done}")
            
            # Explore this subtree further if not done
            if not done:
                if self.verbose:
                    print(f"  - Looking ahead {self.num_steps_ahead} steps")
                
                # Perform look-ahead simulation (h = num_steps_ahead)
                leaf_node = self._simulate_ahead(
                    child_node, 
                    task_description,
                    kwargs,
                    additional_env_build_kwargs ,
                    current_depth=1
                )
                
                if self.verbose:
                    print(f"  - Best leaf node reward: {leaf_node.reward}")
                
                # Update best leaf node if better reward
                if leaf_node.reward > best_reward:
                    best_reward = leaf_node.reward
                    best_leaf_node = leaf_node
                    
                    if self.verbose:
                        print(f"  - New best reward found: {best_reward}")
            
            # If done but reward is better than current best
            elif child_node.reward > best_reward:
                best_reward = child_node.reward
                best_leaf_node = child_node
                
                if self.verbose:
                    print(f"  - Task completed with reward: {best_reward}")
        
        # Store the tree and best trajectory
        self.simulation_tree = root_node
        self.best_reward = best_reward
        
        # Backtrack to find the best initial action
        if best_leaf_node:
            self.best_trajectory = self._backtrack_to_root(best_leaf_node)
            best_initial_action = initial_actions[best_leaf_node.original_action_idx]
            
            if self.verbose:
                print(f"\n[TwoSimulatorPlanner] Selected best initial action with index {best_leaf_node.original_action_idx}")
                print(f"  - Best reward: {best_reward}")
                print(f"  - World vector: {best_initial_action['world_vector']}")
                print(f"  - Rotation: {best_initial_action['rot_axangle']}")
                print(f"  - Gripper: {best_initial_action['gripper']}")
            
            return best_initial_action
        
        # Fallback to the first action if no simulation was successful
        if self.verbose:
            print(f"\n[TwoSimulatorPlanner] No good action found, falling back to first action")
        
        return initial_actions[0] if initial_actions else None
    
    def _simulate_ahead(self, node, task_description, kwargs, additional_env_build_kwargs, current_depth=1):
        """
        Recursively simulate ahead from a node.
        
        Args:
            node: The current node
            task_description: The task description
            current_depth: Current depth in the tree
            
        Returns:
            best_leaf: The best leaf node in this subtree
        """
        # If we've reached the maximum depth or this is a terminal state, return this node
        if current_depth >= self.num_steps_ahead or node.reward == float('inf'):
            if self.verbose:
                print(f"    [Depth {current_depth}] Reached max depth or terminal state, stopping")
            return node
        
        if self.verbose:
            print(f"    [Depth {current_depth}] Simulating ahead, sampling {self.num_candidates} candidates")
        
        # Sample candidate actions from this state (typically more than we'll use)
        candidate_actions = self.sample_actions_from_model(
            node.image, 
            task_description, 
            self.num_candidates
        )
        
        # Select the best candidate actions
        best_actions = self.select_best_actions(
            node.state, 
            candidate_actions, 
            self.num_best_actions, kwargs, additional_env_build_kwargs
        )
        
        best_leaf = node
        best_reward = node.reward
        
        # Explore each of the best actions
        for i, action in enumerate(best_actions):
            if self.verbose:
                print(f"    [Depth {current_depth}] Exploring action {i+1}/{len(best_actions)}")
            
            # Simulate the action
            next_state, obs,  reward, next_image, done = self.simulate_action(node.state, action, kwargs, additional_env_build_kwargs)
            
            if self.verbose:
                print(f"      - Step reward: {reward}")
                print(f"      - Cumulative reward: {node.reward + reward}")
                print(f"      - Done: {done}")
            
            # Create a child node with cumulative reward
            child_node = SimulationNode(
                next_state, 
                next_image, 
                parent=node, 
                action=action, 
                reward=node.reward + reward,  # Accumulate rewards along the path
                depth=current_depth + 1
            )
            child_node.original_action_idx = node.original_action_idx  # Propagate original action index
            node.add_child(child_node)
            
            # Continue simulation if not done
            if not done:
                leaf_node = self._simulate_ahead(
                    child_node, 
                    task_description, 
                    kwargs, additional_env_build_kwargs,
                    current_depth + 1
                )
                
                # Update best leaf if needed
                if leaf_node.reward > best_reward:
                    if self.verbose:
                        print(f"      - New best leaf found with reward: {leaf_node.reward}")
                    best_reward = leaf_node.reward
                    best_leaf = leaf_node
            
            # If done but reward is better than current best
            elif child_node.reward > best_reward:
                if self.verbose:
                    print(f"      - Task completed with better reward: {child_node.reward}")
                best_reward = child_node.reward
                best_leaf = child_node
        
        if self.verbose:
            print(f"    [Depth {current_depth}] Best reward in subtree: {best_reward}")
        
        return best_leaf
    
    def _backtrack_to_root(self, node):
        """
        Backtrack from a leaf node to the root to find the trajectory.
        
        Args:
            node: The leaf node
            
        Returns:
            trajectory: List of (state, action) pairs from root to leaf
        """
        trajectory = []
        current = node
        
        # Traverse up the tree from leaf to root
        while current.parent:
            trajectory.append((current.parent.state, current.action))
            current = current.parent
        
        # Reverse to get from root to leaf
        trajectory.reverse()
        return trajectory
    
    def plan(self, env, image, task_description, kwargs, additional_env_build_kwargs):
        """
        Plan the best action to take from the current state.
        
        Args:
            env: The current environment state (first simulator)
            image: The current image observation
            task_description: Optional updated task description
            
        Returns:
            best_action: The best action to take
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting planning process")
            print("="*80)
        
        # Update task description if provided
        if task_description is not None:
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Updated task description: {task_description}")
            self.task_description = task_description
            self.model.reset(task_description)
        
        # Record time for performance monitoring
        import time
        start_time = time.time()


        # print('env plan = ', env_name)
        
        
        # print('env plan = ', env_name)
        
        # Build simulation tree and get the best action
        best_action = self.build_simulation_tree(env, image, self.task_description, kwargs, additional_env_build_kwargs)
        
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Selected action:")
            print(f"  - World vector: {best_action['world_vector']}")
            print(f"  - Rotation: {best_action['rot_axangle']}")
            print(f"  - Gripper: {best_action['gripper']}")
            print("="*80 + "\n")
        
        return best_action
    
    def plan_trajectory(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs, trajectory_length=10, num_trajectories=5):
        """
        Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
        Args:
            env: The current environment state (first simulator)
            image: The current image observation
            task_description: Optional updated task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_trajectories: Number of different trajectories to evaluate
            
        Returns:
            best_trajectory: List of actions forming the best trajectory
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
            print(f"[TwoSimulatorPlanner] Evaluating {num_trajectories} trajectories of length {trajectory_length}")
            print("="*80)
        
        # Update task description if provided
        if task_description is not None:
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Updated task description: {task_description}")
            self.task_description = task_description
            self.model.reset(task_description)
        
        # Record time for performance monitoring
        import time
        start_time = time.time()
        
        # Track best trajectory and its final reward
        best_trajectory = None
        best_final_reward = float('-inf')
        
        # Generate and evaluate multiple trajectories
        for traj_idx in range(num_trajectories):
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{num_trajectories}")
            
            # Initialize trajectory for this run
            trajectory_actions = []
            current_env, current_obs, _ = self.get_to_state(env_name, action_list,env_reset_options, kwargs, additional_env_build_kwargs)
            # current_env = self.copy_state(env_name, kwargs, additional_env_build_kwargs)
            # current_env = 
            current_image = image
            
            # Generate a sequence of actions for this trajectory
            for step in range(trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{trajectory_length}")
                
                # Sample actions from the model with temperature to ensure diversity
                actions = self.sample_actions_from_model(
                    current_image, 
                    self.task_description, 
                    num_samples=1,  # Just sample one action at a time for the trajectory
                    temperature=self.temperature
                )
                
                if not actions:
                    if self.verbose:
                        print("  - Failed to sample actions, ending trajectory early")
                    break
                    
                action = actions[0]
                trajectory_actions.append(action)
                
                # Simulate this action to update the environment state for the next step
                current_env, current_obs, reward, current_image, done = self.simulate_action(
                    current_env, action, kwargs, additional_env_build_kwargs
                )
                
                computed_reward = self.compute_reward(current_env)
                # print(f"  - Simulated step reward: {computed_reward}")
    
                
                if self.verbose:
                    print(f"  - Simulated step reward: {reward}")
                
                # If task is done, we can stop this trajectory
                if done:
                    if self.verbose:
                        print(f"  - Task completed after {step+1} steps")
                    break
            
            # Compute final reward for this trajectory
            final_reward = self.compute_reward(current_env)
            
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Trajectory {traj_idx+1} final reward: {final_reward}")
            
            # print reward for the trajectory
            print(f"Trajectory {traj_idx+1} final reward: {final_reward}")
            # Update best trajectory if this one is better
            if final_reward > best_final_reward:
                best_final_reward = final_reward
                best_trajectory = trajectory_actions
                
                if self.verbose:
                    print(f"[TwoSimulatorPlanner] New best trajectory found with reward: {final_reward}")
        
        print(f"Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
            print("="*80 + "\n")
        
        return best_trajectory, best_final_reward
        
        return best_trajectory or []  # Return empty list if no trajectory was found
    
    def plan_trajectory_reward_qwen(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs, trajectory_length=10, num_trajectories=5):
        """
        Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
        Args:
            env: The current environment state (first simulator)
            image: The current image observation
            task_description: Optional updated task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_trajectories: Number of different trajectories to evaluate
            
        Returns:
            best_trajectory: List of actions forming the best trajectory
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
            print(f"[TwoSimulatorPlanner] Evaluating {num_trajectories} trajectories of length {trajectory_length}")
            print("="*80)
        
        # Update task description if provided
        if task_description is not None:
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Updated task description: {task_description}")
            self.task_description = task_description
            self.model.reset(task_description)
        
        # Record time for performance monitoring
        import time
        start_time = time.time()
        
        
        
        trajectories = []
        final_images = []
        
        # Generate and evaluate multiple trajectories
        for traj_idx in range(num_trajectories):
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{num_trajectories}")
            
            # Initialize trajectory for this run
            trajectory_actions = []
            current_env, current_obs, start_image = self.get_to_state(env_name, action_list,env_reset_options, kwargs, additional_env_build_kwargs)
            # current_env = self.copy_state(env_name, kwargs, additional_env_build_kwargs)
            # current_env = 
            current_image = image
            
            # Generate a sequence of actions for this trajectory
            for step in range(trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{trajectory_length}")
                
                # Sample actions from the model with temperature to ensure diversity
                actions = self.sample_actions_from_model(
                    current_image, 
                    self.task_description, 
                    num_samples=1,  # Just sample one action at a time for the trajectory
                    temperature=self.temperature
                )
                
                if not actions:
                    if self.verbose:
                        print("  - Failed to sample actions, ending trajectory early")
                    break
                    
                action = actions[0]
                trajectory_actions.append(action)
                
                # Simulate this action to update the environment state for the next step
                current_env, current_obs, reward, current_image, done = self.simulate_action(
                    current_env, action, kwargs, additional_env_build_kwargs
                )
                
                # print(f"  - Simulated step reward: {computed_reward}")
    
                
                if self.verbose:
                    print(f"  - Simulated step reward: {reward}")
                
                # If task is done, we can stop this trajectory
                if done:
                    if self.verbose:
                        print(f"  - Task completed after {step+1} steps")
                    break
            
            # Compute final reward for this trajectory
            final_image = get_image_from_maniskill2_obs_dict(current_env, current_obs)
            
            trajectories.append(trajectory_actions)
            final_images.append(final_image)
            
        
        
        best_image_idx, best_image_reward = self.find_best_image(self.task_description, final_images, start_image)
        
        best_trajectory = trajectories[best_image_idx]
        best_final_reward = best_image_reward
        
        
        print(f"Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
            print("="*80 + "\n")
        
        return best_trajectory or []  # Return empty list if no trajectory was found
    

    
    def debug_trajectory_planning(self, env, image, task_description, kwargs, additional_env_build_kwargs, dummy):
        """
        Debug method to test if environment copying and simulation work correctly
        by testing two predefined trajectories.
        """
        
        if self.verbose:
            print("\n" + "="*80)
            print("[DEBUG] Testing environment copying and trajectory simulation")
            print("="*80)
        
        # Create two copies of the environment
        env_good = self.copy_state(env, kwargs, additional_env_build_kwargs)
        env_bad = self.copy_state(env, kwargs, additional_env_build_kwargs)
        
        # Predefined "good" action (customize for your environment)
        good_action = {
            "world_vector": np.array([0.05, 0.0, 0.02]),
            "rot_axangle": np.array([0.0, 0.0, 0.0]),
            "gripper": np.array([1.0])
        }
        
        # Predefined "bad" action (customize for your environment)
        bad_action = {
            "world_vector": np.array([-0.05, 0.0, -0.02]),
            "rot_axangle": np.array([0.3, 0.0, 0.0]),
            "gripper": np.array([0.0])
        }
        
        # Simulate good trajectory
        if self.verbose:
            print("\n[DEBUG] Simulating 'good' trajectory")
        
        good_rewards = []
        current_env = env_good
        current_image = image
        
        for step in range(5):  # 5 steps with the good action
            if self.verbose:
                print(f"  Step {step+1}: Applying good action")
            
            next_env, next_obs, reward, next_image, done = self.simulate_action(
                current_env, good_action, kwargs, additional_env_build_kwargs
            )
            
            good_rewards.append(reward)
            
            if self.verbose:
                print(f"    Reward: {reward}")
                print(f"    Done: {done}")
            
            current_env = next_env
            current_image = next_image
            
            if done:
                break
        
        # Calculate final reward for good trajectory
        good_final_reward = self.compute_reward(current_env)
        
        # Simulate bad trajectory
        if self.verbose:
            print("\n[DEBUG] Simulating 'bad' trajectory")
        
        bad_rewards = []
        current_env = env_bad
        current_image = image
        
        for step in range(5):  # 5 steps with the bad action
            if self.verbose:
                print(f"  Step {step+1}: Applying bad action")
            
            next_env, next_obs, reward, next_image, done = self.simulate_action(
                current_env, bad_action, kwargs, additional_env_build_kwargs
            )
            
            bad_rewards.append(reward)
            
            if self.verbose:
                print(f"    Reward: {reward}")
                print(f"    Done: {done}")
            
            current_env = next_env
            current_image = next_image
            
            if done:
                break
        
        # Calculate final reward for bad trajectory
        bad_final_reward = self.compute_reward(current_env)
        
        # Compare and report results
        # if self.verbose:
        print("\n[DEBUG] Results comparison:")
        print(f"  Good trajectory final reward: {good_final_reward}")
        print(f"  Bad trajectory final reward: {bad_final_reward}")
        print(f"  Difference: {good_final_reward - bad_final_reward}")
        
        if good_final_reward > bad_final_reward:
            print("\n[DEBUG] TEST PASSED: Good trajectory produced better reward")
            print("  Environment copying and simulation appear to be working correctly")
        else:
            print("\n[DEBUG] TEST FAILED: Bad trajectory did not produce better reward")
            print("  This may indicate an issue with environment copying, simulation, or reward function")
        
        return {
            "good_rewards": good_rewards,
            "bad_rewards": bad_rewards,
            "good_final_reward": good_final_reward,
            "bad_final_reward": bad_final_reward,
            "test_passed": good_final_reward > bad_final_reward
        }