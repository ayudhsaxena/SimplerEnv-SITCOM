from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from simpler_env.policies.sitcom.simulation_node import SimulationNode
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
from simpler_env.reward_scripts import calculate_affordance_metrics
import os
import datetime
import tempfile
import cv2
import uuid
import json
import pathlib
import torch
from world_model import DynamicsModel
import time
import matplotlib.pyplot as plt


from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)

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
        )
        
        print("HERE")
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
        self.episode_id = -1
        
        # Reset internal state
        self.reset()
        
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
    
    def reset(self, task_description=None, episode_id=None):
        """
        Reset the planner.
        
        Args:
            task_description: Optional task description
        """
        self.simulation_tree = None
        self.best_trajectory = None
        self.best_reward = float('-inf')
        if episode_id is not None:
            self.episode_id = episode_id
        else :
            self.episode_id = -1
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
        
        for action in action_list:
            obs, reward, done, truncated, info = env.step(
                np.concatenate(
                    [action["world_vector"], action["rot_axangle"], action["gripper"]]
                ),
            )
        
        return env, obs
        

    def copy_state(self, state, kwargs, additional_env_build_kwargs, env_reset_options):
        if self.verbose:
            print("[TwoSimulatorPlanner] Creating environment copy")
        
        # Create a new environment instance with the same configuration
        from simpler_env.utils.env.env_builder import build_maniskill2_env
        
        # Create the new environment instance
        state_copy = build_maniskill2_env(
            self.env_name,
             **additional_env_build_kwargs,
            **kwargs,
        )
        
        # Reset the environment
        state_copy.reset(options=env_reset_options)
        
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
        raw_actions = []
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
            raw_actions.append(raw_action)
            
            if self.verbose:
                print(f"    - Action vector: {action['world_vector']}")
                print(f"    - Rotation: {action['rot_axangle']}")
                print(f"    - Gripper: {action['gripper']}")
        
        return raw_actions, actions
    
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

        
        return state, reward, image, done
    
    def compute_reward(self, state=None, image=None, action=None, trajectory_images=None):
        """
        Compute reward for a state-action pair.
        
        Args:
            state: The current state
            image: The current image (optional)
            action: The action (optional)
            rewards_history: List of rewards from previous best trajectories (optional)
            
        Returns:
            reward: The computed reward
        """
        # Pass rewards_history to the reward function if it accepts it
        import inspect
        
        # Check if the reward function accepts a rewards_history parameter
        reward_function_params = inspect.signature(self.reward_function).parameters
        
        if 'trajectory_images' in reward_function_params:
            return self.reward_function(trajectory_images=trajectory_images)
        elif 'image' in reward_function_params:
            return self.reward_function(state, image)
        else:
            return self.reward_function(state)
    
    def plan_trajectory(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs, trajectory_length=10, num_trajectories=5, rewards_history=None):
        """
        Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
        Args:
            env_name: The environment name
            action_list: List of actions taken so far
            env_reset_options: Options for environment reset
            image: The current image observation
            task_description: Optional updated task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_trajectories: Number of different trajectories to evaluate
            rewards_history: List of rewards from previous best trajectories
            
        Returns:
            tuple: (best_trajectory, best_final_reward)
                - best_trajectory: List of actions forming the best trajectory
                - best_final_reward: The reward achieved by the best trajectory
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
            print(f"[TwoSimulatorPlanner] Evaluating {num_trajectories} trajectories of length {trajectory_length}")
            if rewards_history:
                print(f"[TwoSimulatorPlanner] Using rewards history: {rewards_history}")
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
        best_metrics = {
            'reward': float('-inf'),
        }
        most_recent_metric = {'reward': -1}
        if len(rewards_history) > 0:
            most_recent_metric = copy.deepcopy(rewards_history[-1])
        # Generate and evaluate multiple trajectories
        for traj_idx in range(num_trajectories):
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{num_trajectories}")
            
            # Initialize trajectory for this run
            trajectory_actions = []
            current_env = self.get_to_state(env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs)
            current_image = image
            
            trajectory_images = []
            # Generate a sequence of actions for this trajectory
            for step in range(trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{trajectory_length}")
                
                # Sample actions from the model with temperature to ensure diversity
                actions = self.sample_actions_from_model(
                    current_image, 
                    self.task_description, 
                    num_samples=1,
                    temperature=self.temperature
                )
                
                if not actions:
                    if self.verbose:
                        print("  - Failed to sample actions, ending trajectory early")
                    break
                    
                action = actions[0]
                trajectory_actions.append(action)
                
                # Simulate this action to update the environment state for the next step
                current_env, reward, current_image, done = self.simulate_action(
                    current_env, action, kwargs, additional_env_build_kwargs
                )
                trajectory_images.append(current_image)
                
                # # Use compute_reward instead of the direct reward from simulate_action
                # # Pass the rewards_history to the reward function
                # computed_reward = self.compute_reward(current_env, current_image, action, rewards_history)
                
                # if self.verbose:
                #     print(f"  - Simulated step reward: {computed_reward}")
                
                # If task is done, we can stop this trajectory
                if done:
                    if self.verbose:
                        print(f"  - Task completed after {step+1} steps")
                    break
            # Compute final reward for this trajectory
            metrics = self.compute_reward(trajectory_images=trajectory_images)
            if metrics is None :
                metrics = most_recent_metric
            final_reward = metrics['reward']
            metrics['image'] = current_image

            if self.verbose:
                print(f"[TwoSimulatorPlanner] Trajectory {traj_idx+1} final reward: {final_reward}")
            
            print(f"Trajectory {traj_idx+1} final reward: {final_reward}")
            
            # Update best trajectory if this one is better
            if final_reward > best_metrics['reward']:
                best_trajectory = trajectory_actions
                best_metrics = metrics
                if self.verbose:
                    print(f"[TwoSimulatorPlanner] New best trajectory found with reward: {final_reward}")
        
        print(f"Best trajectory has {len(best_trajectory or [])} actions with final reward: {best_metrics['reward']}")
        self.save_metrics_metrics_for_best_trajectory(best_metrics)
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory or [])} actions with final reward: {best_metrics['reward']}")
            print("="*80 + "\n")
        
        return best_trajectory or [], best_metrics  # Return empty list if no trajectory was found, plus the best reward
    
    def load_world_model(self):
        """
        Load the world model for simulation.
        """
        
        # Initialize the world model
        self.world_model = DynamicsModel(
            dim=768,
            image_size=224,
            patch_size=14,
            spatial_depth=8,
            dim_head=64,
            heads=16,
            use_lpips_loss=True,
        )
        
        # Load the model checkpoint
        pretrain_ckpt = "/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/results/dyna1_simpl_ft_1/vae.pt"
        ckpt = torch.load(pretrain_ckpt, map_location="cpu")["model"]
        self.world_model.load_state_dict(ckpt)
        
        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.world_model = self.world_model.to(self.device)
        self.world_model.eval()
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] World model loaded successfully to {self.device}")

    def prepare_image_for_world_model(self, image, image_size=(224, 224)):
        """
        Convert an image to a tensor suitable for the world model and resize it.
        
        Args:
            image: The image to convert (numpy array or tensor)
            image_size: Target size to resize the image to as a tuple (height, width) (default: (224, 224))
            
        Returns:
            image_tensor: The resized image as a tensor on the correct device
        """
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        if isinstance(image, np.ndarray):
            # Image is a numpy array, convert to tensor
            if image.ndim == 3 and image.shape[2] == 3:  # HWC format
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # Convert to CHW
            else:
                # Already in CHW format
                image_tensor = torch.from_numpy(image).float()
        else:
            # Image is already a tensor
            image_tensor = image.float()
        
        # Ensure proper dimensions
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Resize the image to the target size
        if image_tensor.shape[-2] != image_size[0] or image_tensor.shape[-1] != image_size[1]:
            image_tensor = F.interpolate(
                image_tensor, 
                size=image_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        # Move to the correct device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor

    def prepare_action_for_world_model(self, action):
        """
        Convert an action to a tensor suitable for the world model.
        
        Args:
            action: The action dictionary with 'world_vector', 'rot_axangle', and 'gripper' keys
            
        Returns:
            action_tensor: The action as a tensor on the correct device
        """
        
        # Extract the components and concatenate them
        action_list = (
            action["world_vector"].tolist() + 
            action["rotation_delta"].tolist() + 
            action["open_gripper"].tolist()
        )

        self.action_mean = torch.tensor([
            0.00023341893393080682,
            0.0001300494186580181,
            -0.0001276240509469062,
            -0.00015565630747005343,
            -0.0004039352061226964,
            0.00023557755048386753,
            0.5764579772949219
        ], dtype=torch.float32, device=self.device).reshape(-1)
        self.action_std = torch.tensor([
            0.009765958413481712,
            0.01368918176740408,
            0.012667348608374596,
            0.02853415347635746,
            0.03063797391951084,
            0.07691441476345062,
            0.4973689615726471
        ], dtype=torch.float32, device=self.device).reshape(-1)
        # Convert to tensor
        action_tensor = torch.tensor(action_list, dtype=torch.float32, device=self.device)
        action_tensor = (action_tensor - self.action_mean) / self.action_std
        
        return action_tensor

    def simulate_batch_with_world_model(self, images, actions):
        """
        Simulate a batch of actions using the world model.
        
        Args:
            images: List of current images (numpy arrays or tensors)
            actions: List of actions to simulate
            
        Returns:
            next_images: List of predicted next images (in the same format as the input)
        """
        
        # Determine if images are numpy arrays or tensors
        is_numpy = isinstance(images[0], np.ndarray)
        
        # Convert all images to tensors
        image_tensors = [self.prepare_image_for_world_model(img) for img in images]
        image_tensor_batch = torch.cat(image_tensors, dim=0)
        # Convert all actions to tensors
        action_tensors = [self.prepare_action_for_world_model(action) for action in actions]
        action_tensor_batch = torch.stack(action_tensors)
         
        # Predict the next images
        with torch.no_grad():
            next_image_tensor_batch = self.world_model(image_tensor_batch, action_tensor_batch, None, return_recons_only=True)
        
        
        # Convert back to the original format
        if is_numpy:
            next_images = [(tensor.to(torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8) for tensor in next_image_tensor_batch]
        else:
            next_images = [tensor.to(torch.float32).squeeze(0) for tensor in next_image_tensor_batch]
        # breakpoint()
        return next_images

    def plan_trajectory_with_world_model(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs, trajectory_length=10, num_trajectories=5, rewards_history=None):
        """
        Plan and evaluate multiple trajectories using a world model instead of a simulator.
        
        Args:
            env_name: The environment name
            action_list: List of actions taken so far
            env_reset_options: Options for environment reset
            image: The current image observation
            task_description: The task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_trajectories: Number of different trajectories to evaluate
            rewards_history: List of rewards from previous best trajectories
            
        Returns:
            tuple: (best_trajectory, best_metrics)
                - best_trajectory: List of actions forming the best trajectory
                - best_metrics: The metrics achieved by the best trajectory, including reward
        """
 
        
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning with world model")
            print(f"[TwoSimulatorPlanner] Evaluating {num_trajectories} trajectories of length {trajectory_length}")
            print("="*80)
        
        # Update task description if provided
        if task_description is not None:
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Updated task description: {task_description}")
            self.task_description = task_description
            self.model.reset(task_description)
        
        # Record time for performance monitoring
        start_time = time.time()
        
        # Load the world model if not already loaded
        if not hasattr(self, 'world_model'):
            self.load_world_model()
        
        current_env, obs = self.get_to_state(env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs)
        # Track best trajectory and its metrics
        best_trajectory = None
        best_metrics = {
            'reward': float('-inf'),
        }
        most_recent_metric = {'reward': -1}
        if rewards_history and len(rewards_history) > 0:
            most_recent_metric = copy.deepcopy(rewards_history[-1])
        
        # Generate trajectories with batch processing
        batch_size = min(num_trajectories, 10)  # Process in smaller batches to avoid OOM
        all_trajectories = []
        
        for batch_start in range(0, num_trajectories, batch_size):
            batch_end = min(batch_start + batch_size, num_trajectories)
            batch_size_actual = batch_end - batch_start
            
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Processing trajectories {batch_start+1}-{batch_end} (batch size {batch_size_actual})")
            
            # Initialize trajectories for this batch
            batch_trajectory_actions = [[] for _ in range(batch_size_actual)]
            batch_trajectory_images = [[image] for _ in range(batch_size_actual)]
            
            # Start with the same initial image for all trajectories
            current_images = [image] * batch_size_actual
            
            # Generate trajectories step by step
            for step in range(trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{trajectory_length}")
                
                # Sample actions for each trajectory in the batch
                batch_actions = []
                raw_batch_actions = []
                for i in range(batch_size_actual):
                    raw_actions, actions = self.sample_actions_from_model(
                        current_images[i],
                        self.task_description,
                        num_samples=1,
                        temperature=self.temperature
                    )
                    
                    if not actions:
                        batch_actions.append(None)
                    else:
                        action = actions[0]
                        raw_action = raw_actions[0]
                        batch_trajectory_actions[i].append(action)
                        raw_batch_actions.append(raw_action)

                # Filter out trajectories with invalid actions
                valid_indices = [i for i, action in enumerate(raw_batch_actions) if action is not None]
                if not valid_indices:
                    if self.verbose:
                        print("  - No valid actions sampled, ending step early")
                    break
                
                # Prepare batch for world model simulation
                valid_images = [current_images[i] for i in valid_indices]
                valid_actions = [raw_batch_actions[i] for i in valid_indices]
                
                # Simulate all valid actions in a batch
                next_images = self.simulate_batch_with_world_model(valid_images, valid_actions)
                
                # Update states for trajectories with valid actions
                for j, i in enumerate(valid_indices):
                    current_images[i] = next_images[j]
                    batch_trajectory_images[i].append(next_images[j])
            
            # Append the completed trajectories to the overall list
            for i in range(batch_size_actual):
                all_trajectories.append((batch_trajectory_actions[i], batch_trajectory_images[i]))
                
        best_traj_images = all_trajectories[0][1]
        # Evaluate all trajectories
        for traj_idx, (trajectory_actions, trajectory_images) in enumerate(all_trajectories):
            # Skip empty trajectories
            if not trajectory_actions:
                continue

            eval_env, _ = self.get_to_state(env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs)

            # Execute the trajectory in the actual simulator to get the final state
            for action in trajectory_actions:
                eval_env, _, _, _ = self.simulate_action(
                    eval_env, action, kwargs, additional_env_build_kwargs
                )
 
            metrics = self.compute_reward(eval_env)
            # # Compute metrics for this trajectory
            # metrics = self.compute_reward(trajectory_images=trajectory_images)
            if metrics is None:
                metrics = most_recent_metric
            final_reward = metrics['reward']
            metrics['image'] = trajectory_images[-1]
            
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Trajectory {traj_idx+1} final reward: {final_reward}")
            
            print(f"Trajectory {traj_idx+1} final reward: {final_reward}")
            
            # Update best trajectory if this one is better
            if final_reward > best_metrics['reward']:
                best_trajectory = trajectory_actions
                best_traj_images = trajectory_images
                best_metrics = metrics
                if self.verbose:
                    print(f"[TwoSimulatorPlanner] New best trajectory found with reward: {final_reward}")
        
        print(f"Best trajectory has {len(best_trajectory or [])} actions with final reward: {best_metrics['reward']}")
        #self.save_metrics_metrics_for_best_trajectory(best_metrics)
        current_image = get_image_from_maniskill2_obs_dict(current_env, obs)
        eval_env, _ = self.get_to_state(env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs)
        sim_images = [current_image] + self.get_simulator_rollout_images(eval_env, best_trajectory, kwargs, additional_env_build_kwargs, env_reset_options)
        self.save_trajectory_images(best_trajectory, best_traj_images, sim_images, os.path.join(self.logging_dir, "planning", f"episode_{self.episode_id}"))
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory or [])} actions with final reward: {best_metrics['reward']}")
            print("="*80 + "\n")
        
        return best_trajectory or [], best_metrics  # Return empty list if no trajectory was found, plus the best metrics
    def save_metrics_metrics_for_best_trajectory(self,metrics):
        image = metrics['image']
        if image is None:
            print("Image cant be none")
            return 
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            # Convert RGB to BGR for cv2.imwrite
            if image.ndim == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            cv2.imwrite(tmp_path, image_bgr)
        # Create output directory for visualizations
        output_base_dir = f"/data/user_data/ayudhs/random/multimodal/SimplerEnv-SITCOM/outputs/affordance_visualizations/{self.episode_id}"
        os.makedirs(output_base_dir, exist_ok=True)

        # Generate unique identifiers
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create output directories for target objects
        carrot_output_dir = os.path.join(output_base_dir, f"carrot_{timestamp}_{pathlib.Path(tmp_path).stem}")
        os.makedirs(carrot_output_dir, exist_ok=True)
        try:
            _ = calculate_affordance_metrics(
                img_path=tmp_path,
                target_object="carrot",
                gripper_name="gripper",
                visualize=True,
                save_metrics=True,
                output_dir=carrot_output_dir,
            )
            # Save metrics to JSON file
            metrics_output_path = os.path.join(carrot_output_dir, "metrics.json")

            # Function to convert numpy types and other non-serializable objects to Python native types
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, tuple):
                    return list(obj)
                elif hasattr(obj, 'tolist'):  # Handle other numpy types with tolist method
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items() if k != 'image'}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj

            # Convert metrics to JSON serializable format (exclude the image)
            serializable_metrics = {k: convert_for_json(v) for k, v in metrics.items() if k != 'image'}

            # Save to JSON file
            with open(metrics_output_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)

            print(f"Saved metrics to {metrics_output_path}")
        except Exception as e:
            print(f"Error saving metrics {metrics}: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def save_trajectory_images(self, best_traj_actions, wm_images, sim_images, output_dir=None):

        # Generate unique identifiers
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print(f"[TwoSimulatorPlanner] Saving trajectory visualizations to {output_dir}")
    
        # Save simulator images as a horizontal strip
        if sim_images and len(sim_images) > 0:
            # Create horizontal strip of simulator images
            fig, axes = plt.subplots(1, len(sim_images), figsize=(len(sim_images) * 4, 4))
            if len(sim_images) == 1:
                axes = [axes]  # Make iterable for single image case
            
            for i, (ax, img) in enumerate(zip(axes, sim_images)):
                if img is not None:
                    # Convert if tensor
                    if torch.is_tensor(img):
                        img = img.detach().cpu().numpy()
                    # Handle different formats
                    if img.shape[0] == 3:  # CHW format
                        img = np.transpose(img, (1, 2, 0))
                    
                    ax.imshow(img.astype(np.uint8))
                    ax.set_title(f"Step {i}")
                    ax.axis('off')
            
            plt.tight_layout()
            strip_path = os.path.join(output_dir, "simulator_trajectory_strip.png")
            plt.savefig(strip_path)
            plt.close(fig)
        
        # Save world model images as a horizontal strip
        if wm_images and len(wm_images) > 0:
            fig, axes = plt.subplots(1, len(wm_images), figsize=(len(wm_images) * 4, 4))
            if len(wm_images) == 1:
                axes = [axes]  # Make iterable for single image case
            
            for i, (ax, img) in enumerate(zip(axes, wm_images)):
                if img is not None:
                    # Convert if tensor
                    if torch.is_tensor(img):
                        img = img.detach().cpu().numpy()
                    # Handle different formats
                    if img.shape[0] == 3:  # CHW format
                        img = np.transpose(img, (1, 2, 0))
                    
                    ax.imshow(img.astype(np.uint8))
                    ax.set_title(f"Step {i}")
                    ax.axis('off')
            
            plt.tight_layout()
            strip_path = os.path.join(output_dir, "world_model_trajectory_strip.png")
            plt.savefig(strip_path)
            plt.close(fig)

        # Save action metadata
        if best_traj_actions:
            action_file = os.path.join(output_dir, "actions.json")
            with open(action_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_actions = []
                for action in best_traj_actions:
                    serializable_action = {
                        "world_vector": action["world_vector"].tolist(),
                        "rot_axangle": action["rot_axangle"].tolist(),
                        "gripper": action["gripper"].tolist()
                    }
                    serializable_actions.append(serializable_action)
                json.dump(serializable_actions, f, indent=2)

    def get_simulator_rollout_images(self, eval_env, trajectory_actions, kwargs, additional_env_build_kwargs, env_reset_options):
        # Create a copy of the environment for final reward evaluation
        
        images = []
        # Execute the trajectory in the actual simulator to get the final state
        for action in trajectory_actions:
            eval_env, _, image, _ = self.simulate_action(
                eval_env, action, kwargs, additional_env_build_kwargs
            )
            images.append(image)
        
        return images
        
        # Compute metrics for this trajectory
    def debug_trajectory_planning(self, env, image, task_description, kwargs, additional_env_build_kwargs, dummy):
        """
        Debug method to test if environment copying and simulation work correctly
        by testing two predefined trajectories.
        """
        import numpy as np
        
        
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
            
            next_env, reward, next_image, done = self.simulate_action(
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
            
            next_env, reward, next_image, done = self.simulate_action(
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