from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from simpler_env.policies.sitcom.simulation_node import SimulationNode
from simpler_env.policies.openvla.openvla_model import OpenVLAInference
from PIL import Image
from simpler_env.policies.sitcom.gemini_reward import gemini_reward


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
        rewarder=None,
        num_initial_actions=10,  # A parameter
        horizon_per_action=5,    # Horizon parameter
        num_steps_ahead=3,       # h parameter
        num_candidates=5,        # Number of candidate actions to sample
        trajectory_length=10,      # Number of best actions to select
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
        
        # Initialize the rewarder
        self.rewarder = rewarder
        
        # Set up reward function or use default
        if reward_function is None:
            self.reward_function = self._default_reward_function
        else:
            self.reward_function = reward_function
            
        # self.gemini_reward_function = gemini_reward
        
        # Hyperparameters
        self.num_initial_actions = num_initial_actions  # A
        self.horizon_per_action = horizon_per_action    # Horizon
        self.num_steps_ahead = num_steps_ahead          # h
        self.num_candidates = num_candidates
        self.trajectory_length = trajectory_length
        self.temperature = temperature
        
        # Visualization settings
        self.render_tree = render_tree
        self.logging_dir = logging_dir
        self.verbose = verbose
        
        # Task description
        self.task_description = None
        
        # Reset internal state
        self.reset()
        
        self.point_wise_error = []
        self.ranking_error = []
        
    
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
            print(f"  - Trajectory length: {trajectory_length}")
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
    
    import numpy as np
    
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

    # def ndcg_at_k(self, oracle_rewards, gemini_rewards, k=None):
    #     """
    #     Compute NDCG@k treating oracle_rewards as ground truth relevances.
    #     If k is None we use all trajectories.
    #     """
    #     if k is None:
    #         k = len(oracle_rewards)
        
    #     # Normalize oracle_rewards to be non-negative
    #     min_reward = min(oracle_rewards)
    #     normalized_oracle = [r - min_reward for r in oracle_rewards]
        
    #     # ideal DCG: sort oracle descending
    #     ideal = sorted(normalized_oracle, reverse=True)[:k]
        
    #     def dcg(rs):
    #         return sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rs))
        
    #     idcg = dcg(ideal)
        
    #     # DCG of the gemini-ranked list
    #     order = np.argsort(gemini_rewards)[::-1][:k]
    #     rels = [normalized_oracle[i] for i in order]
        
    #     return dcg(rels) / idcg if idcg > 0 else 0.0
    
    def ndcg_at_k(self, oracle_rewards, gemini_rewards, k=None):
        """
        Compute NDCG@k treating oracle_rewards as ground truth relevances.
        Handles ties in gemini_rewards by averaging over all possible orderings.
        """
        if k is None:
            k = len(oracle_rewards)
        
        # Normalize oracle_rewards to be non-negative
        min_reward = min(oracle_rewards)
        normalized_oracle = [r - min_reward for r in oracle_rewards]
        
        # Calculate ideal DCG
        ideal = sorted(normalized_oracle, reverse=True)[:k]
        
        def dcg(rs):
            return sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(rs))
        
        idcg = dcg(ideal)
        if idcg <= 0:
            return 0.0
        
        # Group indices by their gemini_rewards value
        score_to_indices = {}
        for i, score in enumerate(gemini_rewards):
            if score not in score_to_indices:
                score_to_indices[score] = []
            score_to_indices[score].append(i)
        
        # Calculate average NDCG over all possible orderings
        from itertools import permutations
        import random
        
        # Get unique scores in descending order
        unique_scores = sorted(score_to_indices.keys(), reverse=True)
        
        # If there are too many permutations, sample a reasonable number
        MAX_PERMS = 100  # Adjust as needed
        
        total_ndcg = 0.0
        num_perms = 0
        
        # Generate permutations of index positions for each score group
        pos_assignments = []
        remaining_positions = list(range(len(oracle_rewards)))
        
        # Assign positions to each score group (higher scores get earlier positions)
        for score in unique_scores:
            indices = score_to_indices[score]
            num_indices = len(indices)
            
            # Take positions from the beginning of remaining_positions
            positions_for_this_score = remaining_positions[:num_indices]
            remaining_positions = remaining_positions[num_indices:]
            
            # Generate permutations of indices within this score group
            if len(indices) > 1:
                # If too many permutations, sample randomly
                all_perms = list(permutations(indices))
                if len(all_perms) > MAX_PERMS:
                    perms_to_use = random.sample(all_perms, MAX_PERMS)
                else:
                    perms_to_use = all_perms
                
                pos_assignments.append((positions_for_this_score, perms_to_use))
            else:
                # No permutation needed for a single item
                pos_assignments.append((positions_for_this_score, [tuple(indices)]))
        
        # Calculate NDCG for each permutation combination
        from itertools import product
        
        # Extract just the permutations for each score group
        perm_lists = [perms for _, perms in pos_assignments]
        
        # Calculate total number of permutation combinations
        total_perms = 1
        for perms in perm_lists:
            total_perms *= len(perms)
        
        # If too many total permutations, sample a subset
        if total_perms > MAX_PERMS:
            sampled_perms = 0
            while sampled_perms < MAX_PERMS:
                # Create a random ordering
                ordering = [0] * len(oracle_rewards)
                for (positions, perms) in pos_assignments:
                    perm = random.choice(perms)
                    for pos, idx in zip(positions, perm):
                        if pos < k:  # Only care about positions within k
                            ordering[pos] = idx
                
                # Calculate NDCG for this ordering
                rels = [normalized_oracle[ordering[i]] for i in range(min(k, len(ordering)))]
                total_ndcg += dcg(rels) / idcg
                sampled_perms += 1
            
            return total_ndcg / sampled_perms
        else:
            # Generate all permutation combinations (reasonable number)
            for perm_combination in product(*perm_lists):
                # Create ordering based on this permutation combination
                ordering = [0] * len(oracle_rewards)
                for i, (positions, _) in enumerate(pos_assignments):
                    for pos, idx in zip(positions, perm_combination[i]):
                        if pos < k:  # Only care about positions within k
                            ordering[pos] = idx
                
                # Calculate NDCG for this ordering
                rels = [normalized_oracle[ordering[i]] for i in range(min(k, len(ordering)))]
                total_ndcg += dcg(rels) / idcg
                num_perms += 1
            
            return total_ndcg / num_perms if num_perms > 0 else 0.0
    
    def compare_rewards(self, oracle_reward, gemini_reward):
        """
        Compare rewards from oracle and Gemini.
        
        Args:
            oracle_reward: Reward from the oracle
            gemini_reward: Reward from Gemini
            
        Returns:
            Tuple of (oracle_reward, gemini_reward)
        """
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Comparing rewards:")
            print(f"  - Oracle reward: {oracle_reward}")
            print(f"  - Gemini reward: {gemini_reward}")
            
        # if oracle reward < 1 then gemini reward should be 0
        # if oracle reward < 8 then gemini reward should be 1
        # if oracle reward > 8 then gemini reward should be 2
        if oracle_reward < 1:
            oracle_gemini_reward = 0
        elif oracle_reward < 8:
            oracle_gemini_reward = 1
        else:
            oracle_gemini_reward = 2
        
        # Compare the two rewards
        error = abs(oracle_gemini_reward - gemini_reward)
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Error between scaled rewards: {error}")
        
        
        return error
    
    
    # def plan_trajectory(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs):
    #     """
    #     Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
    #     Args:
    #         env: The current environment state (first simulator)
    #         image: The current image observation
    #         task_description: Optional updated task description
    #         kwargs: Additional arguments for environment building
    #         additional_env_build_kwargs: Additional environment building arguments
    #         trajectory_length: The number of actions to include in each trajectory
    #         num_candidates: Number of different trajectories to evaluate
            
    #     Returns:
    #         best_trajectory: List of actions forming the best trajectory
    #     """
    #     if self.verbose:
    #         print("\n" + "="*80)
    #         print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
    #         print(f"[TwoSimulatorPlanner] Evaluating {self.num_candidates} trajectories of length {self.trajectory_length}")
    #         print("="*80)
        
    #     # Update task description if provided
    #     if task_description is not None:
    #         if self.verbose:
    #             print(f"[TwoSimulatorPlanner] Updated task description: {task_description}")
    #         self.task_description = task_description
    #         self.model.reset(task_description)
        
    #     # Record time for performance monitoring
    #     import time
    #     start_time = time.time()
        
    #     # Track best trajectory and its final reward
    #     best_trajectory = None
    #     best_final_reward = float('-inf')
        
    #     tot_trajectory_error = 0
        
    #     oracle_rewards = []
    #     gemini_rewards = []
        
    #     # Generate and evaluate multiple trajectories
    #     for traj_idx in range(self.num_candidates):
    #         if self.verbose:
    #             print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{self.num_candidates}")
            
    #         # Initialize trajectory for this run
    #         trajectory_actions = []
    #         current_env, current_obs, start_image = self.get_to_state(env_name, action_list,env_reset_options, kwargs, additional_env_build_kwargs)
    #         # current_env = self.copy_state(env_name, kwargs, additional_env_build_kwargs)
    #         # current_env = 
    #         current_image = image
            
    #         # Generate a sequence of actions for this trajectory
    #         for step in range(self.trajectory_length):
    #             if self.verbose:
    #                 print(f"  - Planning step {step+1}/{self.trajectory_length}")
                
    #             # Sample actions from the model with temperature to ensure diversity
    #             actions = self.sample_actions_from_model(
    #                 current_image, 
    #                 self.task_description, 
    #                 num_samples=1,  # Just sample one action at a time for the trajectory
    #                 temperature=self.temperature
    #             )
                
    #             if not actions:
    #                 if self.verbose:
    #                     print("  - Failed to sample actions, ending trajectory early")
    #                 break
                    
    #             action = actions[0]
    #             trajectory_actions.append(action)
                
    #             # Simulate this action to update the environment state for the next step
    #             current_env, current_obs, reward, current_image, done = self.simulate_action(
    #                 current_env, action, kwargs, additional_env_build_kwargs
    #             )
                
    #             computed_reward = self.compute_reward(current_env)
    #             # print(f"  - Simulated step reward: {computed_reward}")
    
                
    #             if self.verbose:
    #                 print(f"  - Simulated step reward: {reward}")
                
    #             # If task is done, we can stop this trajectory
    #             if done:
    #                 if self.verbose:
    #                     print(f"  - Task completed after {step+1} steps")
    #                 break
                
    #         # Compute final reward for this trajectory
            
            
    #         # Compute final reward for this trajectory by oracle
    #         final_reward = self.compute_reward(current_env)
            
            
            
    #         if self.verbose:
    #             print(f"[TwoSimulatorPlanner] Trajectory {traj_idx+1} final reward: {final_reward}")
            
    #         # print reward for the trajectory
    #         print(f"Trajectory {traj_idx+1} final reward: {final_reward}")
    #         # Update best trajectory if this one is better
    #         if final_reward > best_final_reward:
    #             best_final_reward = final_reward
    #             best_trajectory = trajectory_actions
                
    #             if self.verbose:
    #                 print(f"[TwoSimulatorPlanner] New best trajectory found with reward: {final_reward}")
        
    #     # average_trajectory_error = tot_trajectory_error / num_candidates
        
    #     # Compute NDCG@k
    #     # ndcg_score = self.ndcg_at_k(oracle_rewards, gemini_rewards)
        
    #     # if self.verbose:
    #     #     print(f"[TwoSimulatorPlanner] NDCG@k: {ndcg_score}")
    #     # # Store the trajectory error for analysis
    #     # self.ranking_error.append(ndcg_score)        
    #     # self.point_wise_error.append(average_trajectory_error)
    #     # print(f"Average trajectory error: {average_trajectory_error}")
    #     print(f"Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
    #     # Calculate planning time
    #     planning_time = time.time() - start_time
        
    #     if self.verbose:
    #         print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
    #         print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
    #         print("="*80 + "\n")
        
    #     return best_trajectory, best_final_reward

    
    def plan_trajectory(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs):
        """
        Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
        Args:
            env: The current environment state (first simulator)
            image: The current image observation
            task_description: Optional updated task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_candidates: Number of different trajectories to evaluate
            
        Returns:
            best_trajectory: List of actions forming the best trajectory
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
            print(f"[TwoSimulatorPlanner] Evaluating {self.num_candidates} trajectories of length {self.trajectory_length}")
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
        
        tot_trajectory_error = 0
        
        oracle_rewards = []
        gemini_rewards = []
        
        # Generate and evaluate multiple trajectories
        for traj_idx in range(self.num_candidates):
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{self.num_candidates}")
            
            # Initialize trajectory for this run
            trajectory_actions = []
            current_env, current_obs, start_image = self.get_to_state(env_name, action_list,env_reset_options, kwargs, additional_env_build_kwargs)
            # current_env = self.copy_state(env_name, kwargs, additional_env_build_kwargs)
            # current_env = 
            current_image = image
            
            # Generate a sequence of actions for this trajectory
            for step in range(self.trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{self.trajectory_length}")
                
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
                
                # computed_reward = self.compute_reward(current_env)
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
            # breakpoint()
            final_reward = self.rewarder.get_reward(start_image, final_image, task_description)  
            # breakpoint()
            # Compute final reward for this trajectory by oracle
            final_reward_oracle = self.compute_reward(current_env)
            
            print("FInal Reward from GRM:", final_reward)
            
            # error analysis
            error = self.compare_rewards(final_reward_oracle, final_reward)
            
            tot_trajectory_error += error
            
            oracle_rewards.append(final_reward_oracle)
            gemini_rewards.append(final_reward)
            
            
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
        
        average_trajectory_error = tot_trajectory_error / self.num_candidates
        # breakpoint()
        # Compute NDCG@k
        ndcg_score = self.ndcg_at_k(oracle_rewards, gemini_rewards)
        # breakpoint()

        # Store the trajectory error for analysis
        self.ranking_error.append(ndcg_score)        
        self.point_wise_error.append(average_trajectory_error)
        print(f"Average trajectory error: {average_trajectory_error}")
        print(f"[TwoSimulatorPlanner] NDCG@k: {ndcg_score}")
        print(f"Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
        # Calculate planning time
        planning_time = time.time() - start_time
        
        if self.verbose:
            print(f"[TwoSimulatorPlanner] Trajectory planning completed in {planning_time:.2f} seconds")
            print(f"[TwoSimulatorPlanner] Best trajectory has {len(best_trajectory)} actions with final reward: {best_final_reward}")
            print("="*80 + "\n")
        
        return best_trajectory, best_final_reward
        
    
    def plan_trajectory_reward_qwen(self, env_name, action_list, env_reset_options, image, task_description, kwargs, additional_env_build_kwargs):
        """
        Plan and evaluate multiple trajectories, returning the best one based on final reward.
        
        Args:
            env: The current environment state (first simulator)
            image: The current image observation
            task_description: Optional updated task description
            kwargs: Additional arguments for environment building
            additional_env_build_kwargs: Additional environment building arguments
            trajectory_length: The number of actions to include in each trajectory
            num_candidates: Number of different trajectories to evaluate
            
        Returns:
            best_trajectory: List of actions forming the best trajectory
        """
        if self.verbose:
            print("\n" + "="*80)
            print(f"[TwoSimulatorPlanner] Starting trajectory planning process")
            print(f"[TwoSimulatorPlanner] Evaluating {self.num_candidates} trajectories of length {self.trajectory_length}")
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
        for traj_idx in range(self.num_candidates):
            if self.verbose:
                print(f"[TwoSimulatorPlanner] Generating trajectory {traj_idx+1}/{self.num_candidates}")
            
            # Initialize trajectory for this run
            trajectory_actions = []
            current_env, current_obs, start_image = self.get_to_state(env_name, action_list,env_reset_options, kwargs, additional_env_build_kwargs)
            # current_env = self.copy_state(env_name, kwargs, additional_env_build_kwargs)
            # current_env = 
            current_image = image
            
            # Generate a sequence of actions for this trajectory
            for step in range(self.trajectory_length):
                if self.verbose:
                    print(f"  - Planning step {step+1}/{self.trajectory_length}")
                
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
    