from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from simpler_env.policies.sitcom.simulation_node import SimulationNode
from simpler_env.policies.openvla.openvla_model import OpenVLAInference

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
            action_scale=action_scale
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
        state_copy = self.copy_state(state, kwargs, additional_env_build_kwargs)
        
        # Simulate the action using the model (second simulator)
        # For ManiSkill2, we need to concatenate action components
        action_array = np.concatenate([
            action["world_vector"], 
            action["rot_axangle"], 
            action["gripper"]
        ])
        
        # Step the environment with the action
        obs, reward, done, truncated, info = state_copy.step(action_array)
        
        if self.verbose:
            print(f"  - Reward: {reward}")
            print(f"  - Done: {done}")
            if done:
                print(f"  - Task completed in simulation!")
        
        # Extract the image from the observation
        image = get_image_from_maniskill2_obs_dict(state_copy, obs)
        
        return state_copy, reward, image, done
    
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
            next_state, _, _, _ = self.simulate_action(state, action, kwargs, additional_env_build_kwargs)
            
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
            next_state, reward, next_image, done = self.simulate_action(root_state, action, kwargs, additional_env_build_kwargs)
            
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
            next_state, reward, next_image, done = self.simulate_action(node.state, action, kwargs, additional_env_build_kwargs)
            
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
    
    def plan_trajectory(self, env, image, task_description, kwargs, additional_env_build_kwargs, trajectory_length=10, num_trajectories=5):
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
            current_env = self.copy_state(env, kwargs, additional_env_build_kwargs)
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
                next_env, reward, next_image, done = self.simulate_action(
                    current_env, action, kwargs, additional_env_build_kwargs
                )
                
                # Update the current state and image
                current_env = next_env
                current_image = next_image
                
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