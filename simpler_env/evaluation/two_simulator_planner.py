from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import copy
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
from simulation_node import SimulationNode

class TwoSimulatorPlanner:
    """Planning algorithm using two simulators."""
    
    def __init__(
        self,
        env,
        model,
        openvla_model,
        task_description,
        reward_function,
        num_initial_actions=10,  # A parameter
        horizon_per_action=5,    # Horizon parameter
        num_steps_ahead=3,       # h parameter
        num_candidates=5,        # Number of candidate actions to sample
        num_best_actions=3,      # Number of best actions to select
        temperature=1.0,         # Temperature for sampling
        render_tree=False,       # Whether to render the tree
        logging_dir="./results/planning",
    ):
        """
        Initialize the planner.
        
        Args:
            env: The first simulator (environment)
            model: The second simulator (dynamics model)
            openvla_model: The OpenVLA model for sampling actions
            task_description: The task description
            reward_function: Function to compute reward (state, action) -> reward
            num_initial_actions: Number of initial actions to sample (A)
            horizon_per_action: Number of actions to consider for each state (Horizon)
            num_steps_ahead: Number of simulation steps to look ahead (h)
            num_candidates: Number of candidate actions to sample
            num_best_actions: Number of best actions to select
            temperature: Temperature for sampling
            render_tree: Whether to render the tree
            logging_dir: Directory for logging
        """
        self.env = env
        self.model = model
        self.openvla_model = openvla_model
        self.task_description = task_description
        self.reward_function = reward_function
        
        # Hyperparameters
        self.num_initial_actions = num_initial_actions
        self.horizon_per_action = horizon_per_action
        self.num_steps_ahead = num_steps_ahead
        self.num_candidates = num_candidates
        self.num_best_actions = num_best_actions
        self.temperature = temperature
        
        # Visualization
        self.render_tree = render_tree
        self.logging_dir = logging_dir
        
        # Reset internal state
        self.reset()
    
    def reset(self):
        """Reset the planner."""
        self.simulation_tree = None
        self.best_trajectory = None
        self.best_reward = float('-inf')
        
    def sample_actions_from_openvla(self, image, num_actions, temp=None):
        """
        Sample actions from OpenVLA model.
        
        Args:
            image: The current image observation
            num_actions: Number of actions to sample
            temp: Temperature for sampling (override default if provided)
            
        Returns:
            List of sampled actions
        """
        temperature = temp if temp is not None else self.temperature
        # This is a placeholder for how you would call your OpenVLA model
        # The actual implementation will depend on your OpenVLA API
        return self.openvla_model.sample_actions(
            image, 
            self.task_description, 
            num_samples=num_actions,
            temperature=temperature
        )
    
    def simulate_action(self, state, action):
        """
        Simulate an action using the second simulator (model).
        
        Args:
            state: The current state
            action: The action to simulate
            
        Returns:
            next_state, reward, image
        """
        # Create a copy of the state to avoid modifying the original
        model_copy = copy.deepcopy(self.model)
        
        # Process the action similar to what's done in the evaluation code
        raw_action, processed_action = model_copy.step(state["image"], self.task_description, override_action=action)
        
        # Simulate the action using the model (second simulator)
        next_state, reward, done, truncated, info = model_copy.simulate(
            state,
            np.concatenate([
                processed_action["world_vector"], 
                processed_action["rot_axangle"], 
                processed_action["gripper"]
            ])
        )
        
        # Get the image from the next state
        image = next_state.get("image", None)
        if image is None and "obs" in next_state:
            # Extract image using the utility function if not directly available
            image = get_image_from_maniskill2_obs_dict(None, next_state["obs"])
        
        return next_state, reward, image, done
    
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
    
    def select_best_actions(self, state, candidate_actions, num_best):
        """
        Select the best actions based on the reward function.
        
        Args:
            state: The current state
            candidate_actions: List of candidate actions
            num_best: Number of best actions to select
            
        Returns:
            best_actions: List of best actions
        """
        # Compute rewards for all candidate actions
        rewards = []
        for action in candidate_actions:
            next_state, _, _, _ = self.simulate_action(state, action)
            reward = self.compute_reward(next_state)
            rewards.append((action, reward))
        
        # Sort by reward (descending) and select the best
        rewards.sort(key=lambda x: x[1], reverse=True)
        return [action for action, _ in rewards[:num_best]]
    
    def build_simulation_tree(self, root_state, root_image):
        """
        Build a simulation tree by exploring possible actions.
        
        Args:
            root_state: The initial state
            root_image: The initial image
            
        Returns:
            root_node: The root node of the tree
        """
        # Create the root node
        root_node = SimulationNode(root_state, root_image)
        
        # Sample initial actions
        initial_actions = self.sample_actions_from_openvla(
            root_image, 
            self.num_initial_actions
        )
        
        best_leaf_node = None
        best_reward = float('-inf')
        
        # For each initial action, simulate and build a subtree
        for i, action in enumerate(initial_actions):
            # Simulate the action
            next_state, reward, next_image, done = self.simulate_action(root_state, action)
            
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
            
            # Explore this subtree further if not done
            if not done:
                # Perform look-ahead simulation
                leaf_node = self._simulate_ahead(child_node)
                
                # Update best leaf node if needed
                if leaf_node.reward > best_reward:
                    best_reward = leaf_node.reward
                    best_leaf_node = leaf_node
        
        # Store the tree and best trajectory
        self.simulation_tree = root_node
        self.best_reward = best_reward
        
        # Backtrack to find the best initial action
        if best_leaf_node:
            self.best_trajectory = self._backtrack_to_root(best_leaf_node)
            best_initial_action = initial_actions[best_leaf_node.original_action_idx]
            return best_initial_action
        
        # Fallback to the first action if no simulation was successful
        return initial_actions[0] if initial_actions else None
    
    def _simulate_ahead(self, node, current_depth=1):
        """
        Recursively simulate ahead from a node.
        
        Args:
            node: The current node
            current_depth: Current depth in the tree
            
        Returns:
            best_leaf: The best leaf node in this subtree
        """
        # If we've reached the maximum depth or this is a terminal state, return this node
        if current_depth >= self.num_steps_ahead or node.reward == float('inf'):
            return node
        
        # Sample candidate actions from this state
        candidate_actions = self.sample_actions_from_openvla(
            node.image, 
            self.num_candidates
        )
        
        # Select the best candidate actions
        best_actions = self.select_best_actions(
            node.state, 
            candidate_actions, 
            self.num_best_actions
        )
        
        best_leaf = node
        best_reward = node.reward
        
        # Explore each of the best actions
        for action in best_actions:
            # Simulate the action
            next_state, reward, next_image, done = self.simulate_action(node.state, action)
            
            # Create a child node
            child_node = SimulationNode(
                next_state, 
                next_image, 
                parent=node, 
                action=action, 
                reward=node.reward + reward,  # Accumulate rewards
                depth=current_depth + 1
            )
            child_node.original_action_idx = node.original_action_idx  # Propagate original action index
            node.add_child(child_node)
            
            # Continue simulation if not done
            if not done:
                leaf_node = self._simulate_ahead(child_node, current_depth + 1)
                
                # Update best leaf if needed
                if leaf_node.reward > best_reward:
                    best_reward = leaf_node.reward
                    best_leaf = leaf_node
            elif child_node.reward > best_reward:
                # Terminal state with better reward
                best_reward = child_node.reward
                best_leaf = child_node
        
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
        
        while current.parent:
            trajectory.append((current.parent.state, current.action))
            current = current.parent
        
        # Reverse to get from root to leaf
        trajectory.reverse()
        return trajectory
    
    def plan(self, obs, task_description=None):
        """
        Plan the best action to take from the current state.
        
        Args:
            obs: The current observation
            task_description: Optional updated task description
            
        Returns:
            best_action: The best action to take
        """
        # Update task description if provided
        if task_description is not None:
            self.task_description = task_description
        
        # Extract state and image from observation
        state = {"obs": obs}  # Wrap observation in a state dict
        image = get_image_from_maniskill2_obs_dict(self.env, obs)
        
        # Build simulation tree and get the best action
        best_action = self.build_simulation_tree(state, image)
        
        return best_action
    
    def visualize_tree(self, save_path=None):
        """
        Visualize the simulation tree.
        
        Args:
            save_path: Path to save the visualization
        """
        if not self.render_tree:
            return
        
        # Implement visualization as needed
        # This could use libraries like networkx and matplotlib
        pass