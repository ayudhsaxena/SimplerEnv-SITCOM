from simpler_env.policies.sitcom.two_simulator_planner import TwoSimulatorPlanner
from simpler_env.policies.sitcom.general_rewarder import GeneralRewarder
from memory.reward_memory import RewardMemory
import numpy as np
from collections import deque
import os

class SITCOMInference:
    """
    Wrapper model that uses the TwoSimulatorPlanner for decision making.
    This class implements the same interface as the original model used in evaluation.
    """
    
    def __init__(
        self,
        env_name,
        saved_model_path: str = "openvla/openvla-7b",
        reward_function=None,
        num_initial_actions=10,
        horizon_per_action=5,
        num_steps_ahead=3,
        num_candidates=5,
        num_best_actions=3,
        temperature=1.0,
        render_tree=False,
        logging_dir="./results/planning",
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        trajectory_length: int = 10,  # New parameter for trajectory length
    ):
        """
        Initialize the wrapper for the planning model.
        
        Args:
            saved_model_path: Path to the OpenVLA model
            reward_function: Function to compute reward
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
            trajectory_length: Length of trajectory to request from planner
        """
        
        API_KEY = os.environ.get("GEMINI_API_KEY")
        # breakpoint()
        
        # TODO check this
        memory = RewardMemory.load("./memory/reward_memory_state.pkl", api_key=API_KEY)
        
        ### create rewarder
        self.rewarder = GeneralRewarder(memory, api_key=API_KEY, num_examples=4)
        
        # Create the planner
        self.planner = TwoSimulatorPlanner(
            env_name=env_name,
            saved_model_path=saved_model_path,
            reward_function=reward_function,
            rewarder=self.rewarder,
            num_initial_actions=num_initial_actions,
            horizon_per_action=horizon_per_action,
            num_steps_ahead=num_steps_ahead,
            num_candidates=num_candidates,
            num_best_actions=num_best_actions,
            temperature=temperature,
            render_tree=render_tree,
            logging_dir=logging_dir,
            policy_setup=policy_setup,
            action_scale=action_scale,
            verbose=False
        )
        
        
        
        self.task_description = None
        self.action_buffer = deque()  # Buffer to store trajectory actions
        self.trajectory_length = trajectory_length
    
    def reset(self, task_description: str) -> None:
        """
        Reset the model with a new task description.
        
        Args:
            task_description: The task description
        """
        self.task_description = task_description
        self.planner.reset(task_description)
        self.action_buffer.clear()  # Clear the action buffer on reset
    
    def step(self, image, task_description, current_env_name, action_list, env_reset_options, kwargs, additional_env_build_kwargs):
        """
        Take a step with the model.
        
        Args:
            image: The current image observation
            task_description: The task description
            current_env: The current environment (first simulator)
            
        Returns:
            raw_action: The raw action from the model
            action: The processed action
        """
        # Update task description if changed
        if task_description != self.task_description:
            self.task_description = task_description
            self.planner.reset(task_description)
            self.action_buffer.clear()  # Clear buffer when task changes
        
        # Check if buffer is empty
        if not self.action_buffer:
            # Get a new trajectory from the planner
            trajectory, reward = self.planner.plan_trajectory(
                current_env_name,
                action_list,
                env_reset_options,
                image, 
                task_description, 
                kwargs, 
                additional_env_build_kwargs,
                self.trajectory_length
            )
            
            print(f"Trajectory reward: {reward}")
            
            # Fill the buffer with the trajectory actions
            for action in trajectory:
                self.action_buffer.append(action)
        
        # Get the next action from the buffer
        best_action = self.action_buffer.popleft() if self.action_buffer else None
        
        # If buffer was empty and planner couldn't provide a trajectory, fall back to single action planning
        if best_action is None:
            best_action = self.planner.plan(current_env, image, task_description, kwargs, additional_env_build_kwargs)
        
        # Raw action would be needed for compatibility with the evaluation framework
        raw_action = {
            "world_vector": best_action["world_vector"],
            "rotation_delta": best_action.get("rotation_delta", np.zeros(3)),
            "open_gripper": best_action.get("open_gripper", np.zeros(1))
        }
        
        return raw_action, best_action
    
    def visualize_epoch(self, actions, images, save_path=None):
        """
        Visualize the epoch.
        
        Args:
            actions: List of actions
            images: List of images
            save_path: Path to save the visualization
        """
        # Delegate to the model's visualize_epoch method
        # This is needed for compatibility with the evaluation framework
        if hasattr(self.planner.model, 'visualize_epoch'):
            return self.planner.model.visualize_epoch(actions, images, save_path)