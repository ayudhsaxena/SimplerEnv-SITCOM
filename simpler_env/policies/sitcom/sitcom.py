from simpler_env.policies.sitcom.two_simulator_planner import TwoSimulatorPlanner
import numpy as np

class SITCOMInference:
    """
    Wrapper model that uses the TwoSimulatorPlanner for decision making.
    This class implements the same interface as the original model used in evaluation.
    """
    
    def __init__(
        self,
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
        """
        # Create the planner
        self.planner = TwoSimulatorPlanner(
            saved_model_path=saved_model_path,
            reward_function=reward_function,
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
        )
        
        self.task_description = None
    
    def reset(self, task_description: str) -> None:
        """
        Reset the model with a new task description.
        
        Args:
            task_description: The task description
        """
        self.task_description = task_description
        self.planner.reset(task_description)
    
    def step(self, image, task_description, current_env):
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
        
        # Use the planner to get the best action
        best_action = self.planner.plan(current_env, image, task_description)
        
        # Raw action would be needed for compatibility with the evaluation framework
        # For simplicity, we use the same action as both raw and processed
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