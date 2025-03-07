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
        Initialize the planning model.
        
        Args:
            env: The environment (first simulator)
            model: The dynamics model (second simulator)
            openvla_model: The OpenVLA model
            reward_function: Function to compute reward
            num_initial_actions: Number of initial actions to sample (A)
            horizon_per_action: Number of actions to consider for each state (Horizon)
            num_steps_ahead: Number of simulation steps to look ahead (h)
            num_candidates: Number of candidate actions to sample
            num_best_actions: Number of best actions to select
            temperature: Temperature for sampling
            render_tree: Whether to render the tree
            logging_dir: Directory for logging
        """
        
        # Create default reward function if not provided
        if reward_function is None:
            def default_reward_function(state, action=None):
                # Example reward function based on distance
                # You'll need to implement this based on your specific environment
                gripper_pos = state.get("gripper_position", None)
                object_pos = state.get("object_position", None)
                plate_pos = state.get("plate_position", None)
                
                if gripper_pos is None or object_pos is None:
                    return 0.0
                
                # Calculate distance between gripper and object
                distance = np.linalg.norm(gripper_pos - object_pos)
                
                # If object is grabbed, measure distance to plate
                is_grabbed = state.get("is_grabbed", False)
                if is_grabbed and plate_pos is not None:
                    distance = np.linalg.norm(gripper_pos - plate_pos)
                
                # Convert distance to reward (closer is better)
                return -distance
            
            reward_function = default_reward_function
        
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

    
    def reset(self, task_description: str) -> None:
        """
        Reset the model with a new task description.
        
            task_description: The task description
        """
        self.task_description = task_description
        self.planner.reset()
    
    def step(self, image, task_description, current_env):
        """
        Take a step with the model.
        
        Args:
            image: The current image observation
            task_description: The task description
            override_action: Optional action to override planning
            
        Returns:
            raw_action: The raw action from the model
            action: The processed action
        """
        # Update task description if changed
        if task_description != self.task_description:
            self.task_description = task_description
      
        best_action = self.planner.plan(current_env, image, task_description=task_description)

        return best_action
    
    def visualize_epoch(self, actions, images, save_path=None):
        """
        Visualize the epoch.
        
        Args:
            actions: List of actions
            images: List of images
            save_path: Path to save the visualization
        """
        # Delegate to the original model
        return self.model.visualize_epoch(actions, images, save_path)