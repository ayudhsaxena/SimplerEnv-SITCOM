from two_simulator_planner import TwoSimulatorPlanner

class PlanningModel:
    """
    Wrapper model that uses the TwoSimulatorPlanner for decision making.
    This class implements the same interface as the original model used in evaluation.
    """
    
    def __init__(
        self,
        env,
        model,
        openvla_model,
        reward_function=None,
        num_initial_actions=10,
        horizon_per_action=5,
        num_steps_ahead=3,
        num_candidates=5,
        num_best_actions=3,
        temperature=1.0,
        render_tree=False,
        logging_dir="./results/planning",
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
        self.env = env
        self.model = model
        self.openvla_model = openvla_model
        
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
            env=env,
            model=model,
            openvla_model=openvla_model,
            task_description=None,  # Will be set in reset
            reward_function=reward_function,
            num_initial_actions=num_initial_actions,
            horizon_per_action=horizon_per_action,
            num_steps_ahead=num_steps_ahead,
            num_candidates=num_candidates,
            num_best_actions=num_best_actions,
            temperature=temperature,
            render_tree=render_tree,
            logging_dir=logging_dir,
        )
        
        # Additional state
        self.task_description = None
        self.last_action = None
    
    def reset(self, task_description):
        """
        Reset the model with a new task description.
        
        Args:
            task_description: The task description
        """
        self.task_description = task_description
        self.planner.task_description = task_description
        self.planner.reset()
        self.model.reset(task_description)
        self.last_action = None
    
    def step(self, image, task_description, override_action=None):
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
            self.planner.task_description = task_description
        
        # Get the observation from the image
        # This assumes your environment can construct an observation from an image
        # You may need to adapt this to your specific environment
        obs = self.env.get_observation_from_image(image)
        
        # Plan the best action or use override
        if override_action is not None:
            best_action = override_action
        else:
            best_action = self.planner.plan(obs, task_description)
        
        # Process the action through the model to get the correct format
        raw_action, action = self.model.step(image, task_description, override_action=best_action)
        
        self.last_action = action
        return raw_action, action
    
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