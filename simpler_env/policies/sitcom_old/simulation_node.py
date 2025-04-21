class SimulationNode:
    """Node in the simulation tree."""
    
    def __init__(self, state, image, parent=None, action=None, reward=0, depth=0):
        self.state = state  # Environment state
        self.image = image  # Image observation
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.children = []  # Child nodes
        self.reward = reward  # Reward at this node
        self.depth = depth  # Depth in the tree
        self.original_action_idx = -1  # Index of the original action (for backtracking)
        
    def add_child(self, child):
        """Add a child node to this node."""
        self.children.append(child)
        return child