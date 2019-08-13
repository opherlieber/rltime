class ExplorationManager():
    """Base class for exploration managers"""

    def __init__(self, total_actors=None):
        self.total_actors = total_actors

    def remap_actions(self, actions, actor_indices, action_space, progress):
        """Remaps a set of actions based on the specific exploration policy

        Args:
            actions: The actions (vector, 1 value for each env)
            actor_indices: Actor indices for each actor, should match <actions>
                in size
            action_space: The action space for the ENV
            progress: The training progress in [0,1] range

        Returns: (actions, stats dict)
        """
        raise NotImplementedError
