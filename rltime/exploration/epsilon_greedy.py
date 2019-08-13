import numpy as np
from .base import ExplorationManager


class EpsilonGreedyExplorationManager(ExplorationManager):
    """Epsilon greedy exploration

    Remaps to a random action with epsilon probability, decayed from a base
    value to a minimum up to the requested exploration fraction
    (Out of total training), with optional per-actor exponent
    """

    def __init__(self, eps_final, exploration_fraction, eps_prob=1.0,
                 eps_start=1.0, eps_min=0., per_actor_exponent_factor=0,
                 **kwargs):
        """Initialize the exploration manager

        Args:
            eps_final: The final epsilon value to decay to (The final base
                value if using per-actor exponent). Can also be a list of
                values in which case <eps_prob> must be a list of the same size
                summing up to 1.0 and defines the probability to use each
                eps_final value. This allows to achieve exploration as in the
                A3C paper with different minimums.
            exploration_fraction: The fraction of the training period in [0,1]
                over which to decay the epsilon value
            eps_prob: Required only if eps_final is a list and not a single
                value, in this case this is also a list of the same size
                summing up to 1.0
            eps_start: The starting epsilon value at time=0
            eps_min: The absolute minimal epsilon value to use, even if
                per-actor exponentiation resulted in a lower value.
            per_actor_exponent_factor: If >0 uses an exponentially lowered
                epsilon per actor as in the APEX paper (In this case
                total_actors must be initialized, greater than 1, and
                actor_indices must be passed in to remap_actions()).
                In this case typically eps_start can equal eps_final (In the
                APEX paper it's eps_start=eps_final=0.4, with
                per_actor_exponent_factor=7), or eps_final can be also be lower
                to ensure more actors follow policy more often towards the end
                of training
        """
        super().__init__(**kwargs)
        self.eps_start = eps_start
        self.eps_probs = [eps_prob] \
            if not isinstance(eps_prob, list) \
            else eps_prob
        self.eps_finals = [eps_final] \
            if not isinstance(eps_final, list) \
            else eps_final
        self.exploration_fraction = exploration_fraction
        self.eps_min = eps_min
        self.per_actor_exponent_factor = per_actor_exponent_factor

        if per_actor_exponent_factor > 0:
            assert(self.total_actors is not None), \
                "EpsilonGreedyExplorationManager per_actor_exponent_factor " \
                "requires the total_actors to be passed in"
            assert(self.total_actors > 1), \
                "EpsilonGreedyExplorationManager per_actor_exponent_factor " \
                "requires at least 2 actors"
        assert(exploration_fraction <= 1.0)

    def _get_eps(self, progress):
        """Returns current epsilon to use based on the training progress"""
        # If we a got a list of final values to use, sample from them
        index = np.random.choice(range(len(self.eps_probs)), p=self.eps_probs)
        min_val = self.eps_finals[index]
        if progress >= self.exploration_fraction:
            return min_val
        fraction = (progress/self.exploration_fraction)
        return self.eps_start - fraction*(self.eps_start - min_val)

    def remap_actions(self, actions, actor_indices, action_space, progress):
        eps = self._get_eps(progress)
        actuals = []  # The actual eps used for each actor, for logging
        for i in range(len(actions)):
            if not self.per_actor_exponent_factor:
                actual_eps = eps
            else:
                actor_index = actor_indices[i]
                assert(actor_index < self.total_actors), \
                    "EpsilonGreedyExplorationManager got an out of range " \
                    f"({self.total_actors}) actor index: {actor_index}"
                actual_eps = eps ** (
                    1 + ((actor_index / (self.total_actors-1)) *
                         self.per_actor_exponent_factor))
            if actual_eps < self.eps_min:
                actual_eps = self.eps_min

            # Remap the action to a random one with actual_eps probability
            if np.random.rand() < actual_eps:
                # TODO: This should probably be action_space.sample() to
                # support also non-discrete, need to check performance before
                # changing
                actions[i] = np.random.randint(0, action_space.n)
            actuals.append(actual_eps)
        # Return the (possibly) remapped actions, as well as the epsilon values
        # used (For logging purposes only)
        return actions, {"eps": np.array(actuals)}
