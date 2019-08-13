from .acting_interface import ActingInterface


class ActorWrapper(ActingInterface):
    """Wrapper for a created actor

    Allows overriding only specific actor methods while passing through the
    rest, similar to gym wrappers
    """

    def __init__(self, actor):
        super().__init__(*actor.get_spaces())
        self._actor = actor

    def get_samples(self, min_samples):
        return self._actor.get_samples(min_samples)

    def get_env_count(self):
        return self._actor.get_env_count()

    def set_actor_policy(self, actor_policy):
        return self._actor.set_actor_policy(actor_policy)

    def update_state(self, progress, policy_state=None):
        return self._actor.update_state(progress, policy_state)

    def close(self):
        return self._actor.close()
