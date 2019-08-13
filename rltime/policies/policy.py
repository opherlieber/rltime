from rltime.general.type_registry import get_registered_type


class Policy:
    """Base class for any policy implementation

    These are the minimal functions needed for the policy to work with the
    policy trainer and actors
    """

    def actor_predict(self, state, timesteps=1):
        """Performs a prediction for acting on the given input state
        (i.e. 'action selection')

        Returns a dictionay containing at least an 'actions' vector matching
        the input batch size, as well as any other outputs the training may
        need later.
        Acting prediction includes all postprocessing required for action
        selection (e.g. argmax, sampling from distribution etc...)
        """
        raise NotImplementedError

    def get_creator(self, cuda=False):
        """Returns a function that creates a copy of this policy"""
        raise NotImplementedError

    def get_state(self):
        """Returns a copy of current state/weights of this policy

        Can be in any (picklable) format that can later be passed to
        load_state() of this policy or a copy of it
        """
        raise NotImplementedError

    def load_state(self):
        """Loads the state/weights previously returned by get_state()"""
        raise NotImplementedError

    def is_recurrent(self):
        """Returns whether this policy has any recurrent layers"""
        raise NotImplementedError

    def make_state(self, inp, initials):
        """Makes the model input state from the given input/observation

        Args:
            inp: The input observation (Batched on axis=0), which matches the
                observation space used to initialize the policy/model
            initials: Boolean indication for each batch item, signifying if
                it's the initial/starting observation of a new episode

        Returns: The full policy input state, including the obervation itself
            and any additional relevant model inputs such as RNN hidden state
        """
        raise NotImplementedError

    def _create_model_from_config(self, config, observation_space):
        """Creates a model from the given model-config (Dictionary containing
        the 'type' and 'args')"""
        model_cls = get_registered_type("models", config.get("type"))
        return model_cls(
            observation_space=observation_space, **config.get("args"))

    def get_state_store(self, is_async_storage):
        """Returns an optional state store for storing and stacking states

        Args:
            is_async_storage: Whether the state storage and loading is in a
                separate process than the training process
        """
        return None
