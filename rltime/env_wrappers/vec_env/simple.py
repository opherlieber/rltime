from .common import VecEnv, stack_vec_obs
import numpy as np


class SimpleVecEnv(VecEnv):
    """A simple vec-env not using sub-processes.

    Can be faster then the 'SubProc' version for simple envs like cartpole
    """
    def __init__(self, envs):
        VecEnv.__init__(
          self, len(envs), envs[0].observation_space, envs[0].action_space)
        self.envs = envs
        self.name = envs[0].name if hasattr(envs[0],'name') else "noname"

    def reset(self):
        obs = []
        for env in self.envs:
            obs += [env.reset()]
        return stack_vec_obs(obs)

    def step_async(self, actions):
        results = []

        for (env, a) in zip(self.envs, actions):
            o, r, d, i = env.step(a)
            if d:
                o = env.reset()
            results += [(o, r, d, i)]
        obs, rews, dones, infos = zip(*results)
        stacked_obs = stack_vec_obs(obs)
        self.results = stacked_obs, np.stack(rews), np.stack(dones), infos

    def render(self, mode='human'):
        for env in self.envs:
            env.render(mode=mode)

    def step_wait(self):
        return self.results

    def close(self):
        """Clean up the environments' resources."""
        for env in self.envs:
            env.close()


def make_simple_vec_env(creator, num_env, base_seed=0):
    """Creates a simple vec-env with the given env-creator and amount of
    envs"""
    def make_env(rank):
        env = creator()
        env.seed(base_seed + rank)
        return env
    return SimpleVecEnv([make_env(i) for i in range(num_env)])