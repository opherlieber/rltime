""" Contains wrappers for atari environments (Mostly from OpenAI baselines) """
import gym
from .common import MaxAndSkipEnv, wrap_visual


class NoopResetEnv(gym.Wrapper):
    """Issues random noop actions at the start of each episode"""
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    """Takes action on reset for environments that are fixed until firing."""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.

    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few
            # frames so it's important to keep lives > 0, so that we only reset
            # once the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def wrap_atari(env, noop_actions=True, episodic_life=True, frame_skip=4,
               frame_skip_max=2, stack=4, warp_size=(84, 84, 1)):
    """Wraps an atari ENV

    Args:
        noop_actions: Whether to issue up to 30 noop actions on each new
            episode
        episodic_life: Whether to treat each life-loss as an end-of-episode
        frame_skip: Amount of frame-skipping to perform on each step
        frame_skip_max: Amount of frames to combine using 'max' for the
            obvservation during the frame_skip
        stack: Amount of observations to use for stacking (Usually this is 4
            for atari unless using an LSTM model in which case 1 is sometimes
            used)
        warp_size: The size to which to resize the observation image to,
            defaults to 84x84x1 (grayscale) which is the standard for atari
    """
    if noop_actions:
        env = NoopResetEnv(env, noop_max=30)
    if frame_skip and frame_skip > 1:
        env = MaxAndSkipEnv(
            env, skip=frame_skip, skip_max=frame_skip_max, skip_action=None)
    if episodic_life:
        env = EpisodicLifeEnv(env)

    # This handles ENVs which require pressing 'fire' to start after a new life
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    return wrap_visual(env, stack=stack, warp_size=warp_size)
