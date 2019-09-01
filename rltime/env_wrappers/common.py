import gym
import numpy as np
import cv2
import importlib
import os
import datetime
import pathlib

from rltime.general.backend import get_channel_axis
from gym.wrappers.monitoring.video_recorder import ImageEncoder
from gym.wrappers.time_limit import TimeLimit


class EpisodeTracker(gym.Wrapper):
    """Tracks the actual episode length/rewards before any processing such as
    'episodic life', clipping etc.., and passes it back in the 'info'"""
    def __init__(self, env):
        super(EpisodeTracker, self).__init__(env)

    def reset(self):
        self.reward = 0
        self.len = 0
        return self.env.reset()

    def step(self, action):
        o, r, d, info = self.env.step(action)
        self.reward += r
        self.len += 1
        if "episode_info" not in info:  # If not already handled by the ENV
            info = {
                **info,
                "episode_info": {
                    "reward": self.reward,
                    "length": self.len,
                    "done": d
                }
            }
        return o, r, d, info


class MaxAndSkipEnv(gym.Wrapper):
    """Implements frame-skipping common for atari and other visual envs"""
    def __init__(self, env, skip=4, skip_max=2, skip_action=None):
        """skip_action is the action to use when skipping.

        'None' means repeat the same action which was sent
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (skip_max,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self._max = skip_max
        self._skip_action = skip_action \
            if (skip_action is None or skip_action >= 0) \
            else (self.action_space.n + skip_action)
        assert(self._skip_action is None or
               self._skip_action < self.action_space.n)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i >= self._skip - self._max:
                self._obs_buffer[i-(self._skip - self._max)] = obs

            total_reward += reward
            if done:
                break
            if i == 0 and self._skip_action is not None:
                action = self._skip_action

        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        return self.env.reset()


class WarpFrame(gym.ObservationWrapper):
    """Warps the frame to specified size.

    Note that width and height axes are swapped, i.e. the input size is
    in the form (width, height, channels) while the observation shape will be
    (height, width, channels)
    """
    def __init__(self, env, size=(84, 84, 1)):
        gym.ObservationWrapper.__init__(self, env)
        if isinstance(size, list):
            size = tuple(size)
        self.size = size
        assert(len(size) == 3)
        assert(size[-1] in (1, 3)), "WarpFrame supports 1 or 3 channel output"

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(self.size[1], self.size[0], self.size[2]),
            dtype=np.uint8)

    def observation(self, frame):
        if self.size[-1] == 1:
            # Grayscale it
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        frame = cv2.resize(frame, self.size[:-1], interpolation=cv2.INTER_AREA)
        if self.size[-1] == 1:
            frame = np.expand_dims(frame, -1)
        return frame


class WindowedEnv(gym.Wrapper):
    """Implemens a frame-stacking wrapper

    Supports multiple input channels (e.g. RGB) and different input and output
    channel axes which by default will be auto-selected (For example swapped to
    0 for pytorch)
    """
    def __init__(self, env, window, in_channel_axis=-1, out_channel_axis=None):
        super(WindowedEnv, self).__init__(env)
        self.action_space = self.env.action_space
        self.in_channel_axis = in_channel_axis

        assert(in_channel_axis in [0, -1])
        obs_shape = env.observation_space.shape
        self.in_channels = obs_shape[in_channel_axis]
        self.main_shape = obs_shape[:-1] \
            if in_channel_axis == -1 \
            else obs_shape[1:]

        self.window = window
        # If not specified, output using the backends channel axis
        # (E.g. pytorch will be 0)
        self.out_channel_axis = out_channel_axis \
            if out_channel_axis is not None \
            else get_channel_axis()
        if self.out_channel_axis == 0:
            shape = (self.window*self.in_channels,) + self.main_shape
        elif self.out_channel_axis == -1:
            shape = self.main_shape + (self.window*self.in_channels,)
        else:
            assert(False)
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low.flat[0],
            high=self.env.observation_space.high.flat[0],
            shape=shape,
            dtype=self.env.observation_space.dtype)

    def make_state(self, s):
        # If the input and output channel axes don't match, need to swap them
        if self.in_channel_axis != self.out_channel_axis:
            if self.out_channel_axis == 0:
                s = np.transpose(
                    s, (len(s.shape)-1,) + tuple(range(len(s.shape)-1)))
            else:
                s = np.transpose(s, tuple(range(1, len(s.shape)))+(0,))
        # 'roll' the window based on the amount of channels (usually 1 or 3)
        # and place the new observation in the last/new slot
        self.value = np.roll(
            self.value, shift=-self.in_channels, axis=self.out_channel_axis)
        insert_pos = self.window*self.in_channels - self.in_channels
        if self.out_channel_axis == 0:
            self.value[insert_pos:, ...] = s
        elif self.out_channel_axis == -1:
            self.value[..., insert_pos:] = s
        else:
            assert(False)

        return self.value

    def reset(self):
        s = self.env.reset()
        self.value = np.zeros(self.observation_space.shape, dtype=s.dtype)
        return self.make_state(s)

    def step(self, a):
        s, r, d, p = self.env.step(a)
        s = self.make_state(s)
        return s, r, d, p


class DelayedEnding(gym.Wrapper):
    """Inserts an artificial delay to the end of episodes

    Delays the final reward and done indication by the given amount of delay
    steps (During this final delay period, the last observation is repeated
    and reward is 0, until the delay ends and the original final reward and
    done indication are given)
    """

    def __init__(self, env, delay):
        super(DelayedEnding, self).__init__(env)
        self.delay = delay

    def reset(self):
        self.delay_remain = 0
        return self.env.reset()

    def step(self, action):
        if self.delay_remain > 0:
            self.delay_remain -= 1
            if self.delay_remain == 0:
                return self.delayed_result
            else:
                return self.delayed_result[0], 0, False, {}

        o, r, d, i = self.env.step(action)
        if d and self.delay > 0:
            self.delay_remain = self.delay
            self.delayed_result = (o, r, d, i)
            d = False
            r = 0

        return o, r, d, i


class ExtraFeaturesEnvWrapper(gym.Wrapper):
    """Adds extra linear features to the observation space

    E.g.: Last reward, action and timestep. Typically for use as extra inputs
    to an LSTM or FC layer in the model.

    The new observation space will be a tuple-space with the 1st space being
    the original observation and the 2nd one being a 1D box space of the
    requested extra features.

    Actions will be one-hot vectorized if the action-space is discrete, and the
    absolute values if the action-space is box

    By default timestep scale is 1/100000, assuming Atari max-episode-steps of
    400K with frame-skip of 4 (So max 100K timesteps)
    This means the 'timestep' feature will be between [0,1] relative to the
    episode position out of the max steps.

    If 'use_real_done' is True, the real ENV done will be used for the timestep
    reset (And not 'simulated' dones like 'end episode on life loss')
    If clip_reward=True the reward values are clipped to [-1,0,1]. This is ONLY
    for the feature vector, the actual reward the ENV returns remains
    as-received.
    """
    def __init__(self, env, include_action=True, include_reward=True,
                 include_timestep=True, clip_reward=True,
                 timestep_scale=1./100000, use_real_done=True):
        gym.Wrapper.__init__(self, env)

        self._timestep_scale = timestep_scale
        self._include_action = include_action
        self._include_reward = include_reward
        self._include_timestep = include_timestep
        self._use_real_done = use_real_done
        self._is_real_done = True
        self._clip_reward = clip_reward
        extra_space = gym.spaces.Box(
            shape=(self._calc_size(),), dtype="float32",
            low=-np.inf, high=np.inf)
        base_space = self.env.observation_space
        # If space is already a tuple extend it to a longer tuple (instead
        # of nested tuple spaces)
        if not isinstance(base_space, gym.spaces.Tuple):
            base_space = (base_space,)
        self.observation_space = gym.spaces.Tuple(
            (*base_space, extra_space))

    def _calc_size(self):
        """Calculate the size/length of the feature vector"""
        size = 0
        if self._include_action:
            size += self.env.action_space.n \
                if isinstance(self.env.action_space, gym.spaces.Discrete) \
                else np.prod(self.env.action_space.shape)
        if self._include_reward:
            size += 1
        if self._include_timestep:
            size += 1
        return size

    def _make_features(self):
        """Generates the current feature vector"""
        vals = []
        if self._include_action:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                # One-hot vector of the last action
                action_vec = np.zeros(
                    (self.env.action_space.n,), dtype="float32")
                action_vec[self._last_action] = 1
            else:
                # Box-space, use the absolute aciton values reshaped to 1D
                action_vec = self._last_action.reshape(-1)
            vals.append(action_vec)

        if self._include_timestep:
            vals.append([self._timesteps_since_reset * self._timestep_scale])
        if self._include_reward:
            vals.append([self._last_reward])
        return np.concatenate(vals).astype('float32')

    def _make_obs(self, base_obs):
        """Creates the combined observation given the base observation"""

        # Support cases where obs is already a tuple, in which case we extend
        # the tuple otherwise create a new 2-tuple with original and extra
        if not isinstance(base_obs, tuple):
            base_obs = (base_obs,)
        return (*base_obs, self._make_features())

    def reset(self):
        if not self._use_real_done or self._is_real_done:
            self._timesteps_since_reset = 0
        self._is_real_done = False
        self._last_action = 0
        self._last_reward = 0

        return self._make_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._last_action = action
        self._last_reward = reward \
            if not self._clip_reward \
            else np.sign(reward)
        if self._use_real_done:
            assert('episode_info' in info), \
                "ExtraFeaturesEnvWrapper 'use_real_done' " \
                "requires EpisodeTracker in the ENV stack"
            self._is_real_done = info['episode_info']['done']

        self._timesteps_since_reset += 1
        return (self._make_obs(obs), reward, done, info)


class EpisodeRecorder(gym.Wrapper):
    """Wrapper for recording episodes to MP4 or jpeg-frames"""
    def __init__(self, env, path, fps=60, header=None):
        """Initialize the wrapper

        Args:
            env: The env to wrap and record
            path: The location where to place recordings
            fps: The FPS to encode the MP4 at
            header: Optional prefix for the output file name
        """
        super(EpisodeRecorder, self).__init__(env)
        self.fps = fps
        self.path = path
        self.header = header
        self.encoder = None
        os.makedirs(path, exist_ok=True)

    def _get_output_name(self):
        """Gets a unique name for the output video, starting from 00000.mp4

        (Handles collisions with parallel ENVs)
        """
        base_name = self.header + "_" if self.header is not None else ""
        i = 0
        while True:
            try:
                path = os.path.join(
                    self.path, f"{base_name}{str(i).zfill(5)}.mp4")
                pathlib.Path(path).touch(exist_ok=False)
                return path
            except FileExistsError:
                i += 1
                continue

    def _init_new(self, shape):
        self._close_last()
        self.cur_path = self._get_output_name()
        self.encoder = ImageEncoder(self.cur_path, shape, self.fps)

    def _dump_frame(self, frame):
        self.encoder.capture_frame(frame)

    def _close_last(self):
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None

    def close(self):
        self._close_last()
        self.env.close()

    def reset(self):
        obs = self.env.reset()
        # Create a new output
        self._init_new(obs.shape)

        self._dump_frame(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._dump_frame(obs)

        return obs, reward, done, info


def wrap_visual(env, stack=4, warp_size=None):
    """General wrapper for any visual environment

    Does frame-stacking and optional warp/resize.
    Also ensures the output channel axis is correct according to the backend
    being used.
    """
    if warp_size is not None:
        env = WarpFrame(env, size=warp_size)

    # Note we need to wrap this also if stack=1 to ensure the correct output
    # channel axis is used (For example pytorch needs it swapped to axis=0)
    # Output channel will be auto-detected by 'WindowedEnv' class
    env = WindowedEnv(env, stack, in_channel_axis=-1)
    return env


def make_env_creator(env_type, wrappers=[], imports=[],
                     max_episode_steps=None, **kwargs):
    """Returns a function for creating a given ENV

    Args:
        env_type: The ENV to use. If it's a string it creates a registered GYM
            env, otherwise it should be a python callable used to create the
            ENV (e.g. a python function or class)
        wrappers: A list of wrappers to apply to the ENV. Each wrapper should
            have the format {'type' : type, 'args' : {}} where 'type' is a
            callable (e.g. function or class) and args are optional arguments
            to the callable (In addition to the env being wrapped)
        imports: Python modules to import before creating the ENV (e.g.
            'gym_ple', 'retro'), for example these modules may register the ENV
            with gym
        max_episode_steps: If set configures/overrides the max steps for each
            episode
        kwargs: Additional args which will be passed to the env creation itself
            (Valid only if env_type is a callable and not a string)

    Returns: A function which creates a new instance of the ENV
    """
    def create():
        for import_name in imports:
            importlib.import_module(import_name)
        if isinstance(env_type, str):
            assert(not kwargs), "ENV kwargs not supported for gym envs"
            env = gym.make(env_type)
        elif callable(env_type):
            env = env_type(**kwargs)
        else:
            raise ValueError(
                "make_env_creator() expected either a string or "
                f"callable for 'env_type', got {type(env_type)}")

        # Limit the max steps per episode if requested
        if max_episode_steps is not None:
            if hasattr(env, "_max_episode_steps"):
                # Use the '_max_episode_steps' if available from dym. This is
                # to allow increasing the limit for example in cartpole.
                # (The TimeLimit option can only decrease the limit)
                env._max_episode_steps = max_episode_steps
            else:
                env = TimeLimit(env, max_episode_steps)

        # Always begin with EpisodeTracker so that the training gets the real
        # rewards/dones before any additional wrappers process them
        env = EpisodeTracker(env)

        # Apply all requested wrappers
        for wrapper in wrappers:
            wrapper_type = wrapper.get("type")
            wrapper_args = wrapper.get("args", {})
            env = wrapper_type(env, **wrapper_args)
        return env
    return create
